""" This module is responsible for evaluating rendering, trajectory and reconstruction metrics"""
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torchvision
import json
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image
from tqdm import tqdm

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.evaluation.evaluate_merged_map import (RenderFrames, merge_submaps,
                                                refine_global_map)
from src.evaluation.evaluate_reconstruction import evaluate_reconstruction, clean_mesh
from src.evaluation.evaluate_trajectory import evaluate_trajectory
from src.utils.io_utils import load_config, save_dict_to_json, log_metrics_to_wandb
from src.utils.mapper_utils import calc_psnr, create_point_cloud
from src.utils.utils import (get_render_settings, np2torch,
                             render_gaussian_model, setup_seed,
                             torch2np, filter_depth_outliers)


def compute_ause(error_map, uncertainty_map, n_bins=100):
    """
    Compute Area Under the Sparsification Error (AUSE) with mathematical rigor.
    """
    error_flat = error_map.flatten()
    unc_flat = uncertainty_map.flatten()
    n_pixels = error_flat.numel()

    # Fractions tensor (0.0 to 1.0 inclusive for exact endpoints)
    fractions = torch.linspace(0.0, 1.0, n_bins + 1, device=error_flat.device)
    
    # Sort descending
    oracle_order = torch.argsort(error_flat, descending=True)
    pred_order = torch.argsort(unc_flat, descending=True)
    
    oracle_cumsum = torch.cumsum(error_flat[oracle_order], dim=0)
    pred_cumsum = torch.cumsum(error_flat[pred_order], dim=0)

    total_error = oracle_cumsum[-1]
    
    oracle_curve = torch.zeros_like(fractions)
    pred_curve = torch.zeros_like(fractions)
    
    for i, frac in enumerate(fractions):
        n_remove = int(frac.item() * n_pixels)
        n_remain = n_pixels - n_remove
        
        # Prevent division by zero at 100% removal
        if n_remain == 0:
            oracle_curve[i] = 0.0
            pred_curve[i] = 0.0
            break
            
        if n_remove > 0:
            oracle_remaining = (total_error - oracle_cumsum[n_remove - 1]) / n_remain
            pred_remaining = (total_error - pred_cumsum[n_remove - 1]) / n_remain
        else:
            oracle_remaining = total_error / n_remain
            pred_remaining = total_error / n_remain
            
        oracle_curve[i] = oracle_remaining
        pred_curve[i] = pred_remaining

    # Use PyTorch's native trapezoidal integration for mathematical correctness
    diff = pred_curve - oracle_curve
    ause = torch.trapz(diff, x=fractions).item()
    
    return ause


class Evaluator(object):

    def __init__(self, checkpoint_path, config_path, config=None, save_render=False, novel_view_render=False) -> None:
        self.novel_view_render = novel_view_render
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config
        setup_seed(self.config["seed"])

        self.checkpoint_path = Path(checkpoint_path)
        self.use_wandb = self.config["use_wandb"]
        self.device = "cuda"
        self.dataset = get_dataset(self.config["dataset_name"])({**self.config["data"], **self.config["cam"]})
        self.scene_name = self.config["data"]["scene_name"]
        self.dataset_name = self.config["dataset_name"]
        self.gt_poses = np.array(self.dataset.poses)
        self.width, self.height = self.dataset.width, self.dataset.height
        self.save_render = save_render
        if self.save_render:
            self.render_path = self.checkpoint_path / "rendered_imgs"
            self.render_path.mkdir(exist_ok=True, parents=True)

        self.estimated_c2w = torch2np(torch.load(self.checkpoint_path / "estimated_c2w.ckpt", map_location=self.device))
        self.submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*')))
        self.exposures_ab = None
        if (self.checkpoint_path / "exposures_ab.ckpt").exists():
            self.exposures_ab = torch2np(torch.load(self.checkpoint_path / "exposures_ab.ckpt", map_location=self.device))
            print(f"Loaded trained exposures paramters for scene {self.scene_name}")
        print('estimated_c2w.shape',self.estimated_c2w.shape)
    def run_trajectory_eval(self):
        """ Evaluates the estimated trajectory """
        print("Running trajectory evaluation...")
        evaluate_trajectory(self.estimated_c2w, self.gt_poses, self.checkpoint_path)

    def run_rendering_eval(self):
        """ Renders the submaps and global splats and evaluates the PSNR, LPIPS, SSIM, depth L1 and AUSE metrics."""
        print("Running rendering evaluation...")
        psnr, lpips, ssim, depth_l1 = [], [], [], []
        ause_color_list, ause_depth_list = [], []
        color_transform = torchvision.transforms.ToTensor()
        lpips_model = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', normalize=True).to(self.device)
        opt_settings = OptimizationParams(ArgumentParser(
            description="Training script parameters"))

        submaps_paths = sorted(
            list((self.checkpoint_path / "submaps").glob('*.ckpt')))
        for submap_path in tqdm(submaps_paths):
            submap = torch.load(submap_path, map_location=self.device)
            gaussian_model = GaussianModel()
            gaussian_model.training_setup(opt_settings)
            gaussian_model.restore_from_params(
                submap["gaussian_params"], opt_settings)

            for keyframe_id in submap["submap_keyframes"]:

                _, gt_color, gt_depth, _ = self.dataset[keyframe_id]
                gt_color = color_transform(gt_color).to(self.device)
                gt_depth = np2torch(gt_depth).to(self.device)

                estimate_c2w = self.estimated_c2w[keyframe_id]
                estimate_w2c = np.linalg.inv(estimate_c2w)
                render_dict = render_gaussian_model(
                    gaussian_model, get_render_settings(self.width, self.height, None, estimate_w2c),
                    return_variance=True)
                rendered_color, rendered_depth = render_dict["color"].detach(
                ), render_dict["depth"].detach()
                rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)
                if self.save_render:
                    torchvision.utils.save_image(
                        rendered_color, self.render_path / f"{keyframe_id:05d}.png")

                mse_loss = torch.nn.functional.mse_loss(
                    rendered_color, gt_color)
                psnr_value = (-10. * torch.log10(mse_loss)).item()
                lpips_value = lpips_model(
                    rendered_color[None], gt_color[None]).item()
                ssim_value = ms_ssim(
                    rendered_color[None], gt_color[None], data_range=1.0, size_average=True).item()
                depth_l1_value = torch.abs(
                    (rendered_depth - gt_depth)).mean().item()

                # AUSE computation
                color_error = (rendered_color - gt_color).pow(2).mean(dim=0)  # [H, W]
                color_uncertainty = render_dict["render_color_variance"].detach().mean(dim=0)  # [H, W]
                ause_color_list.append(compute_ause(color_error, color_uncertainty))

                depth_error = (rendered_depth - gt_depth).abs()  # [H, W]
                depth_uncertainty = render_dict["render_depth_variance"].detach().squeeze(0)  # [H, W]
                ause_depth_list.append(compute_ause(depth_error, depth_uncertainty))

                psnr.append(psnr_value)
                lpips.append(lpips_value)
                ssim.append(ssim_value)
                depth_l1.append(depth_l1_value)

        num_frames = len(psnr)
        metrics = {
            "psnr": sum(psnr) / num_frames,
            "lpips": sum(lpips) / num_frames,
            "ssim": sum(ssim) / num_frames,
            "depth_l1_train_view": sum(depth_l1) / num_frames,
            "ause_color": sum(ause_color_list) / num_frames,
            "ause_depth": sum(ause_depth_list) / num_frames,
            "num_renders": num_frames
        }
        save_dict_to_json(metrics, "rendering_metrics.json",
                          directory=self.checkpoint_path)

        x = list(range(len(psnr)))
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        axs[0].plot(x, psnr, label="PSNR")
        axs[0].legend()
        axs[0].set_title("PSNR")
        axs[1].plot(x, ssim, label="SSIM")
        axs[1].legend()
        axs[1].set_title("SSIM")
        axs[2].plot(x, depth_l1, label="Depth L1 (Train view)")
        axs[2].legend()
        axs[2].set_title("Depth L1 Render")
        axs[3].plot(x, ause_color_list, label="AUSE Color")
        axs[3].legend()
        axs[3].set_title("AUSE Color")
        axs[4].plot(x, ause_depth_list, label="AUSE Depth")
        axs[4].legend()
        axs[4].set_title("AUSE Depth")
        plt.tight_layout()
        plt.savefig(str(self.checkpoint_path /
                    "rendering_metrics.png"), dpi=300)
        print(metrics)

    def run_reconstruction_eval(self):
        """Fuse rendered panoramas into a point cloud and evaluate reconstruction.

        Perspective TSDF integration assumes a pinhole model, which does not
        apply to equirectangular panoramas.  Instead every keyframe's rendered
        omnidirectional RGB-D is unprojected with the same spherical convention
        used by the rasterizer, accumulated in world space, and (optionally)
        converted to a Poisson mesh.
        """
        print("Running reconstruction evaluation...")

        mesh_dir = self.checkpoint_path / "mesh"
        mesh_dir.mkdir(exist_ok=True, parents=True)
        opt_settings = OptimizationParams(ArgumentParser(
            description="Training script parameters"))

        submaps_paths = sorted(
            list((self.checkpoint_path / "submaps").glob('*.ckpt')))
        all_points = []
        for submap_path in tqdm(submaps_paths, desc="Fusing submaps"):
            submap = torch.load(submap_path, map_location=self.device)
            gaussian_model = GaussianModel()
            gaussian_model.training_setup(opt_settings)
            gaussian_model.restore_from_params(
                submap["gaussian_params"], opt_settings)
            print(gaussian_model.get_size())

            for keyframe_id in submap["submap_keyframes"]:
                estimate_c2w = np.asarray(self.estimated_c2w[keyframe_id])
                estimate_w2c = np.linalg.inv(estimate_c2w)
                render_dict = render_gaussian_model(
                    gaussian_model,
                    get_render_settings(self.width, self.height, None,
                                        estimate_w2c))

                rendered_color = torch.clamp(
                    render_dict["color"].detach(), 0.0, 1.0)
                rendered_color = torch2np(rendered_color.permute(1, 2, 0))
                rendered_depth = torch2np(render_dict["depth"].detach())
                rendered_alpha = torch2np(render_dict["alpha"].detach())

                valid = (
                    np.isfinite(rendered_depth)
                    & (rendered_depth > 0)
                    & (rendered_alpha > 0.5))
                if not np.any(valid):
                    continue

                points = create_point_cloud(
                    rendered_color, rendered_depth, None, estimate_c2w)
                all_points.append(points[valid.reshape(-1)])

        if not all_points:
            print("No valid rendered depth found; skipping reconstruction.")
            return

        points = np.concatenate(all_points, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6].astype(np.float64))
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        pc_path = mesh_dir / "reconstructed_pointcloud.ply"
        o3d.io.write_point_cloud(str(pc_path), pcd)
        print(f"Reconstructed point cloud saved to {pc_path}")

        file_name = pc_path
        try:
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9)
            densities = np.asarray(densities)
            if densities.size:
                mesh.remove_vertices_by_mask(
                    densities < np.quantile(densities, 0.05))
            mesh = clean_mesh(mesh)
            mesh_path = mesh_dir / "cleaned_mesh.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), mesh)
            print(f"Reconstructed mesh saved to {mesh_path}")
            file_name = mesh_path
        except Exception as exc:  # meshing is optional; keep the point cloud
            print(f"Poisson meshing failed ({exc}); keeping point cloud only.")

        recon_cfg = self.config.get("reconstruction", {})
        gt_mesh_path = recon_cfg.get(
            "gt_mesh_path", self.config["data"].get("gt_mesh_path"))
        gt_pc_path = recon_cfg.get(
            "gt_pc_path", self.config["data"].get("gt_pc_path"))
        if (file_name != pc_path and gt_mesh_path and gt_pc_path
                and Path(gt_mesh_path).exists() and Path(gt_pc_path).exists()):
            evaluate_reconstruction(file_name, gt_mesh_path, gt_pc_path,
                                    self.checkpoint_path)
        else:
            print("No ground-truth reconstruction paths configured; "
                  "skipping mesh metrics.")


    def run_global_map_eval(self, init_from='mesh'):
        """ Merges the map, evaluates it over training and novel views 
        
        Args:
            init_from (str, optional): 'mesh' or 'splats'. Initialization method for the global splats. Defaults to mesh vertices reconstructed before.
        """
        print("Running global map evaluation...")

        training_frames = RenderFrames(
            self.dataset, self.estimated_c2w, self.height, self.width, self.exposures_ab)
        training_frames = DataLoader(
            training_frames, batch_size=1, shuffle=True)
        len_frames = len(training_frames)
        training_frames = cycle(training_frames)
        merged_cloud = merge_submaps(self.submaps_paths) if init_from == 'splats' else None

        # Omnidirectional cameras have no pinhole intrinsics; TSDF export is
        # therefore disabled and the mesh is built from the point cloud.
        refined_merged_gaussian_model = refine_global_map(merged_cloud, training_frames, 30000, export_refine_mesh=False,
                                                          output_dir=self.checkpoint_path, len_frames=len_frames, 
                                                          o3d_intrinsic=None)
        ply_path = self.checkpoint_path / \
            f"{self.config['data']['scene_name']}_global_splats.ply"
        refined_merged_gaussian_model.save_ply(ply_path)
        print(f"Refined global splats saved to {ply_path}")

        if self.config["dataset_name"] == "scannetpp":
            # "NVS evaluation only supported for scannetpp"

            eval_config = deepcopy(self.config)
            print(f"✨ Eval NVS for scene {self.config['data']['scene_name']}...")
            (self.checkpoint_path / "nvs_eval").mkdir(exist_ok=True, parents=True)
            eval_config["data"]["use_train_split"] = False
            test_set = get_dataset(eval_config["dataset_name"])({**eval_config["data"], **eval_config["cam"]})
            test_poses = torch.stack([torch.from_numpy(test_set[i][3]) for i in range(len(test_set))], dim=0)
            test_frames = RenderFrames(test_set, test_poses, self.height, self.width)

            psnr_list = []
            for i in tqdm(range(len(test_set))):
                gt_color, _, render_settings = (
                    test_frames[i]["color"],
                    test_frames[i]["depth"],
                    test_frames[i]["render_settings"])
                render_dict = render_gaussian_model(refined_merged_gaussian_model, render_settings)
                rendered_color, _ = (render_dict["color"].permute(1, 2, 0), render_dict["depth"],)
                rendered_color = torch.clip(rendered_color, 0, 1)
                save_image(rendered_color.permute(2, 0, 1), self.checkpoint_path / f"nvs_eval/{i:04d}.jpg")
                psnr = calc_psnr(gt_color, rendered_color)
                psnr_list.append(psnr.item())
            print(f"PSNR List: {psnr_list}")
            print(f"Avg. NVS PSNR: {np.array(psnr_list).mean()}")
            with open(self.checkpoint_path / 'nvs_eval' / "results.json", "w") as f:
                data = {"avg_nvs_psnr": np.mean(psnr_list)}
                json.dump(data, f, indent=4)
        
        else: # evaluate rendering performance on the global submap
            print("Running rendering evaluation on global map ...")
            psnr, lpips, ssim, depth_l1 = [], [], [], []
            ause_color_list, ause_depth_list = [], []
            color_transform = torchvision.transforms.ToTensor()
            lpips_model = LearnedPerceptualImagePatchSimilarity(
                net_type='alex', normalize=True).to(self.device)

            submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*.ckpt')))

            if self.novel_view_render:
                # Render every frame in the dataset with the merged global Gaussian model.
                frame_iter = tqdm(range(len(self.dataset)))
            else:
                # Default: only keyframes collected across all submaps.
                kf_ids = []
                for sp in submaps_paths:
                    s = torch.load(sp, map_location=self.device)
                    kf_ids.extend(s["submap_keyframes"])
                frame_iter = tqdm(kf_ids)

            render_image_dir = self.checkpoint_path / "render_image"
            render_depth_dir = self.checkpoint_path / "render_depth"
            render_img_uncert_dir = self.checkpoint_path / "render_img_uncert"
            render_depth_uncert_dir = self.checkpoint_path / "render_depth_uncert"
            gt_depth_uncert_dir = self.checkpoint_path / "gt_depth_uncert"
            gt_color_uncert_dir = self.checkpoint_path / "gt_color_uncert"
            for d in (render_image_dir, render_depth_dir,
                      render_img_uncert_dir, render_depth_uncert_dir, gt_depth_uncert_dir, gt_color_uncert_dir):
                d.mkdir(exist_ok=True, parents=True)

            for keyframe_id in frame_iter:

                    _, gt_color, gt_depth, _ = self.dataset[keyframe_id]
                    gt_color = color_transform(gt_color).to(self.device)
                    gt_depth = np2torch(gt_depth).to(self.device)

                    estimate_c2w = self.estimated_c2w[keyframe_id]
                    estimate_w2c = np.linalg.inv(estimate_c2w)
                    render_dict = render_gaussian_model(
                        refined_merged_gaussian_model, get_render_settings(self.width, self.height, None, estimate_w2c),
                        return_variance=True)
                    rendered_color, rendered_depth = render_dict["color"].detach(
                    ), render_dict["depth"].detach()
                    rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)
                    if self.save_render:
                        torchvision.utils.save_image(
                            rendered_color, self.render_path / f"{keyframe_id:05d}.png")

                    mse_loss = torch.nn.functional.mse_loss(
                        rendered_color, gt_color)
                    psnr_value = (-10. * torch.log10(mse_loss)).item()
                    lpips_value = lpips_model(
                        rendered_color[None], gt_color[None]).item()
                    ssim_value = ms_ssim(
                        rendered_color[None], gt_color[None], data_range=1.0, size_average=True).item()
                    depth_l1_value = torch.abs(
                        (rendered_depth - gt_depth)).mean().item()

                    # AUSE computation
                    color_error = (rendered_color - gt_color).abs().mean(dim=0)
                    color_uncertainty = render_dict["render_color_variance"].detach().mean(dim=0)
                    ause_color_list.append(compute_ause(color_error, color_uncertainty))

                    depth_error = (rendered_depth - gt_depth).abs()
                    depth_uncertainty = render_dict["render_depth_variance"].detach().squeeze(0)
                    ause_depth_list.append(compute_ause(depth_error, depth_uncertainty))

                    # Persist per-frame renders + uncertainty maps for offline AUSE/MAE plotting.
                    fid = f"{keyframe_id:05d}"
                    torchvision.utils.save_image(
                        rendered_color, render_image_dir / f"{fid}.png")
                    torchvision.utils.save_image(
                        gt_color, gt_color_uncert_dir / f"{fid}.png")
                    np.save(render_depth_dir / f"{fid}.npy",
                            rendered_depth.cpu().numpy().astype(np.float32))
                    np.save(render_img_uncert_dir / f"{fid}.npy",
                            color_uncertainty.cpu().numpy().astype(np.float32))
                    np.save(render_depth_uncert_dir / f"{fid}.npy",
                            depth_uncertainty.cpu().numpy().astype(np.float32))
                    np.save(gt_depth_uncert_dir / f"{fid}.npy",
                            gt_depth.cpu().numpy().astype(np.float32))

                    psnr.append(psnr_value)
                    lpips.append(lpips_value)
                    ssim.append(ssim_value)
                    depth_l1.append(depth_l1_value)

            num_frames = len(psnr)
            metrics = {
                "psnr": sum(psnr) / num_frames,
                "lpips": sum(lpips) / num_frames,
                "ssim": sum(ssim) / num_frames,
                "depth_l1_train_view": sum(depth_l1) / num_frames,
                "ause_color": sum(ause_color_list) / num_frames,
                "ause_depth": sum(ause_depth_list) / num_frames,
                "num_renders": num_frames
            }
            metrics_name = ("rendering_metrics_global_novel_view.json"
                            if self.novel_view_render else "rendering_metrics_global.json")
            save_dict_to_json(metrics, metrics_name,
                            directory=self.checkpoint_path)

            x = list(range(len(psnr)))
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            axs[0].plot(x, psnr, label="PSNR")
            axs[0].legend()
            axs[0].set_title("PSNR")
            axs[1].plot(x, ssim, label="SSIM")
            axs[1].legend()
            axs[1].set_title("SSIM")
            axs[2].plot(x, depth_l1, label="Depth L1 (Train view)")
            axs[2].legend()
            axs[2].set_title("Depth L1 Render")
            axs[3].plot(x, ause_color_list, label="AUSE Color")
            axs[3].legend()
            axs[3].set_title("AUSE Color")
            axs[4].plot(x, ause_depth_list, label="AUSE Depth")
            axs[4].legend()
            axs[4].set_title("AUSE Depth")
            plt.tight_layout()
            plot_name = ("rendering_metrics_global_novel_view.png"
                         if self.novel_view_render else "rendering_metrics_global.png")
            plt.savefig(str(self.checkpoint_path / plot_name), dpi=300)
            print(metrics)

    def run(self):
        """ Runs the general evaluation flow """

        print("Starting evaluation...🍺")

        try:
            self.run_trajectory_eval()
        except Exception:
            print("Could not run trajectory eval")
            traceback.print_exc()

        try:
            self.run_reconstruction_eval()
        except Exception:
            print("Could not run reconstruction eval")
            traceback.print_exc()
            
        try:
            self.run_global_map_eval()
        except Exception:
            print("Could not run global map eval")
            traceback.print_exc()

        if self.use_wandb: 
            evals = ["rendering_metrics.json", "reconstruction_metrics.json", "ate_aligned.json", "nvs_eval/results.json"]
            log_metrics_to_wandb(evals, self.checkpoint_path, "Evaluation")
