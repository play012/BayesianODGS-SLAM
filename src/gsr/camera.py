import numpy as np
import torch
from PIL import Image
from torch import nn

from src.gsr.loss import image_gradient, image_gradient_mask
from src.utils.graphics_utils import getWorld2View2


class Camera(nn.Module):
    """Omnidirectional (equirectangular) camera.

    The pose is stored as an OpenCV-style world-to-camera transform ``R, T``
    (``p_C = R p_W + T``).  Pose optimisation uses a left SE(3) perturbation
    ``T_CW <- Exp([rho, theta]) T_CW``; ``cam_trans_delta`` holds ``rho`` and
    ``cam_rot_delta`` holds ``theta``, matching the rasterizer's Jacobians.
    """

    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.image_height = image_height
        self.image_width = image_width

        # theta (rotation) and rho (translation) of the SE(3) perturbation.
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

    @staticmethod
    def init_from_dataset(dataset, idx):
        _, gt_color, gt_depth, gt_c2w = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            torch.from_numpy(np.linalg.inv(gt_c2w)).float(),
            dataset.height,
            dataset.width,
            device=getattr(dataset, "device", "cuda:0"),
        )

    @staticmethod
    def init_from_gui(uid, T, H, W):
        return Camera(uid, None, None, T, H, W)

    @property
    def world_view_transform(self):
        """Column-major world-to-camera matrix consumed by the rasterizer."""
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def camera_center(self):
        return -self.R.transpose(0, 1) @ self.T

    def update_RT(self, R, t):
        self.R = R.detach().clone().to(device=self.device)
        self.T = t.detach().clone().to(device=self.device)

    def compute_grad_mask(self, config):
        training_cfg = config.get("Training", {})
        edge_threshold = training_cfg.get(
            "edge_threshold", config.get("tracking", {}).get("edge_threshold", 1.0))
        dataset_type = config.get("Dataset", {}).get(
            "type", config.get("dataset_name", ""))

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if dataset_type in ("replica", "replica360"):
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row): (r + 1) * int(h / row),
                        c * int(w / col): (c + 1) * int(w / col),
                    ]
                    th_median = block.median()
                    block[block > (th_median * multiplier)] = 1
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None

    @property
    def get_T(self):
        T = torch.eye(4, device=self.device).float()
        T[:3, :3] = self.R
        T[:3, 3] = self.T
        return T

    @property
    def get_T_gt(self):
        T = torch.eye(4, device=self.device).float()
        T[:3, :3] = self.R_gt
        T[:3, 3] = self.T_gt
        return T

    def load_rgb(self, image=None):
        if image is None and hasattr(self, "rgb_path"):
            if str(self.rgb_path).lower().endswith(".npy"):
                rgb = np.load(self.rgb_path)
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(rgb)
            else:
                img = Image.open(self.rgb_path).convert("RGB")
            if img.size != (self.image_width, self.image_height):
                img = img.resize(
                    (self.image_width, self.image_height), Image.BILINEAR)
            self.original_image = (
                torch.from_numpy(np.array(img)).permute(2, 0, 1).cuda().float()
                / 255.0
            )
            self.compute_grad_mask(self.config)

        if image is not None:
            self.original_image = image
            self.compute_grad_mask(self.config)
