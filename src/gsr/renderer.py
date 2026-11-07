#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from omni_var_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from src.entities.gaussian_model import GaussianModel
from src.utils.gaussian_model_utils import eval_sh


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
    return_variance: bool = False,
    variance_mode: str = "scalar",
):
    """Render an omnidirectional Gaussian model.

    Background tensor (``bg_color``) must be on GPU.  The returned ``depth``
    and ``opacity`` maps are ``[H, W]``; RGB is ``[3, H, W]`` and the optional
    variance map is ``[3, H, W]``.
    """
    if pc.get_xyz().shape[0] == 0:
        return None

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz(), dtype=pc.get_xyz().dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mapping_mode=True,
        tracking_mode=True,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz()
    means2D = screenspace_points
    opacity = pc.get_opacity()

    color_variance = None
    if return_variance:
        color_variance = pc.get_color_variance()
        if color_variance.shape[-1] == 1:
            color_variance = color_variance.expand(-1, 3)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        if pc.get_scaling().shape[-1] == 1:
            scales = pc.get_scaling().repeat(1, 3)
        else:
            scales = pc.get_scaling()
        rotations = pc.get_rotation()

    shs = None
    colors_precomp = None
    if override_color is not None:
        colors_precomp = override_color
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features().transpose(1, 2).view(
            -1, 3, (pc.max_sh_degree + 1) ** 2
        )
        dir_pp = pc.get_xyz() - viewpoint_camera.camera_center.repeat(
            pc.get_features().shape[0], 1
        )
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features()

    def _select(tensor, idx):
        return tensor[idx] if (tensor is not None and idx is not None) else tensor

    render_args = {
        "means3D": _select(means3D, mask),
        "means2D": _select(means2D, mask),
        "opacities": _select(opacity, mask),
        "shs": _select(shs, mask),
        "colors_precomp": _select(colors_precomp, mask),
        "scales": _select(scales, mask),
        "rotations": _select(rotations, mask),
        "cov3D_precomp": _select(cov3D_precomp, mask),
        "theta": viewpoint_camera.cam_rot_delta,
        "rho": viewpoint_camera.cam_trans_delta,
        "color_variance": _select(color_variance, mask),
        "return_variance": return_variance,
    }

    (rendered_image, depth, opacity_map, radii, psi, lat, lon,
     n_touched, variance) = rasterizer(**render_args)

    result = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity_map,
        "psi": psi,
        "lat": lat,
        "lon": lon,
        "n_touched": n_touched,
        "render_color_variance": None,
        "render_depth_variance": None,
    }

    if return_variance:
        no_gs_mask = opacity_map < 0.6
        result["variance"] = variance
        result["render_color_variance"] = (variance + 1e-3).masked_fill(
            no_gs_mask, 100.0)
        result["render_depth_variance"] = (
            variance.mean(dim=0, keepdim=True) + 1e-3
        ).masked_fill(no_gs_mask, 100.0)

    return result
