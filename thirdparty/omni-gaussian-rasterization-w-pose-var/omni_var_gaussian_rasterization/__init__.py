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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta,
    rho,
    raster_settings,
    color_variance=None,
    return_variance=False,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
        color_variance,
        return_variance,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
        color_variance,
        return_variance,
    ):
        # Handle optional per-splat variance.
        ctx.return_variance = bool(return_variance)
        if color_variance is None:
            color_variance = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
        ctx.input_shapes = (
            means3D.shape,
            means2D.shape,
            sh.shape,
            colors_precomp.shape,
            opacities.shape,
            scales.shape,
            rotations.shape,
            cov3Ds_precomp.shape,
            theta.shape,
            rho.shape,
            color_variance.shape,
        )

        # Restructure arguments the way that the C++ lib expects them.
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            color_variance,
            return_variance,
        )

        # Invoke C++/CUDA rasterizer.
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                (num_rendered, color, depth, opacity, radii, psi, lat, lon,
                 n_touched, variance, geomBuffer, binningBuffer, imgBuffer) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            (num_rendered, color, depth, opacity, radii, psi, lat, lon,
             n_touched, variance, geomBuffer, binningBuffer, imgBuffer) = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward.
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            color_variance,
            color,
            depth,
            variance,
        )

        # Omni outputs (depth/opacity/radii/psi/lat/lon/n_touched) plus
        # the optional per-pixel variance map.
        return color, depth, opacity, radii, psi, lat, lon, n_touched, variance

    @staticmethod
    def backward(
        ctx,
        grad_out_color,
        grad_out_depth,
        grad_out_opacity,
        grad_radii,
        grad_psi,
        grad_lat,
        grad_lon,
        grad_n_touched,
        grad_variance,
    ):
        # Restore necessary values from context.
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh,
         geomBuffer, binningBuffer, imgBuffer, color_variance, color, depth,
         variance) = ctx.saved_tensors

        # Autograd passes None for outputs that did not contribute to the
        # loss. The CUDA backward always needs a dense colour/depth gradient.
        if grad_out_color is None:
            grad_out_color = torch.zeros_like(color)
        if grad_out_depth is None:
            grad_out_depth = torch.zeros_like(depth)
        if grad_variance is None or not ctx.return_variance:
            grad_variance = torch.empty(0, device=means3D.device, dtype=means3D.dtype)

        # Restructure args as the C++ method expects them.
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            grad_out_color,
            grad_out_depth,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
            color_variance,
            grad_variance,
            color,
            variance,
            raster_settings.mapping_mode,
            raster_settings.tracking_mode,
        )

        # Compute gradients for relevant tensors by invoking the backward method.
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)  # Copy them before they can be corrupted
            try:
                (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
                 grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations,
                 grad_tau, grad_color_variance) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            (grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D,
             grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations,
             grad_tau, grad_color_variance) = _C.rasterize_gaussians_backward(*args)

        # SE(3) pose gradients: [rho (translation), theta (rotation)].
        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3]
        grad_theta = grad_tau[3:]

        def _match(grad, shape):
            """Return a gradient with the input's shape, or None if unused."""
            if grad is None:
                return None
            numel = 1
            for dim in shape:
                numel *= int(dim)
            if numel != grad.numel():
                return None
            return grad.reshape(shape)

        (means3D_shape, means2D_shape, sh_shape, colors_shape, opacities_shape,
         scales_shape, rotations_shape, cov3D_shape, theta_shape, rho_shape,
         color_variance_shape) = ctx.input_shapes

        grads = (
            _match(grad_means3D, means3D_shape),                    # means3D
            _match(grad_means2D, means2D_shape),                    # means2D
            _match(grad_sh, sh_shape),                              # sh
            _match(grad_colors_precomp, colors_shape),              # colors_precomp
            _match(grad_opacities, opacities_shape),                # opacities
            _match(grad_scales, scales_shape),                      # scales
            _match(grad_rotations, rotations_shape),                # rotations
            _match(grad_cov3Ds_precomp, cov3D_shape),               # cov3Ds_precomp
            _match(grad_theta, theta_shape),                        # theta
            _match(grad_rho, rho_shape),                            # rho
            None,                                                   # raster_settings
            _match(grad_color_variance, color_variance_shape),      # color_variance
            None,                                                   # return_variance
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    mapping_mode: bool = True
    tracking_mode: bool = True
    debug: bool = False


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on the omnidirectional near-sphere test).
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix)

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        theta=None,
        rho=None,
        color_variance=None,
        return_variance=False,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide exactly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine.
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            theta,
            rho,
            raster_settings,
            color_variance,
            return_variance,
        )
