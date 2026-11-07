from scipy.ndimage import median_filter
import os
import random

import numpy as np
import open3d as o3d
import torch
from omni_var_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def setup_seed(seed: int) -> None:
    """ Sets the seed for generating random numbers to ensure reproducibility across multiple runs.
    Args:
        seed: The seed value to set for random number generators in torch, numpy, and random.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def torch2np(tensor: torch.Tensor) -> np.ndarray:
    """ Converts a PyTorch tensor to a NumPy ndarray.
    Args:
        tensor: The PyTorch tensor to convert.
    Returns:
        A NumPy ndarray with the same data and dtype as the input tensor.
    """
    return tensor.detach().cpu().numpy()


def np2torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Converts a NumPy ndarray to a PyTorch tensor.
    Args:
        array: The NumPy ndarray to convert.
        device: The device to which the tensor is sent. Defaults to 'cpu'.

    Returns:
        A PyTorch tensor with the same data as the input array.
    """
    return torch.from_numpy(array).float().to(device)


def np2ptcloud(pts: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    """converts numpy array to point cloud
    Args:
        pts (ndarray): point cloud
    Returns:
        (PointCloud): resulting point cloud
    """
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None:
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    return cloud


def dict2device(dict: dict, device: str = "cpu") -> dict:
    """Sends all tensors in a dictionary to a specified device.
    Args:
        dict: The dictionary containing tensors.
        device: The device to send the tensors to. Defaults to 'cpu'.
    Returns:
        The dictionary with all tensors sent to the specified device.
    """
    for k, v in dict.items():
        if isinstance(v, torch.Tensor):
            dict[k] = v.to(device)
    return dict


def get_render_settings(w, h, intrinsics=None, w2c=None, near=0.01, far=100,
                        sh_degree=0, mapping_mode=True, tracking_mode=True):
    """Build omnidirectional (equirectangular) rasterizer settings.

    The perspective intrinsics argument is accepted only for API compatibility
    and is intentionally ignored: projection is fully determined by the
    equirectangular image size and the world-to-camera pose.

    Args:
        w (int): Panorama width in pixels.
        h (int): Panorama height in pixels.
        intrinsics: Unused; retained for backwards-compatible call sites.
        w2c (array): 4x4 world-to-camera transformation matrix (OpenCV camera).
        near, far (float): Unused; retained for backwards-compatible call sites.
        sh_degree (int): Active spherical-harmonics degree.
        mapping_mode (bool): Enable map/geometry gradients.
        tracking_mode (bool): Enable the analytic SE(3) pose Jacobian.

    Returns:
        GaussianRasterizationSettings for the omnidirectional rasterizer.
    """
    w2c = torch.as_tensor(w2c, dtype=torch.float32, device="cuda")
    viewmatrix = w2c.transpose(0, 1).contiguous()
    cam_center = torch.inverse(w2c)[:3, 3]
    return GaussianRasterizationSettings(
        image_height=int(h),
        image_width=int(w),
        bg=torch.tensor([0, 0, 0], device="cuda").float(),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        sh_degree=sh_degree,
        campos=cam_center,
        prefiltered=False,
        mapping_mode=mapping_mode,
        tracking_mode=tracking_mode,
        debug=False)


def update_render_settings_pose(render_settings, w2c):
    """Return a copy of ``render_settings`` with a new world-to-camera pose."""
    w2c = torch.as_tensor(w2c, dtype=torch.float32,
                          device=render_settings.viewmatrix.device)
    return render_settings._replace(
        viewmatrix=w2c.transpose(0, 1).contiguous(),
        campos=torch.inverse(w2c)[:3, 3])


def render_gaussian_model(gaussian_model, render_settings,
                          override_means_3d=None, override_means_2d=None,
                          override_scales=None, override_rotations=None,
                          override_opacities=None, override_colors=None,
                          return_variance=True, theta=None, rho=None):
    """
    Renders a Gaussian model with the omnidirectional variance/pose rasterizer.

    Args:
        gaussian_model: A Gaussian model object that provides the scene fields.
        render_settings: Omnidirectional ``GaussianRasterizationSettings``.
        override_*: Optional per-splat overrides.
        return_variance (bool): Whether to render the per-pixel RGB variance map.
        theta (torch.Tensor | None): Rotation part of the SE(3) pose perturbation.
            When provided, the rasterizer writes dL/dtheta into ``theta.grad``.
        rho (torch.Tensor | None): Translation part of the SE(3) perturbation.
    Returns:
        dict with:
            color                 [3, H, W]
            depth                 [H, W]  (omnidirectional radial depth)
            alpha                 [H, W]  (accumulated coverage)
            radii                 [P]
            means2D               [P, 3]
            variance              [3, H, W] or None
            render_color_variance [3, H, W] or None
            render_depth_variance [1, H, W] or None (epistemic proxy)
            psi / lat / lon / n_touched (per-splat omnidirectional auxiliaries)
    """
    renderer = GaussianRasterizer(raster_settings=render_settings)

    if override_means_3d is None:
        means3D = gaussian_model.get_xyz()
    else:
        means3D = override_means_3d

    if override_means_2d is None:
        means2D = torch.zeros_like(
            means3D, dtype=means3D.dtype, requires_grad=True, device="cuda")
        means2D.retain_grad()
    else:
        means2D = override_means_2d

    if override_opacities is None:
        opacities = gaussian_model.get_opacity()
    else:
        opacities = override_opacities

    shs, colors_precomp = None, None
    if override_colors is not None:
        colors_precomp = override_colors
    else:
        shs = gaussian_model.get_features()

    # The rasterizer expects one variance per colour channel.  The Gaussian
    # model stores a single shared value; expand it so autograd sums the
    # per-channel gradients back into the [P, 1] parameter.
    color_variance = None
    if return_variance:
        color_variance = gaussian_model.get_color_variance()
        if color_variance.shape[-1] == 1:
            color_variance = color_variance.expand(-1, 3)

    result = renderer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        shs=shs,
        colors_precomp=colors_precomp,
        scales=gaussian_model.get_scaling() if override_scales is None else override_scales,
        rotations=gaussian_model.get_rotation() if override_rotations is None else override_rotations,
        cov3D_precomp=None,
        theta=theta,
        rho=rho,
        color_variance=color_variance,
        return_variance=return_variance,
    )

    color, depth, alpha, radii, psi, lat, lon, n_touched, variance = result

    render_dict = {
        "color": color,
        "depth": depth,
        "radii": radii,
        "means2D": means2D,
        "alpha": alpha,
        "opacity": alpha,
        "psi": psi,
        "lat": lat,
        "lon": lon,
        "n_touched": n_touched,
        "variance": variance if return_variance else None,
        "render_color_variance": None,
        "render_depth_variance": None,
    }

    if return_variance:
        no_gs_mask = alpha < 0.6  # [H, W]: pixels with no Gaussian coverage
        render_color_variance = (variance + 1e-3).masked_fill(no_gs_mask, 100.0)
        # The rasterizer exposes per-splat appearance variance only.  Use the
        # channel-averaged epistemic variance as the depth-uncertainty proxy;
        # the learnable sensor-uncertainty MLP supplies the metric component.
        render_depth_variance = (variance.mean(dim=0, keepdim=True) + 1e-3).masked_fill(
            no_gs_mask, 100.0)
        render_dict["render_color_variance"] = render_color_variance
        render_dict["render_depth_variance"] = render_depth_variance

    return render_dict


def batch_search_faiss(indexer, query_points, k):
    """
    Perform a batch search on a IndexIVFFlat indexer to circumvent the search size limit of 65535.

    Args:
        indexer: The FAISS indexer object.
        query_points: A tensor of query points.
        k (int): The number of nearest neighbors to find.

    Returns:
        distances (torch.Tensor): The distances of the nearest neighbors.
        ids (torch.Tensor): The indices of the nearest neighbors.
    """
    split_pos = torch.split(query_points, 65535, dim=0)
    distances_list, ids_list = [], []

    for split_p in split_pos:
        distance, id = indexer.search(split_p.float(), k)
        distances_list.append(distance.clone())
        ids_list.append(id.clone())
    distances = torch.cat(distances_list, dim=0)
    ids = torch.cat(ids_list, dim=0)

    return distances, ids


def filter_depth_outliers(depth_map, kernel_size=3, threshold=1.0):
    median_filtered = median_filter(depth_map, size=kernel_size)
    abs_diff = np.abs(depth_map - median_filtered)
    outlier_mask = abs_diff > threshold
    depth_map_filtered = np.where(outlier_mask, median_filtered, depth_map)
    return depth_map_filtered


def get_gpu_memory_stats(device=None):
    """
    Get current GPU memory statistics.
    
    Args:
        device: torch device, device string (e.g., 'cuda:0'), or device index. If None, uses current device.
    
    Returns:
        dict: Dictionary with memory statistics in MB
    """
    if not torch.cuda.is_available():
        return None
    
    # Handle different device input types
    if device is None:
        device_idx = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device_idx = device.index if device.index is not None else torch.cuda.current_device()
    elif isinstance(device, str):
        # Handle string like 'cuda:0' or 'cuda'
        if ':' in device:
            device_idx = int(device.split(':')[1])
        else:
            device_idx = torch.cuda.current_device()
    else:
        # Assume it's an integer device index
        device_idx = device
    
    stats = {
        'allocated_mb': torch.cuda.memory_allocated(device_idx) / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved(device_idx) / 1024**2,
        'max_allocated_mb': torch.cuda.max_memory_allocated(device_idx) / 1024**2,
        'max_reserved_mb': torch.cuda.max_memory_reserved(device_idx) / 1024**2,
    }
    return stats


def reset_peak_gpu_memory(device=None):
    """
    Reset peak GPU memory statistics.
    
    Args:
        device: torch device, device string (e.g., 'cuda:0'), or device index. If None, uses current device.
    """
    if not torch.cuda.is_available():
        return
    
    # Handle different device input types
    if device is None:
        device_idx = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device_idx = device.index if device.index is not None else torch.cuda.current_device()
    elif isinstance(device, str):
        # Handle string like 'cuda:0' or 'cuda'
        if ':' in device:
            device_idx = int(device.split(':')[1])
        else:
            device_idx = torch.cuda.current_device()
    else:
        # Assume it's an integer device index
        device_idx = device
    
    torch.cuda.reset_peak_memory_stats(device_idx)


def format_memory_mb(mb):
    """
    Format memory in MB to human-readable string.
    
    Args:
        mb: Memory in megabytes
    
    Returns:
        str: Formatted string (e.g., "1024.00 MB" or "1.00 GB")
    """
    if mb >= 1024:
        return f"{mb/1024:.2f} GB"
    else:
        return f"{mb:.2f} MB"
