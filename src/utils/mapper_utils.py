
import cv2
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch


def compute_opt_views_distribution(keyframes_num, iterations_num, current_frame_iter) -> np.ndarray:
    """ Computes the probability distribution for selecting views based on the current iteration.
    Args:
        keyframes_num: The total number of keyframes.
        iterations_num: The total number of iterations planned.
        current_frame_iter: The current iteration number.
    Returns:
        An array representing the probability distribution of keyframes.
    """
    if keyframes_num == 1:
        return np.array([1.0])
    prob = np.full(keyframes_num, (iterations_num - current_frame_iter) / (keyframes_num - 1))
    prob[0] = current_frame_iter
    prob /= prob.sum()
    return prob


def compute_camera_frustum_corners(depth_map: np.ndarray, pose: np.ndarray,
                                   intrinsics: np.ndarray = None) -> np.ndarray:
    """Axis-aligned bounds of the observed omnidirectional shell.

    A 360-degree camera has no view frustum: every direction is visible.  The
    only useful spatial bound is the radial shell of observed depths.  We
    return the eight corners of the cube that encloses that shell around the
    camera centre, so callers can reuse the existing AABB helper.

    Args:
        depth_map: Equirectangular depth map of shape ``(H, W)``.
        pose: Camera-to-world matrix of shape ``(4, 4)``.
        intrinsics: Unused; kept for backwards-compatible call sites.
    Returns:
        ``(8, 3)`` array of world-space bounds.
    """
    valid = depth_map[depth_map > 0]
    if valid.size == 0:
        radius = 1e-3
    else:
        radius = float(valid.max())
    center = np.asarray(pose[:3, 3], dtype=np.float64)
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
    ], dtype=np.float64)
    return center[None, :] + radius * signs


def compute_frustum_aabb(frustum_corners: torch.Tensor):
    """ Computes a mask indicating which points lie inside a given axis-aligned bounding box (AABB).
    Args:
        points: An array of 3D points.
        min_corner: The minimum corner of the AABB.
        max_corner: The maximum corner of the AABB.
    Returns:
        A boolean array indicating whether each point lies inside the AABB.
    """
    return torch.min(frustum_corners, axis=0).values, torch.max(frustum_corners, axis=0).values


def points_inside_aabb_mask(points: np.ndarray, min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
    """ Computes a mask indicating which points lie inside the camera frustum.
    Args:
        points: A tensor of 3D points.
        frustum_planes: A tensor representing the planes of the frustum.
    Returns:
        A boolean tensor indicating whether each point lies inside the frustum.
    """
    return (
        (points[:, 0] >= min_corner[0])
        & (points[:, 0] <= max_corner[0])
        & (points[:, 1] >= min_corner[1])
        & (points[:, 1] <= max_corner[1])
        & (points[:, 2] >= min_corner[2])
        & (points[:, 2] <= max_corner[2]))


def compute_frustum_point_ids(pts: torch.Tensor, frustum_corners: torch.Tensor, device: str = "cuda"):
    """Select points inside the omnidirectional observation shell.

    For a 360-degree camera there are no side planes, so the visibility region
    is the axis-aligned box enclosing the observed radial depths.  The name is
    kept for compatibility with the perspective pipeline it replaces.

    Args:
        pts: A tensor of 3D points.
        frustum_corners: ``(8, 3)`` bounds from
            :func:`compute_camera_frustum_corners`.
        device: The computation device ("cuda" or "cpu").
    Returns:
        Indices of points lying inside the shell bounds.
    """
    if pts.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64, device=device)
    pts = pts.to(device)
    frustum_corners = frustum_corners.to(device)

    min_corner, max_corner = compute_frustum_aabb(frustum_corners)
    inside_aabb_mask = points_inside_aabb_mask(pts, min_corner, max_corner)
    return torch.where(inside_aabb_mask)[0]


def sample_pixels_based_on_gradient(image: np.ndarray, num_samples: int) -> np.ndarray:
    """ Samples pixel indices based on the gradient magnitude of an image.
    Args:
        image: The image from which to sample pixels.
        num_samples: The number of pixels to sample.
    Returns:
        Indices of the sampled pixels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize the gradient magnitude to create a probability map
    prob_map = grad_magnitude / np.sum(grad_magnitude)

    # Flatten the probability map
    prob_map_flat = prob_map.flatten()

    # Sample pixel indices based on the probability map
    sampled_indices = np.random.choice(prob_map_flat.size, size=num_samples, p=prob_map_flat)
    return sampled_indices.T


def compute_new_points_ids(frustum_points: torch.Tensor, new_pts: torch.Tensor,
                           radius: float = 0.03, device: str = "cpu") -> torch.Tensor:
    """ Having newly initialized points, decides which of them should be added to the submap.
        For every new point, if there are no neighbors within the radius in the frustum points,
        it is added to the submap.
    Args:
        frustum_points: Point within a current frustum of the active submap of shape (N, 3)
        new_pts: New 3D Gaussian means which are about to be added to the submap of shape (N, 3)
        radius: Radius whithin which the points are considered to be neighbors
        device: Execution device
    Returns:
        Indicies of the new points that should be added to the submap of shape (N)
    """
    if frustum_points.shape[0] == 0:
        return torch.arange(new_pts.shape[0])
    if device == "cpu":
        pts_index = faiss.IndexFlatL2(3)
    else:
        pts_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss.IndexFlatL2(3))
    frustum_points = frustum_points.to(device)
    new_pts = new_pts.to(device)
    pts_index.add(frustum_points)

    split_pos = torch.split(new_pts, 65535, dim=0)
    distances, ids = [], []
    for split_p in split_pos:
        distance, id = pts_index.search(split_p.float(), 8)
        distances.append(distance)
        ids.append(id)
    distances = torch.cat(distances, dim=0)
    ids = torch.cat(ids, dim=0)
    neighbor_num = (distances < radius).sum(axis=1).int()
    pts_index.reset()
    return torch.where(neighbor_num == 0)[0]


def rotation_to_euler(R: torch.Tensor) -> torch.Tensor:
    """
    Converts a rotation matrix to Euler angles.
    Args:
        R: A rotation matrix.
    Returns:
        Euler angles corresponding to the rotation matrix.
    """
    sy = torch.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    return torch.tensor([x, y, z]) * (180 / np.pi)


def exceeds_motion_thresholds(current_c2w: torch.Tensor, last_submap_c2w: torch.Tensor,
                              rot_thre: float = 50, trans_thre: float = 0.5) -> bool:
    """  Checks if a camera motion exceeds certain rotation and translation thresholds
    Args:
        current_c2w: The current camera-to-world transformation matrix.
        last_submap_c2w: The last submap's camera-to-world transformation matrix.
        rot_thre: The rotation threshold for triggering a new submap.
        trans_thre: The translation threshold for triggering a new submap.

    Returns:
        A boolean indicating whether a new submap is required.
    """
    delta_pose = torch.matmul(torch.linalg.inv(last_submap_c2w).float(), current_c2w.float())
    translation_diff = torch.norm(delta_pose[:3, 3])
    rot_euler_diff_deg = torch.abs(rotation_to_euler(delta_pose[:3, :3]))
    exceeds_thresholds = (translation_diff > trans_thre) or torch.any(rot_euler_diff_deg > rot_thre)
    return exceeds_thresholds.item()


def geometric_edge_mask(rgb_image: np.ndarray, dilate: bool = True, RGB: bool = False) -> np.ndarray:
    """ Computes an edge mask for an RGB image using geometric edges.
    Args:
        rgb_image: The RGB image.
        dilate: Whether to dilate the edges.
        RGB: Indicates if the image format is RGB (True) or BGR (False).
    Returns:
        An edge mask of the input image.
    """
    # Convert the image to grayscale as Canny edge detection requires a single channel image
    gray_image = cv2.cvtColor(
        rgb_image, cv2.COLOR_BGR2GRAY if not RGB else cv2.COLOR_RGB2GRAY)
    if gray_image.dtype != np.uint8:
        gray_image = gray_image.astype(np.uint8)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200, apertureSize=3, L2gradient=True)
    # Define the structuring element for dilation, you can change the size for a thicker/thinner mask
    if dilate:
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def calc_psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """ Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        img1: The first image.
        img2: The second image.
    Returns:
        The PSNR value.
    """
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).mean()


def create_point_cloud(image: np.ndarray, depth: np.ndarray,
                       intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Unproject an equirectangular RGB-D frame into a world-space cloud.

    The projection follows the rasterizer's convention:
    ``lon = atan2(x, z)``, ``lat = atan2(-y, sqrt(x^2 + z^2))`` and
    ``u = (lon / pi + 1) * W / 2``, ``v = (0.5 - lat / pi) * H``.  Depth is the
    Euclidean radial distance from the camera centre.

    Args:
        image: RGB image of shape ``(H, W, 3)``.
        depth: Radial depth map of shape ``(H, W)``.
        intrinsics: Unused; kept for backwards-compatible call sites.
        pose: Camera-to-world matrix of shape ``(4, 4)``.
    Returns:
        ``(N, 6)`` array of ``(x, y, z, r, g, b)``.
    """
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Pixel (u, v) maps to lon = (2u / W - 1) * pi and
    # lat = (0.5 - v / H) * pi, matching the rasterizer's point_image
    # convention (pixel centres are at integer image coordinates).
    lon = (u.astype(np.float64) / width - 0.5) * 2.0 * np.pi
    lat = (0.5 - v.astype(np.float64) / height) * np.pi
    cos_lat = np.cos(lat)

    # Unit direction in the camera frame.
    dx = cos_lat * np.sin(lon)
    dy = -np.sin(lat)
    dz = cos_lat * np.cos(lon)

    points_cam = np.stack((dx * depth, dy * depth, dz * depth), axis=-1)
    points_cam = points_cam.reshape(-1, 3)

    rotation = pose[:3, :3].astype(np.float64)
    translation = pose[:3, 3].astype(np.float64)
    posed_points = points_cam @ rotation.T + translation

    colors = image.reshape(-1, 3)
    return np.concatenate((posed_points, colors), axis=-1).astype(np.float32)
