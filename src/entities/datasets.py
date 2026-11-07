import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import json
import imageio
import trimesh


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.depth_scale = dataset_config["depth_scale"]

        model = dataset_config.get(
            "camera_model", dataset_config.get("model", "equirectangular"))
        self.camera_model = str(model).lower()
        self.is_omni = self.camera_model in (
            "omni", "omnidirectional", "equirectangular", "360")
        if not self.is_omni:
            raise NotImplementedError(
                "This branch renders with the omnidirectional rasterizer. "
                "Set cam.camera_model: equirectangular (and provide 2:1 "
                "panoramic RGB-D data).")

        if self.is_omni:
            # Omnidirectional RGB-D is expected to be a 2:1 equirectangular
            # panorama.  Perspective intrinsics are deliberately not used.
            self.fx = None
            self.fy = None
            self.cx = None
            self.cy = None
            self.intrinsics = None
            self.distortion = None
            self.crop_edge = 0
            self.fovx = 2 * math.pi
            self.fovy = math.pi

        self.color_paths = []
        self.depth_paths = []

    def _resize_omni(self, color_data: np.ndarray, depth_data: np.ndarray):
        """Resize an equirectangular frame to the configured panorama size."""
        if color_data.shape[:2] != (self.height, self.width):
            color_data = cv2.resize(
                color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if depth_data.shape[:2] != (self.height, self.width):
            depth_data = cv2.resize(
                depth_data, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return color_data, depth_data

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)


class Replica(BaseDataset):

    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(
            list((self.dataset_path / "results").glob("frame*.jpg")))
        self.depth_paths = sorted(
            list((self.dataset_path / "results").glob("depth*.png")))
        self.load_poses(self.dataset_path / "traj.txt")
        print(f"Loaded {len(self.color_paths)} frames")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        if self.is_omni:
            color_data, depth_data = self._resize_omni(color_data, depth_data)
        return index, color_data, depth_data, self.poses[index]


class TUM_RGBD(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.dataset_path, frame_rate=32)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))
        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths = [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            poses += [c2w.astype(np.float32)]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None and not self.is_omni:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        if self.is_omni:
            color_data, depth_data = self._resize_omni(color_data, depth_data)
        else:
            edge = self.crop_edge
            if edge > 0:
                color_data = color_data[edge:-edge, edge:-edge]
                depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index]


class ScanNet(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(list(
            (self.dataset_path / "rgb").glob("*.png")), key=lambda x: int(os.path.basename(x)[-9:-4]))
        self.depth_paths = sorted(list(
            (self.dataset_path / "depth").glob("*.TIFF")), key=lambda x: int(os.path.basename(x)[-10:-5]))
        self.n_img = len(self.color_paths)
        self.load_poses(self.dataset_path / "gt_pose.txt")

    def load_poses(self, path):
        self.poses = []
        pose_data = np.loadtxt(path, delimiter=" ", dtype=np.unicode_, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)
        for i in range(self.n_img):
            quat = pose_vecs[i][4:]
            trans = pose_vecs[i][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            pose = T
            self.poses.append(pose)

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None and not self.is_omni:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = cv2.resize(color_data, (self.width, self.height))

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        depth_data = cv2.resize(
            depth_data, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        edge = self.crop_edge
        if not self.is_omni and edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index]


class ScanNetPP(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.use_train_split = dataset_config["use_train_split"]
        self.train_test_split = json.load(open(f"{self.dataset_path}/dslr/train_test_lists.json", "r"))
        if self.use_train_split:
            self.image_names = self.train_test_split["train"]
        else:
            self.image_names = self.train_test_split["test"]
        self.load_data()

    def load_data(self):
        self.poses = []
        cams_path = self.dataset_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        cams_metadata = json.load(open(str(cams_path), "r"))
        frames_key = "frames" if self.use_train_split else "test_frames"
        frames_metadata = cams_metadata[frames_key]
        frame2idx = {frame["file_path"]: index for index, frame in enumerate(frames_metadata)}
        P = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(np.float32)
        for image_name in self.image_names:
            frame_metadata = frames_metadata[frame2idx[image_name]]
            # if self.ignore_bad and frame_metadata['is_bad']:
            #     continue
            color_path = str(self.dataset_path / "dslr" / "undistorted_images" / image_name)
            depth_path = str(self.dataset_path / "dslr" / "undistorted_depths" / image_name.replace('.JPG', '.png'))
            self.color_paths.append(color_path)
            self.depth_paths.append(depth_path)
            c2w = np.array(frame_metadata["transform_matrix"]).astype(np.float32)
            c2w = P @ c2w @ P.T
            self.poses.append(c2w)

    def __len__(self):
        if self.use_train_split:
            return len(self.image_names) if self.frame_limit < 0 else int(self.frame_limit)
        else:
            return len(self.image_names)

    def __getitem__(self, index):
        color_data = np.asarray(imageio.imread(self.color_paths[index]), dtype=float)
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = color_data.astype(np.uint8)

        depth_data = np.asarray(imageio.imread(self.depth_paths[index]), dtype=np.int64)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index]


class Replica360(Replica):
    """Replica-style equirectangular panoramas (``results/frame*.jpg``)."""

    def __init__(self, dataset_config: dict):
        config = dict(dataset_config)
        config.setdefault("camera_model", "equirectangular")
        super().__init__(config)


class OmniRGBD(BaseDataset):
    """Generic equirectangular RGB-D sequence.

    Expected layout::

        <input_path>/rgb/*.png|jpg
        <input_path>/depth/*.png|tiff
        <input_path>/poses.txt

    Each pose line may contain 16 row-major values (camera-to-world), 12
    row-major values (3x4 camera-to-world), 8 values ``timestamp tx ty tz qx
    qy qz qw``, or 7 values ``tx ty tz qx qy qz qw``.
    """

    def __init__(self, dataset_config: dict):
        config = dict(dataset_config)
        config.setdefault("camera_model", "equirectangular")
        super().__init__(config)
        rgb_dir = self.dataset_path / "rgb"
        depth_dir = self.dataset_path / "depth"
        self.color_paths = sorted(
            [p for p in rgb_dir.glob("*") if p.suffix.lower() in
             (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")])
        self.depth_paths = sorted(
            [p for p in depth_dir.glob("*") if p.suffix.lower() in
             (".png", ".tif", ".tiff", ".npy")])
        if len(self.color_paths) == 0 or len(self.depth_paths) == 0:
            raise FileNotFoundError(
                f"Expected rgb/ and depth/ images under {self.dataset_path}")
        if len(self.color_paths) != len(self.depth_paths):
            raise ValueError("rgb/ and depth/ must contain the same number of frames")
        self.load_poses()

    def _pose_path(self):
        for name in ("poses.txt", "traj.txt", "pose.txt", "groundtruth.txt"):
            candidate = self.dataset_path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No poses.txt/traj.txt found under {self.dataset_path}")

    def load_poses(self):
        from scipy.spatial.transform import Rotation

        path = self._pose_path()
        rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
        self.poses = []
        for line in rows:
            line = np.atleast_1d(line)
            pose = np.eye(4)
            if line.size == 16:
                pose = line.reshape(4, 4)
            elif line.size == 12:
                pose[:3, :4] = line.reshape(3, 4)
            elif line.size in (7, 8):
                values = line[-7:]
                pose[:3, 3] = values[:3]
                pose[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
            else:
                raise ValueError(
                    f"Unsupported pose row with {line.size} values in {path}")
            self.poses.append(pose.astype(np.float32))

    def __getitem__(self, index):
        if self.color_paths[index].suffix.lower() == ".npy":
            color_data = np.load(self.color_paths[index])
            if color_data.dtype != np.uint8:
                # Keep the pipeline's uint8 RGB convention so that callers can
                # scale by 1/255 unconditionally.
                scale = 255.0 if float(color_data.max()) <= 1.0 else 1.0
                color_data = np.clip(color_data * scale, 0, 255).astype(np.uint8)
        else:
            color_data = cv2.imread(str(self.color_paths[index]))
            color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        if self.depth_paths[index].suffix.lower() == ".npy":
            depth_data = np.load(self.depth_paths[index]).astype(np.float32)
            depth_data = depth_data / self.depth_scale
        else:
            depth_data = cv2.imread(
                str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
            depth_data = depth_data.astype(np.float32) / self.depth_scale
        color_data, depth_data = self._resize_omni(color_data, depth_data)
        return index, color_data, depth_data, self.poses[index]


def get_dataset(dataset_name: str):
    if dataset_name == "replica":
        return Replica
    elif dataset_name == "replica360":
        return Replica360
    elif dataset_name in ("omni", "omnidirectional", "equirectangular"):
        return OmniRGBD
    elif dataset_name == "tum_rgbd":
        return TUM_RGBD
    elif dataset_name == "scan_net":
        return ScanNet
    elif dataset_name == "scannetpp":
        return ScanNetPP
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
