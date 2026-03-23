"""TUM RGB-D dataset loader."""

import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

from .base import BaseLocalizationDataset

# Hardcoded intrinsics per freiburg camera
TUM_INTRINSICS = {
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


class TUMDataset(BaseLocalizationDataset):
    """TUM RGB-D dataset.

    Directory structure:
        data_root/
            rgb/           — PNG images (640x480)
            depth/         — 16-bit PNG depth (scale=5000)
            groundtruth.txt — timestamp tx ty tz qx qy qz qw
            rgb.txt        — timestamp filename
    """

    def __init__(
        self,
        data_dir: str,
        max_depth: float = 10.0,
        stride: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.max_depth = max_depth

        # Detect freiburg camera
        name = self.data_dir.name.lower()
        if "freiburg1" in name:
            self._cam = "freiburg1"
        elif "freiburg2" in name:
            self._cam = "freiburg2"
        elif "freiburg3" in name:
            self._cam = "freiburg3"
        else:
            self._cam = "freiburg1"

        fx, fy, cx, cy = TUM_INTRINSICS[self._cam]
        self._K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float64
        )
        self._width = 640
        self._height = 480

        # Parse groundtruth
        gt_poses, gt_timestamps = self._parse_groundtruth()
        # Parse rgb list
        rgb_files, rgb_timestamps = self._parse_rgb_list()
        # Parse depth list (timestamps differ from RGB in TUM)
        depth_files, depth_timestamps = self._parse_depth_list()
        # Associate: nearest GT pose for each RGB frame
        self._frames = self._associate(
            rgb_files, rgb_timestamps, gt_poses, gt_timestamps
        )
        # Associate depth frames with RGB frames by nearest timestamp
        self._associate_depth(depth_files, depth_timestamps)
        # Apply stride
        self._frames = self._frames[::stride]

    def _parse_groundtruth(self):
        gt_file = self.data_dir / "groundtruth.txt"
        poses, timestamps = [], []
        with open(gt_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                ts = float(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = (
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                )
                # Build 4x4 camera-to-world from quaternion
                pose = self._quat_to_matrix(tx, ty, tz, qx, qy, qz, qw)
                poses.append(pose)
                timestamps.append(ts)
        return np.stack(poses), np.array(timestamps)

    def _parse_rgb_list(self):
        rgb_file = self.data_dir / "rgb.txt"
        files, timestamps = [], []
        with open(rgb_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                timestamps.append(float(parts[0]))
                files.append(parts[1])
        return files, np.array(timestamps)

    def _parse_depth_list(self):
        depth_file = self.data_dir / "depth.txt"
        files, timestamps = [], []
        if not depth_file.exists():
            return files, np.array([])
        with open(depth_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                timestamps.append(float(parts[0]))
                files.append(parts[1])
        return files, np.array(timestamps)

    def _associate_depth(self, depth_files, depth_ts, max_diff=0.02):
        """Associate each RGB frame with its nearest depth frame."""
        if len(depth_files) == 0 or len(depth_ts) == 0:
            for frame in self._frames:
                frame["depth_file"] = None
            return
        for frame in self._frames:
            rgb_t = frame["timestamp"]
            diffs = np.abs(depth_ts - rgb_t)
            idx = np.argmin(diffs)
            if diffs[idx] < max_diff:
                frame["depth_file"] = depth_files[idx]
            else:
                frame["depth_file"] = None

    def _associate(self, rgb_files, rgb_ts, gt_poses, gt_ts, max_diff=0.02):
        """Associate RGB frames with nearest GT pose."""
        frames = []
        for i, t in enumerate(rgb_ts):
            diffs = np.abs(gt_ts - t)
            idx = np.argmin(diffs)
            if diffs[idx] < max_diff:
                frames.append(
                    {
                        "rgb_file": rgb_files[i],
                        "pose": gt_poses[idx],
                        "timestamp": t,
                    }
                )
        return frames

    @staticmethod
    def _quat_to_matrix(tx, ty, tz, qx, qy, qz, qw):
        """Quaternion (qx,qy,qz,qw) + translation -> 4x4 c2w matrix."""
        R = np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
            ]
        )
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> dict:
        frame = self._frames[idx]

        # Load RGB
        rgb_path = self.data_dir / frame["rgb_file"]
        img = PILImage.open(rgb_path).convert("RGB")
        img = torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, 3]

        # Load depth if available (associated by timestamp in __init__)
        depth = None
        depth_file = frame.get("depth_file", None)
        if depth_file is not None:
            depth_path = self.data_dir / depth_file
            if depth_path.exists():
                d = PILImage.open(depth_path)
                depth = torch.from_numpy(np.array(d)).float() / 5000.0  # meters
                depth[depth > self.max_depth] = 0.0

        pose = torch.from_numpy(frame["pose"]).to(torch.float64)

        return {
            "image": img,
            "pose": pose,
            "K": self._K.clone(),
            "depth": depth,
            "timestamp": frame["timestamp"],
        }

    def get_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_trans = np.array([f["pose"][:3, 3] for f in self._frames])
        margin = 0.5
        bounds_min = torch.from_numpy(all_trans.min(axis=0) - margin).float()
        bounds_max = torch.from_numpy(all_trans.max(axis=0) + margin).float()
        return bounds_min, bounds_max

    def get_intrinsics(self) -> torch.Tensor:
        return self._K.clone()

    @property
    def image_size(self) -> tuple[int, int]:
        return (self._width, self._height)
