"""Replica dataset loader (NICE-SLAM format)."""

import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

from .base import BaseLocalizationDataset

# Constant intrinsics for Replica (NICE-SLAM renders)
REPLICA_FX = 600.0
REPLICA_FY = 600.0
REPLICA_CX = 599.5
REPLICA_CY = 339.5
REPLICA_WIDTH = 1200
REPLICA_HEIGHT = 680


class ReplicaDataset(BaseLocalizationDataset):
    """Replica dataset in NICE-SLAM format.

    Directory structure:
        data_root/
            results/    — frame000000.jpg, frame000001.jpg, ...
            depth/      — depth000000.png (16-bit, scale=6553.5)
            traj.txt    — N lines, each 16 floats (row-major 4x4 c2w)
    """

    def __init__(
        self,
        data_dir: str,
        max_depth: float = 10.0,
        stride: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.max_depth = max_depth

        self._K = torch.tensor(
            [
                [REPLICA_FX, 0, REPLICA_CX],
                [0, REPLICA_FY, REPLICA_CY],
                [0, 0, 1],
            ],
            dtype=torch.float64,
        )

        # Parse trajectory
        self._poses = self._parse_trajectory()
        # Find available frames
        self._frame_indices = list(range(0, len(self._poses), stride))

    def _parse_trajectory(self) -> list[np.ndarray]:
        traj_file = self.data_dir / "traj.txt"
        poses = []
        with open(traj_file) as f:
            for line in f:
                values = list(map(float, line.strip().split()))
                if len(values) == 16:
                    pose = np.array(values).reshape(4, 4)
                    poses.append(pose)
        return poses

    def __len__(self) -> int:
        return len(self._frame_indices)

    def __getitem__(self, idx: int) -> dict:
        frame_idx = self._frame_indices[idx]

        # Load RGB
        rgb_path = self.data_dir / "results" / f"frame{frame_idx:06d}.jpg"
        if not rgb_path.exists():
            rgb_path = self.data_dir / "results" / f"frame{frame_idx:06d}.png"
        img = PILImage.open(rgb_path).convert("RGB")
        img = torch.from_numpy(np.array(img)).float() / 255.0

        # Load depth
        depth = None
        depth_path = self.data_dir / "depth" / f"depth{frame_idx:06d}.png"
        if depth_path.exists():
            d = PILImage.open(depth_path)
            depth = torch.from_numpy(np.array(d)).float() / 6553.5  # meters
            depth[depth > self.max_depth] = 0.0

        pose = torch.from_numpy(self._poses[frame_idx]).to(torch.float64)

        return {
            "image": img,
            "pose": pose,
            "K": self._K.clone(),
            "depth": depth,
            "timestamp": frame_idx,
        }

    def get_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        all_trans = np.array([p[:3, 3] for p in self._poses])
        margin = 0.5
        bounds_min = torch.from_numpy(all_trans.min(axis=0) - margin).float()
        bounds_max = torch.from_numpy(all_trans.max(axis=0) + margin).float()
        return bounds_min, bounds_max

    def get_intrinsics(self) -> torch.Tensor:
        return self._K.clone()

    @property
    def image_size(self) -> tuple[int, int]:
        return (REPLICA_WIDTH, REPLICA_HEIGHT)
