"""Convert Replica dataset (NICE-SLAM format) to COLMAP format for gsplat."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import numpy as np
from pathlib import Path

from semantic_pf_loc.utils.colmap_utils import (
    Camera, Image, c2w_to_colmap,
    write_cameras_binary, write_images_binary, write_points3D_binary,
)
from semantic_pf_loc.datasets.replica import (
    REPLICA_FX, REPLICA_FY, REPLICA_CX, REPLICA_CY,
    REPLICA_WIDTH, REPLICA_HEIGHT,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to Replica scene directory (e.g., data/replica/room0)")
    parser.add_argument("--stride", type=int, default=10, help="Frame subsampling stride")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    W, H = REPLICA_WIDTH, REPLICA_HEIGHT
    fx, fy, cx, cy = REPLICA_FX, REPLICA_FY, REPLICA_CX, REPLICA_CY

    # Parse trajectory
    traj_file = data_dir / "traj.txt"
    poses = []
    with open(traj_file) as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 16:
                poses.append(np.array(values).reshape(4, 4))

    # Create COLMAP directory structure
    sparse_dir = data_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"

    # Symlink images directory to results
    if not images_dir.exists():
        results_dir = data_dir / "results"
        if results_dir.exists():
            images_dir.symlink_to(results_dir)

    # Write cameras.bin
    camera = Camera(id=1, model="PINHOLE", width=W, height=H,
                    params=np.array([fx, fy, cx, cy]))
    write_cameras_binary([camera], sparse_dir / "cameras.bin")

    # Write images.bin (subsampled)
    colmap_images = []
    img_id = 1
    for frame_idx in range(0, len(poses), args.stride):
        c2w = poses[frame_idx]
        qvec, tvec = c2w_to_colmap(c2w)
        img_name = f"frame{frame_idx:06d}.jpg"
        # Check if png instead
        if not (data_dir / "results" / img_name).exists():
            img_name = f"frame{frame_idx:06d}.png"
        colmap_images.append(
            Image(id=img_id, qvec=qvec, tvec=tvec, camera_id=1, name=img_name)
        )
        img_id += 1

    write_images_binary(colmap_images, sparse_dir / "images.bin")

    # Write empty points3D.bin
    write_points3D_binary(path=sparse_dir / "points3D.bin")

    print(f"COLMAP sparse model written to {sparse_dir}")
    print(f"  Camera: PINHOLE {W}x{H} fx={fx} fy={fy} cx={cx} cy={cy}")
    print(f"  Images: {len(colmap_images)} (stride={args.stride} from {len(poses)} total)")


if __name__ == "__main__":
    main()
