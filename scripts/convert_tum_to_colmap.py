"""Convert TUM RGB-D dataset to COLMAP format for gsplat training."""

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
from semantic_pf_loc.datasets.tum import TUMDataset, TUM_INTRINSICS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to TUM sequence directory")
    parser.add_argument("--stride", type=int, default=1, help="Frame subsampling stride")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Detect freiburg camera
    name = data_dir.name.lower()
    for cam_name in ["freiburg1", "freiburg2", "freiburg3"]:
        if cam_name in name:
            fx, fy, cx, cy = TUM_INTRINSICS[cam_name]
            break
    else:
        fx, fy, cx, cy = TUM_INTRINSICS["freiburg1"]

    W, H = 640, 480

    # Load dataset to get associated frames
    dataset = TUMDataset(str(data_dir), stride=args.stride)

    # Create COLMAP directory structure
    sparse_dir = data_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"

    # Symlink images directory to rgb
    if not images_dir.exists():
        rgb_dir = data_dir / "rgb"
        if rgb_dir.exists():
            images_dir.symlink_to(rgb_dir)

    # Write cameras.bin (single PINHOLE camera)
    camera = Camera(id=1, model="PINHOLE", width=W, height=H,
                    params=np.array([fx, fy, cx, cy]))
    write_cameras_binary([camera], sparse_dir / "cameras.bin")

    # Write images.bin
    colmap_images = []
    for i in range(len(dataset)):
        frame = dataset._frames[i]
        c2w = frame["pose"]
        qvec, tvec = c2w_to_colmap(c2w)
        # Image name: just the filename from rgb path
        img_name = Path(frame["rgb_file"]).name
        colmap_images.append(
            Image(id=i + 1, qvec=qvec, tvec=tvec, camera_id=1, name=img_name)
        )

    write_images_binary(colmap_images, sparse_dir / "images.bin")

    # Write empty points3D.bin
    write_points3D_binary(path=sparse_dir / "points3D.bin")

    print(f"COLMAP sparse model written to {sparse_dir}")
    print(f"  Camera: PINHOLE {W}x{H} fx={fx} fy={fy} cx={cx} cy={cy}")
    print(f"  Images: {len(colmap_images)}")


if __name__ == "__main__":
    main()
