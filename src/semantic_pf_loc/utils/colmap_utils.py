"""COLMAP binary format I/O utilities."""

import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

CAMERA_MODEL_IDS = {"PINHOLE": 1, "SIMPLE_PINHOLE": 0}
CAMERA_MODEL_PARAMS = {"PINHOLE": 4, "SIMPLE_PINHOLE": 3}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # [fx, fy, cx, cy] for PINHOLE


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # [qw, qx, qy, qz] COLMAP w-first
    tvec: np.ndarray  # [tx, ty, tz] world-to-camera
    camera_id: int
    name: str


def write_cameras_binary(cameras: list[Camera], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras:
            model_id = CAMERA_MODEL_IDS[cam.model]
            f.write(struct.pack("<i", cam.id))
            f.write(struct.pack("<i", model_id))
            f.write(struct.pack("<Q", cam.width))
            f.write(struct.pack("<Q", cam.height))
            for p in cam.params:
                f.write(struct.pack("<d", p))


def write_images_binary(images: list[Image], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images:
            f.write(struct.pack("<I", img.id))
            for q in img.qvec:
                f.write(struct.pack("<d", q))
            for t in img.tvec:
                f.write(struct.pack("<d", t))
            f.write(struct.pack("<I", img.camera_id))
            f.write(img.name.encode("utf-8"))
            f.write(b"\x00")
            f.write(struct.pack("<Q", 0))  # num_points2D = 0


def write_points3D_binary(
    points: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    path: Path = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        if points is None:
            f.write(struct.pack("<Q", 0))
            return
        n = len(points)
        f.write(struct.pack("<Q", n))
        if colors is None:
            colors = np.zeros((n, 3), dtype=np.uint8)
        for i in range(n):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<ddd", *points[i]))
            f.write(struct.pack("<BBB", *colors[i]))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", 0))


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            model_name = "PINHOLE"
            for name, mid in CAMERA_MODEL_IDS.items():
                if mid == model_id:
                    model_name = name
                    break
            num_params = CAMERA_MODEL_PARAMS[model_name]
            params = np.array(
                [struct.unpack("<d", f.read(8))[0] for _ in range(num_params)]
            )
            cameras[cam_id] = Camera(
                id=cam_id, model=model_name, width=width, height=height, params=params
            )
    return cameras


def read_images_binary(path: Path) -> dict[int, Image]:
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(
                [struct.unpack("<d", f.read(8))[0] for _ in range(4)]
            )
            tvec = np.array(
                [struct.unpack("<d", f.read(8))[0] for _ in range(3)]
            )
            camera_id = struct.unpack("<I", f.read(4))[0]
            name_bytes = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_bytes += ch
            name = name_bytes.decode("utf-8")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)  # skip 2D points
            images[img_id] = Image(
                id=img_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name
            )
    return images


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> COLMAP quaternion [qw, qx, qy, qz]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion [qw, qx, qy, qz] -> 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ]
    )


def c2w_to_colmap(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-to-world 4x4 -> COLMAP (qvec, tvec) world-to-camera."""
    R_c2w = c2w[:3, :3]
    t_c2w = c2w[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    qvec = rotmat_to_qvec(R_w2c)
    return qvec, t_w2c
