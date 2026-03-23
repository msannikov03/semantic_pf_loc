"""3D Gaussian Splatting map management."""

import torch
import numpy as np
from pathlib import Path
from plyfile import PlyData


class GaussianMap:
    """Loads and stores 3D Gaussian Splatting map parameters."""

    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        sh_coeffs: torch.Tensor,
        sh_degree: int = 3,
        scene_bounds: tuple | None = None,
        device: str = "cuda",
    ):
        self.means = means.to(device)          # [N, 3]
        self.quats = quats.to(device)          # [N, 4] wxyz
        self.scales = scales.to(device)        # [N, 3] log-scale
        self.opacities = opacities.to(device)  # [N] logit-space
        self.sh_coeffs = sh_coeffs.to(device)  # [N, K, 3]
        self.sh_degree = sh_degree
        self.scene_bounds = scene_bounds
        self.device = device

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: str = "cuda") -> "GaussianMap":
        """Load from training checkpoint dict."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        bounds = ckpt.get("scene_bounds", None)
        return cls(
            means=ckpt["means"],
            quats=ckpt["quats"],
            scales=ckpt["scales"],
            opacities=ckpt["opacities"],
            sh_coeffs=ckpt["sh_coeffs"],
            sh_degree=ckpt.get("sh_degree", 3),
            scene_bounds=bounds,
            device=device,
        )

    @classmethod
    def from_ply(cls, ply_path: str, sh_degree: int = 3, device: str = "cuda") -> "GaussianMap":
        """Load from PLY file (standard 3DGS format)."""
        plydata = PlyData.read(ply_path)
        vertex = plydata["vertex"]

        means = np.stack(
            [vertex["x"], vertex["y"], vertex["z"]], axis=-1
        )
        # Quaternions: stored as rot_0 (w), rot_1 (x), rot_2 (y), rot_3 (z)
        quats = np.stack(
            [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
            axis=-1,
        )
        scales = np.stack(
            [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1
        )
        opacities = vertex["opacity"]

        # SH coefficients: f_dc_0..2 + f_rest_0..N
        sh_dc = np.stack(
            [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1
        )  # [N, 3]
        num_sh = (sh_degree + 1) ** 2
        sh_rest_names = [f"f_rest_{i}" for i in range(3 * (num_sh - 1))]
        if sh_rest_names and sh_rest_names[0] in vertex.data.dtype.names:
            sh_rest = np.stack([vertex[n] for n in sh_rest_names], axis=-1)
            sh_rest = sh_rest.reshape(-1, num_sh - 1, 3)  # [N, K-1, 3]
            sh_coeffs = np.concatenate(
                [sh_dc[:, np.newaxis, :], sh_rest], axis=1
            )  # [N, K, 3]
        else:
            sh_coeffs = sh_dc[:, np.newaxis, :]  # [N, 1, 3]

        return cls(
            means=torch.from_numpy(means).float(),
            quats=torch.from_numpy(quats).float(),
            scales=torch.from_numpy(scales).float(),
            opacities=torch.from_numpy(opacities).float(),
            sh_coeffs=torch.from_numpy(sh_coeffs).float(),
            sh_degree=sh_degree,
            device=device,
        )

    def save_checkpoint(self, path: str, step: int = 0) -> None:
        """Save as training checkpoint."""
        torch.save(
            {
                "means": self.means.cpu(),
                "quats": self.quats.cpu(),
                "scales": self.scales.cpu(),
                "opacities": self.opacities.cpu(),
                "sh_coeffs": self.sh_coeffs.cpu(),
                "sh_degree": self.sh_degree,
                "scene_bounds": self.scene_bounds,
                "step": step,
            },
            path,
        )

    def get_render_params(self) -> dict:
        """Parameters ready for gsplat.rasterization()."""
        return {
            "means": self.means,
            "quats": self.quats,
            "scales": torch.exp(self.scales),
            "opacities": torch.sigmoid(self.opacities),
            "colors": self.sh_coeffs,
        }
