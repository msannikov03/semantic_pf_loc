"""Batched differentiable renderer using gsplat."""

import torch
from gsplat import rasterization

from .gaussian_map import GaussianMap


class BatchRenderer:
    """Wraps gsplat.rasterization for batched multi-viewpoint rendering."""

    def __init__(
        self,
        gaussian_map: GaussianMap,
        width: int = 320,
        height: int = 240,
        near_plane: float = 0.01,
        far_plane: float = 100.0,
        sh_degree: int | None = None,
        chunk_size: int = 100,
    ):
        self.gmap = gaussian_map
        self.width = width
        self.height = height
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.sh_degree = sh_degree if sh_degree is not None else gaussian_map.sh_degree
        self.chunk_size = chunk_size

    def render_batch(
        self,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        backgrounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Render C viewpoints in one batched call (with chunking for memory).

        Args:
            viewmats: [C, 4, 4] world-to-camera matrices
            Ks: [C, 3, 3] camera intrinsics (scaled for render resolution)
            backgrounds: [C, 3] or None

        Returns:
            renders: [C, H, W, 3] RGB images
            alphas: [C, H, W, 1] alpha maps
            meta: dict from last chunk
        """
        C = viewmats.shape[0]
        params = self.gmap.get_render_params()

        if C <= self.chunk_size:
            return self._render_chunk(viewmats, Ks, params, backgrounds)

        all_renders = []
        all_alphas = []
        meta = {}

        for start in range(0, C, self.chunk_size):
            end = min(start + self.chunk_size, C)
            chunk_viewmats = viewmats[start:end]
            chunk_Ks = Ks[start:end]
            chunk_bg = backgrounds[start:end] if backgrounds is not None else None

            renders, alphas, meta = self._render_chunk(
                chunk_viewmats, chunk_Ks, params, chunk_bg
            )
            all_renders.append(renders)
            all_alphas.append(alphas)

        return torch.cat(all_renders, dim=0), torch.cat(all_alphas, dim=0), meta

    def _render_chunk(
        self,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        params: dict,
        backgrounds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Render a chunk of viewpoints."""
        kwargs = {
            "means": params["means"],
            "quats": params["quats"],
            "scales": params["scales"],
            "opacities": params["opacities"],
            "colors": params["colors"],
            "viewmats": viewmats,
            "Ks": Ks,
            "width": self.width,
            "height": self.height,
            "near_plane": self.near_plane,
            "far_plane": self.far_plane,
            "sh_degree": self.sh_degree,
            "packed": False,
            "render_mode": "RGB",
        }
        if backgrounds is not None:
            kwargs["backgrounds"] = backgrounds

        renders, alphas, meta = rasterization(**kwargs)
        return renders, alphas, meta

    def render_single(
        self,
        viewmat: torch.Tensor,
        K: torch.Tensor,
    ) -> torch.Tensor:
        """Render a single viewpoint. Returns [H, W, 3]."""
        renders, _, _ = self.render_batch(
            viewmat.unsqueeze(0), K.unsqueeze(0)
        )
        return renders[0]

    def update_resolution(self, width: int, height: int) -> None:
        """Change render resolution (e.g., for adaptive PF)."""
        self.width = width
        self.height = height
