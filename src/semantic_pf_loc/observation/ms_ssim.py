"""MS-SSIM observation model — more discriminative than single-scale SSIM.

Multi-scale SSIM captures structural similarity at multiple resolutions,
making it better at discriminating poses in textureless scenes where
single-scale SSIM plateaus.
"""

import torch
from pytorch_msssim import ms_ssim

from .base import ObservationModel


class MSSSIMObservation(ObservationModel):
    """Multi-Scale SSIM observation model."""

    def __init__(self, data_range: float = 1.0, temperature: float = 20.0):
        self.data_range = data_range
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "ms_ssim"

    def compute_log_weights(
        self,
        rendered_images: torch.Tensor,
        observation: dict,
    ) -> torch.Tensor:
        """Compute MS-SSIM between each rendered image and the query.

        Args:
            rendered_images: [C, H, W, 3] float [0, 1]
            observation: {"image": [H, W, 3]}
        Returns:
            [C] log-weights
        """
        query = observation["image"]  # [H, W, 3]
        C = rendered_images.shape[0]
        H, W = rendered_images.shape[1], rendered_images.shape[2]

        # pytorch_msssim expects [B, C, H, W]
        rendered = rendered_images.permute(0, 3, 1, 2)  # [C, 3, H, W]
        target = query.permute(2, 0, 1).unsqueeze(0).expand(C, -1, -1, -1)  # [C, 3, H, W]

        min_dim = min(H, W)

        # The pytorch_msssim.ms_ssim has a hardcoded assertion:
        #   smaller_side > (win_size - 1) * (2**4)
        # regardless of how many scales (weights) we use.
        # So we must pick win_size small enough that the assertion passes.
        # For min_dim=120: need (win_size-1)*16 < 120 -> win_size < 8.5 -> win_size=7
        # For min_dim=240: need (win_size-1)*16 < 240 -> win_size < 16 -> win_size=11 (default)

        if min_dim < 32:
            # Too small for any MS-SSIM, fall back to single-scale SSIM
            from pytorch_msssim import ssim
            ssim_vals = ssim(
                rendered, target,
                data_range=self.data_range,
                size_average=False,
            )
            return self.temperature * torch.log(ssim_vals.clamp(min=1e-8))

        # Pick win_size to satisfy assertion: (win_size-1)*16 < min_dim
        max_win = min_dim // 16  # integer division
        win_size = min(11, max_win)
        if win_size % 2 == 0:
            win_size -= 1  # must be odd
        win_size = max(win_size, 3)  # minimum 3

        # Number of useful scales: each scale halves resolution.
        # At scale i, min_dim is min_dim / 2^i. Need min_dim / 2^i >= win_size.
        # But we use default 5 weights — the library handles it.
        # With fewer weights we get fewer scales which is fine.
        # Pick weights based on what the resolution can support:
        # After k downsamplings, size = min_dim / 2^k. Need size >= win_size.
        max_levels = 1
        dim = min_dim
        while dim // 2 >= win_size and max_levels < 5:
            dim = dim // 2
            max_levels += 1

        # Standard MS-SSIM weights (Wang 2003)
        all_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = all_weights[:max_levels]

        # Compute per-image MS-SSIM
        msssim_vals = ms_ssim(
            rendered,
            target,
            data_range=self.data_range,
            size_average=False,
            win_size=win_size,
            weights=weights,
        )  # [C]

        return self.temperature * torch.log(msssim_vals.clamp(min=1e-8))
