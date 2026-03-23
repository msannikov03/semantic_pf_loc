"""SSIM-based observation model (NuRF baseline)."""

import torch
from pytorch_msssim import ssim

from .base import ObservationModel


class SSIMObservation(ObservationModel):
    """SSIM structural similarity observation model."""

    def __init__(self, data_range: float = 1.0, temperature: float = 20.0):
        self.data_range = data_range
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "ssim"

    def compute_log_weights(
        self,
        rendered_images: torch.Tensor,
        observation: dict,
    ) -> torch.Tensor:
        """Compute SSIM between each rendered image and the query.

        Args:
            rendered_images: [C, H, W, 3] float [0, 1]
            observation: {"image": [H, W, 3]}
        Returns:
            [C] log-weights
        """
        query = observation["image"]  # [H, W, 3]
        C = rendered_images.shape[0]

        # pytorch_msssim expects [B, C, H, W]
        rendered = rendered_images.permute(0, 3, 1, 2)  # [C, 3, H, W]
        target = query.permute(2, 0, 1).unsqueeze(0).expand(C, -1, -1, -1)  # [C, 3, H, W]

        # Compute per-image SSIM
        # size_average=False returns per-image scores
        ssim_vals = ssim(
            rendered,
            target,
            data_range=self.data_range,
            size_average=False,
        )  # [C]

        return self.temperature * torch.log(ssim_vals.clamp(min=1e-8))
