"""LPIPS observation model for particle filter.

LPIPS (Learned Perceptual Image Patch Similarity) uses VGG features at
multiple scales and is far more discriminative than SSIM for pose-dependent
appearance changes, especially in textureless scenes.
"""

import torch
import torch.nn.functional as F
import lpips

from .base import ObservationModel


class LPIPSObservation(ObservationModel):
    """Uses LPIPS (learned perceptual similarity) for particle weighting.

    LPIPS compares images using VGG features at multiple scales,
    making it far more discriminative than pixel-level SSIM in
    textureless or self-similar scenes.

    Note: LPIPS returns a DISTANCE (lower = more similar), so we
    convert to log-weights via log_weight = -temperature * distance.
    """

    def __init__(
        self,
        net: str = "vgg",
        temperature: float = 5.0,
        chunk_size: int = 50,
        device: str = "cuda",
    ):
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.device = device
        # lpips.LPIPS returns a DISTANCE (lower = more similar)
        self.loss_fn = lpips.LPIPS(net=net).to(device).eval()
        for p in self.loss_fn.parameters():
            p.requires_grad_(False)

    @property
    def name(self) -> str:
        return "lpips"

    @property
    def requires_image(self) -> bool:
        return True

    def compute_log_weights(
        self,
        rendered_images: torch.Tensor,
        observation: dict,
    ) -> torch.Tensor:
        """Compute LPIPS distance between each rendered image and the query.

        Args:
            rendered_images: [C, H, W, 3] float [0, 1]
            observation: {"image": [H, W, 3]}
        Returns:
            [C] log-weights (unnormalized)
        """
        query = observation["image"]  # [H, W, 3]
        C = rendered_images.shape[0]

        # LPIPS expects [B, 3, H, W] in range [-1, 1]
        rendered = rendered_images.permute(0, 3, 1, 2) * 2.0 - 1.0  # [C, 3, H, W]
        target = query.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0  # [1, 3, H, W]

        # Resize target to match rendered if needed
        rh, rw = rendered.shape[2], rendered.shape[3]
        th, tw = target.shape[2], target.shape[3]
        if rh != th or rw != tw:
            target = F.interpolate(
                target, size=(rh, rw), mode="bilinear", align_corners=False
            )

        target = target.expand(C, -1, -1, -1)  # [C, 3, H, W]

        # Compute LPIPS distance in chunks to avoid OOM
        with torch.no_grad():
            distances = []
            for i in range(0, C, self.chunk_size):
                end = min(i + self.chunk_size, C)
                d = self.loss_fn(rendered[i:end], target[i:end])
                # d shape: [chunk, 1, 1, 1] -> squeeze to [chunk]
                distances.append(d.view(-1))
            distances = torch.cat(distances)  # [C]

        # Convert distance to log-weight (lower distance = higher weight)
        log_weights = -self.temperature * distances

        return log_weights
