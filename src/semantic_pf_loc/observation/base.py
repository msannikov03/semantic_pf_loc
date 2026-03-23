"""Abstract base class for observation models."""

from abc import ABC, abstractmethod
import torch


class ObservationModel(ABC):
    """Base class for particle filter observation models."""

    @abstractmethod
    def compute_log_weights(
        self,
        rendered_images: torch.Tensor,
        observation: dict,
    ) -> torch.Tensor:
        """Compute log-weights for particles.

        Args:
            rendered_images: [C, H, W, 3] float32 in [0, 1]
            observation: {"image": [H,W,3]} or {"text": str}
        Returns:
            [C] log-weights (unnormalized)
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def requires_image(self) -> bool:
        return True
