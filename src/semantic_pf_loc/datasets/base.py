"""Abstract base class for localization datasets."""

from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset


class BaseLocalizationDataset(Dataset, ABC):
    """Base dataset providing RGB images with ground-truth poses."""

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Returns dict with keys:
        - image: [H, W, 3] float32 in [0, 1]
        - pose: [4, 4] float64 camera-to-world
        - K: [3, 3] float64 intrinsics
        - depth: [H, W] float32 meters (or None)
        - timestamp: float or int
        """
        ...

    @abstractmethod
    def get_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(bounds_min [3], bounds_max [3]) scene spatial bounds."""
        ...

    @abstractmethod
    def get_intrinsics(self) -> torch.Tensor:
        """Camera intrinsics [3, 3]."""
        ...

    @property
    @abstractmethod
    def image_size(self) -> tuple[int, int]:
        """(width, height)."""
        ...

    def get_train_indices(self, stride: int = 1) -> list[int]:
        return list(range(0, len(self), stride))

    def get_eval_indices(self, stride: int = 5) -> list[int]:
        return list(range(0, len(self), stride))
