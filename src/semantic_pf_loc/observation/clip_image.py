"""CLIP image-to-image observation model."""

import torch
import torch.nn.functional as F
import open_clip

from .base import ObservationModel

# CLIP normalization constants
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


class CLIPImageObservation(ObservationModel):
    """CLIP feature similarity between rendered and observed images."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
        temperature: float = 10.0,
    ):
        self.device = device
        self.temperature = temperature

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device).eval()

        self._mean = CLIP_MEAN.to(device).view(1, 3, 1, 1)
        self._std = CLIP_STD.to(device).view(1, 3, 1, 1)

    @property
    def name(self) -> str:
        return "clip_image"

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for CLIP: resize to 224x224, normalize.
        Args: images [B, H, W, 3] float [0, 1]
        Returns: [B, 3, 224, 224]
        """
        x = images.permute(0, 3, 1, 2)  # [B, 3, H, W]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        return x

    @torch.no_grad()
    def compute_log_weights(
        self,
        rendered_images: torch.Tensor,
        observation: dict,
    ) -> torch.Tensor:
        """CLIP cosine similarity between rendered and query images.

        Args:
            rendered_images: [C, H, W, 3]
            observation: {"image": [H, W, 3]}
        Returns:
            [C] log-weights
        """
        query = observation["image"].unsqueeze(0)  # [1, H, W, 3]

        # Preprocess
        rendered_clip = self._preprocess(rendered_images)  # [C, 3, 224, 224]
        query_clip = self._preprocess(query)  # [1, 3, 224, 224]

        # Encode all in one batch
        all_images = torch.cat([rendered_clip, query_clip], dim=0)  # [C+1, 3, 224, 224]
        features = self.model.encode_image(all_images)  # [C+1, dim]
        features = F.normalize(features, dim=-1)

        rendered_feats = features[:-1]  # [C, dim]
        query_feat = features[-1:]  # [1, dim]

        # Cosine similarity
        sim = (rendered_feats @ query_feat.T).squeeze(-1)  # [C]

        return self.temperature * sim
