"""CLIP text-guided observation model."""

import torch
import torch.nn.functional as F
import open_clip

from .base import ObservationModel

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


class CLIPTextObservation(ObservationModel):
    """CLIP text-to-image similarity for text-guided localization."""

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
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self._mean = CLIP_MEAN.to(device).view(1, 3, 1, 1)
        self._std = CLIP_STD.to(device).view(1, 3, 1, 1)

        # Cache for text embeddings
        self._text_cache: dict[str, torch.Tensor] = {}

    @property
    def name(self) -> str:
        return "clip_text"

    @property
    def requires_image(self) -> bool:
        return False

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        x = images.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        return x

    @torch.no_grad()
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        if text not in self._text_cache:
            tokens = self.tokenizer([text]).to(self.device)
            text_feat = self.model.encode_text(tokens)  # [1, dim]
            text_feat = F.normalize(text_feat, dim=-1)
            self._text_cache[text] = text_feat
        return self._text_cache[text]

    @torch.no_grad()
    def compute_log_weights(
        self,
        rendered_images: torch.Tensor,
        observation: dict,
    ) -> torch.Tensor:
        """CLIP cosine similarity between rendered images and text.

        Args:
            rendered_images: [C, H, W, 3]
            observation: {"text": str}
        Returns:
            [C] log-weights
        """
        text = observation["text"]
        text_feat = self._get_text_embedding(text)  # [1, dim]

        rendered_clip = self._preprocess(rendered_images)
        image_feats = self.model.encode_image(rendered_clip)  # [C, dim]
        image_feats = F.normalize(image_feats, dim=-1)

        sim = (image_feats @ text_feat.T).squeeze(-1)  # [C]

        return self.temperature * sim
