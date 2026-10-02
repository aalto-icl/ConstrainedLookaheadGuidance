"""Differentiable edit rewards used by CLAG."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _preprocess_clip(image: torch.Tensor) -> torch.Tensor:
    """Convert a differentiable image tensor from [-1, 1] to CLIP input."""
    image = F.interpolate(
        (image + 1.0) / 2.0,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )
    mean = image.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (image - mean) / std


class CLIPContrastiveReward:
    """CLIP(image, target prompt) minus CLIP(image, source prompt)."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self._text_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _text_features(
        self,
        source_prompt: str,
        edit_prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (source_prompt, edit_prompt)
        if key not in self._text_cache:
            inputs = self.processor(
                text=[source_prompt, edit_prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                features = F.normalize(features, dim=-1)
            self._text_cache[key] = (features[0:1], features[1:2])
        return self._text_cache[key]

    def __call__(
        self,
        image: torch.Tensor,
        source_prompt: str,
        edit_prompt: str,
    ) -> torch.Tensor:
        image_features = self.model.get_image_features(
            pixel_values=_preprocess_clip(image.to(self.device))
        )
        image_features = F.normalize(image_features, dim=-1)
        source_features, edit_features = self._text_features(source_prompt, edit_prompt)
        positive = (image_features * edit_features).sum(dim=-1).mean()
        negative = (image_features * source_features).sum(dim=-1).mean()
        return positive - negative
