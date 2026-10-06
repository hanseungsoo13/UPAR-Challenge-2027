"""Vanilla CLIP ViT-B/32 image/text encoder used by AttriVision."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_open_clip() -> Any:
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "AttriVision requires open_clip_torch. Recreate the conda environment "
            "from environment.yml or run `pip install open_clip_torch`."
        ) from exc
    return open_clip


class AttriVision(nn.Module):
    """Thin, fully fine-tunable wrapper around OpenCLIP's vanilla ViT-B/32."""

    def __init__(self, pretrained: str | None = "openai",
                 model_name: str = "ViT-B-32-quickgelu") -> None:
        super().__init__()
        open_clip = _import_open_clip()
        self.model_name = model_name
        self.pretrained = pretrained
        self.clip = open_clip.create_model(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name, context_length=77)

        output_dim = int(getattr(self.clip.visual, "output_dim", 0))
        if output_dim != 512:
            raise ValueError(f"{model_name} must produce 512-D image features, got {output_dim}")
        # Task-2 supervised head.  Keeping it on the wrapper makes the
        # retrieval path use exactly the same image representation as training.
        self.binary_head = nn.Linear(output_dim, 40)

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts)
        if tokens.ndim != 2 or tokens.shape[1] != 77:
            raise RuntimeError(f"Expected token shape [N,77], got {tuple(tokens.shape)}")
        return tokens

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.clip.encode_image(images), dim=-1)

    def binary_logits_from_features(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.binary_head(image_features.float())

    def binary_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.binary_logits_from_features(self.encode_image(images))

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.clip.encode_text(tokens), dim=-1)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.clip.logit_scale

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if text_tokens.ndim != 2 or text_tokens.shape[1] != 77:
            raise ValueError(f"Expected independent text tokens [T,77], got {tuple(text_tokens.shape)}")
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_tokens)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * image_features @ text_features.T
        return image_features, text_features, logits
