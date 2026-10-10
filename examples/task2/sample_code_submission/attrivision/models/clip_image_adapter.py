"""Small trainable image adapter on top of a frozen AttriVision CLIP model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attrivision import AttriVision


class ResidualImageAdapter(nn.Module):
    """Identity-initialized bottleneck adapter for normalized CLIP features."""

    def __init__(self, feature_dim: int = 512, bottleneck_dim: int = 128) -> None:
        super().__init__()
        if feature_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("feature_dim and bottleneck_dim must be positive")
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        # Start from the original CLIP embedding.  Only the residual branch is
        # learned during the feasibility experiment.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.gelu(self.down(features.float())))
        return F.normalize(features.float() + residual, dim=-1)


class FrozenCLIPImageAdapter(nn.Module):
    """Expose adapted image features while keeping the base CLIP fixed."""

    def __init__(
        self,
        base_model: AttriVision,
        bottleneck_dim: int = 128,
    ) -> None:
        super().__init__()
        self.base = base_model
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.image_adapter = ResidualImageAdapter(512, bottleneck_dim)

    @property
    def model_name(self) -> str:
        return self.base.model_name

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.base.logit_scale.detach()

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        return self.base.tokenize(texts)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_features = self.base.encode_image(images)
        return self.image_adapter(base_features)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.base.encode_text(tokens)
