"""ConvNeXt model, ratio-weighted BCE, and exponential moving average."""
from __future__ import annotations

import copy
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Base_Weights, convnext_base

from .config import NUM_ATTRIBUTES


class UPARModel(nn.Module):
    """ImageNet ConvNeXt-B + average pooling + dropout + linear classifier."""

    def __init__(self, num_attributes: int = NUM_ATTRIBUTES, dropout: float = 0.7,
                 pretrained: bool = True) -> None:
        super().__init__()
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = convnext_base(weights=weights)
        feature_dim = backbone.classifier[-1].in_features
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feature_norm = backbone.classifier[0]
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_attributes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_norm(self.avgpool(self.features(images)))
        return self.classifier(self.dropout(self.flatten(features)))


class WeightedBCELoss(nn.Module):
    """Ratio-weighted BCE with UPAR binary label smoothing."""

    def __init__(self, positive_ratios: Sequence[float], alpha: float = 0.05,
                 weighted: bool = True) -> None:
        super().__init__()
        ratios = torch.as_tensor(positive_ratios, dtype=torch.float32)
        if ratios.shape != (NUM_ATTRIBUTES,):
            raise ValueError("positive_ratios must have shape [40]")
        if not 0.0 <= alpha < 0.5:
            raise ValueError("label smoothing alpha must be in [0, 0.5)")
        self.register_buffer("positive_ratios", ratios)
        self.alpha = alpha
        self.weighted = weighted

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = (1.0 - self.alpha) * targets + self.alpha * (1.0 - targets)
        loss = F.binary_cross_entropy_with_logits(logits, smoothed, reduction="none")
        if self.weighted:
            weights = torch.where(
                targets > 0.5,
                torch.exp(1.0 - self.positive_ratios),
                torch.exp(self.positive_ratios),
            )
            loss = loss * weights
        return loss.mean()


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = dict(model.named_parameters())
        for name, value in self.module.named_parameters():
            value.mul_(self.decay).add_(current[name].detach(), alpha=1.0 - self.decay)
        ema_buffers = dict(self.module.named_buffers())
        for name, value in model.named_buffers():
            ema_buffers[name].copy_(value.detach())
