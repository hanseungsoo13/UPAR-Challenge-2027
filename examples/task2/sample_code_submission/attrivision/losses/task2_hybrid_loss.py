"""Task-2-aligned semantic prototype and attribute-set contrastive objective."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal_clip_loss import _directional_loss


@dataclass
class Task2LossOutput:
    loss: torch.Tensor
    prototype: torch.Tensor
    set_contrastive: torch.Tensor
    i2t: torch.Tensor
    t2i: torch.Tensor


class Task2HybridLoss(nn.Module):
    """Supervise every semantic state and the complete retrieval query.

    The prototype term is a class-balanced sigmoid focal loss over all text
    states. The set term constructs the same mean text descriptor used during
    Task 2 evaluation and aligns it bidirectionally with the gallery image.
    """

    def __init__(self, positive_ratios: torch.Tensor, focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0, balance_max_weight: float = 10.0,
                 prototype_weight: float = 1.0, set_weight: float = 1.0) -> None:
        super().__init__()
        if positive_ratios.ndim != 1:
            raise ValueError("positive_ratios must have shape [S]")
        if focal_alpha < 0 or focal_gamma < 0 or balance_max_weight < 1:
            raise ValueError("invalid focal or balancing hyperparameter")
        if prototype_weight < 0 or set_weight < 0 or prototype_weight + set_weight == 0:
            raise ValueError("at least one hybrid loss weight must be positive")
        ratios = positive_ratios.float().clamp(1e-4, 1.0 - 1e-4)
        positive_weights = (0.5 / ratios).clamp(max=balance_max_weight)
        negative_weights = (0.5 / (1.0 - ratios)).clamp(max=balance_max_weight)
        self.register_buffer("positive_weights", positive_weights)
        self.register_buffer("negative_weights", negative_weights)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.prototype_weight = prototype_weight
        self.set_weight = set_weight

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor,
                semantic_labels: torch.Tensor,
                logit_scale: torch.Tensor) -> Task2LossOutput:
        if image_features.ndim != 2 or text_features.ndim != 2:
            raise ValueError("image and text features must be matrices")
        if semantic_labels.shape != (len(image_features), len(text_features)):
            raise ValueError("semantic labels must have shape [B,S]")
        targets = semantic_labels.to(dtype=torch.float32)
        scale = logit_scale.exp().clamp(max=100.0)
        prototype_logits = scale * image_features @ text_features.T

        probabilities = torch.sigmoid(prototype_logits.float())
        pt = torch.where(targets.bool(), probabilities, 1.0 - probabilities)
        balancing = torch.where(
            targets.bool(), self.positive_weights, self.negative_weights,
        )
        focal = (1.0 - pt).pow(self.focal_gamma)
        binary_ce = F.binary_cross_entropy_with_logits(
            prototype_logits.float(), targets, reduction="none",
        )
        prototype = (self.focal_alpha * balancing * focal * binary_ce).mean()

        counts = targets.sum(dim=1, keepdim=True)
        if (counts == 0).any():
            raise ValueError("every image must have at least one semantic state")
        query_features = F.normalize((targets @ text_features.float()) / counts, dim=-1)
        set_logits = scale.float() * image_features.float() @ query_features.T
        overlap = targets @ targets.T
        semantic_counts = targets.sum(dim=1)
        # Binary rows are identical iff their intersection equals both sizes.
        # This avoids constructing a temporary [B,B,S] equality tensor.
        positive_mask = (
            (overlap == semantic_counts[:, None])
            & (overlap == semantic_counts[None, :])
        )
        positive_mask.fill_diagonal_(True)
        i2t = _directional_loss(
            set_logits, positive_mask, True, self.focal_alpha, self.focal_gamma,
        )
        t2i = _directional_loss(
            set_logits.T, positive_mask.T, True, self.focal_alpha, self.focal_gamma,
        )
        set_contrastive = 0.5 * (i2t + t2i)
        total = self.prototype_weight * prototype + self.set_weight * set_contrastive
        return Task2LossOutput(total, prototype, set_contrastive, i2t, t2i)
