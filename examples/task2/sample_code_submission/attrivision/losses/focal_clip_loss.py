"""Symmetric diagonal or multi-positive CLIP/Focal-CLIP objectives."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContrastiveLossOutput:
    loss: torch.Tensor
    i2t: torch.Tensor
    t2i: torch.Tensor


def _directional_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    focal: bool,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    if logits.shape != positive_mask.shape:
        raise ValueError("logits and positive_mask must have identical shapes")
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every contrastive anchor must have at least one positive")
    log_prob = F.log_softmax(logits.float(), dim=1)
    probability = log_prob.exp()
    weights = positive_mask.to(log_prob.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True)
    modulation = (1.0 - probability).pow(gamma) if focal else torch.ones_like(probability)
    scale = alpha if focal else 1.0
    return (-(scale * modulation * log_prob * weights).sum(dim=1)).mean()


class FocalCLIPLoss(nn.Module):
    def __init__(self, loss: str = "focal_clip", target: str = "multi_positive",
                 alpha: float = 1.0, gamma: float = 2.0) -> None:
        super().__init__()
        if loss not in {"clip", "focal_clip"}:
            raise ValueError("loss must be 'clip' or 'focal_clip'")
        if target not in {"diagonal", "multi_positive"}:
            raise ValueError("target must be 'diagonal' or 'multi_positive'")
        if alpha < 0 or gamma < 0:
            raise ValueError("focal alpha and gamma must be non-negative")
        self.focal = loss == "focal_clip"
        self.target = target
        self.alpha = alpha
        self.gamma = gamma

    def positive_mask(self, logits: torch.Tensor, labels: torch.Tensor,
                      selected_attributes: torch.Tensor,
                      text_owners: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"Expected [B,T] logits, got {tuple(logits.shape)}")
        batch_size, text_count = logits.shape
        if labels.ndim != 2 or labels.shape[0] != batch_size:
            raise ValueError("labels must have shape [B,A] matching the image logits")
        if selected_attributes.shape != (text_count, labels.shape[1]):
            raise ValueError("selected_attributes must have shape [T,A]")
        if text_owners.shape != (text_count,) or text_owners.dtype != torch.long:
            raise ValueError("text_owners must be a LongTensor with shape [T]")
        if text_count == 0 or (text_owners < 0).any() or (text_owners >= batch_size).any():
            raise ValueError("text_owners contains an invalid image index")
        owned_are_positive = (
            labels[text_owners].bool() & selected_attributes.bool()
        ).any(dim=1)
        if not owned_are_positive.all():
            raise ValueError("Every text must describe a true semantic state of its owner image")

        if self.target == "diagonal":
            positive_mask = torch.zeros_like(logits, dtype=torch.bool)
            positive_mask[text_owners, torch.arange(text_count, device=logits.device)] = True
        else:
            positive_mask = (labels.float() @ selected_attributes.float().T) > 0
        return positive_mask

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                selected_attributes: torch.Tensor,
                text_owners: torch.Tensor) -> ContrastiveLossOutput:
        positive_mask = self.positive_mask(
            logits, labels, selected_attributes, text_owners,
        )
        i2t = _directional_loss(logits, positive_mask, self.focal, self.alpha, self.gamma)
        t2i = _directional_loss(logits.T, positive_mask.T, self.focal, self.alpha, self.gamma)
        return ContrastiveLossOutput(0.5 * (i2t + t2i), i2t, t2i)
