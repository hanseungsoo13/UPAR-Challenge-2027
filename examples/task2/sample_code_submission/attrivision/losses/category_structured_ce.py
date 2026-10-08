"""Category-local soft-target cross entropy for Native52 geometry."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoryStructuredCELoss(nn.Module):
    """Average equally weighted CE terms over the 12 semantic categories."""

    def __init__(self, category_indices: Sequence[Sequence[int]],
                 temperature: float = 0.01) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("Category CE temperature must be positive")
        groups = [tuple(int(index) for index in group) for group in category_indices]
        covered = [index for group in groups for index in group]
        if len(groups) != 12 or len(covered) != 52 or sorted(covered) != list(range(52)):
            raise ValueError("Category CE requires a 12-category partition of states 0..51")
        if any(not group for group in groups):
            raise ValueError("Category CE groups cannot be empty")
        self.groups = groups
        self.temperature = float(temperature)

    def forward(self, image_features: torch.Tensor, state_features: torch.Tensor,
                semantic_targets: torch.Tensor) -> torch.Tensor:
        if image_features.ndim != 2 or state_features.ndim != 2:
            raise ValueError("Category CE expects image [B,D] and state [52,D] features")
        if state_features.shape != (52, image_features.shape[1]):
            raise ValueError(
                f"Expected state features [52,{image_features.shape[1]}], got {tuple(state_features.shape)}"
            )
        if semantic_targets.shape != (image_features.shape[0], 52):
            raise ValueError(f"Expected semantic targets [B,52], got {tuple(semantic_targets.shape)}")

        logits = (image_features.float() @ state_features.float().T) / self.temperature
        targets = semantic_targets.float()
        category_losses = []
        for indices in self.groups:
            category_target = targets[:, indices]
            counts = category_target.sum(dim=1, keepdim=True)
            if (counts <= 0).any():
                raise ValueError("Every sample must have an active state in every category")
            category_target = category_target / counts
            log_probabilities = F.log_softmax(logits[:, indices], dim=1)
            category_losses.append(-(category_target * log_probabilities).sum(dim=1))
        return torch.stack(category_losses, dim=1).mean()
