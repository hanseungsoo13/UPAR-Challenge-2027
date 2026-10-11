"""Category-local FCE objective for AttrQuery-FrozenCLIP."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CategoryLocalFCEOutput:
    loss: torch.Tensor
    category_loss: torch.Tensor
    category_accuracy: torch.Tensor
    valid_counts: torch.Tensor


class CategoryLocalFCELoss(nn.Module):
    """Focal category-local contrastive loss over a 52-state vocabulary.

    Each category feature is compared only with the states in its own group.
    The returned loss is averaged over valid image/category pairs, so a large
    category cannot dominate merely because it contains more states.
    """

    def __init__(
        self,
        category_indices: Sequence[Sequence[int]],
        temperature: float = 0.01,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("category-local FCE temperature must be positive")
        if gamma < 0:
            raise ValueError("focal gamma must be non-negative")
        self.groups = tuple(tuple(int(index) for index in group) for group in category_indices)
        covered = [index for group in self.groups for index in group]
        if (
            len(self.groups) != 12
            or len(covered) != 52
            or sorted(covered) != list(range(52))
            or any(not group for group in self.groups)
        ):
            raise ValueError("category-local FCE requires a 12-way partition of states 0..51")
        self.temperature = float(temperature)
        self.gamma = float(gamma)

    def forward(
        self,
        category_features: torch.Tensor,
        text_prototypes: torch.Tensor,
        semantic_targets: torch.Tensor,
        valid_categories: torch.Tensor | None = None,
    ) -> CategoryLocalFCEOutput:
        if category_features.ndim != 3:
            raise ValueError(
                f"Expected category features [B,12,D], got {tuple(category_features.shape)}"
            )
        batch_size, category_count, feature_dim = category_features.shape
        if category_count != len(self.groups):
            raise ValueError(f"Expected {len(self.groups)} category features, got {category_count}")
        if text_prototypes.shape != (52, feature_dim):
            raise ValueError(
                f"Expected text prototypes [52,{feature_dim}], got {tuple(text_prototypes.shape)}"
            )
        if semantic_targets.shape != (batch_size, 52):
            raise ValueError(
                f"Expected semantic targets [B,52], got {tuple(semantic_targets.shape)}"
            )
        if valid_categories is not None and valid_categories.shape != (batch_size, category_count):
            raise ValueError(
                f"Expected valid category mask [B,{category_count}], got {tuple(valid_categories.shape)}"
            )

        features = F.normalize(category_features.float(), dim=-1)
        prototypes = F.normalize(text_prototypes.float(), dim=-1)
        targets = semantic_targets.float()
        if valid_categories is None:
            valid_categories = torch.ones(
                batch_size, category_count, dtype=torch.bool, device=targets.device,
            )
        else:
            valid_categories = valid_categories.bool()

        total_loss = features.new_zeros(())
        total_valid = features.new_zeros(())
        category_losses: list[torch.Tensor] = []
        category_accuracies: list[torch.Tensor] = []
        valid_counts: list[torch.Tensor] = []

        for category_index, group in enumerate(self.groups):
            indices = torch.as_tensor(group, device=features.device, dtype=torch.long)
            category_target = targets[:, indices]
            target_count = category_target.sum(dim=1)
            valid = valid_categories[:, category_index] & (target_count > 0)
            count = valid.sum()
            valid_counts.append(count.to(features.dtype))
            if not bool(valid.any()):
                category_losses.append(features.new_zeros(()))
                category_accuracies.append(features.new_zeros(()))
                continue

            logits = torch.matmul(features[valid, category_index], prototypes[indices].T)
            logits = logits / self.temperature
            target = category_target[valid] / target_count[valid].unsqueeze(1)
            log_probability = F.log_softmax(logits.float(), dim=1)
            probability = log_probability.exp()
            target_probability = (target * probability).sum(dim=1).clamp(0.0, 1.0)
            negative_log_likelihood = -(target * log_probability).sum(dim=1)
            focal_weight = (1.0 - target_probability).pow(self.gamma)
            losses = focal_weight * negative_log_likelihood
            category_loss = losses.mean()
            category_losses.append(category_loss)
            total_loss = total_loss + losses.sum()
            total_valid = total_valid + count.to(total_valid.dtype)

            prediction = logits.argmax(dim=1)
            target_indices = target.argmax(dim=1)
            category_accuracies.append((prediction == target_indices).float().mean())

        loss = total_loss / total_valid.clamp_min(1.0)
        return CategoryLocalFCEOutput(
            loss=loss,
            category_loss=torch.stack(category_losses),
            category_accuracy=torch.stack(category_accuracies),
            valid_counts=torch.stack(valid_counts),
        )
