"""Mixed categorical/multi-label objective used by A7-mixed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal_clip_loss import _directional_loss


@dataclass
class MixedStateLossOutput:
    loss: torch.Tensor
    prototype: torch.Tensor
    set_contrastive: torch.Tensor
    i2t: torch.Tensor
    t2i: torch.Tensor
    single_ce: torch.Tensor
    multi_bce: torch.Tensor
    consistency: torch.Tensor


class MixedStateHybridLoss(nn.Module):
    """Supervise categorical and multi-label state groups separately.

    Single-label groups use a category-local softmax cross entropy.  Multi-label
    groups use class-balanced focal BCE, while retaining the same set-level
    image/query contrastive term used by the Task-2 hybrid objective.
    """

    def __init__(
        self,
        positive_ratios: torch.Tensor,
        group_specs: Sequence[tuple[str, Sequence[int]]],
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        balance_max_weight: float = 10.0,
        prototype_weight: float = 0.25,
        set_weight: float = 1.0,
        consistency_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if positive_ratios.ndim != 1:
            raise ValueError("positive_ratios must have shape [S]")
        if focal_alpha < 0 or focal_gamma < 0 or balance_max_weight < 1:
            raise ValueError("invalid focal or balancing hyperparameter")
        if prototype_weight < 0 or set_weight < 0 or prototype_weight + set_weight == 0:
            raise ValueError("at least one hybrid loss weight must be positive")
        if consistency_weight < 0:
            raise ValueError("consistency_weight cannot be negative")
        self.group_specs = [
            (str(kind), tuple(int(index) for index in indices))
            for kind, indices in group_specs
        ]
        if not self.group_specs or any(kind not in {"single", "multi"} for kind, _ in self.group_specs):
            raise ValueError("group_specs must contain single or multi groups")
        covered = [index for _, indices in self.group_specs for index in indices]
        if sorted(covered) != list(range(len(positive_ratios))) or len(set(covered)) != len(covered):
            raise ValueError("group_specs must partition all semantic states")
        ratios = positive_ratios.float().clamp(1e-4, 1.0 - 1e-4)
        self.register_buffer("positive_weights", (0.5 / ratios).clamp(max=balance_max_weight))
        self.register_buffer(
            "negative_weights", (0.5 / (1.0 - ratios)).clamp(max=balance_max_weight),
        )
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.prototype_weight = float(prototype_weight)
        self.set_weight = float(set_weight)
        self.consistency_weight = float(consistency_weight)

    @staticmethod
    def _category_balanced_query(
        text_features: torch.Tensor,
        targets: torch.Tensor,
        group_specs: Sequence[tuple[str, Sequence[int]]],
    ) -> torch.Tensor:
        """Average states within each category, then average categories."""
        accumulator = text_features.new_zeros((len(targets), text_features.shape[1]))
        valid_groups = text_features.new_zeros((len(targets), 1))
        for _, indices in group_specs:
            group_targets = targets[:, indices]
            counts = group_targets.sum(dim=1, keepdim=True)
            valid = counts > 0
            state_mean = group_targets @ text_features[list(indices)] / counts.clamp_min(1.0)
            accumulator = accumulator + torch.where(valid, F.normalize(state_mean, dim=-1), 0.0)
            valid_groups = valid_groups + valid.to(valid_groups.dtype)
        if (valid_groups <= 0).any():
            raise ValueError("Every query must contain at least one category state")
        return F.normalize(accumulator / valid_groups, dim=-1)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        semantic_labels: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> MixedStateLossOutput:
        if image_features.ndim != 2 or text_features.ndim != 2:
            raise ValueError("image and text features must be matrices")
        if semantic_labels.shape != (len(image_features), len(text_features)):
            raise ValueError("semantic labels must have shape [B,S]")
        targets = semantic_labels.float()
        scale = logit_scale.exp().clamp(max=100.0)
        prototype_logits = scale * image_features.float() @ text_features.float().T

        single_terms: list[torch.Tensor] = []
        multi_terms: list[torch.Tensor] = []
        consistency_terms: list[torch.Tensor] = []
        for kind, indices in self.group_specs:
            group_logits = prototype_logits[:, list(indices)]
            group_targets = targets[:, list(indices)]
            counts = group_targets.sum(dim=1, keepdim=True)
            valid = counts[:, 0] > 0
            if not valid.any():
                continue
            if kind == "single":
                normalized = group_targets[valid] / counts[valid].clamp_min(1.0)
                log_prob = F.log_softmax(group_logits[valid].float(), dim=1)
                single_terms.append(-(normalized * log_prob).sum(dim=1).mean())
                continue

            logits = group_logits[valid].float()
            local_targets = group_targets[valid]
            probabilities = torch.sigmoid(logits)
            pt = torch.where(local_targets.bool(), probabilities, 1.0 - probabilities)
            balancing = torch.where(
                local_targets.bool(),
                self.positive_weights[list(indices)],
                self.negative_weights[list(indices)],
            )
            binary_ce = F.binary_cross_entropy_with_logits(
                logits, local_targets, reduction="none",
            )
            focal = (1.0 - pt).pow(self.focal_gamma)
            multi_terms.append((self.focal_alpha * balancing * focal * binary_ce).mean())

            # ``other`` is the final state for hair/lower_type.  It is a
            # residual state, so it should not be simultaneously high with a
            # known state even though BCE is otherwise multi-label.
            if len(indices) > 1 and (
                (kind == "multi" and len(indices) in {3, 4})
            ):
                known = probabilities[:, :-1]
                residual = probabilities[:, -1]
                consistency_terms.append(
                    F.relu(residual + known.max(dim=1).values - 1.0).mean()
                )

        if not single_terms or not multi_terms:
            raise ValueError("Mixed A7 requires both single and multi supervision terms")
        single_ce = torch.stack(single_terms).mean()
        multi_bce = torch.stack(multi_terms).mean()
        consistency = (
            torch.stack(consistency_terms).mean()
            if consistency_terms else prototype_logits.new_zeros(())
        )
        prototype = single_ce + multi_bce + self.consistency_weight * consistency

        query_features = self._category_balanced_query(text_features, targets, self.group_specs)
        set_logits = scale.float() * image_features.float() @ query_features.T
        overlap = targets @ targets.T
        semantic_counts = targets.sum(dim=1)
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
        return MixedStateLossOutput(
            total, prototype, set_contrastive, i2t, t2i,
            single_ce, multi_bce, consistency,
        )
