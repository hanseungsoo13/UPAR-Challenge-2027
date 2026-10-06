"""Zero-shot binary attribute probabilities from paired CLIP prompts."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from upar.retrieval import autocast

from ..datasets.attribute_prompts import prompt_pairs_for_attributes


@torch.inference_mode()
def encode_prompt_pairs(model: Any, attribute_names: Sequence[str],
                        device: torch.device, amp: bool) -> torch.Tensor:
    """Encode 40 negative/positive prompt pairs as normalized [40,2,D]."""
    negative, positive = prompt_pairs_for_attributes(attribute_names)
    # Interleave the pair so index 0 is negative and index 1 is positive.
    prompts = [text for pair in zip(negative, positive) for text in pair]
    model.eval()
    with autocast(device, amp):
        features = model.encode_text(model.tokenize(prompts).to(device))
    return features.float().cpu().reshape(len(attribute_names), 2, -1)


def learned_inverse_temperature(model: Any) -> float:
    """Return exp(logit_scale)=1/T with CLIP's conventional upper bound."""
    return float(model.logit_scale.detach().float().exp().clamp(max=100.0).cpu())


def paired_attribute_probabilities(
    gallery_features: torch.Tensor,
    prompt_features: torch.Tensor,
    inverse_temperature: float,
) -> np.ndarray:
    """Compute softmax([s-,s+]/T)_+ for every gallery image and attribute."""
    if gallery_features.ndim != 2 or prompt_features.ndim != 3:
        raise ValueError("Expected gallery [G,D] and prompts [A,2,D]")
    if prompt_features.shape[1] != 2 or gallery_features.shape[1] != prompt_features.shape[2]:
        raise ValueError(
            f"Incompatible gallery/prompt shapes: {gallery_features.shape}, {prompt_features.shape}"
        )
    if inverse_temperature <= 0:
        raise ValueError("inverse_temperature must be positive")
    # [G,D] x [A,2,D] -> [G,A,2]. Encoders already L2-normalize features,
    # so these are cosine similarities before temperature scaling.
    logits = torch.einsum("gd,akd->gak", gallery_features.float(), prompt_features.float())
    probabilities = (logits * inverse_temperature).softmax(dim=-1)[..., 1]
    result = probabilities.numpy().astype(np.float32, copy=False)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise RuntimeError("Paired-prompt inference produced invalid probabilities")
    return result
