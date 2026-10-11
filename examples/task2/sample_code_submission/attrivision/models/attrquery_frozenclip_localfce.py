"""AttrQuery-FrozenCLIP-LocalFCE model.

The A7 OpenCLIP model is kept as an unmodified, frozen submodule.  A forward
hook on the final ViT transformer output exposes the patch sequence without
changing the original CLIP forward path.  Only the category queries,
cross-attention head, and the new feature projection are trainable.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from upar.retrieval import autocast

from ..datasets.attribute_prompts import CategoryPromptMapper
from .attrivision import AttriVision


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _visual_width(visual: nn.Module) -> int:
    for candidate in (
        getattr(visual, "width", None),
        getattr(getattr(visual, "transformer", None), "width", None),
        getattr(getattr(visual, "trunk", None), "embed_dim", None),
    ):
        if candidate is not None:
            value = int(candidate)
            if value > 0:
                return value
    class_embedding = getattr(visual, "class_embedding", None)
    if torch.is_tensor(class_embedding) and class_embedding.numel() > 0:
        return int(class_embedding.numel())
    conv1 = getattr(visual, "conv1", None)
    out_channels = getattr(conv1, "out_channels", None)
    if out_channels is not None and int(out_channels) > 0:
        return int(out_channels)
    raise RuntimeError(
        "Unable to infer the OpenCLIP visual transformer width; "
        "the AttrQuery experiment requires a ViT-like image encoder."
    )


def _global_width(base: AttriVision, text_dim: int) -> int:
    value = getattr(base.clip.visual, "output_dim", None)
    return int(value) if value is not None and int(value) > 0 else text_dim


def _spatial_hook_target(visual: nn.Module) -> nn.Module:
    transformer = getattr(visual, "transformer", None)
    if transformer is not None:
        return transformer
    trunk = getattr(visual, "trunk", None)
    blocks = getattr(trunk, "blocks", None)
    if blocks is not None and len(blocks):
        return blocks[-1]
    raise RuntimeError(
        "The selected CLIP visual tower has no transformer/trunk block from "
        "which final spatial tokens can be extracted."
    )


def _batch_first_sequence(
    sequence: torch.Tensor, inputs: tuple[Any, ...], batch_size: int,
) -> torch.Tensor:
    if sequence.ndim != 3:
        raise RuntimeError(
            "Expected a 3-D ViT token sequence from the final transformer, "
            f"got {tuple(sequence.shape)}"
        )
    input_tensor = _first_tensor(inputs)
    input_shape = tuple(input_tensor.shape) if input_tensor is not None else ()
    if len(input_shape) == 3:
        if input_shape[0] == batch_size and sequence.shape[0] == batch_size:
            return sequence
        if input_shape[1] == batch_size and sequence.shape[1] == batch_size:
            return sequence.transpose(0, 1)
    if sequence.shape[0] == batch_size and sequence.shape[1] != batch_size:
        return sequence
    if sequence.shape[1] == batch_size:
        return sequence.transpose(0, 1)
    raise RuntimeError(
        "Could not identify the batch dimension in the captured ViT token "
        f"sequence {tuple(sequence.shape)} for batch size {batch_size}."
    )


def build_text_prototypes(
    base: AttriVision,
    mapper: CategoryPromptMapper,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    """Encode and normalize the 52 A7 text states exactly once."""
    base.eval()
    tokens = base.tokenize(mapper.prompts).to(device)
    with torch.inference_mode(), autocast(device, amp):
        prototypes = base.encode_text(tokens).float()
    prototypes = F.normalize(prototypes, dim=-1)
    if prototypes.ndim != 2 or prototypes.shape[0] != 52 or not torch.isfinite(prototypes).all():
        raise RuntimeError(f"Invalid 52-state text prototype tensor: {tuple(prototypes.shape)}")
    return prototypes.detach()


class CategoryCrossAttention(nn.Module):
    """One learnable query per category attending to the frozen patch sequence."""

    def __init__(
        self,
        num_categories: int,
        visual_dim: int,
        text_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
    ) -> None:
        super().__init__()
        if num_categories <= 0 or visual_dim <= 0 or text_dim <= 0:
            raise ValueError("Category attention dimensions must be positive")
        if visual_dim % num_heads:
            raise ValueError(
                f"visual_dim={visual_dim} must be divisible by num_heads={num_heads}"
            )
        self.category_queries = nn.Parameter(torch.empty(num_categories, visual_dim))
        nn.init.normal_(self.category_queries, std=0.02)
        self.attention = nn.MultiheadAttention(
            visual_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(visual_dim)
        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            hidden = 4 * visual_dim
            self.ffn = nn.Sequential(
                nn.Linear(visual_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, visual_dim),
            )
            self.norm2 = nn.LayerNorm(visual_dim)
        self.output_projection = nn.Linear(visual_dim, text_dim)

    def forward(
        self, patch_tokens: torch.Tensor, return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError(f"Expected patch tokens [B,N,D], got {tuple(patch_tokens.shape)}")
        batch_size = patch_tokens.shape[0]
        queries = self.category_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, attention = self.attention(
            queries, patch_tokens, patch_tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        output = self.norm1(queries + self.dropout(attended))
        if self.use_ffn:
            output = self.norm2(output + self.dropout(self.ffn(output)))
        features = F.normalize(self.output_projection(output.float()), dim=-1)
        if return_attention:
            if attention is None:
                raise RuntimeError("Cross-attention weights were requested but not returned")
            return features, attention
        return features


class AttrQueryFrozenCLIPLocalFCE(nn.Module):
    """Frozen A7 CLIP plus category-local visual features.

    ``feature_mode='local'`` returns one text-space feature per category,
    shaped ``[B,12,512]`` for the current A7 checkpoint.  ``feature_mode``
    ``'global'`` is the requested control mode: one projected global CLIP
    feature is shared by all twelve category slots.
    """

    def __init__(
        self,
        base_model: AttriVision,
        attribute_names: Sequence[str],
        text_prototypes: torch.Tensor,
        feature_mode: str = "local",
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
    ) -> None:
        super().__init__()
        if feature_mode not in {"local", "global"}:
            raise ValueError("feature_mode must be 'local' or 'global'")
        if text_prototypes.ndim != 2 or text_prototypes.shape[0] != 52:
            raise ValueError(
                f"Expected normalized text prototypes [52,D], got {tuple(text_prototypes.shape)}"
            )
        self.base = base_model
        self.attribute_names = list(attribute_names)
        self.mapper = CategoryPromptMapper(self.attribute_names)
        self.category_groups = tuple(tuple(group) for group in self.mapper.category_indices())
        self.feature_mode = feature_mode
        self.visual_dim = _visual_width(self.base.clip.visual)
        self.text_dim = int(text_prototypes.shape[1])
        self.num_categories = len(self.category_groups)
        self.num_states = int(text_prototypes.shape[0])
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.use_ffn = bool(use_ffn)
        self.base_model_name = self.base.model_name
        self.spatial_hook_target_name = type(_spatial_hook_target(self.base.clip.visual)).__name__

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

        if feature_mode == "local":
            self.local_head = CategoryCrossAttention(
                self.num_categories, self.visual_dim, self.text_dim,
                num_heads=num_heads, dropout=dropout, use_ffn=use_ffn,
            )
        else:
            self.global_projection = nn.Linear(
                _global_width(self.base, self.text_dim), self.text_dim,
            )
        self.register_buffer("text_prototypes", F.normalize(text_prototypes.float(), dim=-1))

    def train(self, mode: bool = True) -> "AttrQueryFrozenCLIPLocalFCE":
        # ``model.train()`` recursively switches every child to train mode.
        # Restore the frozen CLIP to eval mode on every call.
        super().train(mode)
        self.base.eval()
        return self

    def _frozen_global(self, images: torch.Tensor) -> torch.Tensor:
        self.base.eval()
        with torch.no_grad():
            features = self.base.encode_image(images)
        return features.float().detach()

    def _frozen_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        self.base.eval()
        target = _spatial_hook_target(self.base.clip.visual)
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            sequence = _first_tensor(output)
            if sequence is None:
                raise RuntimeError("The CLIP transformer hook did not return a tensor")
            captured["tokens"] = _batch_first_sequence(sequence, inputs, images.shape[0])

        handle = target.register_forward_hook(hook)
        try:
            with torch.no_grad():
                # Calling the original CLIP image path preserves its projection,
                # normalization, and all model state behavior.
                _ = self.base.encode_image(images)
        finally:
            handle.remove()
        sequence = captured.get("tokens")
        if sequence is None:
            raise RuntimeError("No spatial token sequence was captured from the CLIP ViT")
        visual = self.base.clip.visual
        has_cls = getattr(visual, "class_embedding", None) is not None
        patches = sequence[:, 1:] if has_cls else sequence
        if patches.ndim != 3 or patches.shape[1] <= 0:
            raise RuntimeError(f"Invalid CLIP patch token shape after CLS removal: {tuple(patches.shape)}")
        if patches.shape[-1] != self.visual_dim:
            raise RuntimeError(
                f"Captured patch dimension {patches.shape[-1]} differs from visual dimension {self.visual_dim}"
            )
        return patches.float().detach()

    def category_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "global":
            global_features = self._frozen_global(images)
            output = self.global_projection(global_features)
            output = F.normalize(output.float(), dim=-1)
            return output.unsqueeze(1).expand(-1, self.num_categories, -1)
        patch_tokens = self._frozen_patch_tokens(images)
        return self.local_head(patch_tokens)

    def category_features_with_attention(
        self, images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Return local features and per-head ``[B,H,C,N]`` attention weights.

        This method is inference-only by contract and is intentionally kept
        separate from the regular training forward path.
        """
        if self.feature_mode != "local":
            raise RuntimeError("Attention diagnostics require feature_mode='local'")
        patch_tokens = self._frozen_patch_tokens(images)
        result = self.local_head(patch_tokens, return_attention=True)
        features, attention = result
        patch_count = patch_tokens.shape[1]
        visual = self.base.clip.visual
        conv1 = getattr(visual, "conv1", None)
        stride = getattr(conv1, "stride", (1, 1))
        if isinstance(stride, int):
            stride_h = stride_w = int(stride)
        else:
            stride_h, stride_w = int(stride[0]), int(stride[1])
        grid_h = (int(images.shape[-2]) + stride_h - 1) // stride_h
        grid_w = (int(images.shape[-1]) + stride_w - 1) // stride_w
        if grid_h * grid_w != patch_count:
            side = int(round(patch_count ** 0.5))
            if side * side != patch_count:
                raise RuntimeError(
                    f"Cannot reconstruct patch grid for {patch_count} tokens "
                    f"from input {tuple(images.shape[-2:])}"
                )
            grid_h = grid_w = side
        if attention.shape[-1] != patch_count:
            raise RuntimeError(
                f"Attention source length {attention.shape[-1]} differs from patch count {patch_count}"
            )
        return features, attention, (grid_h, grid_w)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.category_features(images)

    def category_state_scores(self, images: torch.Tensor) -> torch.Tensor:
        """Return all state similarities as ``[B,12,52]``."""
        features = self.category_features(images)
        return torch.einsum("bcd,sd->bcs", features.float(), self.text_prototypes.float())

    def encode_text(self, _tokens: torch.Tensor | None = None) -> torch.Tensor:
        """Expose cached prototypes; text encoding is deliberately not trainable."""
        return self.text_prototypes

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        frozen = sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)
        return {"total": total, "frozen": frozen, "trainable": total - frozen}

    def assert_frozen_backbone(self) -> None:
        if any(parameter.requires_grad for parameter in self.base.parameters()):
            raise RuntimeError("Frozen CLIP contains a parameter with requires_grad=True")
        if self.base.training:
            raise RuntimeError("Frozen CLIP must remain in eval mode")
