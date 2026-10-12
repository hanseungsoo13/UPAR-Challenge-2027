"""AttrQuery AttributeAdapter variants.

The original AttrQuery local head is intentionally kept intact.  This module
contains a drop-in local head with three zero-initialized transformations:

* ``FeatureAdapter`` applies a bottleneck residual MLP to the attended feature;
* ``CategoryFiLM`` creates channel-wise scale/shift from the category query;
* ``PatchFiLM`` modulates the projected value of every patch/head before the
  existing attention weights aggregate it.

``PatchFiLM`` implements the actual ``nn.MultiheadAttention`` tensor layout:
Q, K and V are projected, reshaped to ``[B,H,C/N,D_head]``, and only V is
modulated.  Its attention probabilities are the same softmax probabilities as
the original head.  The ``NoAdapter`` path calls the original PyTorch module
directly, which provides an exact control condition.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attrquery_frozenclip_localfce import (
    CategoryCrossAttention,
    _batch_first_sequence,
    _first_tensor,
    _spatial_hook_target,
    _visual_width,
)
from .attrivision import AttriVision
from ..datasets.attribute_prompts import CategoryPromptMapper


ADAPTER_ARCHITECTURES = ("NoAdapter", "FeatureAdapter", "CategoryFiLM", "PatchFiLM")


def _zero_initialize(module: nn.Module) -> None:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)


class BottleneckResidualMLP(nn.Module):
    """Identity-initialized residual bottleneck adapter."""

    def __init__(self, width: int, bottleneck_ratio: int = 4) -> None:
        super().__init__()
        hidden = max(1, int(width) // int(bottleneck_ratio))
        self.down = nn.Linear(width, hidden)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden, width)
        # The residual branch is exactly zero at initialization.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.width = int(width)
        self.hidden = int(hidden)
        self.bottleneck_ratio = int(bottleneck_ratio)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.up(self.activation(self.down(values)))


class CategoryQueryFiLM(nn.Module):
    """Zero-initialized channel-wise FiLM from one query per category."""

    def __init__(self, query_width: int, feature_width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(query_width, 2 * feature_width)
        _zero_initialize(self.projection)
        self.feature_width = int(feature_width)

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.projection(query).chunk(2, dim=-1)
        return (1.0 + gamma) * values + beta


class PatchValueFiLM(nn.Module):
    """Per-head query/patch conditioning for value aggregation.

    The network receives the concatenated projected query and projected patch
    key for each ``(head, category, patch)`` pair and emits one gain per value
    channel.  A small category chunk avoids materializing all pairs at once for
    large training batches.  The final layer is zero initialized, therefore
    the initial modulation is ``1 + 0.1*tanh(0) == 1``.
    """

    def __init__(
        self,
        head_width: int,
        bottleneck_ratio: int = 4,
        delta: float = 0.1,
        category_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        hidden = max(1, int(head_width) // int(bottleneck_ratio))
        self.down = nn.Linear(2 * head_width, hidden)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden, head_width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.head_width = int(head_width)
        self.hidden = int(hidden)
        self.bottleneck_ratio = int(bottleneck_ratio)
        self.delta = float(delta)
        self.category_chunk_size = max(1, int(category_chunk_size))

    def _gain(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        # query: [B,H,C,Dh], key: [B,H,N,Dh]
        condition = torch.cat(
            (
                query.unsqueeze(3).expand(-1, -1, -1, key.shape[2], -1),
                key.unsqueeze(2).expand(-1, -1, query.shape[2], -1, -1),
            ),
            dim=-1,
        )
        return self.up(self.activation(self.down(condition)))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention: torch.Tensor,
        return_gain: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, head_count, category_count, head_width = query.shape
        if key.shape[:2] != (batch_size, head_count) or value.shape != key.shape:
            raise ValueError("PatchFiLM Q/K/V head shapes are inconsistent")
        if attention.shape != (batch_size, head_count, category_count, key.shape[2]):
            raise ValueError("PatchFiLM attention shape is inconsistent with Q/K/V")

        aggregates: list[torch.Tensor] = []
        gains: list[torch.Tensor] = []
        for start in range(0, category_count, self.category_chunk_size):
            stop = min(category_count, start + self.category_chunk_size)
            gain = self._gain(query[:, :, start:stop], key)
            modulation = 1.0 + self.delta * torch.tanh(gain)
            weighted_value = modulation * value.unsqueeze(2)
            aggregates.append(
                (attention[:, :, start:stop].unsqueeze(-1) * weighted_value).sum(dim=3),
            )
            if return_gain:
                gains.append(gain)
        aggregate = torch.cat(aggregates, dim=2)
        return aggregate, (torch.cat(gains, dim=2) if return_gain else None)


class AttributeAdapterCrossAttention(nn.Module):
    """AttrQuery CategoryCrossAttention with an optional feature transform."""

    def __init__(
        self,
        num_categories: int,
        visual_dim: int,
        text_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
        architecture: str = "NoAdapter",
        bottleneck_ratio: int = 4,
        patchfilm_delta: float = 0.1,
        patchfilm_category_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        if architecture not in ADAPTER_ARCHITECTURES:
            raise ValueError(f"Unknown AttributeAdapter architecture: {architecture}")
        if visual_dim % num_heads:
            raise ValueError(f"visual_dim={visual_dim} must be divisible by num_heads={num_heads}")
        self.architecture = architecture
        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.num_categories = int(num_categories)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.use_ffn = bool(use_ffn)
        self.bottleneck_ratio = int(bottleneck_ratio)
        self.patchfilm_delta = float(patchfilm_delta)
        self.patchfilm_category_chunk_size = int(patchfilm_category_chunk_size)

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

        self.feature_adapter: BottleneckResidualMLP | None = None
        self.category_film: CategoryQueryFiLM | None = None
        self.patch_film: PatchValueFiLM | None = None
        head_width = visual_dim // num_heads
        if architecture == "FeatureAdapter":
            self.feature_adapter = BottleneckResidualMLP(visual_dim, bottleneck_ratio)
        elif architecture == "CategoryFiLM":
            self.category_film = CategoryQueryFiLM(visual_dim, visual_dim)
        elif architecture == "PatchFiLM":
            self.patch_film = PatchValueFiLM(
                head_width, bottleneck_ratio, patchfilm_delta,
                patchfilm_category_chunk_size,
            )

    def _project_qkv(
        self, queries: torch.Tensor, patch_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.attention._qkv_same_embed_dim:
            weights = self.attention.in_proj_weight
            bias = self.attention.in_proj_bias
            query, key, value = weights.chunk(3, dim=0)
            query_bias, key_bias, value_bias = bias.chunk(3, dim=0) if bias is not None else (None, None, None)
            return (
                F.linear(queries, query, query_bias),
                F.linear(patch_tokens, key, key_bias),
                F.linear(patch_tokens, value, value_bias),
            )
        return (
            F.linear(queries, self.attention.q_proj_weight, self.attention.in_proj_bias[:self.visual_dim]),
            F.linear(patch_tokens, self.attention.k_proj_weight, self.attention.in_proj_bias[self.visual_dim:2 * self.visual_dim]),
            F.linear(patch_tokens, self.attention.v_proj_weight, self.attention.in_proj_bias[2 * self.visual_dim:]),
        )

    def _patchfilm_attention(
        self, queries: torch.Tensor, patch_tokens: torch.Tensor,
        return_gain: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        query, key, value = self._project_qkv(queries, patch_tokens)
        batch_size, category_count, _ = query.shape
        patch_count = key.shape[1]
        head_width = self.visual_dim // self.num_heads
        query_heads = query.view(batch_size, category_count, self.num_heads, head_width).transpose(1, 2)
        key_heads = key.view(batch_size, patch_count, self.num_heads, head_width).transpose(1, 2)
        value_heads = value.view(batch_size, patch_count, self.num_heads, head_width).transpose(1, 2)
        attention = torch.matmul(query_heads, key_heads.transpose(-2, -1)) * (head_width ** -0.5)
        attention = F.softmax(attention, dim=-1)
        if self.patch_film is None:
            aggregate_heads = torch.matmul(attention, value_heads)
            gain = None
        else:
            dropped_attention = F.dropout(
                attention, p=self.attention.dropout, training=self.training,
            )
            aggregate_heads, gain = self.patch_film(
                query_heads, key_heads, value_heads, dropped_attention, return_gain,
            )
        aggregate = aggregate_heads.transpose(1, 2).contiguous().view(
            batch_size, category_count, self.visual_dim,
        )
        attended = self.attention.out_proj(aggregate)
        return attended, attention, gain

    def forward(
        self,
        patch_tokens: torch.Tensor,
        return_attention: bool = False,
        return_patch_gain: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if patch_tokens.ndim != 3:
            raise ValueError(f"Expected patch tokens [B,N,D], got {tuple(patch_tokens.shape)}")
        batch_size = patch_tokens.shape[0]
        queries = self.category_queries.unsqueeze(0).expand(batch_size, -1, -1)
        patch_gain: torch.Tensor | None = None
        if self.architecture == "NoAdapter" or self.architecture in {"FeatureAdapter", "CategoryFiLM"}:
            attended, attention = self.attention(
                queries, patch_tokens, patch_tokens,
                need_weights=return_attention,
                average_attn_weights=False,
            )
        else:
            attended, attention, patch_gain = self._patchfilm_attention(
                queries, patch_tokens, return_patch_gain,
            )
        if self.feature_adapter is not None:
            attended = self.feature_adapter(attended)
        if self.category_film is not None:
            attended = self.category_film(queries, attended)
        output = self.norm1(queries + self.dropout(attended))
        if self.use_ffn:
            output = self.norm2(output + self.dropout(self.ffn(output)))
        features = F.normalize(self.output_projection(output.float()), dim=-1)
        if return_attention:
            if attention is None:
                raise RuntimeError("Cross-attention weights were requested but not returned")
            return features, attention, patch_gain
        return features

    def adapter_parameters(self) -> list[nn.Parameter]:
        modules: list[nn.Module] = []
        if self.feature_adapter is not None:
            modules.append(self.feature_adapter)
        if self.category_film is not None:
            modules.append(self.category_film)
        if self.patch_film is not None:
            modules.append(self.patch_film)
        return [parameter for module in modules for parameter in module.parameters()]

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter_parameters())


class AttrQueryAttributeAdapter(nn.Module):
    """Frozen A7 CLIP plus one of the four AttributeAdapter local heads."""

    def __init__(
        self,
        base_model: AttriVision,
        attribute_names: Sequence[str],
        text_prototypes: torch.Tensor,
        architecture: str = "NoAdapter",
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
        bottleneck_ratio: int = 4,
        patchfilm_delta: float = 0.1,
        patchfilm_category_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        if architecture not in ADAPTER_ARCHITECTURES:
            raise ValueError(f"Unknown AttributeAdapter architecture: {architecture}")
        if text_prototypes.ndim != 2 or text_prototypes.shape[0] != 52:
            raise ValueError(f"Expected text prototypes [52,D], got {tuple(text_prototypes.shape)}")
        self.base = base_model
        self.attribute_names = list(attribute_names)
        mapper = CategoryPromptMapper(attribute_names)
        self.category_groups = tuple(tuple(group) for group in mapper.category_indices())
        self.feature_mode = "local"
        self.architecture = architecture
        self.visual_dim = _visual_width(base_model.clip.visual)
        self.text_dim = int(text_prototypes.shape[1])
        self.num_categories = len(self.category_groups)
        self.num_states = int(text_prototypes.shape[0])
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.use_ffn = bool(use_ffn)
        self.bottleneck_ratio = int(bottleneck_ratio)
        self.patchfilm_delta = float(patchfilm_delta)
        self.patchfilm_category_chunk_size = int(patchfilm_category_chunk_size)
        self.base_model_name = base_model.model_name
        self.local_head = AttributeAdapterCrossAttention(
            self.num_categories, self.visual_dim, self.text_dim,
            num_heads=num_heads, dropout=dropout, use_ffn=use_ffn,
            architecture=architecture, bottleneck_ratio=bottleneck_ratio,
            patchfilm_delta=patchfilm_delta,
            patchfilm_category_chunk_size=patchfilm_category_chunk_size,
        )
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()
        self.register_buffer("text_prototypes", F.normalize(text_prototypes.float(), dim=-1))

    def train(self, mode: bool = True) -> "AttrQueryAttributeAdapter":
        super().train(mode)
        self.base.eval()
        return self

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
            raise RuntimeError(f"Invalid patch token shape: {tuple(patches.shape)}")
        if patches.shape[-1] != self.visual_dim:
            raise RuntimeError(
                f"Captured patch dimension {patches.shape[-1]} differs from visual dimension {self.visual_dim}"
            )
        return patches.float().detach()

    def category_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.local_head(self._frozen_patch_tokens(images))

    def category_features_with_attention(
        self, images: torch.Tensor, return_patch_gain: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], torch.Tensor | None]:
        patches = self._frozen_patch_tokens(images)
        features, attention, gain = self.local_head(
            patches, return_attention=True, return_patch_gain=return_patch_gain,
        )
        visual = self.base.clip.visual
        conv1 = getattr(visual, "conv1", None)
        stride = getattr(conv1, "stride", (1, 1))
        if isinstance(stride, int):
            stride_h = stride_w = int(stride)
        else:
            stride_h, stride_w = int(stride[0]), int(stride[1])
        grid_h = (int(images.shape[-2]) + stride_h - 1) // stride_h
        grid_w = (int(images.shape[-1]) + stride_w - 1) // stride_w
        if grid_h * grid_w != patches.shape[1]:
            side = int(round(patches.shape[1] ** 0.5))
            if side * side != patches.shape[1]:
                raise RuntimeError(f"Cannot reconstruct patch grid for {patches.shape[1]} tokens")
            grid_h = grid_w = side
        return features, attention, (grid_h, grid_w), gain

    def category_state_scores(self, images: torch.Tensor) -> torch.Tensor:
        features = self.category_features(images)
        return torch.einsum("bcd,sd->bcs", features.float(), self.text_prototypes.float())

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        frozen = sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)
        adapter = self.local_head.adapter_parameter_count()
        trainable = total - frozen
        return {
            "total": total, "frozen": frozen, "trainable": trainable,
            "adapter": adapter,
            "base_total": sum(parameter.numel() for parameter in self.base.parameters()),
            "local_head_total": sum(parameter.numel() for parameter in self.local_head.parameters()),
        }

    def set_trainability(self, stage: int) -> None:
        if stage not in {1, 2}:
            raise ValueError("stage must be 1 or 2")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if stage == 1:
            for parameter in self.local_head.adapter_parameters():
                parameter.requires_grad_(True)
        else:
            for parameter in self.local_head.parameters():
                parameter.requires_grad_(True)
        self.base.eval()

    def assert_trainability(self, stage: int) -> None:
        self.assert_frozen_backbone()
        adapter_ids = {id(parameter) for parameter in self.local_head.adapter_parameters()}
        for name, parameter in self.named_parameters():
            expected = (
                id(parameter) in adapter_ids
                if stage == 1 else name.startswith("local_head.")
            )
            if bool(parameter.requires_grad) != expected:
                raise RuntimeError(
                    f"Unexpected trainability for {name}: requires_grad={parameter.requires_grad}, expected={expected}"
                )
        if stage == 1 and self.architecture == "NoAdapter":
            raise RuntimeError("NoAdapter has no Stage-1 trainable parameters; evaluate the existing LocalFCE checkpoint")

    def assert_frozen_backbone(self) -> None:
        if any(parameter.requires_grad for parameter in self.base.parameters()):
            raise RuntimeError("AttributeAdapter CLIP backbone contains a trainable parameter")
        if self.base.training:
            raise RuntimeError("AttributeAdapter CLIP backbone must remain in eval mode")


def build_reference_local_head(
    model: AttrQueryAttributeAdapter,
) -> CategoryCrossAttention:
    """Build a plain existing head for the identity-forward check."""
    reference = CategoryCrossAttention(
        model.num_categories, model.visual_dim, model.text_dim,
        num_heads=model.num_heads, dropout=model.dropout_rate, use_ffn=model.use_ffn,
    )
    state = {
        key: value.detach().cpu()
        for key, value in model.local_head.state_dict().items()
        if not key.startswith(("feature_adapter.", "category_film.", "patch_film."))
    }
    reference.load_state_dict(state, strict=True)
    return reference


@torch.no_grad()
def assert_identity_initialization(
    model: AttrQueryAttributeAdapter,
    device: torch.device,
    seed: int = 1729,
) -> float:
    """Verify every condition equals the existing head before training."""
    previous_mode = model.training
    model.eval()
    reference = build_reference_local_head(model).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    patches = torch.randn(
        2, 49, model.visual_dim, generator=generator, device=device,
    )
    actual = model.local_head(patches)
    expected = reference(patches)
    difference = float((actual - expected).abs().max().cpu())
    if not torch.allclose(actual, expected, atol=2e-4, rtol=2e-4):
        raise RuntimeError(
            f"{model.architecture} is not identity-initialized; max output difference={difference:.6g}"
        )
    if previous_mode:
        model.train()
    return difference
