"""End-to-end From-CLIP AttrQuery JointFCE with identity PatchFiLM values.

The original ``CategoryCrossAttention`` is constructed first with its usual
seeded initialization. PatchFiLM is then attached to those same query,
attention, normalization, FFN, and projection modules, so every shared local
head weight matches FromScratchJointFCE exactly. Its initialization draws are
isolated from the random-number stream used by the rest of the JointFCE run.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attrquery_attribute_adapter import PatchValueFiLM
from .attrquery_frozenclip_localfce import CategoryCrossAttention
from .attrivision import AttriVision
from .attrquery_fromscratch_jointfce import (
    AttrQueryFromScratchJointFCE,
)


ARCHITECTURE_NAME = "attrquery_patchfilm_joint_fromclip"


class PatchFiLMCrossAttention(nn.Module):
    """Category cross-attention that modulates per-head values before sum."""

    def __init__(
        self,
        reference: CategoryCrossAttention,
        bottleneck_ratio: int = 4,
        delta: float = 0.1,
        category_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        # Reuse, rather than reinitialize, every module that exists in the
        # original LocalFCE head. The only new parameters are PatchFiLM's.
        self.category_queries = reference.category_queries
        self.attention = reference.attention
        self.dropout = reference.dropout
        self.norm1 = reference.norm1
        self.use_ffn = reference.use_ffn
        if self.use_ffn:
            self.ffn = reference.ffn
            self.norm2 = reference.norm2
        self.output_projection = reference.output_projection
        self.visual_dim = int(reference.attention.embed_dim)
        self.num_heads = int(reference.attention.num_heads)
        self.dropout_rate = float(reference.dropout.p)
        self.patch_film = PatchValueFiLM(
            self.visual_dim // self.num_heads,
            bottleneck_ratio=bottleneck_ratio,
            delta=delta,
            category_chunk_size=category_chunk_size,
        )

    def _attention(
        self,
        queries: torch.Tensor,
        patches: torch.Tensor,
        return_patch_gain: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        attention_module = self.attention
        if not attention_module._qkv_same_embed_dim:
            raise RuntimeError("PatchFiLM expects the shared-dimension OpenCLIP attention layout")
        weights = attention_module.in_proj_weight.chunk(3, dim=0)
        if attention_module.in_proj_bias is None:
            biases: tuple[torch.Tensor | None, ...] = (None, None, None)
        else:
            biases = attention_module.in_proj_bias.chunk(3, dim=0)
        query, key, value = (
            F.linear(queries, weights[0], biases[0]),
            F.linear(patches, weights[1], biases[1]),
            F.linear(patches, weights[2], biases[2]),
        )

        batch, category_count, _ = query.shape
        patch_count = key.shape[1]
        head_width = self.visual_dim // self.num_heads
        query_heads = query.reshape(batch, category_count, self.num_heads, head_width).transpose(1, 2)
        key_heads = key.reshape(batch, patch_count, self.num_heads, head_width).transpose(1, 2)
        value_heads = value.reshape(batch, patch_count, self.num_heads, head_width).transpose(1, 2)

        # Preserve the original Q/K and softmax attention computation. Only
        # V is modulated, independently for each (head, category, patch).
        attention = torch.matmul(query_heads, key_heads.transpose(-2, -1)) * (head_width ** -0.5)
        attention = F.softmax(attention, dim=-1)
        dropped_attention = F.dropout(
            attention, p=attention_module.dropout, training=self.training,
        )
        aggregate_heads, raw_gain = self.patch_film(
            query_heads, key_heads, value_heads, dropped_attention,
            return_gain=return_patch_gain,
        )
        aggregate = aggregate_heads.transpose(1, 2).contiguous().reshape(
            batch, category_count, self.visual_dim,
        )
        attended = attention_module.out_proj(aggregate)
        return attended, attention, raw_gain

    def forward(
        self,
        patch_tokens: torch.Tensor,
        return_attention: bool = False,
        return_patch_gain: bool = False,
    ) -> Any:
        if patch_tokens.ndim != 3:
            raise ValueError(f"Expected patch tokens [B,N,D], got {tuple(patch_tokens.shape)}")
        batch = patch_tokens.shape[0]
        queries = self.category_queries.unsqueeze(0).expand(batch, -1, -1)
        attended, attention, raw_gain = self._attention(queries, patch_tokens, return_patch_gain)
        output = self.norm1(queries + self.dropout(attended))
        if self.use_ffn:
            output = self.norm2(output + self.dropout(self.ffn(output)))
        features = F.normalize(self.output_projection(output.float()), dim=-1)
        if return_patch_gain:
            if attention is None:
                raise RuntimeError("PatchFiLM attention was not produced")
            return features, attention, raw_gain
        if return_attention:
            return features, attention
        return features


class AttrQueryPatchFiLMJointFromCLIP(AttrQueryFromScratchJointFCE):
    """From-original-CLIP JointFCE with learnable category-conditioned V."""

    def __init__(
        self,
        base_model: AttriVision,
        attribute_names: Sequence[str],
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
        freeze_text_encoder: bool = False,
        freeze_logit_scale: bool = False,
        bottleneck_ratio: int = 4,
        patchfilm_delta: float = 0.1,
        patchfilm_category_chunk_size: int = 4,
    ) -> None:
        # This creates the original JointFCE local head in its original order.
        super().__init__(
            base_model, attribute_names, num_heads, dropout, use_ffn,
            freeze_text_encoder, freeze_logit_scale,
        )
        jointfce_rng_state = torch.get_rng_state()
        self.local_head = PatchFiLMCrossAttention(
            self.local_head,
            bottleneck_ratio=bottleneck_ratio,
            delta=patchfilm_delta,
            category_chunk_size=patchfilm_category_chunk_size,
        )
        # Added adapter initialization must not shift the shared JointFCE RNG
        # stream used by data workers and dropout during the controlled run.
        torch.set_rng_state(jointfce_rng_state)
        self.bottleneck_ratio = int(bottleneck_ratio)
        self.patchfilm_delta = float(patchfilm_delta)
        self.patchfilm_category_chunk_size = int(patchfilm_category_chunk_size)
        self.assert_trainability()

    def patchfilm_parameters(self) -> list[nn.Parameter]:
        patch_film = getattr(self.local_head, "patch_film", None)
        return list(patch_film.parameters()) if patch_film is not None else []

    def parameter_counts(self) -> dict[str, int]:
        counts = super().parameter_counts()
        counts["patchfilm_trainable"] = sum(
            parameter.numel() for parameter in self.patchfilm_parameters()
            if parameter.requires_grad
        )
        counts["jointfce_local_head_trainable_excluding_patchfilm"] = (
            counts["local_head_trainable"] - counts["patchfilm_trainable"]
        )
        return counts

    def assert_trainability(self) -> None:
        super().assert_trainability()
        patch_film = getattr(self.local_head, "patch_film", None)
        if patch_film is not None and any(not parameter.requires_grad for parameter in patch_film.parameters()):
            raise RuntimeError("Every PatchFiLM parameter must be trainable")

    def gradient_flow_flags(self) -> dict[str, bool]:
        flags = super().gradient_flow_flags()
        flags["patchfilm"] = bool(self.patchfilm_parameters()) and all(
            parameter.grad is not None for parameter in self.patchfilm_parameters()
        )
        return flags

    def joint_features_with_patch_diagnostics(
        self, images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int], torch.Tensor]:
        global_features, patches, grid = self._capture_patch_tokens(images, return_attention=True)
        local_features, attention, raw_gain = self.local_head(
            patches, return_attention=True, return_patch_gain=True,
        )
        if raw_gain is None:
            raise RuntimeError("PatchFiLM diagnostics did not return modulation gains")
        return F.normalize(global_features.float(), dim=-1), local_features, attention, grid, raw_gain


def _check_identity_initialization(
    head: PatchFiLMCrossAttention,
    num_categories: int,
    text_dim: int,
    device: torch.device,
    seed: int = 1729,
) -> float:
    rng_before = torch.get_rng_state()
    was_training = head.training
    try:
        reference = CategoryCrossAttention(
            num_categories, head.visual_dim, text_dim,
            num_heads=head.num_heads, dropout=head.dropout_rate,
            use_ffn=head.use_ffn,
        ).to(device).eval()
        plain_state = {
            key: value.detach().cpu()
            for key, value in head.state_dict().items()
            if not key.startswith("patch_film.")
        }
        reference.load_state_dict(plain_state, strict=True)
        generator = torch.Generator(device=device).manual_seed(seed)
        patches = torch.randn(2, 49, head.visual_dim, generator=generator, device=device)
        head.eval()
        with torch.inference_mode():
            actual = head(patches)
            expected = reference(patches)
    finally:
        torch.set_rng_state(rng_before)
        if was_training:
            head.train()
    difference = float((actual - expected).abs().max().cpu())
    if not torch.allclose(actual, expected, atol=2e-4, rtol=2e-4):
        raise RuntimeError(
            "Identity-initialized PatchFiLM differs from NoAdapter: "
            f"max_abs_diff={difference:.6g}"
        )
    return difference


def build_model_from_pretrained(
    attribute_names: Sequence[str],
    device: torch.device,
    pretrained_tag: str = "openai",
    model_name: str = "ViT-B-32-quickgelu",
    num_heads: int = 8,
    dropout: float = 0.1,
    use_ffn: bool = True,
    freeze_text_encoder: bool = False,
    freeze_logit_scale: bool = False,
    bottleneck_ratio: int = 4,
    patchfilm_delta: float = 0.1,
    patchfilm_category_chunk_size: int = 4,
) -> tuple[AttrQueryPatchFiLMJointFromCLIP, float]:
    if not pretrained_tag:
        raise ValueError("PatchFiLM-JointFromCLIP requires original pretrained CLIP weights")
    base = AttriVision(pretrained=pretrained_tag, model_name=model_name)
    model = AttrQueryPatchFiLMJointFromCLIP(
        base, attribute_names, num_heads=num_heads, dropout=dropout,
        use_ffn=use_ffn, freeze_text_encoder=freeze_text_encoder,
        freeze_logit_scale=freeze_logit_scale,
        bottleneck_ratio=bottleneck_ratio, patchfilm_delta=patchfilm_delta,
        patchfilm_category_chunk_size=patchfilm_category_chunk_size,
    ).to(device)
    model.assert_trainability()
    identity_difference = _check_identity_initialization(
        model.local_head, model.num_categories, model.text_dim, device,
    )
    return model, identity_difference


def build_model_for_checkpoint(
    payload: dict[str, Any], device: torch.device,
) -> AttrQueryPatchFiLMJointFromCLIP:
    config = dict(payload["training_config"])
    base = AttriVision(pretrained=None, model_name=payload["base_model_name"])
    model = AttrQueryPatchFiLMJointFromCLIP(
        base, payload["attribute_names"],
        num_heads=int(payload.get("num_heads", config.get("num_heads", 8))),
        dropout=float(payload.get("dropout", config.get("dropout", 0.1))),
        use_ffn=bool(payload.get("use_ffn", config.get("use_ffn", True))),
        freeze_text_encoder=bool(config.get("freeze_text_encoder", False)),
        freeze_logit_scale=bool(config.get("freeze_logit_scale", False)),
        bottleneck_ratio=int(payload.get("bottleneck_ratio", config.get("bottleneck_ratio", 4))),
        patchfilm_delta=float(payload.get("patchfilm_delta", config.get("patchfilm_delta", 0.1))),
        patchfilm_category_chunk_size=int(
            payload.get("patchfilm_category_chunk_size", config.get("patchfilm_category_chunk_size", 4))
        ),
    )
    model.pretrained_tag = payload.get("initialization", {}).get("pretrained_tag", "openai")
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if missing or unexpected:
        raise ValueError(
            f"PatchFiLM checkpoint/model mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.to(device)
    model.assert_trainability()
    return model


def load_experiment_model(
    checkpoint: str,
    device: torch.device,
) -> tuple[AttrQueryPatchFiLMJointFromCLIP, dict[str, Any]]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("architecture") != ARCHITECTURE_NAME:
        raise ValueError(f"Not a {ARCHITECTURE_NAME} checkpoint: {checkpoint}")
    if payload.get("initialization", {}).get("source") != "original_pretrained_clip":
        raise ValueError("PatchFiLM checkpoint must document original pretrained CLIP initialization")
    for key in ("model_state_dict", "attribute_names", "base_model_name", "training_config"):
        if key not in payload:
            raise ValueError(f"PatchFiLM checkpoint is missing {key}")
    return build_model_for_checkpoint(payload, device), payload


def checkpoint_payload(
    model: AttrQueryPatchFiLMJointFromCLIP,
    epoch: int,
    metrics: dict[str, Any],
    training_config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    best_score: float | None = None,
    best_epoch: int = 0,
    stale_evaluations: int = 0,
    rng_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .attrquery_fromscratch_jointfce import checkpoint_payload as jointfce_payload

    payload = jointfce_payload(
        model, epoch, metrics, training_config, optimizer, scheduler, scaler,
        best_score, best_epoch, stale_evaluations, rng_state,
    )
    payload["architecture"] = ARCHITECTURE_NAME
    payload["patchfilm"] = {
        "bottleneck_ratio": model.bottleneck_ratio,
        "delta": model.patchfilm_delta,
        "category_chunk_size": model.patchfilm_category_chunk_size,
        "initialization": "zero_output_identity",
        "conditioning": "projected_category_query_and_patch_key_per_head",
        "applied_before_value_aggregation": True,
    }
    payload["initialization"]["patchfilm_checkpoint_used"] = False
    return payload


def save_checkpoint(path: Any, payload: dict[str, Any]) -> None:
    from .attrquery_fromscratch_jointfce import save_checkpoint as save

    save(path, payload)
