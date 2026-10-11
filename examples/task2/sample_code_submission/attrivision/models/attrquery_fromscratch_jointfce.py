"""From-scratch end-to-end AttrQuery + A7 JointFCE model.

Unlike the partial/frozen AttrQuery experiments, this module constructs a new
OpenCLIP model from its original pretrained weights.  No A7 or AttrQuery
checkpoint is accepted during initialization.  The visual encoder, text
encoder, CLIP temperature, and the new category-query head are trainable by
default, matching the original A7 CLI policy.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets.attribute_prompts import CategoryPromptMapper
from ..models.attrivision import AttriVision
from .attrquery_frozenclip_localfce import (
    CategoryCrossAttention,
    _batch_first_sequence,
    _first_tensor,
    _spatial_hook_target,
    _visual_width,
)


ARCHITECTURE_NAME = "attrquery_fromscratch_jointfce"


def category_names(mapper: CategoryPromptMapper) -> list[str]:
    names = [spec[0] for spec in mapper._MULTI_GROUPS]
    names.extend(spec[0] for spec in mapper._BINARY_GROUPS)
    if len(names) != 12:
        raise RuntimeError(f"Expected 12 category names, got {len(names)}")
    return names


def validate_mapper(mapper: CategoryPromptMapper) -> None:
    if len(mapper.keys) != 52 or len(mapper.prompts) != 52:
        raise RuntimeError("AttrQuery-FromScratchJointFCE requires the exact 52-state vocabulary")
    groups = mapper.category_indices()
    covered = [index for group in groups for index in group]
    if len(groups) != 12 or sorted(covered) != list(range(52)):
        raise RuntimeError("The A7 category mapping is not a 12-category partition of 52 states")


def _text_width(base: AttriVision) -> int:
    projection = getattr(base.clip, "text_projection", None)
    if projection is not None and hasattr(projection, "shape"):
        shape = tuple(projection.shape)
        if shape:
            return int(shape[-1])
    return int(getattr(base.clip.visual, "output_dim", 512))


def _set_text_trainability(base: AttriVision, enabled: bool) -> None:
    """Toggle the OpenCLIP text tower without touching the visual tower."""
    visual_ids = {id(parameter) for parameter in base.clip.visual.parameters()}
    for parameter in base.clip.parameters():
        if id(parameter) not in visual_ids:
            parameter.requires_grad_(enabled)


class AttrQueryFromScratchJointFCE(nn.Module):
    """Global A7 FCE plus category-local FCE with one shared image forward."""

    def __init__(
        self,
        base_model: AttriVision,
        attribute_names: Sequence[str],
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
        freeze_text_encoder: bool = False,
        freeze_logit_scale: bool = False,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.attribute_names = list(attribute_names)
        self.mapper = CategoryPromptMapper(self.attribute_names)
        validate_mapper(self.mapper)
        self.category_groups = tuple(tuple(group) for group in self.mapper.category_indices())
        self.category_names = category_names(self.mapper)
        self.base_model_name = self.base.model_name
        self.pretrained_tag = self.base.pretrained
        self.visual_dim = _visual_width(self.base.clip.visual)
        self.text_dim = _text_width(self.base)
        self.num_categories = len(self.category_groups)
        self.num_states = len(self.mapper.keys)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.use_ffn = bool(use_ffn)
        self.freeze_text_encoder = bool(freeze_text_encoder)
        self.freeze_logit_scale = bool(freeze_logit_scale)
        self.spatial_hook_target_name = type(_spatial_hook_target(self.base.clip.visual)).__name__

        self.local_head = CategoryCrossAttention(
            self.num_categories,
            self.visual_dim,
            self.text_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_ffn=use_ffn,
        )
        self._configure_trainability()
        self.assert_trainability()

    def _configure_trainability(self) -> None:
        # A7 default: all CLIP parameters are trainable.  The two optional
        # switches are explicit departures from that policy for controlled
        # follow-up runs, and are recorded in every checkpoint.
        for parameter in self.base.parameters():
            parameter.requires_grad_(True)
        for parameter in self.local_head.parameters():
            parameter.requires_grad_(True)
        if self.freeze_text_encoder:
            _set_text_trainability(self.base, False)
        if self.freeze_logit_scale:
            self.base.logit_scale.requires_grad_(False)

    def train(self, mode: bool = True) -> "AttrQueryFromScratchJointFCE":
        super().train(mode)
        # Do not force the CLIP base into eval: A7 end-to-end fine-tuning uses
        # the same train/eval mode for the complete pretrained encoder.
        return self

    def _capture_patch_tokens(
        self, images: torch.Tensor, return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, int]]:
        target = _spatial_hook_target(self.base.clip.visual)
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            sequence = _first_tensor(output)
            if sequence is None:
                raise RuntimeError("The CLIP transformer hook did not return a tensor")
            captured["tokens"] = _batch_first_sequence(sequence, inputs, images.shape[0])

        handle = target.register_forward_hook(hook)
        try:
            # This is intentionally one image-encoder call.  Its global output
            # and the hooked patch sequence feed the two downstream branches.
            global_features = self.base.encode_image(images)
        finally:
            handle.remove()
        sequence = captured.get("tokens")
        if sequence is None:
            raise RuntimeError("No spatial token sequence was captured from the CLIP ViT")
        visual = self.base.clip.visual
        has_cls = getattr(visual, "class_embedding", None) is not None
        patches = sequence[:, 1:] if has_cls else sequence
        if patches.ndim != 3 or patches.shape[1] <= 0 or patches.shape[-1] != self.visual_dim:
            raise RuntimeError(
                f"Invalid patch token shape after CLS removal: {tuple(patches.shape)}; "
                f"expected final dimension {self.visual_dim}"
            )
        patch_count = int(patches.shape[1])
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
                    f"Cannot reconstruct patch grid for {patch_count} tokens from "
                    f"input {tuple(images.shape[-2:])}"
                )
            grid_h = grid_w = side
        if return_attention:
            return global_features.float(), patches.float(), (grid_h, grid_w)
        return global_features.float(), patches.float(), (grid_h, grid_w)

    def joint_features(
        self, images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_features, patches, _ = self._capture_patch_tokens(images)
        local_features = self.local_head(patches)
        return F.normalize(global_features.float(), dim=-1), local_features

    def joint_features_with_attention(
        self, images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        global_features, patches, grid = self._capture_patch_tokens(images, True)
        local_features, attention = self.local_head(patches, return_attention=True)
        if attention.shape[-1] != patches.shape[1]:
            raise RuntimeError("Cross-attention patch count does not match captured patch tokens")
        return F.normalize(global_features.float(), dim=-1), local_features, attention, grid

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.base.encode_text(tokens)

    def global_logits_from_features(
        self, image_features: torch.Tensor, text_features: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.base.logit_scale.exp().clamp(max=100.0)
        return scale * image_features @ text_features.T

    def category_state_scores_from_features(
        self, local_features: torch.Tensor, state_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum(
            "bcd,sd->bcs",
            F.normalize(local_features.float(), dim=-1),
            F.normalize(state_features.float(), dim=-1),
        )

    def category_state_scores(
        self, images: torch.Tensor, state_features: torch.Tensor,
    ) -> torch.Tensor:
        _, local_features = self.joint_features(images)
        return self.category_state_scores_from_features(local_features, state_features)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.base.logit_scale

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def parameter_counts(self) -> dict[str, int]:
        def count(parameters: Any) -> int:
            return sum(parameter.numel() for parameter in parameters)

        visual = self.base.clip.visual.parameters()
        local = self.local_head.parameters()
        text = (
            parameter for name, parameter in self.base.clip.named_parameters()
            if not name.startswith("visual.")
        )
        binary = self.base.binary_head.parameters()
        return {
            "total": count(self.parameters()),
            "frozen": count(parameter for parameter in self.parameters() if not parameter.requires_grad),
            "trainable": count(self.trainable_parameters()),
            "visual_backbone_total": count(visual),
            "visual_backbone_trainable": count(
                parameter for parameter in self.base.clip.visual.parameters()
                if parameter.requires_grad
            ),
            "text_encoder_total": count(text),
            "text_encoder_trainable": count(
                parameter for name, parameter in self.base.clip.named_parameters()
                if not name.startswith("visual.") and parameter.requires_grad
            ),
            "binary_head_total": count(binary),
            "binary_head_trainable": count(
                parameter for parameter in self.base.binary_head.parameters()
                if parameter.requires_grad
            ),
            "local_head_trainable": count(local),
        }

    def assert_trainability(self) -> None:
        for parameter in self.local_head.parameters():
            if not parameter.requires_grad:
                raise RuntimeError("A new local query/cross-attention/projection parameter is frozen")
        if self.freeze_logit_scale and self.base.logit_scale.requires_grad:
            raise RuntimeError("freeze_logit_scale requested but logit_scale is trainable")
        if not self.freeze_logit_scale and not self.base.logit_scale.requires_grad:
            raise RuntimeError("A7-compatible logit_scale should be trainable by default")
        for name, parameter in self.base.clip.named_parameters():
            if name.startswith("visual.") and not parameter.requires_grad:
                raise RuntimeError(f"Visual encoder parameter unexpectedly frozen: {name}")
            if not name.startswith("visual.") and self.freeze_text_encoder and parameter.requires_grad:
                if name != "logit_scale" or not self.freeze_logit_scale:
                    raise RuntimeError(f"Text encoder parameter unexpectedly trainable: {name}")

    def gradient_flow_flags(self) -> dict[str, bool]:
        local_ids = {id(parameter) for parameter in self.local_head.parameters()}
        visual_ids = {id(parameter) for parameter in self.base.clip.visual.parameters()}
        text_ids = {
            id(parameter) for name, parameter in self.base.clip.named_parameters()
            if not name.startswith("visual.")
        }
        return {
            "local_head": any(
                parameter.grad is not None and id(parameter) in local_ids
                for parameter in self.parameters()
            ),
            "visual_backbone": any(
                parameter.grad is not None and id(parameter) in visual_ids
                for parameter in self.parameters()
            ),
            "text_encoder": any(
                parameter.grad is not None and id(parameter) in text_ids
                for parameter in self.parameters()
            ),
        }


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
) -> AttrQueryFromScratchJointFCE:
    if not pretrained_tag:
        raise ValueError("FromScratchJointFCE requires an original pretrained CLIP tag")
    base = AttriVision(pretrained=pretrained_tag, model_name=model_name)
    model = AttrQueryFromScratchJointFCE(
        base, attribute_names, num_heads=num_heads, dropout=dropout,
        use_ffn=use_ffn, freeze_text_encoder=freeze_text_encoder,
        freeze_logit_scale=freeze_logit_scale,
    ).to(device)
    model.assert_trainability()
    return model


def _load_payload(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("architecture") != ARCHITECTURE_NAME:
        raise ValueError(f"Not an {ARCHITECTURE_NAME} checkpoint: {path}")
    required = {"model_state_dict", "attribute_names", "base_model_name", "training_config"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"From-scratch checkpoint is missing keys: {sorted(missing)}")
    if payload.get("initialization", {}).get("source") != "original_pretrained_clip":
        raise ValueError("Checkpoint does not document original pretrained CLIP initialization")
    return payload


def load_experiment_model(
    checkpoint: str | Path, device: torch.device,
) -> tuple[AttrQueryFromScratchJointFCE, dict[str, Any]]:
    payload = _load_payload(checkpoint, map_location="cpu")
    attributes = list(payload["attribute_names"])
    config = dict(payload["training_config"])
    # Loading a saved experiment never downloads or uses an A7 checkpoint.  A
    # random base is enough because every parameter is immediately overwritten.
    base = AttriVision(pretrained=None, model_name=payload["base_model_name"])
    model = AttrQueryFromScratchJointFCE(
        base, attributes,
        num_heads=int(payload.get("num_heads", config.get("num_heads", 8))),
        dropout=float(payload.get("dropout", config.get("dropout", 0.1))),
        use_ffn=bool(payload.get("use_ffn", config.get("use_ffn", True))),
        freeze_text_encoder=bool(config.get("freeze_text_encoder", False)),
        freeze_logit_scale=bool(config.get("freeze_logit_scale", False)),
    )
    model.pretrained_tag = payload.get("initialization", {}).get("pretrained_tag", "openai")
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if missing or unexpected:
        raise ValueError(
            f"From-scratch checkpoint/model mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.to(device)
    model.assert_trainability()
    return model, payload


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def checkpoint_payload(
    model: AttrQueryFromScratchJointFCE,
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
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": ARCHITECTURE_NAME,
        "initialization": {
            "source": "original_pretrained_clip",
            "pretrained_tag": model.pretrained_tag,
            "a7_checkpoint_used": False,
            "attrquery_checkpoint_used": False,
        },
        "base_model_name": model.base_model_name,
        "model_state_dict": _cpu_state_dict(model),
        "attribute_names": list(model.attribute_names),
        "category_groups": [list(group) for group in model.category_groups],
        "category_names": list(model.category_names),
        "num_heads": model.num_heads,
        "dropout": model.dropout_rate,
        "use_ffn": model.use_ffn,
        "visual_dim": model.visual_dim,
        "text_dim": model.text_dim,
        "num_categories": model.num_categories,
        "num_states": model.num_states,
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "training_config": dict(training_config),
    }
    if optimizer is not None:
        if scheduler is None or scaler is None or best_score is None:
            raise ValueError("A resumable checkpoint needs optimizer, scheduler, scaler, and best score")
        payload["training_state"] = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "stale_evaluations": int(stale_evaluations),
            "rng": rng_state,
        }
    return payload


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
