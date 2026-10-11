"""AttrQuery-PartialJointFinetune model and checkpoint utilities.

The model is initialized from an existing A7 checkpoint.  The A7 CLIP image
and text towers are retained, while a newly initialized category query head
is trained jointly with the global A7 FCE objective.  Only the requested final
ViT blocks (and, explicitly, the final visual LayerNorm/projection) can be
made trainable; the text tower and binary head remain frozen.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..checkpoint import load_model
from ..datasets.attribute_prompts import CategoryPromptMapper
from ..models.attrquery_frozenclip_localfce import (
    CategoryCrossAttention,
    _batch_first_sequence,
    _first_tensor,
    _spatial_hook_target,
    build_text_prototypes,
)
from ..models.attrivision import AttriVision


ARCHITECTURE_NAME = "attrquery_partial_joint_finetune"


def _vision_blocks(visual: nn.Module) -> nn.ModuleList | nn.Sequential:
    transformer = getattr(visual, "transformer", None)
    blocks = getattr(transformer, "resblocks", None)
    if blocks is not None and len(blocks):
        return blocks
    trunk = getattr(visual, "trunk", None)
    blocks = getattr(trunk, "blocks", None)
    if blocks is not None and len(blocks):
        return blocks
    raise RuntimeError(
        "The selected CLIP visual tower has no transformer block sequence; "
        "partial joint fine-tuning requires a ViT-like image encoder."
    )


def _set_object_requires_grad(value: nn.Module | nn.Parameter, enabled: bool) -> None:
    if isinstance(value, nn.Parameter):
        value.requires_grad_(enabled)
    else:
        for parameter in value.parameters():
            parameter.requires_grad_(enabled)


def _object_parameter_names(prefix: str, value: nn.Module | nn.Parameter) -> list[str]:
    if isinstance(value, nn.Parameter):
        return [prefix]
    return [f"{prefix}.{name}" for name, _ in value.named_parameters()]


class AttrQueryPartialJointFinetune(nn.Module):
    """A7 global FCE plus a trainable category-local cross-attention branch."""

    def __init__(
        self,
        base_model: AttriVision,
        attribute_names: Sequence[str],
        text_prototypes: torch.Tensor,
        unfreeze_last_blocks: int = 0,
        unfreeze_vision_ln_proj: bool = True,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
    ) -> None:
        super().__init__()
        if unfreeze_last_blocks not in {0, 1, 2}:
            raise ValueError("unfreeze_last_blocks must be 0, 1, or 2")
        if text_prototypes.ndim != 2 or text_prototypes.shape[0] != 52:
            raise ValueError(
                f"Expected normalized text prototypes [52,D], got {tuple(text_prototypes.shape)}"
            )

        self.base = base_model
        self.attribute_names = list(attribute_names)
        self.mapper = CategoryPromptMapper(self.attribute_names)
        self.category_groups = tuple(tuple(group) for group in self.mapper.category_indices())
        self.base_model_name = self.base.model_name
        self.visual_dim = _visual_width(self.base.clip.visual)
        self.text_dim = int(text_prototypes.shape[1])
        self.num_categories = len(self.category_groups)
        self.num_states = int(text_prototypes.shape[0])
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.use_ffn = bool(use_ffn)
        self.unfreeze_last_blocks = int(unfreeze_last_blocks)
        self.unfreeze_vision_ln_proj = bool(unfreeze_vision_ln_proj)

        self.local_head = CategoryCrossAttention(
            self.num_categories,
            _visual_width(self.base.clip.visual),
            self.text_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_ffn=use_ffn,
        )
        self.register_buffer("text_prototypes", F.normalize(text_prototypes.float(), dim=-1))

        self._configure_trainability()
        self.assert_trainability()

    def _configure_trainability(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for parameter in self.local_head.parameters():
            parameter.requires_grad_(True)

        if self.unfreeze_last_blocks:
            blocks = _vision_blocks(self.base.clip.visual)
            if self.unfreeze_last_blocks > len(blocks):
                raise ValueError(
                    f"unfreeze_last_blocks={self.unfreeze_last_blocks} exceeds visual depth {len(blocks)}"
                )
            for block in list(blocks)[-self.unfreeze_last_blocks:]:
                _set_object_requires_grad(block, True)

            if self.unfreeze_vision_ln_proj:
                visual = self.base.clip.visual
                ln_post = getattr(visual, "ln_post", None)
                projection = getattr(visual, "proj", None)
                if ln_post is not None:
                    _set_object_requires_grad(ln_post, True)
                if projection is not None:
                    _set_object_requires_grad(projection, True)

    def _selected_visual_modules(self) -> list[tuple[str, nn.Module | nn.Parameter]]:
        if not self.unfreeze_last_blocks:
            return []
        blocks = _vision_blocks(self.base.clip.visual)
        selected: list[tuple[str, nn.Module | nn.Parameter]] = [
            (f"base.clip.visual.{type(blocks).__name__}[-{offset}]", block)
            for offset, block in enumerate(
                reversed(list(blocks)[-self.unfreeze_last_blocks:]), start=1,
            )
        ]
        if self.unfreeze_vision_ln_proj:
            visual = self.base.clip.visual
            if getattr(visual, "ln_post", None) is not None:
                selected.append(("base.clip.visual.ln_post", visual.ln_post))
            if getattr(visual, "proj", None) is not None:
                selected.append(("base.clip.visual.proj", visual.proj))
        return selected

    def train(self, mode: bool = True) -> "AttrQueryPartialJointFinetune":
        # Keep frozen CLIP components deterministic while allowing only the
        # selected visual blocks to use train mode.  The local head follows the
        # requested mode and therefore retains its dropout behavior.
        super().train(mode)
        self.base.eval()
        self.local_head.train(mode)
        for _, module in self._selected_visual_modules():
            if isinstance(module, nn.Module):
                module.train(mode)
        return self

    def _capture_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        target = _spatial_hook_target(self.base.clip.visual)
        captured: dict[str, torch.Tensor] = {}

        def hook(
            _module: nn.Module, inputs: tuple[Any, ...], output: Any,
        ) -> None:
            sequence = _first_tensor(output)
            if sequence is None:
                raise RuntimeError("The CLIP transformer hook did not return a tensor")
            captured["tokens"] = _batch_first_sequence(sequence, inputs, images.shape[0])

        handle = target.register_forward_hook(hook)
        try:
            if self.unfreeze_last_blocks:
                _ = self.base.encode_image(images)
            else:
                with torch.no_grad():
                    _ = self.base.encode_image(images)
        finally:
            handle.remove()

        sequence = captured.get("tokens")
        if sequence is None:
            raise RuntimeError("No spatial token sequence was captured from the A7 ViT")
        visual = self.base.clip.visual
        has_cls = getattr(visual, "class_embedding", None) is not None
        patches = sequence[:, 1:] if has_cls else sequence
        expected_dim = _visual_width(visual)
        if patches.ndim != 3 or patches.shape[1] <= 0 or patches.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Invalid patch tokens after CLS removal: {tuple(patches.shape)}; "
                f"expected final dimension {expected_dim}"
            )
        return patches.float() if self.unfreeze_last_blocks else patches.float().detach()

    def global_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.unfreeze_last_blocks:
            return self.base.encode_image(images).float()
        with torch.no_grad():
            features = self.base.encode_image(images)
        return features.float().detach()

    def local_features(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self._capture_patch_tokens(images)
        return self.local_head(patch_tokens)

    def category_state_scores(self, images: torch.Tensor) -> torch.Tensor:
        features = self.local_features(images)
        return torch.einsum(
            "bcd,sd->bcs", features.float(), self.text_prototypes.float(),
        )

    def global_fce_outputs(
        self, images: torch.Tensor, text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_features = self.global_features(images)
        with torch.no_grad():
            text_features = self.base.encode_text(text_tokens)
        scale = self.base.logit_scale.exp().clamp(max=100.0)
        logits = scale * image_features @ text_features.T
        return image_features, text_features, logits

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.global_features(images), self.local_features(images)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.base.logit_scale

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        local_head = sum(parameter.numel() for parameter in self.local_head.parameters())
        visual_trainable = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and name.startswith("base.clip.visual.")
        )
        text_frozen = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name.startswith("base.clip.") and not name.startswith("base.clip.visual.")
        )
        return {
            "total": total,
            "frozen": total - trainable,
            "trainable": trainable,
            "local_head_trainable": local_head,
            "visual_backbone_trainable": visual_trainable,
            "text_encoder_and_scale_frozen": text_frozen,
        }

    def assert_trainability(self) -> None:
        expected_local = {id(parameter) for parameter in self.local_head.parameters()}
        for parameter in self.local_head.parameters():
            if not parameter.requires_grad:
                raise RuntimeError("A local query/cross-attention/projection parameter is frozen")

        allowed_visual: set[int] = set()
        for _, module in self._selected_visual_modules():
            if isinstance(module, nn.Parameter):
                allowed_visual.add(id(module))
            else:
                allowed_visual.update(id(parameter) for parameter in module.parameters())
        for name, parameter in self.base.named_parameters():
            if parameter.requires_grad and id(parameter) not in allowed_visual:
                raise RuntimeError(f"Unexpected trainable A7 parameter: base.{name}")
        if any(
            parameter.requires_grad
            for name, parameter in self.base.named_parameters()
            if name.startswith("clip.transformer")
            or name.startswith("clip.token_embedding")
            or name.startswith("clip.positional_embedding")
            or name.startswith("clip.ln_final")
            or name.startswith("clip.text_projection")
            or name.startswith("clip.logit_scale")
            or name.startswith("binary_head")
        ):
            raise RuntimeError("Text encoder, logit scale, and binary head must remain frozen")
        if not expected_local:
            raise RuntimeError("The local head has no parameters")

    def assert_gradient_flow(
        self,
        global_loss: torch.Tensor,
        local_loss: torch.Tensor,
    ) -> dict[str, bool]:
        """Return whether each loss reaches the expected parameter groups.

        This is called after ``backward`` in the training loop and is also
        recorded in the first-batch diagnostics.  A frozen-backbone global
        loss is expected not to reach the new local head.
        """
        del global_loss, local_loss
        local_ids = {id(parameter) for parameter in self.local_head.parameters()}
        visual_ids = {
            id(parameter)
            for name, parameter in self.named_parameters()
            if name.startswith("base.clip.visual.") and parameter.requires_grad
        }
        global_local = any(
            parameter.grad is not None and id(parameter) in local_ids
            for parameter in self.trainable_parameters()
        )
        global_visual = any(
            parameter.grad is not None and id(parameter) in visual_ids
            for parameter in self.trainable_parameters()
        )
        return {
            "total_grad_reaches_local_head": global_local,
            "total_grad_reaches_visual": global_visual,
        }


def _visual_width(visual: nn.Module) -> int:
    for candidate in (
        getattr(visual, "width", None),
        getattr(getattr(visual, "transformer", None), "width", None),
        getattr(getattr(visual, "trunk", None), "embed_dim", None),
    ):
        if candidate is not None and int(candidate) > 0:
            return int(candidate)
    class_embedding = getattr(visual, "class_embedding", None)
    if torch.is_tensor(class_embedding) and class_embedding.numel() > 0:
        return int(class_embedding.numel())
    conv1 = getattr(visual, "conv1", None)
    out_channels = getattr(conv1, "out_channels", None)
    if out_channels is not None and int(out_channels) > 0:
        return int(out_channels)
    raise RuntimeError("Unable to infer visual transformer width")


def build_model_from_a7(
    checkpoint: str | Path,
    attribute_names: Sequence[str],
    device: torch.device,
    amp: bool,
    unfreeze_last_blocks: int,
    unfreeze_vision_ln_proj: bool,
    num_heads: int,
    dropout: float,
    use_ffn: bool,
) -> tuple[AttrQueryPartialJointFinetune, dict[str, Any]]:
    """Load only an A7 checkpoint and create fresh local modules."""
    checkpoint = Path(checkpoint).resolve()
    base, payload = load_model(checkpoint, device)
    if payload.get("prompt_mode") != "category_complete":
        raise ValueError("Initialization must be the original A7 category_complete checkpoint")
    if list(payload["attribute_names"]) != list(attribute_names):
        raise ValueError("A7 checkpoint attribute order differs from the training annotations")
    if payload.get("model_name") != "ViT-B-32-quickgelu":
        raise ValueError(
            "AttrQuery-PartialJointFinetune expects the A7 ViT-B/32-quickgelu checkpoint; "
            f"got {payload.get('model_name')!r}"
        )
    mapper = CategoryPromptMapper(attribute_names)
    text_prototypes = build_text_prototypes(base, mapper, device, amp)
    model = AttrQueryPartialJointFinetune(
        base, attribute_names, text_prototypes,
        unfreeze_last_blocks=unfreeze_last_blocks,
        unfreeze_vision_ln_proj=unfreeze_vision_ln_proj,
        num_heads=num_heads, dropout=dropout, use_ffn=use_ffn,
    ).to(device)
    model.assert_trainability()
    return model, {**payload, "checkpoint_path": str(checkpoint)}


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
        raise ValueError(f"Partial joint checkpoint is missing keys: {sorted(missing)}")
    return payload


def load_experiment_model(
    checkpoint: str | Path, device: torch.device,
) -> tuple[AttrQueryPartialJointFinetune, dict[str, Any]]:
    payload = _load_payload(checkpoint, map_location="cpu")
    attributes = list(payload["attribute_names"])
    mapper = CategoryPromptMapper(attributes)
    state = payload["model_state_dict"]
    text_prototypes = state.get("text_prototypes", payload.get("text_prototypes"))
    if text_prototypes is None:
        raise ValueError("Partial joint checkpoint does not contain text prototypes")
    config = payload["training_config"]
    base = AttriVision(pretrained=None, model_name=payload["base_model_name"])
    model = AttrQueryPartialJointFinetune(
        base, attributes, text_prototypes,
        unfreeze_last_blocks=int(config["unfreeze_last_blocks"]),
        unfreeze_vision_ln_proj=bool(config["unfreeze_vision_ln_proj"]),
        num_heads=int(payload.get("num_heads", config.get("num_heads", 8))),
        dropout=float(payload.get("dropout", config.get("dropout", 0.1))),
        use_ffn=bool(payload.get("use_ffn", config.get("use_ffn", True))),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Partial joint checkpoint/model mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.to(device)
    model.assert_trainability()
    return model, payload


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def checkpoint_payload(
    model: AttrQueryPartialJointFinetune,
    epoch: int,
    metrics: dict[str, Any],
    training_config: dict[str, Any],
    base_checkpoint: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    best_score: float | None = None,
    best_epoch: int = 0,
    stale_evaluations: int = 0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": ARCHITECTURE_NAME,
        "base_model_name": model.base_model_name,
        "base_checkpoint": str(Path(base_checkpoint).resolve()),
        "model_state_dict": _cpu_state_dict(model),
        "text_prototypes": model.text_prototypes.detach().cpu(),
        "attribute_names": list(model.attribute_names),
        "category_groups": [list(group) for group in model.category_groups],
        "num_heads": model.num_heads,
        "dropout": model.dropout_rate,
        "use_ffn": model.use_ffn,
        "unfreeze_last_blocks": model.unfreeze_last_blocks,
        "unfreeze_vision_ln_proj": model.unfreeze_vision_ln_proj,
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "training_config": dict(training_config),
        "parameter_counts": model.parameter_counts(),
        "trainable_parameter_names": model.trainable_parameter_names(),
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
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    return payload


def save_checkpoint(
    path: Path,
    model: AttrQueryPartialJointFinetune,
    epoch: int,
    metrics: dict[str, Any],
    training_config: dict[str, Any],
    base_checkpoint: str,
    **training_state: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        checkpoint_payload(
            model, epoch, metrics, training_config, base_checkpoint, **training_state,
        ),
        temporary,
    )
    os.replace(temporary, path)
