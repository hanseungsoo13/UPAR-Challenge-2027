"""Single-encoder CLIP ViT-B/16 model for E1 shared-backbone experiments."""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets.attribute_prompts import CategoryPromptMapper
from ..models.attrivision import AttriVision
from ..models.attrquery_frozenclip_localfce import (
    CategoryCrossAttention,
    _batch_first_sequence,
    _first_tensor,
    _spatial_hook_target,
    _visual_width,
)


MODEL_NAME = "ViT-B-16-quickgelu"
MODEL_ID = "openai/clip-vit-base-patch16"
ARCHITECTURE = "attrivision_attrquery_shared_openclip_vit_b_16"


def _text_width(base: AttriVision) -> int:
    projection = getattr(base.clip, "text_projection", None)
    if projection is not None and hasattr(projection, "shape"):
        shape = tuple(projection.shape)
        if shape:
            return int(shape[-1])
    return int(getattr(base.clip.visual, "output_dim", 512))


class SharedCLIPB16AttrQuery(nn.Module):
    """A single CLIP B/16 supplies both the global and local image features.

    ``stage`` is ``local`` for frozen-encoder E1-0/E1-B training, ``joint``
    for E1-C's vision-only update, and ``full`` for E1-D's complete CLIP
    update. The local decoder follows the repository's existing
    12-category cross-attention head and projects its output into CLIP's text
    embedding space.
    """

    def __init__(
        self,
        base_model: AttriVision,
        attribute_names: Sequence[str],
        *,
        stage: str,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
    ) -> None:
        super().__init__()
        if stage not in {"local", "joint", "full"}:
            raise ValueError("stage must be 'local', 'joint', or 'full'")
        if base_model.model_name != MODEL_NAME:
            raise ValueError(
                f"Shared E1 requires OpenCLIP {MODEL_NAME}, got {base_model.model_name!r}"
            )
        self.base = base_model
        self.attribute_names = list(attribute_names)
        self.mapper = CategoryPromptMapper(self.attribute_names)
        self.category_groups = tuple(tuple(group) for group in self.mapper.category_indices())
        if len(self.mapper.keys) != 52 or len(self.category_groups) != 12:
            raise ValueError("Shared E1 requires the established 52-state/12-category mapping")
        self.stage = stage
        self.visual_dim = _visual_width(self.base.clip.visual)
        self.text_dim = _text_width(self.base)
        self.num_categories = len(self.category_groups)
        self.num_states = len(self.mapper.keys)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.use_ffn = bool(use_ffn)
        self.local_head = CategoryCrossAttention(
            self.num_categories,
            self.visual_dim,
            self.text_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            use_ffn=self.use_ffn,
        )
        self._set_trainability()

    def _set_trainability(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for parameter in self.local_head.parameters():
            parameter.requires_grad_(True)
        if self.stage in {"joint", "full"}:
            for parameter in self.base.clip.visual.parameters():
                parameter.requires_grad_(True)
            self.base.clip.logit_scale.requires_grad_(True)
        if self.stage == "full":
            for name, parameter in self.base.clip.named_parameters():
                if not name.startswith("visual.") and name != "logit_scale":
                    parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "SharedCLIPB16AttrQuery":
        super().train(mode)
        # Keep the unused binary head deterministic. E1-C trains only vision;
        # E1-D puts both CLIP towers in train mode.
        self.base.eval()
        if self.stage == "full":
            self.base.clip.train(mode)
        else:
            self.base.clip.visual.train(mode and self.stage == "joint")
        self.local_head.train(mode)
        return self

    def _capture_shared_features(
        self, images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.base.clip.visual
        target = _spatial_hook_target(visual)
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            sequence = _first_tensor(output)
            if sequence is None:
                raise RuntimeError("CLIP's visual transformer returned no token tensor")
            captured["tokens"] = _batch_first_sequence(sequence, inputs, images.shape[0])

        handle = target.register_forward_hook(hook)
        try:
            if self.stage == "local":
                with torch.no_grad():
                    global_features = self.base.encode_image(images)
            else:
                global_features = self.base.encode_image(images)
        finally:
            handle.remove()

        sequence = captured.get("tokens")
        if sequence is None:
            raise RuntimeError("No spatial tokens were captured from the shared CLIP encoder")
        has_cls = getattr(visual, "class_embedding", None) is not None
        patch_tokens = sequence[:, 1:] if has_cls else sequence
        if (
            patch_tokens.ndim != 3
            or patch_tokens.shape[1] == 0
            or patch_tokens.shape[2] != self.visual_dim
        ):
            raise RuntimeError(
                "Expected CLIP spatial tokens [B,N,768] after removing CLS; "
                f"received {tuple(patch_tokens.shape)}"
            )
        if self.stage == "local":
            patch_tokens = patch_tokens.detach()
        return F.normalize(global_features.float(), dim=-1), patch_tokens.float()

    def joint_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global 512-D features and 12 local 512-D category features."""
        global_features, patch_tokens = self._capture_shared_features(images)
        local_features = self.local_head(patch_tokens)
        return global_features, local_features

    def joint_features_with_tokens(
        self, images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_features, patch_tokens = self._capture_shared_features(images)
        local_features, attention = self.local_head(patch_tokens, return_attention=True)
        if attention.shape[-1] != patch_tokens.shape[1]:
            raise RuntimeError("Cross-attention does not cover every shared patch token")
        return global_features, local_features, patch_tokens

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

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def parameter_counts(self) -> dict[str, int]:
        vision_total = sum(parameter.numel() for parameter in self.base.clip.visual.parameters())
        vision_trainable = sum(
            parameter.numel() for parameter in self.base.clip.visual.parameters()
            if parameter.requires_grad
        )
        text_total = sum(
            parameter.numel() for name, parameter in self.base.clip.named_parameters()
            if not name.startswith("visual.") and name != "logit_scale"
        )
        text_trainable = sum(
            parameter.numel() for name, parameter in self.base.clip.named_parameters()
            if not name.startswith("visual.") and name != "logit_scale" and parameter.requires_grad
        )
        local_total = sum(parameter.numel() for parameter in self.local_head.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.trainable_parameters())
        return {
            "total": total,
            "frozen": total - trainable,
            "trainable": trainable,
            "vision_backbone_total": vision_total,
            "vision_backbone_trainable": vision_trainable,
            "text_encoder_total": text_total,
            "text_encoder_trainable": text_trainable,
            "logit_scale_trainable": int(self.base.clip.logit_scale.requires_grad),
            "local_decoder_trainable": local_total,
            "vision_encoder_instances": 1,
            "visual_hidden_dim": self.visual_dim,
            "joint_embedding_dim": self.text_dim,
            "patch_tokens_at_224": 196,
        }

    def assert_trainability(self) -> None:
        for parameter in self.local_head.parameters():
            if not parameter.requires_grad:
                raise RuntimeError("The AttriQuery local decoder must be trainable")
        visual_ids = {id(parameter) for parameter in self.base.clip.visual.parameters()}
        for name, parameter in self.base.clip.named_parameters():
            if id(parameter) in visual_ids and parameter.requires_grad != (self.stage in {"joint", "full"}):
                raise RuntimeError(f"Unexpected vision trainability for {name}")
            if id(parameter) not in visual_ids and name != "logit_scale":
                expected = self.stage == "full"
                if parameter.requires_grad != expected:
                    raise RuntimeError(f"Unexpected text encoder trainability for {name}")
        if self.base.clip.logit_scale.requires_grad != (self.stage in {"joint", "full"}):
            raise RuntimeError("Global CLIP temperature trainability does not match the stage")
        if any(parameter.requires_grad for parameter in self.base.binary_head.parameters()):
            raise RuntimeError("The unused A7 binary head must stay frozen in E1-B/C/D")
