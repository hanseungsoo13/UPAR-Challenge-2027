"""Backbone adapters and the shared A7/AttriQuery upgrade model.

The Hugging Face dependencies are imported lazily. Constructing this module is
the point at which a user's run may download model files; importing it does not.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from attrivision.checkpoint import load_model as load_a7_model
from attrivision.datasets.attribute_prompts import CategoryPromptMapper
from attrivision.models.attrivision import AttriVision
from attrivision.models.attrquery_frozenclip_localfce import build_text_prototypes


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
DINO_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
CLIP_B16_MODEL_ID = "openai/clip-vit-base-patch16"
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-224"


def _transformers_imports() -> tuple[Any, Any, Any, Any]:
    try:
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        from transformers import CLIPVisionModel
    except ImportError as exc:
        raise ImportError(
            "BackboneUpgrade requires transformers>=4.56.0. Install the optional "
            "requirements-backbone-upgrade.txt dependencies before running an experiment."
        ) from exc
    return AutoImageProcessor, AutoModel, AutoProcessor, CLIPVisionModel


class HFVisualBackbone(nn.Module):
    """Expose global image embeddings and spatial tokens from a ViT backbone."""

    def __init__(self, backbone: str, model_id: str | None = None) -> None:
        super().__init__()
        AutoImageProcessor, AutoModel, AutoProcessor, CLIPVisionModel = _transformers_imports()
        if backbone not in {"clip_b16", "dinov3_b16", "siglip2_b16"}:
            raise ValueError(f"Unsupported local backbone: {backbone}")
        defaults = {
            "clip_b16": CLIP_B16_MODEL_ID,
            "dinov3_b16": DINO_MODEL_ID,
            "siglip2_b16": SIGLIP2_MODEL_ID,
        }
        self.backbone = backbone
        self.model_id = model_id or defaults[backbone]
        self.processor = None
        self.tokenizer = None

        if backbone == "clip_b16":
            self.vision = CLIPVisionModel.from_pretrained(self.model_id)
            self.config = self.vision.config
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.hidden_dim = int(self.config.hidden_size)
            self.text_dim = 512  # aligned with the A7 text embedding space.
            self.num_special_tokens = 1  # CLS; removed before local attention.
            self.register_buffer("image_mean", torch.tensor(self.processor.image_mean).view(1, 3, 1, 1))
            self.register_buffer("image_std", torch.tensor(self.processor.image_std).view(1, 3, 1, 1))
        elif backbone == "dinov3_b16":
            self.vision = AutoModel.from_pretrained(self.model_id)
            self.config = self.vision.config
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.hidden_dim = int(self.config.hidden_size)
            self.text_dim = 512  # DINOv3 reuses the A7 text prototypes.
            self.num_special_tokens = 1 + int(
                getattr(self.config, "num_register_tokens", 4)
            )
            self.register_buffer("image_mean", torch.tensor(self.processor.image_mean).view(1, 3, 1, 1))
            self.register_buffer("image_std", torch.tensor(self.processor.image_std).view(1, 3, 1, 1))
        else:
            # The fixed-resolution SigLIP2 checkpoint currently resolves to a
            # SigLIP (`model_type=siglip`) config and uses the standard 4D image
            # API. NaFlex SigLIP2 checkpoints use a patch sequence instead.
            from transformers import AutoTokenizer

            self.vision = AutoModel.from_pretrained(self.model_id)
            self.siglip_model_type = str(getattr(self.vision.config, "model_type", ""))
            if self.siglip_model_type not in {"siglip", "siglip2"}:
                raise ValueError(
                    f"Unsupported SigLIP model type {self.siglip_model_type!r} "
                    f"for {self.model_id}; expected 'siglip' or 'siglip2'"
                )
            self.config = self.vision.config.vision_config
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.hidden_dim = int(self.config.hidden_size)
            self.text_dim = int(self.vision.config.text_config.projection_size)
            self.num_special_tokens = 0
            image_processor = self.processor.image_processor
            self.register_buffer("image_mean", torch.tensor(image_processor.image_mean).view(1, 3, 1, 1))
            self.register_buffer("image_std", torch.tensor(image_processor.image_std).view(1, 3, 1, 1))

    def encode_text(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        if self.backbone != "siglip2_b16":
            raise RuntimeError("Only SigLIP2 has a text encoder in the local backbone adapter")
        tokens = self.tokenizer(
            list(prompts), padding="max_length", truncation=True, max_length=64,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        self.vision.eval()
        with torch.no_grad():
            encoded = self.vision.text_model(**tokens).pooler_output.float()
        return F.normalize(encoded, dim=-1).detach()

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        std = self.image_std.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std

    def forward(
        self, images: torch.Tensor, *, vision_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected raw RGB images [B,3,H,W], got {tuple(images.shape)}")
        pixel_values = self._normalize(images)
        context = torch.enable_grad() if vision_grad else torch.no_grad()
        with context:
            if self.backbone == "siglip2_b16":
                if self.siglip_model_type == "siglip":
                    # FixRes SigLIP2 checkpoints are backward-compatible with
                    # SigLIP and expect [B,3,H,W], not flattened patch pixels.
                    outputs = self.vision.vision_model(pixel_values=pixel_values)
                else:
                    height, width = pixel_values.shape[-2:]
                    patch = int(self.config.patch_size)
                    if height % patch or width % patch:
                        raise ValueError(
                            f"SigLIP2 input {height}x{width} must be divisible by patch size {patch}"
                        )
                    grid_h, grid_w = height // patch, width // patch
                    # Match Siglip2ImageProcessor.convert_image_to_patches: patch
                    # rows are flattened as (patch_y, patch_x, RGB channel).
                    patch_grid = pixel_values.reshape(
                        images.shape[0], 3, grid_h, patch, grid_w, patch,
                    ).permute(0, 2, 4, 3, 5, 1).reshape(
                        images.shape[0], grid_h * grid_w, patch * patch * 3,
                    )
                    spatial_shapes = torch.tensor(
                        [[grid_h, grid_w]], device=images.device, dtype=torch.long,
                    ).expand(images.shape[0], -1)
                    patch_mask = torch.ones(
                        images.shape[0], grid_h * grid_w, device=images.device, dtype=torch.bool,
                    )
                    outputs = self.vision.vision_model(
                        pixel_values=patch_grid,
                        pixel_attention_mask=patch_mask,
                        spatial_shapes=spatial_shapes,
                    )
                sequence = outputs.last_hidden_state
                if sequence.ndim != 3 or sequence.shape[-1] != self.hidden_dim:
                    raise RuntimeError(
                        f"Unexpected SigLIP spatial hidden-state shape {tuple(sequence.shape)}"
                    )
                patch_tokens = sequence[:, self.num_special_tokens:]
                if outputs.pooler_output is None:
                    raise RuntimeError("SigLIP model did not return its pooled global embedding")
                global_features = F.normalize(outputs.pooler_output.float(), dim=-1)
            else:
                outputs = self.vision(pixel_values=pixel_values)
                sequence = outputs.last_hidden_state
                if sequence.ndim != 3 or sequence.shape[-1] != self.hidden_dim:
                    raise RuntimeError(
                        f"Unexpected {self.backbone} hidden-state shape {tuple(sequence.shape)}"
                    )
                if sequence.shape[1] <= self.num_special_tokens:
                    raise RuntimeError(
                        f"No spatial tokens remain after removing {self.num_special_tokens} "
                        f"special tokens from shape {tuple(sequence.shape)}"
                    )
                patch_tokens = sequence[:, self.num_special_tokens:]
                global_features = None
        if patch_tokens.ndim != 3 or patch_tokens.shape[-1] != self.hidden_dim:
            raise RuntimeError(f"Invalid spatial patch features: {tuple(patch_tokens.shape)}")
        return patch_tokens.float(), None if global_features is None else global_features.float()


class CategoryCrossAttention(nn.Module):
    """Existing category-local decoder, with text-conditioned E3 queries."""

    def __init__(
        self,
        num_categories: int,
        visual_dim: int,
        text_dim: int,
        text_conditioned_queries: bool = False,
        num_text_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
    ) -> None:
        super().__init__()
        if visual_dim <= 0 or text_dim <= 0 or visual_dim % num_text_heads:
            raise ValueError("Invalid cross-attention dimensions or head count")
        self.query_projector = (
            nn.Linear(text_dim, visual_dim) if text_conditioned_queries else None
        )
        self.category_queries = nn.Parameter(torch.empty(num_categories, visual_dim))
        nn.init.normal_(self.category_queries, std=0.02)
        self.attention = nn.MultiheadAttention(
            visual_dim, num_text_heads, dropout=dropout, batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(visual_dim)
        self.use_ffn = bool(use_ffn)
        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(visual_dim, 4 * visual_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(4 * visual_dim, visual_dim),
            )
            self.norm2 = nn.LayerNorm(visual_dim)
        self.output_projection = nn.Linear(visual_dim, text_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        category_text_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        if patch_tokens.ndim != 3 or category_text_prototypes.ndim != 2:
            raise ValueError("Expected patch tokens [B,N,H] and category text [C,D]")
        if self.query_projector is None:
            query = self.category_queries
        else:
            query = (
                self.query_projector(category_text_prototypes.float())
                + self.category_queries
            )
        query = query.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        attended, _ = self.attention(query, patch_tokens, patch_tokens, need_weights=False)
        output = self.norm1(query + self.dropout(attended))
        if self.use_ffn:
            output = self.norm2(output + self.dropout(self.ffn(output)))
        return F.normalize(self.output_projection(output.float()), dim=-1)


class BackboneUpgradeModel(nn.Module):
    """A7 global + independent local backbone for E1/E2, full SigLIP2 for E3."""

    def __init__(
        self,
        *,
        experiment: str,
        backbone: str,
        model_id: str,
        attribute_names: Sequence[str],
        a7_checkpoint: str | None,
        trainable_vision_last_blocks: int = 0,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
    ) -> None:
        super().__init__()
        self.experiment = str(experiment)
        self.backbone_name = str(backbone)
        self.model_id = str(model_id)
        self.attribute_names = list(attribute_names)
        self.mapper = CategoryPromptMapper(self.attribute_names)
        self.category_groups = tuple(tuple(group) for group in self.mapper.category_indices())
        if len(self.mapper.keys) != 52 or len(self.category_groups) != 12:
            raise ValueError("BackboneUpgrade requires the established 52-state/12-category mapper")
        self.global_mode = "siglip2" if backbone == "siglip2_b16" else "a7"
        self.a7_checkpoint = str(a7_checkpoint) if a7_checkpoint else None
        self.local_backbone = HFVisualBackbone(backbone, model_id)
        self.visual_dim = self.local_backbone.hidden_dim
        self.text_dim = self.local_backbone.text_dim
        self.num_categories = len(self.category_groups)
        self.num_states = len(self.mapper.keys)

        device = next(self.local_backbone.parameters()).device
        if self.global_mode == "a7":
            if not self.a7_checkpoint:
                raise ValueError(f"{experiment} requires an A7 best checkpoint")
            from pathlib import Path

            a7_path = Path(self.a7_checkpoint)
            if not a7_path.is_file():
                raise FileNotFoundError(f"A7 checkpoint not found: {a7_path}")
            self.global_a7, a7_payload = load_a7_model(a7_path, device)
            if a7_payload.get("prompt_mode") != "category_complete":
                raise ValueError("E1/E2 require the category_complete A7 checkpoint")
            if list(a7_payload.get("attribute_names", [])) != self.attribute_names:
                raise ValueError("A7 checkpoint attribute order differs from the dataset")
            self.global_a7.eval()
            a7_mapper = CategoryPromptMapper(self.attribute_names)
            prototypes = build_text_prototypes(self.global_a7, a7_mapper, device, amp=False)
            self.register_buffer("state_text_prototypes", prototypes.float())
            self.register_buffer("global_text_prototypes", prototypes.float().clone())
        else:
            self.global_a7 = None
            prototypes = self.local_backbone.encode_text(self.mapper.prompts, device)
            self.register_buffer("state_text_prototypes", prototypes.float())
            self.register_buffer("global_text_prototypes", prototypes.float().clone())

        category_prototypes = []
        for group in self.category_groups:
            indices = torch.as_tensor(group, dtype=torch.long, device=device)
            pooled = self.state_text_prototypes[indices].mean(dim=0)
            category_prototypes.append(F.normalize(pooled, dim=-1))
        self.register_buffer("category_text_prototypes", torch.stack(category_prototypes).float())

        self.local_projector = nn.Linear(self.visual_dim, self.visual_dim)
        self.local_decoder = CategoryCrossAttention(
            self.num_categories, self.visual_dim, self.text_dim,
            text_conditioned_queries=(backbone == "siglip2_b16"),
            num_text_heads=num_heads, dropout=dropout, use_ffn=use_ffn,
        )
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.use_ffn = bool(use_ffn)
        self.trainable_vision_last_blocks = int(trainable_vision_last_blocks)
        self._freeze_and_select_parameters()

    def _freeze_and_select_parameters(self) -> None:
        for parameter in self.local_backbone.parameters():
            parameter.requires_grad_(False)
        if self.global_a7 is not None:
            for parameter in self.global_a7.parameters():
                parameter.requires_grad_(False)

        if self.backbone_name == "siglip2_b16" and self.trainable_vision_last_blocks > 0:
            layers = self.local_backbone.vision.vision_model.encoder.layers
            count = self.trainable_vision_last_blocks
            if count > len(layers):
                raise ValueError(f"Cannot unfreeze {count} blocks; SigLIP2 has {len(layers)}")
            for layer in layers[-count:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)

        for parameter in self.local_projector.parameters():
            parameter.requires_grad_(True)
        for parameter in self.local_decoder.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "BackboneUpgradeModel":
        super().train(mode)
        if self.global_a7 is not None:
            self.global_a7.eval()
        if self.backbone_name == "siglip2_b16":
            self.local_backbone.vision.text_model.eval()
            if self.trainable_vision_last_blocks == 0:
                self.local_backbone.vision.eval()
            else:
                for layer in self.local_backbone.vision.vision_model.encoder.layers:
                    if not any(parameter.requires_grad for parameter in layer.parameters()):
                        layer.eval()
        else:
            self.local_backbone.vision.eval()
        return self

    def forward(self, raw_images: torch.Tensor) -> dict[str, torch.Tensor]:
        vision_grad = self.backbone_name == "siglip2_b16" and self.trainable_vision_last_blocks > 0
        local_tokens, siglip_global = self.local_backbone(raw_images, vision_grad=vision_grad)
        local_tokens = self.local_projector(local_tokens.float())
        local_features = self.local_decoder(local_tokens, self.category_text_prototypes)
        local_scores = torch.einsum(
            "bcd,sd->bcs", local_features.float(), self.state_text_prototypes.float(),
        )

        if self.global_mode == "a7":
            normalized = (raw_images - raw_images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1))
            normalized = normalized / raw_images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
            self.global_a7.eval()
            with torch.no_grad():
                global_features = self.global_a7.encode_image(normalized)
            global_scores = global_features.float() @ self.global_text_prototypes.float().T
        else:
            if siglip_global is None:
                raise RuntimeError("SigLIP2 did not return its pooled global embedding")
            scale = self.local_backbone.vision.logit_scale.float().exp()
            bias = self.local_backbone.vision.logit_bias.float()
            global_scores = (
                scale * (siglip_global.float() @ self.global_text_prototypes.float().T) + bias
            )
        return {
            "global_scores": global_scores.float(),
            "local_scores": local_scores.float(),
            "local_features": local_features.float(),
            "patch_tokens": local_tokens.float(),
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def saveable_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        persistent_buffers = {
            "state_text_prototypes", "global_text_prototypes", "category_text_prototypes",
        }
        return {
            name: value.detach().cpu()
            for name, value in state.items()
            if name in trainable_names or name in persistent_buffers
        }

    def load_saveable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        bad_missing = [
            name for name in missing
            if name in {"state_text_prototypes", "global_text_prototypes", "category_text_prototypes"}
            or any(name == key for key, param in self.named_parameters() if param.requires_grad)
        ]
        if bad_missing or unexpected:
            raise ValueError(
                f"Experiment checkpoint/model mismatch: missing={sorted(bad_missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
