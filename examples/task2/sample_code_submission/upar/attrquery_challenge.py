"""Dependency-free JointFCE inference adapter for Task 2 submissions.

The research checkpoint is converted by ``package_submission.py`` into a
compact payload containing the CLIP visual tower, normalized 52-state text
features, and the trained category cross-attention head.  This module keeps
the submission archive independent from the research-only ``attrivision``
package while preserving the JointFCE Native52 Category-NLL score.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .attrivision_challenge import CLIPVisionTransformer
from .config import REPOSITORY_ROOT, SUBMISSION_DIR, choose_device
from .data import ImagePathDataset
from .retrieval import autocast, reorder_columns


_MULTI_CATEGORY_LAYOUT = (
    (("Age-Young", "Age-Adult", "Age-Old"),
     ("age_young", "age_adult", "age_old"), "age_unknown"),
    (("Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald"),
     ("hair_short", "hair_long", "hair_bald"), "hair_other"),
    (tuple(f"UpperBody-Color-{color}" for color in (
        "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
        "Purple", "Red", "White", "Yellow", "Other",
    )), tuple(f"upper_color_{color}" for color in (
        "black", "blue", "brown", "green", "grey", "orange", "pink",
        "purple", "red", "white", "yellow", "other",
    )), "upper_color_unspecified"),
    (tuple(f"LowerBody-Color-{color}" for color in (
        "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
        "Purple", "Red", "White", "Yellow", "Other",
    )), tuple(f"lower_color_{color}" for color in (
        "black", "blue", "brown", "green", "grey", "orange", "pink",
        "purple", "red", "white", "yellow", "other",
    )), "lower_color_unspecified"),
    (("LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress"),
     ("lower_trousers_shorts", "lower_skirt_dress"), "lower_type_other"),
    (("Accessory-Glasses-Normal", "Accessory-Glasses-Sun"),
     ("glasses_normal", "glasses_sun"), "glasses_none"),
)
_BINARY_CATEGORY_LAYOUT = (
    ("Gender-Female", "gender_woman", "gender_man"),
    ("UpperBody-Length-Short", "upper_sleeves_short", "upper_sleeves_long"),
    ("LowerBody-Length-Short", "lower_length_short", "lower_length_long"),
    ("Accessory-Backpack", "backpack_yes", "backpack_no"),
    ("Accessory-Bag", "bag_yes", "bag_no"),
    ("Accessory-Hat", "hat_yes", "hat_no"),
)


class CategoryCrossAttention(nn.Module):
    """Inference copy of the trained research CategoryCrossAttention head."""

    def __init__(
        self,
        num_categories: int,
        visual_dim: int,
        text_dim: int,
        num_heads: int,
        dropout: float,
        use_ffn: bool,
    ) -> None:
        super().__init__()
        self.category_queries = nn.Parameter(torch.empty(num_categories, visual_dim))
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

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        queries = self.category_queries.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        attended, _ = self.attention(queries, patch_tokens, patch_tokens, need_weights=False)
        output = self.norm1(queries + self.dropout(attended))
        if self.use_ffn:
            output = self.norm2(output + self.dropout(self.ffn(output)))
        return F.normalize(self.output_projection(output.float()), dim=-1)


class JointFCEModel(nn.Module):
    def __init__(
        self,
        quick_gelu: bool,
        visual_state: dict[str, torch.Tensor],
        local_state: dict[str, torch.Tensor],
        visual_dim: int,
        text_dim: int,
        num_categories: int,
        num_heads: int,
        dropout: float,
        use_ffn: bool,
        patch_size: int = 32,
        visual_layers: int = 12,
        visual_heads: int = 12,
    ) -> None:
        super().__init__()
        self.visual = CLIPVisionTransformer(
            quick_gelu,
            patch_size=patch_size,
            visual_dim=visual_dim,
            output_dim=text_dim,
            layers=visual_layers,
            heads=visual_heads,
        )
        self.visual.load_state_dict(visual_state, strict=True)
        self.local_head = CategoryCrossAttention(
            num_categories, visual_dim, text_dim, num_heads, dropout, use_ffn,
        )
        self.local_head.load_state_dict(local_state, strict=True)

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.visual.conv1(images)
        x = x.reshape(images.shape[0], self.visual.visual_dim, -1).permute(0, 2, 1)
        cls = self.visual.class_embedding.to(x.dtype).expand(images.shape[0], 1, -1)
        x = torch.cat((cls, x), dim=1) + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x).permute(1, 0, 2)
        tokens = self.visual.transformer(x).permute(1, 0, 2)
        global_features = F.normalize(
            self.visual.ln_post(tokens[:, 0]) @ self.visual.proj, dim=-1,
        )
        local_features = self.local_head(tokens[:, 1:])
        return global_features, local_features


def _category_groups(semantic_keys: list[str]) -> list[np.ndarray]:
    index = {key: position for position, key in enumerate(semantic_keys)}
    groups: list[np.ndarray] = []
    for _, state_keys, fallback in _MULTI_CATEGORY_LAYOUT:
        groups.append(np.asarray([index[key] for key in (*state_keys, fallback)], dtype=np.int64))
    for _, positive, negative in _BINARY_CATEGORY_LAYOUT:
        groups.append(np.asarray([index[positive], index[negative]], dtype=np.int64))
    covered = np.concatenate(groups)
    if len(groups) != 12 or len(covered) != 52 or len(np.unique(covered)) != 52:
        raise ValueError("JointFCE payload does not contain a Native52 category partition")
    return groups


def _semantic_queries(
    queries: np.ndarray,
    query_names: list[str],
    attribute_names: list[str],
    semantic_keys: list[str],
) -> np.ndarray:
    ordered = reorder_columns(queries, query_names, attribute_names) > 0.5
    attribute_index = {name: index for index, name in enumerate(attribute_names)}
    state_index = {key: index for index, key in enumerate(semantic_keys)}
    states = np.zeros((len(ordered), len(semantic_keys)), dtype=np.float32)
    for columns, state_keys, fallback in _MULTI_CATEGORY_LAYOUT:
        values = ordered[:, [attribute_index[column] for column in columns]]
        for offset, key in enumerate(state_keys):
            states[:, state_index[key]] = values[:, offset]
        states[:, state_index[fallback]] = ~values.any(axis=1)
    for column, positive, negative in _BINARY_CATEGORY_LAYOUT:
        values = ordered[:, attribute_index[column]]
        states[:, state_index[positive]] = values
        states[:, state_index[negative]] = ~values
    return states


def _build_eval_transform(augmentation: str) -> transforms.Compose:
    if augmentation == "resize_pad_crop":
        spatial = [transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True)]
    elif augmentation == "center_crop":
        spatial = [
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
        ]
    else:
        raise ValueError(f"Unknown JointFCE submission augmentation: {augmentation}")
    return transforms.Compose([
        *spatial,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


_MODEL: JointFCEModel | None = None
_DEVICE: torch.device | None = None
_TEXT_FEATURES: torch.Tensor | None = None
_ATTRIBUTE_NAMES: list[str] = []
_SEMANTIC_KEYS: list[str] = []
_GROUPS: list[np.ndarray] = []
_CONFIG: dict[str, Any] = {}


def load_model(payload: dict[str, Any] | None = None) -> None:
    global _MODEL, _DEVICE, _TEXT_FEATURES, _ATTRIBUTE_NAMES, _SEMANTIC_KEYS
    global _GROUPS, _CONFIG
    started = time.perf_counter()
    print("[submission] JointFCE load_model entered", flush=True)
    path = Path(os.environ.get("UPAR_CHECKPOINT", str(REPOSITORY_ROOT / "assets" / "model_best.pth")))
    device = choose_device(os.environ.get("UPAR_DEVICE", "auto"))
    if payload is None:
        print("[submission] JointFCE reading packaged checkpoint on CPU", flush=True)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    architecture = payload.get("architecture")
    print(
        f"[submission] JointFCE checkpoint ready; architecture={architecture}; device={device}",
        flush=True,
    )
    architecture_patch_sizes = {
        "attrivision_jointfce_submission_vit_b_16": 16,
        "attrivision_jointfce_submission_vit_b_32": 32,
        "attrivision_jointfce_submission_vit_l_14": 14,
    }
    if architecture not in architecture_patch_sizes:
        raise ValueError("Invalid packaged JointFCE checkpoint")
    patch_size = int(payload.get("patch_size", architecture_patch_sizes[architecture]))
    if patch_size != architecture_patch_sizes[architecture]:
        raise ValueError("Packaged JointFCE architecture and patch size do not match")
    visual_state = payload["visual_state_dict"]
    local_state = payload["local_head_state_dict"]
    print("[submission] JointFCE building model", flush=True)
    model = JointFCEModel(
        bool(payload.get("quick_gelu", False)), visual_state, local_state,
        int(payload["visual_dim"]), int(payload["text_dim"]),
        int(payload["num_categories"]), int(payload["num_heads"]),
        float(payload["dropout"]), bool(payload["use_ffn"]), patch_size,
        int(payload.get("visual_layers", 12)),
        int(payload.get("visual_heads", 12)),
    )
    print(f"[submission] JointFCE moving model to {device}", flush=True)
    _MODEL, _DEVICE = model.to(device).eval(), device
    _TEXT_FEATURES = F.normalize(payload["text_features"].float(), dim=-1).to(device)
    _ATTRIBUTE_NAMES = list(payload["attribute_names"])
    _SEMANTIC_KEYS = list(payload["semantic_keys"])
    _GROUPS = _category_groups(_SEMANTIC_KEYS)
    _CONFIG = {
        "batch_size": int(os.environ.get(
            "UPAR_BATCH_SIZE", payload.get("default_batch_size", 128),
        )),
        "num_workers": int(os.environ.get(
            "UPAR_NUM_WORKERS", payload.get("default_num_workers", 0),
        )),
        "amp": os.environ.get("UPAR_AMP", "1").lower() not in {"0", "false", "no"},
        "progress_every": max(1, int(os.environ.get("UPAR_PROGRESS_EVERY", "5"))),
        "augmentation": os.environ.get("UPAR_AUGMENTATION", payload.get("augmentation", "resize_pad_crop")),
        "category_temperature": float(os.environ.get(
            "UPAR_CATEGORY_TEMPERATURE", payload.get("category_temperature", 0.01),
        )),
        "fusion_alpha": float(os.environ.get("UPAR_JOINT_FUSION_ALPHA", payload.get("fusion_alpha", 0.5))),
        "retrieval_scoring": os.environ.get(
            "UPAR_RETRIEVAL_SCORING", payload.get("retrieval_scoring", "native52_joint_fusion"),
        ),
    }
    if not 0.0 <= _CONFIG["fusion_alpha"] <= 1.0:
        raise ValueError("UPAR_JOINT_FUSION_ALPHA must be in [0,1]")
    print(
        f"[submission] JointFCE loaded on {device}; batch_size={_CONFIG['batch_size']}; "
        f"num_workers={_CONFIG['num_workers']}; scoring={_CONFIG['retrieval_scoring']}; "
        f"alpha={_CONFIG['fusion_alpha']}; model_init={time.perf_counter() - started:.1f}s",
        flush=True,
    )


def _normalized_query_state_weights(query_states: np.ndarray) -> np.ndarray:
    query_states = np.asarray(query_states, dtype=np.float32)
    weights = np.zeros_like(query_states)
    for indices in _GROUPS:
        targets = query_states[:, indices]
        counts = targets.sum(axis=1, keepdims=True)
        if np.any(counts == 0):
            raise ValueError("Every query category needs an active semantic state")
        weights[:, indices] = targets / counts / len(_GROUPS)
    return weights


def _category_nll_distances_from_features(
    query_weights: np.ndarray | torch.Tensor,
    global_gallery: torch.Tensor,
    local_gallery: torch.Tensor,
) -> np.ndarray:
    assert _TEXT_FEATURES is not None
    temperature = float(_CONFIG["category_temperature"])
    if temperature <= 0:
        raise ValueError("UPAR_CATEGORY_TEMPERATURE must be positive")
    global_scores = (global_gallery @ _TEXT_FEATURES.T).float().cpu().numpy()
    local_scores = torch.einsum(
        "gcd,sd->gcs", local_gallery, _TEXT_FEATURES,
    ).float().cpu().numpy()
    global_probabilities = np.zeros_like(global_scores, dtype=np.float32)
    local_probabilities = np.zeros_like(global_scores, dtype=np.float32)
    for category_index, indices in enumerate(_GROUPS):
        global_values = global_scores[:, indices].astype(np.float64) / temperature
        local_values = local_scores[:, category_index, indices].astype(np.float64) / temperature
        global_values -= global_values.max(axis=1, keepdims=True)
        local_values -= local_values.max(axis=1, keepdims=True)
        global_exp = np.exp(global_values)
        local_exp = np.exp(local_values)
        global_probabilities[:, indices] = (global_exp / global_exp.sum(axis=1, keepdims=True)).astype(np.float32)
        local_probabilities[:, indices] = (local_exp / local_exp.sum(axis=1, keepdims=True)).astype(np.float32)

    scoring = _CONFIG["retrieval_scoring"]
    if scoring in {"native52_joint_fusion", "joint_fusion"}:
        alpha = float(_CONFIG["fusion_alpha"])
        fused = np.exp(
            (1.0 - alpha) * np.log(np.clip(global_probabilities, 1e-12, 1.0))
            + alpha * np.log(np.clip(local_probabilities, 1e-12, 1.0))
        )
    elif scoring in {"native52_global", "global_category_nll"}:
        fused = global_probabilities
    elif scoring in {"native52_local", "local_category_nll"}:
        fused = local_probabilities
    else:
        raise ValueError(f"Unknown JointFCE retrieval scoring: {scoring}")
    log_probabilities = np.log(np.clip(fused, 1e-12, 1.0))
    device = global_gallery.device
    if isinstance(query_weights, torch.Tensor):
        query_weights_tensor = query_weights.to(device=device, dtype=torch.float32)
    else:
        query_weights_tensor = torch.as_tensor(
            query_weights, dtype=torch.float32, device=device,
        )
    log_probabilities_tensor = torch.as_tensor(
        log_probabilities, dtype=torch.float32, device=device,
    )
    distances_tensor = -query_weights_tensor @ log_probabilities_tensor.T
    distances = distances_tensor.float().cpu().numpy()
    if not np.isfinite(distances).all():
        raise RuntimeError("JointFCE Category-NLL distances contain NaN or Inf")
    return distances


def _category_nll_distances(
    queries: np.ndarray,
    query_names: list[str],
    global_gallery: torch.Tensor,
    local_gallery: torch.Tensor,
) -> np.ndarray:
    query_states = _semantic_queries(queries, query_names, _ATTRIBUTE_NAMES, _SEMANTIC_KEYS)
    return _category_nll_distances_from_features(
        _normalized_query_state_weights(query_states), global_gallery, local_gallery,
    )


@torch.inference_mode()
def rank_gallery(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    print(
        f"[submission] JointFCE rank_gallery entered; "
        f"queries={len(sample.get('queries', []))}; "
        f"gallery={len(sample.get('gallery', []))}; "
        f"attributes={len(sample.get('attribute_names', []))}",
        flush=True,
    )
    if _MODEL is None:
        load_model()
    assert _MODEL is not None and _DEVICE is not None
    paths = [item["image_path"] for item in sample["gallery"]]
    loader = DataLoader(
        ImagePathDataset(
            paths,
            [Path.cwd(), SUBMISSION_DIR, REPOSITORY_ROOT, REPOSITORY_ROOT / "data"],
            _build_eval_transform(_CONFIG["augmentation"]),
        ),
        batch_size=_CONFIG["batch_size"],
        num_workers=_CONFIG["num_workers"],
        pin_memory=_DEVICE.type == "cuda",
    )
    print(
        f"[submission] JointFCE gallery loader ready; batches={len(loader)}; "
        f"batch_size={_CONFIG['batch_size']}; num_workers={_CONFIG['num_workers']}",
        flush=True,
    )
    queries = np.asarray(sample["queries"], dtype=np.float32)
    query_states = _semantic_queries(
        queries, list(sample["attribute_names"]), _ATTRIBUTE_NAMES, _SEMANTIC_KEYS,
    )
    query_weights = torch.as_tensor(
        _normalized_query_state_weights(query_states), dtype=torch.float32, device=_DEVICE,
    )
    # Keep only the required query-by-gallery result on CPU. Moving all global
    # and local features to CPU and then back to CUDA created several GB of
    # avoidable traffic and temporary storage for full-gallery retrieval.
    distances = np.empty((len(query_states), len(paths)), dtype=np.float32)
    offset = 0
    started = time.perf_counter()
    for index, images in enumerate(loader, start=1):
        with autocast(_DEVICE, _CONFIG["amp"]):
            global_features, local_features = _MODEL.forward_features(
                images.to(_DEVICE, non_blocking=True),
            )
        end = offset + len(images)
        distances[:, offset:end] = _category_nll_distances_from_features(
            query_weights, global_features.float(), local_features.float(),
        )
        offset = end
        del global_features, local_features
        if index == 1 or index % _CONFIG["progress_every"] == 0 or index == len(loader):
            print(f"[submission] JointFCE gallery batch {index}/{len(loader)}", flush=True)
    if offset != len(paths):
        raise RuntimeError(f"DataLoader returned {offset} images for a gallery of {len(paths)}")
    print(
        f"[submission] JointFCE retrieval complete in {time.perf_counter() - started:.1f}s; "
        f"distances={distances.shape}; alpha={_CONFIG['fusion_alpha']}", flush=True,
    )
    return {"distances": distances}


def predict_attributes(
    gallery: list[dict[str, Any]], attribute_names: list[str],
) -> np.ndarray:
    raise NotImplementedError(
        "AttrQuery JointFCE submission exposes retrieval through rank_gallery(); "
        "it does not expose the legacy independent 40-attribute probability head."
    )
