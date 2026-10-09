"""Dependency-free Codabench inference adapter for AttriVision ViT-B/32."""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .config import DEFAULT_CHECKPOINT, REPOSITORY_ROOT, SUBMISSION_DIR, choose_device
from .data import ImagePathDataset
from .retrieval import autocast, l1_attribute_distances, reorder_columns


class QuickGELU(nn.Module):
    """Activation used by the original OpenAI CLIP checkpoints."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, quick_gelu: bool = False) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(768, 12)
        self.ln_2 = nn.LayerNorm(768)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(768, 3072)),
            ("gelu", QuickGELU() if quick_gelu else nn.GELU()),
            ("c_proj", nn.Linear(3072, 768)),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln_1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, quick_gelu: bool = False) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(quick_gelu) for _ in range(12)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class CLIPVisionTransformer(nn.Module):
    def __init__(self, quick_gelu: bool = False) -> None:
        super().__init__()
        self.class_embedding = nn.Parameter(torch.empty(768))
        self.positional_embedding = nn.Parameter(torch.empty(50, 768))
        self.proj = nn.Parameter(torch.empty(768, 512))
        self.conv1 = nn.Conv2d(3, 768, 32, 32, bias=False)
        self.ln_pre = nn.LayerNorm(768)
        self.transformer = Transformer(quick_gelu)
        self.ln_post = nn.LayerNorm(768)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images).reshape(images.shape[0], 768, -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype).expand(images.shape[0], 1, -1)
        x = torch.cat((cls, x), dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).permute(1, 0, 2)
        x = self.transformer(x).permute(1, 0, 2)
        return F.normalize(self.ln_post(x[:, 0]) @ self.proj, dim=-1)


_MODEL: CLIPVisionTransformer | None = None
_DEVICE: torch.device | None = None
_TEXT_FEATURES: torch.Tensor | None = None
_PAIRED_TEXT_FEATURES: torch.Tensor | None = None
_INVERSE_TEMPERATURE = 1.0
_ATTRIBUTE_NAMES: list[str] = []
_SEMANTIC_KEYS: list[str] = []
_CONFIG: dict[str, Any] = {}

# This is the fixed Native52 layout used by AttriVision's A7 evaluator.  It is
# kept here instead of importing the research package so the packaged
# submission remains dependency-free.
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


def _category_groups(semantic_keys: list[str]) -> list[np.ndarray]:
    state_index = {key: index for index, key in enumerate(semantic_keys)}
    groups: list[np.ndarray] = []
    for _, state_keys, fallback_key in _MULTI_CATEGORY_LAYOUT:
        groups.append(np.asarray(
            [state_index[key] for key in (*state_keys, fallback_key)], dtype=np.int64,
        ))
    for _, positive_key, negative_key in _BINARY_CATEGORY_LAYOUT:
        groups.append(np.asarray(
            [state_index[positive_key], state_index[negative_key]], dtype=np.int64,
        ))
    covered = np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)
    if len(groups) != 12 or len(covered) != 52 or len(np.unique(covered)) != 52:
        raise ValueError("Packaged AttriVision semantic states are not a Native52 partition")
    return groups


def _category_query_states(
    queries: np.ndarray, query_names: list[str],
) -> tuple[np.ndarray, list[np.ndarray]]:
    ordered = reorder_columns(queries, query_names, _ATTRIBUTE_NAMES) > 0.5
    attribute_index = {name: index for index, name in enumerate(_ATTRIBUTE_NAMES)}
    state_index = {key: index for index, key in enumerate(_SEMANTIC_KEYS)}
    states = np.zeros((len(ordered), len(_SEMANTIC_KEYS)), dtype=np.float32)
    groups = _category_groups(_SEMANTIC_KEYS)

    for columns, state_keys, fallback_key in _MULTI_CATEGORY_LAYOUT:
        values = ordered[:, [attribute_index[column] for column in columns]]
        for offset, key in enumerate(state_keys):
            states[:, state_index[key]] = values[:, offset]
        states[:, state_index[fallback_key]] = ~values.any(axis=1)
    for column, positive_key, negative_key in _BINARY_CATEGORY_LAYOUT:
        values = ordered[:, attribute_index[column]]
        states[:, state_index[positive_key]] = values
        states[:, state_index[negative_key]] = ~values

    return states, groups


def _native52_category_nll_distances(
    queries: np.ndarray, query_names: list[str], gallery: torch.Tensor,
) -> np.ndarray:
    if _TEXT_FEATURES is None or len(_SEMANTIC_KEYS) != 52:
        raise ValueError("Category-NLL requires the packaged 52-state text features")
    temperature = float(_CONFIG["category_temperature"])
    if temperature <= 0:
        raise ValueError("Category-NLL temperature must be positive")
    queries = np.asarray(queries, dtype=np.float32)
    if queries.ndim != 2 or not np.isin(queries, (0.0, 1.0)).all():
        raise ValueError("queries must be a binary [Q,40] array")
    query_states, groups = _category_query_states(queries, query_names)
    raw_logits = (gallery.float() @ _TEXT_FEATURES.float().T).numpy()
    probabilities = np.empty_like(raw_logits, dtype=np.float32)
    for indices in groups:
        values = raw_logits[:, indices].astype(np.float64) / temperature
        values -= values.max(axis=1, keepdims=True)
        exponentials = np.exp(values)
        probabilities[:, indices] = (
            exponentials / exponentials.sum(axis=1, keepdims=True)
        ).astype(np.float32)

    result = np.zeros((len(query_states), len(probabilities)), dtype=np.float32)
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
    for indices in groups:
        targets = query_states[:, indices]
        counts = targets.sum(axis=1, keepdims=True)
        if np.any(counts == 0):
            raise ValueError("Every query category needs an active semantic state")
        result -= (targets / counts) @ log_probabilities[:, indices].T
    result /= len(groups)
    if not np.isfinite(result).all():
        raise RuntimeError("Category-NLL distance matrix contains NaN or Inf")
    return result


def _build_eval_transform() -> transforms.Compose:
    if _CONFIG["augmentation"] == "resize_pad_crop":
        spatial = [transforms.Resize(
            (224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True,
        )]
    elif _CONFIG["augmentation"] == "center_crop":
        spatial = [
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
        ]
    else:
        raise ValueError(f"Unknown AttriVision submission augmentation: {_CONFIG['augmentation']}")
    return transforms.Compose([
        *spatial,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


def load_model() -> None:
    global _MODEL, _DEVICE, _TEXT_FEATURES, _PAIRED_TEXT_FEATURES
    global _INVERSE_TEMPERATURE, _ATTRIBUTE_NAMES, _SEMANTIC_KEYS, _CONFIG
    path = Path(os.environ.get("UPAR_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    device = choose_device(os.environ.get("UPAR_DEVICE", "auto"))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "attrivision_submission_vit_b_32":
        raise ValueError("Invalid packaged AttriVision checkpoint")
    model = CLIPVisionTransformer(bool(checkpoint.get("quick_gelu", False)))
    model.load_state_dict(checkpoint["visual_state_dict"], strict=True)
    _MODEL, _DEVICE = model.to(device).eval(), device
    _TEXT_FEATURES = F.normalize(checkpoint["text_features"].float(), dim=-1)
    paired = checkpoint.get("paired_text_features")
    _PAIRED_TEXT_FEATURES = (
        F.normalize(paired.float(), dim=-1) if paired is not None else None
    )
    _INVERSE_TEMPERATURE = float(checkpoint.get("inverse_temperature", 1.0))
    _ATTRIBUTE_NAMES = list(checkpoint["attribute_names"])
    _SEMANTIC_KEYS = list(checkpoint["semantic_keys"])
    _CONFIG = {
        "batch_size": int(os.environ.get("UPAR_BATCH_SIZE", "128")),
        "num_workers": int(os.environ.get("UPAR_NUM_WORKERS", "0")),
        "amp": os.environ.get("UPAR_AMP", "1").lower() not in {"0", "false", "no"},
        "progress_every": max(1, int(os.environ.get("UPAR_PROGRESS_EVERY", "5"))),
        "retrieval_scoring": os.environ.get(
            "UPAR_RETRIEVAL_SCORING", checkpoint.get("retrieval_scoring", "cosine_set")
        ),
        "category_temperature": float(os.environ.get(
            "UPAR_CATEGORY_TEMPERATURE", checkpoint.get("category_temperature", 0.01)
        )),
        "augmentation": os.environ.get(
            "UPAR_AUGMENTATION", checkpoint.get("augmentation", "center_crop")
        ),
    }
    print(
        f"[submission] AttriVision loaded on {device}; batch_size={_CONFIG['batch_size']}; "
        f"scoring={_CONFIG['retrieval_scoring']}", flush=True,
    )


def _semantic_queries(queries: np.ndarray, names: list[str]) -> torch.Tensor:
    queries = reorder_columns(queries, names, _ATTRIBUTE_NAMES) > 0.5
    ai = {name: i for i, name in enumerate(_ATTRIBUTE_NAMES)}
    si = {key: i for i, key in enumerate(_SEMANTIC_KEYS)}
    result = np.zeros((len(queries), len(si)), dtype=np.float32)
    groups = [
        (("Age-Young", "Age-Adult", "Age-Old"), ("age_young", "age_adult", "age_old"), "age_unknown"),
        (("Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald"), ("hair_short", "hair_long", "hair_bald"), "hair_other"),
        (tuple(f"UpperBody-Color-{c}" for c in ("Black","Blue","Brown","Green","Grey","Orange","Pink","Purple","Red","White","Yellow","Other")), tuple(f"upper_color_{c}" for c in ("black","blue","brown","green","grey","orange","pink","purple","red","white","yellow","other")), "upper_color_unspecified"),
        (tuple(f"LowerBody-Color-{c}" for c in ("Black","Blue","Brown","Green","Grey","Orange","Pink","Purple","Red","White","Yellow","Other")), tuple(f"lower_color_{c}" for c in ("black","blue","brown","green","grey","orange","pink","purple","red","white","yellow","other")), "lower_color_unspecified"),
        (("LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress"), ("lower_trousers_shorts", "lower_skirt_dress"), "lower_type_other"),
        (("Accessory-Glasses-Normal", "Accessory-Glasses-Sun"), ("glasses_normal", "glasses_sun"), "glasses_none"),
    ]
    for columns, keys, fallback in groups:
        values = queries[:, [ai[c] for c in columns]]
        for j, key in enumerate(keys): result[:, si[key]] = values[:, j]
        result[:, si[fallback]] = ~values.any(1)
    for column, yes, no in (("Gender-Female","gender_woman","gender_man"), ("UpperBody-Length-Short","upper_sleeves_short","upper_sleeves_long"), ("LowerBody-Length-Short","lower_length_short","lower_length_long"), ("Accessory-Backpack","backpack_yes","backpack_no"), ("Accessory-Bag","bag_yes","bag_no"), ("Accessory-Hat","hat_yes","hat_no")):
        values = queries[:, ai[column]]
        result[:, si[yes]], result[:, si[no]] = values, ~values
    return torch.from_numpy(result)


@torch.inference_mode()
def rank_gallery(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    if _MODEL is None: load_model()
    assert _MODEL is not None and _DEVICE is not None and _TEXT_FEATURES is not None
    paths = [item["image_path"] for item in sample["gallery"]]
    transform = _build_eval_transform()
    loader = DataLoader(
        ImagePathDataset(
            paths, [Path.cwd(), SUBMISSION_DIR, REPOSITORY_ROOT, REPOSITORY_ROOT / "data"],
            transform,
        ),
        batch_size=_CONFIG["batch_size"],
        num_workers=_CONFIG["num_workers"],
        pin_memory=_DEVICE.type == "cuda",
    )
    chunks, started = [], time.perf_counter()
    for index, images in enumerate(loader, 1):
        with autocast(_DEVICE, _CONFIG["amp"]): chunks.append(_MODEL(images.to(_DEVICE)).float().cpu())
        if index % _CONFIG["progress_every"] == 0 or index == len(loader): print(f"[submission] gallery batch {index}/{len(loader)}", flush=True)
    gallery = torch.cat(chunks)
    queries = np.asarray(sample["queries"], dtype=np.float32)
    if _CONFIG["retrieval_scoring"] == "native52_category_nll":
        distances = _native52_category_nll_distances(
            queries, list(sample["attribute_names"]), gallery,
        )
        print(
            f"[submission] Native52 Category-NLL retrieval complete in "
            f"{time.perf_counter()-started:.1f}s; distances={distances.shape}", flush=True,
        )
        return {"distances": distances}
    if _CONFIG["retrieval_scoring"] == "paired_l1":
        if _PAIRED_TEXT_FEATURES is None:
            raise ValueError("Packaged checkpoint does not contain paired prompt features")
        logits = torch.einsum("gd,akd->gak", gallery, _PAIRED_TEXT_FEATURES)
        probabilities = (logits * _INVERSE_TEMPERATURE).softmax(dim=-1)[..., 1].numpy()
        ordered_queries = reorder_columns(
            queries, list(sample["attribute_names"]), _ATTRIBUTE_NAMES,
        )
        distances = l1_attribute_distances(ordered_queries, probabilities)
        print(
            f"[submission] paired-L1 retrieval complete in {time.perf_counter()-started:.1f}s; "
            f"probabilities={probabilities.shape}, distances={distances.shape}", flush=True,
        )
        return {"distances": distances}
    if _CONFIG["retrieval_scoring"] != "cosine_set":
        raise ValueError(f"Unknown retrieval scoring: {_CONFIG['retrieval_scoring']}")
    weights = _semantic_queries(queries, list(sample["attribute_names"]))
    query_features = F.normalize((weights @ _TEXT_FEATURES) / weights.sum(1, keepdim=True), dim=-1)
    similarities = (query_features @ gallery.T).numpy().astype(np.float32, copy=False)
    print(f"[submission] retrieval complete in {time.perf_counter()-started:.1f}s; shape={similarities.shape}", flush=True)
    return {"similarities": similarities}
