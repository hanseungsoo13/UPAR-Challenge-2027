"""Stable API used by the challenge ingestion program."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .config import DEFAULT_CHECKPOINT, REPOSITORY_ROOT, SUBMISSION_DIR, PreprocessingConfig, choose_device
from .data import build_eval_transform
from .modeling import UPARModel
from .retrieval import infer_probabilities, l1_attribute_distances, reorder_columns

_MODEL: UPARModel | None = None
_DEVICE: torch.device | None = None
_TRANSFORM: Any = None
_ATTRIBUTE_NAMES: list[str] = []
_INFERENCE_CONFIG: dict[str, Any] = {}
_ATTRIVISION = False


def load_model() -> None:
    """Load the submission model once into module-level state."""
    global _MODEL, _DEVICE, _TRANSFORM, _ATTRIBUTE_NAMES, _INFERENCE_CONFIG, _ATTRIVISION
    started = time.perf_counter()
    requested_device = os.environ.get("UPAR_DEVICE", "auto")
    device = choose_device(requested_device)
    checkpoint_path = os.environ.get("UPAR_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    preview = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if preview.get("architecture") == "attrivision_submission_vit_b_32":
        from . import attrivision_challenge
        attrivision_challenge.load_model()
        _ATTRIVISION = True
        return
    print(
        f"[submission] loading model: requested_device={requested_device}, "
        f"resolved_device={device}, cuda_available={torch.cuda.is_available()}, "
        f"torch={torch.__version__}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"[submission] CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"[submission] checkpoint: {checkpoint_path}", flush=True)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    preprocessing = PreprocessingConfig.from_dict(checkpoint["preprocessing"])
    _MODEL = model
    _DEVICE = device
    _TRANSFORM = build_eval_transform(preprocessing)
    _ATTRIBUTE_NAMES = list(checkpoint["attribute_names"])
    _INFERENCE_CONFIG = {
        "batch_size": int(os.environ.get("UPAR_BATCH_SIZE", "128")),
        "num_workers": int(os.environ.get("UPAR_NUM_WORKERS", "0")),
        "amp": os.environ.get("UPAR_AMP", "1").lower() not in {"0", "false", "no"},
        "progress_every": max(1, int(os.environ.get("UPAR_PROGRESS_EVERY", "5"))),
    }
    print(
        f"[submission] model loaded in {time.perf_counter() - started:.1f}s; "
        f"batch_size={_INFERENCE_CONFIG['batch_size']}, "
        f"num_workers={_INFERENCE_CONFIG['num_workers']}, "
        f"amp={_INFERENCE_CONFIG['amp']}",
        flush=True,
    )


def predict_attributes(gallery: list[dict[str, Any]], attribute_names: list[str]) -> np.ndarray:
    """Return probabilities in the exact attribute order requested by the server."""
    if _MODEL is None:
        load_model()
    assert _MODEL is not None and _DEVICE is not None
    paths: list[str] = []
    for index, item in enumerate(gallery):
        if "image_path" not in item:
            raise KeyError(f"gallery[{index}] has no 'image_path'")
        paths.append(str(item["image_path"]))
    print(f"[submission] starting gallery inference for {len(paths)} images", flush=True)
    started = time.perf_counter()
    probabilities = infer_probabilities(
        _MODEL,
        paths,
        [Path.cwd(), SUBMISSION_DIR, REPOSITORY_ROOT, REPOSITORY_ROOT / "data"],
        _TRANSFORM,
        _DEVICE,
        _INFERENCE_CONFIG["batch_size"],
        _INFERENCE_CONFIG["num_workers"],
        _INFERENCE_CONFIG["amp"],
        _INFERENCE_CONFIG["progress_every"],
        "[submission] gallery",
    )
    probabilities = reorder_columns(
        probabilities,
        _ATTRIBUTE_NAMES,
        attribute_names,
    ).astype(np.float32, copy=False)
    print(
        f"[submission] gallery inference complete in {time.perf_counter() - started:.1f}s; "
        f"shape={probabilities.shape}, dtype={probabilities.dtype}",
        flush=True,
    )
    return probabilities


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    """Return a [queries, gallery] L1 attribute-distance matrix."""
    if _MODEL is None and not _ATTRIVISION:
        load_model()
    if _ATTRIVISION:
        from . import attrivision_challenge
        return attrivision_challenge.rank_gallery(sample)
    missing = {"gallery", "attribute_names", "queries"} - set(sample)
    if missing:
        raise KeyError(f"sample is missing required fields: {sorted(missing)}")
    print(
        f"[submission] rank_gallery called: queries={len(sample['queries'])}, "
        f"gallery={len(sample['gallery'])}, attributes={len(sample['attribute_names'])}",
        flush=True,
    )
    probabilities = predict_attributes(sample["gallery"], list(sample["attribute_names"]))
    queries = np.asarray(sample["queries"], dtype=np.float32)
    print("[submission] computing query-gallery distances", flush=True)
    started = time.perf_counter()
    distances = l1_attribute_distances(queries, probabilities)
    print(
        f"[submission] distances complete in {time.perf_counter() - started:.1f}s; "
        f"shape={distances.shape}, dtype={distances.dtype}, finite={np.isfinite(distances).all()}",
        flush=True,
    )
    print("[submission] rank_gallery returning distances", flush=True)
    return {"distances": distances}
