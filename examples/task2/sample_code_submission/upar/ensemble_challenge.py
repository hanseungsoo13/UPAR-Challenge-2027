"""Inference adapter for score-level ensembles of portable JointFCE models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import SUBMISSION_DIR


_METHOD: str | None = None
_MEMBERS: list[dict[str, Any]] = []
_ATTRIBUTE_NAMES: list[str] = []
_SHARED_CONFIG: dict[str, Any] = {}


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(payload: dict[str, Any] | None = None) -> None:
    """Read and validate the small ensemble manifest; member weights load on demand."""
    global _METHOD, _MEMBERS, _ATTRIBUTE_NAMES, _SHARED_CONFIG
    manifest_path = SUBMISSION_DIR / "assets" / "model_best.pth"
    print("[submission] Ensemble load_model entered", flush=True)
    if payload is None:
        print("[submission] Ensemble reading manifest on CPU", flush=True)
        payload = _load_payload(manifest_path)
    if payload.get("architecture") != "attrivision_jointfce_ensemble_v1":
        raise ValueError("Invalid packaged JointFCE ensemble manifest")
    method = payload.get("ensemble_method")
    if method not in {"mean", "query_zscore_mean"}:
        raise ValueError(f"Unknown ensemble method: {method!r}")
    members = payload.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError("A JointFCE ensemble needs at least two members")
    attribute_names = payload.get("attribute_names")
    if not isinstance(attribute_names, list) or len(attribute_names) != 40:
        raise ValueError("Ensemble manifest must contain the 40 ordered attribute names")

    resolved_members: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    submission_root = SUBMISSION_DIR.resolve()
    for member in members:
        if not isinstance(member, dict) or not isinstance(member.get("path"), str):
            raise ValueError("Every ensemble member must have a relative asset path")
        relative = Path(member["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Ensemble member paths must stay inside the submission archive")
        resolved = (SUBMISSION_DIR / relative).resolve()
        try:
            resolved.relative_to(submission_root)
        except ValueError as exc:
            raise ValueError("Ensemble member path escapes the submission archive") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"Ensemble member is missing: {relative.as_posix()}")
        normalized = relative.as_posix()
        if normalized in seen_paths:
            raise ValueError(f"Duplicate ensemble member path: {normalized}")
        seen_paths.add(normalized)
        resolved_members.append({**member, "resolved_path": resolved})

    _METHOD = method
    _MEMBERS = resolved_members
    _ATTRIBUTE_NAMES = list(attribute_names)
    _SHARED_CONFIG = dict(payload.get("shared_config") or {})
    print(
        f"[submission] Ensemble manifest ready; method={_METHOD}; "
        f"members={len(_MEMBERS)}; inference=sequential",
        flush=True,
    )


@torch.inference_mode()
def rank_gallery(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run each member and return the selected mean of its query-gallery distances."""
    if _METHOD is None:
        load_model()
    assert _METHOD is not None
    if not {"queries", "gallery", "attribute_names"} <= set(sample):
        missing = {"queries", "gallery", "attribute_names"} - set(sample)
        raise KeyError(f"sample is missing required fields: {sorted(missing)}")
    if list(sample["attribute_names"]) != _ATTRIBUTE_NAMES:
        raise ValueError("Ensemble and challenge attribute order do not match")

    print(
        f"[submission] Ensemble rank_gallery entered; method={_METHOD}; "
        f"queries={len(sample['queries'])}; gallery={len(sample['gallery'])}; "
        f"members={len(_MEMBERS)}",
        flush=True,
    )
    from . import attrquery_challenge

    distances_by_member: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for index, member in enumerate(_MEMBERS, start=1):
        path: Path = member["resolved_path"]
        print(
            f"[submission] Ensemble member {index}/{len(_MEMBERS)} loading: "
            f"{member.get('source_checkpoint', path.name)}",
            flush=True,
        )
        payload = _load_payload(path)
        try:
            if payload.get("architecture") != "attrivision_jointfce_submission_vit_l_14":
                raise ValueError(f"Ensemble member {index} is not a JointFCE ViT-L/14 model")
            if list(payload.get("attribute_names", [])) != _ATTRIBUTE_NAMES:
                raise ValueError(f"Ensemble member {index} has a different attribute order")
            for key, expected in _SHARED_CONFIG.items():
                if payload.get(key) != expected:
                    raise ValueError(
                        f"Ensemble member {index} has incompatible {key}: "
                        f"{payload.get(key)!r} != {expected!r}"
                    )

            attrquery_challenge.load_model(payload=payload)
            result = attrquery_challenge.rank_gallery(sample)
            distances = np.asarray(result["distances"], dtype=np.float32)
            if distances.ndim != 2 or not np.isfinite(distances).all():
                raise RuntimeError(f"Ensemble member {index} returned invalid distances")
            if expected_shape is None:
                expected_shape = distances.shape
            elif distances.shape != expected_shape:
                raise RuntimeError(
                    f"Ensemble member {index} returned {distances.shape}, "
                    f"expected {expected_shape}"
                )
            distances_by_member.append(distances)
            print(
                f"[submission] Ensemble member {index}/{len(_MEMBERS)} complete; "
                f"distances={distances.shape}",
                flush=True,
            )
        finally:
            attrquery_challenge.reset_model()
            del payload

    if not distances_by_member:
        raise RuntimeError("No ensemble member produced distances")
    if _METHOD == "mean":
        combined = np.zeros_like(distances_by_member[0], dtype=np.float32)
        for distances in distances_by_member:
            combined += distances
        combined /= len(distances_by_member)
    else:
        standardized = []
        for distances in distances_by_member:
            mean = distances.mean(axis=1, keepdims=True)
            std = distances.std(axis=1, keepdims=True)
            standardized.append((distances - mean) / np.maximum(std, 1e-6))
        combined = np.mean(np.stack(standardized, axis=0), axis=0, dtype=np.float32)
    if not np.isfinite(combined).all():
        raise RuntimeError("Ensemble produced NaN or Inf distances")
    print(
        f"[submission] Ensemble complete; method={_METHOD}; "
        f"members={len(distances_by_member)}; distances={combined.shape}",
        flush=True,
    )
    return {"distances": combined}


def predict_attributes(
    gallery: list[dict[str, Any]], attribute_names: list[str],
) -> np.ndarray:
    raise NotImplementedError(
        "JointFCE ensembles expose retrieval through rank_gallery(), not "
        "the independent attribute-probability interface."
    )
