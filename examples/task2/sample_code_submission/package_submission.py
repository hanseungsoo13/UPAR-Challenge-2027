"""Build a Codabench-ready Task 2 code-submission archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

import torch


SUBMISSION_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SUBMISSION_DIR.parents[2]
CONVNEXT_CHECKPOINT_KEYS = {
    "model_state_dict",
    "attribute_names",
    "preprocessing",
    "positive_ratios",
}
EXCLUDED_NAMES = {"package_submission.py"}
EXCLUDED_PARTS = {"__pycache__", ".git", "attrivision"}


def load_checkpoint_metadata(source: Path | BinaryIO) -> dict[str, Any]:
    if isinstance(source, Path) and not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(source, map_location="cpu")
    architecture = checkpoint.get("architecture")
    if architecture == "attrivision_openclip_vit_b_32":
        required = {"model_state_dict", "attribute_names", "model_name", "prompt_mode"}
    elif architecture == "attrivision_openclip_vit_l_14":
        required = {"model_state_dict", "attribute_names", "model_name", "prompt_mode"}
    elif architecture == "attrivision_attrquery_shared_openclip_vit_b_16":
        required = {
            "model_state_dict", "attribute_names", "model_name", "model_stage",
            "experiment", "state_prompt_texts", "state_keys", "category_groups",
        }
    elif architecture == "attrivision_attrquery_shared_openclip_vit_l_14":
        required = {
            "model_state_dict", "attribute_names", "model_name", "model_stage",
            "experiment", "state_prompt_texts", "state_keys", "category_groups",
        }
    elif architecture == "attrquery_fromscratch_jointfce":
        required = {
            "model_state_dict", "attribute_names", "base_model_name",
            "category_groups", "num_heads", "dropout", "use_ffn",
        }
    elif architecture == "attrivision_submission_vit_b_32":
        required = {"visual_state_dict", "text_features", "semantic_keys", "attribute_names"}
    elif architecture == "attrivision_submission_vit_b_16":
        required = {"visual_state_dict", "text_features", "semantic_keys", "attribute_names"}
    elif architecture == "attrivision_submission_vit_l_14":
        required = {
            "visual_state_dict", "text_features", "semantic_keys", "attribute_names",
            "patch_size", "visual_dim", "output_dim", "visual_layers", "visual_heads",
        }
    elif architecture == "attrivision_jointfce_submission_vit_b_32":
        required = {
            "visual_state_dict", "local_head_state_dict", "text_features",
            "semantic_keys", "attribute_names", "category_groups",
        }
    elif architecture == "attrivision_jointfce_submission_vit_b_16":
        required = {
            "visual_state_dict", "local_head_state_dict", "text_features",
            "semantic_keys", "attribute_names", "category_groups",
            "patch_size", "fusion_alpha",
        }
    elif architecture == "attrivision_jointfce_submission_vit_l_14":
        required = {
            "visual_state_dict", "local_head_state_dict", "text_features",
            "semantic_keys", "attribute_names", "category_groups",
            "patch_size", "fusion_alpha", "visual_dim", "text_dim",
        }
    elif architecture == "attrivision_jointfce_ensemble_v1":
        required = {
            "format_version", "model_kind", "ensemble_method",
            "members", "attribute_names", "shared_config",
        }
    else:
        required = CONVNEXT_CHECKPOINT_KEYS
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    attributes = checkpoint["attribute_names"]
    if len(attributes) != 40 or len(set(attributes)) != 40:
        raise ValueError("Checkpoint must contain 40 unique attribute names")
    return {
        "epoch": checkpoint.get("epoch"),
        "model_kind": checkpoint.get("model_kind"),
        "metrics": checkpoint.get("metrics", {}),
        "architecture": checkpoint.get("architecture"),
    }


def _jointfce_submission_checkpoint(
    checkpoint: Path,
    category_temperature: float,
    augmentation: str,
    fusion_alpha: float = 0.5,
    retrieval_scoring: str = "native52_joint_fusion",
) -> tuple[bytes, dict[str, Any]]:
    """Convert a research JointFCE checkpoint to a dependency-free payload."""
    from attrivision.datasets.attribute_prompts import CategoryPromptMapper
    from attrivision.models.attrivision import AttriVision

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "attrquery_fromscratch_jointfce":
        raise ValueError("JointFCE packaging requires an attrquery_fromscratch_jointfce checkpoint")
    initialization = payload.get("initialization") or {}
    if initialization.get("a7_checkpoint_used") or initialization.get("attrquery_checkpoint_used"):
        raise ValueError("JointFCE submission packaging rejects A7/AttrQuery-initialized checkpoints")
    if category_temperature <= 0 or not 0.0 <= fusion_alpha <= 1.0:
        raise ValueError("Invalid JointFCE category temperature or fusion alpha")
    if retrieval_scoring not in {
        "native52_joint_fusion", "native52_global", "native52_local",
    }:
        raise ValueError(f"Unknown JointFCE retrieval scoring: {retrieval_scoring}")
    if augmentation not in {"center_crop", "resize_pad_crop"}:
        raise ValueError(f"Unknown JointFCE submission augmentation: {augmentation}")

    state = payload["model_state_dict"]
    base_state = {
        key.removeprefix("base."): value
        for key, value in state.items()
        if key.startswith("base.")
    }
    visual_state = {
        key.removeprefix("base.clip.visual."): value.detach().cpu()
        for key, value in state.items()
        if key.startswith("base.clip.visual.")
    }
    local_state = {
        key.removeprefix("local_head."): value.detach().cpu()
        for key, value in state.items()
        if key.startswith("local_head.")
    }
    if not visual_state or not local_state:
        raise ValueError("JointFCE checkpoint lacks visual or local-head weights")

    base = AttriVision(pretrained=None, model_name=payload["base_model_name"])
    missing, unexpected = base.load_state_dict(base_state, strict=True)
    if missing or unexpected:
        raise ValueError(
            f"JointFCE base checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    base.eval()
    mapper = CategoryPromptMapper(payload["attribute_names"])
    with torch.inference_mode():
        text_features = base.encode_text(
            base.tokenize(mapper.prompts),
        ).float().cpu()
    converted = {
        "format_version": 1,
        "architecture": "attrivision_jointfce_submission_vit_b_32",
        "quick_gelu": payload["base_model_name"].endswith("-quickgelu"),
        "visual_state_dict": visual_state,
        "local_head_state_dict": local_state,
        "text_features": text_features,
        "semantic_keys": list(mapper.keys),
        "attribute_names": list(payload["attribute_names"]),
        "category_groups": [list(group) for group in mapper.category_indices()],
        "visual_dim": int(payload.get("visual_dim", 768)),
        "text_dim": int(payload.get("text_dim", 512)),
        "num_categories": int(payload.get("num_categories", 12)),
        "num_states": int(payload.get("num_states", 52)),
        "num_heads": int(payload.get("num_heads", 8)),
        "dropout": float(payload.get("dropout", 0.1)),
        "use_ffn": bool(payload.get("use_ffn", True)),
        "retrieval_scoring": retrieval_scoring,
        "fusion_alpha": float(fusion_alpha),
        "category_temperature": float(category_temperature),
        "augmentation": augmentation,
        "epoch": payload.get("epoch"),
        "metrics": payload.get("metrics", {}),
        "source_architecture": payload.get("architecture"),
    }
    stream = BytesIO()
    torch.save(converted, stream)
    metadata = {
        "epoch": payload.get("epoch"),
        "model_kind": "attrquery_fromscratch_jointfce",
        "metrics": payload.get("metrics", {}),
        "architecture": payload.get("architecture"),
        "packaged_architecture": converted["architecture"],
    }
    return stream.getvalue(), metadata


def _shared_e1_submission_checkpoint(
    checkpoint: Path,
    expected_experiment: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Convert shared CLIP B/16 E1-B variants and E1-C checkpoints to the portable adapter payload."""
    from attrivision.datasets.attribute_prompts import CategoryPromptMapper
    from attrivision.models.attrivision import AttriVision

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "attrivision_attrquery_shared_openclip_vit_b_16":
        raise ValueError("E1-B packaging requires a shared CLIP ViT-B/16 checkpoint")
    experiment = payload.get("experiment")
    expected_stage = {
        "E1-B": "local",
        "E1-B1": "local",
        "E1-B2": "local",
        "E1-C": "joint",
    }
    if experiment not in expected_stage or payload.get("model_stage") != expected_stage[experiment]:
        raise ValueError("Shared E1 submission packaging supports E1-B variants at local stage or E1-C at joint stage")
    if expected_experiment is not None and experiment != expected_experiment:
        raise ValueError(f"Expected {expected_experiment}, found {experiment!r}")
    if payload.get("model_name") != "ViT-B-16-quickgelu":
        raise ValueError("Shared E1 submission packaging requires OpenAI CLIP ViT-B/16")
    if int(payload.get("epoch", 0)) < 1:
        raise ValueError("Shared E1 submission packaging rejects an untrained checkpoint")

    attributes = list(payload["attribute_names"])
    mapper = CategoryPromptMapper(attributes)
    if list(payload.get("state_keys", [])) != list(mapper.keys):
        raise ValueError("Shared E1 checkpoint semantic state order differs from the submission mapper")
    if list(payload.get("state_prompt_texts", [])) != list(mapper.prompts):
        raise ValueError("Shared E1 checkpoint prompt order differs from the submission mapper")
    expected_groups = [list(group) for group in mapper.category_indices()]
    if payload.get("category_groups") != expected_groups:
        raise ValueError("Shared E1 checkpoint category partition differs from the Native52 mapper")

    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Shared E1 checkpoint has no model_state_dict")
    base_state = {
        key.removeprefix("base."): value.detach().cpu()
        for key, value in state.items()
        if key.startswith("base.")
    }
    visual_state = {
        key.removeprefix("base.clip.visual."): value.detach().cpu()
        for key, value in state.items()
        if key.startswith("base.clip.visual.")
    }
    local_state = {
        key.removeprefix("local_head."): value.detach().cpu()
        for key, value in state.items()
        if key.startswith("local_head.")
    }
    if not visual_state or not local_state:
        raise ValueError("Shared E1 checkpoint lacks visual or local-head weights")

    base = AttriVision(pretrained=None, model_name=payload["model_name"])
    missing, unexpected = base.load_state_dict(base_state, strict=True)
    if missing or unexpected:
        raise ValueError(
            f"Shared E1 CLIP B/16 checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    base.eval()
    with torch.inference_mode():
        text_features = base.encode_text(base.tokenize(mapper.prompts)).float().cpu()

    training_config = payload.get("training_config", {})
    category_temperature = float(training_config.get("category_temperature", 0.01))
    fusion_alpha = float(training_config.get("selection_alpha", 0.5))
    if category_temperature <= 0 or abs(fusion_alpha - 0.5) > 1e-9:
        raise ValueError("Shared E1 submission requires category temperature > 0 and fusion alpha 0.5")
    converted = {
        "format_version": 1,
        "architecture": "attrivision_jointfce_submission_vit_b_16",
        "patch_size": 16,
        "quick_gelu": True,
        "visual_state_dict": visual_state,
        "local_head_state_dict": local_state,
        "text_features": text_features,
        "semantic_keys": list(mapper.keys),
        "attribute_names": attributes,
        "category_groups": expected_groups,
        "visual_dim": int(payload.get("visual_hidden_dim", 768)),
        "text_dim": int(payload.get("joint_embedding_dim", 512)),
        "num_categories": len(expected_groups),
        "num_states": len(mapper.keys),
        "num_heads": int(training_config.get("num_heads", 8)),
        "dropout": float(training_config.get("dropout", 0.1)),
        "use_ffn": bool(training_config.get("use_ffn", True)),
        "retrieval_scoring": "native52_joint_fusion",
        "fusion_alpha": fusion_alpha,
        "category_temperature": category_temperature,
        "augmentation": "resize_pad_crop",
        "default_batch_size": 128,
        "default_num_workers": 0,
        "epoch": payload.get("epoch"),
        "metrics": payload.get("metrics", {}),
        "source_architecture": payload.get("architecture"),
        "source_experiment": experiment,
    }
    stream = BytesIO()
    torch.save(converted, stream)
    metadata = {
        "epoch": payload.get("epoch"),
        "model_kind": f"attrivision_{experiment.lower().replace('-', '_')}_shared_clip_b16",
        "metrics": payload.get("metrics", {}),
        "architecture": payload.get("architecture"),
        "packaged_architecture": converted["architecture"],
    }
    return stream.getvalue(), metadata


def _attrivision_a7_l14_submission_checkpoint(
    checkpoint: Path,
) -> tuple[bytes, dict[str, Any]]:
    """Convert an AttriVision A7 CLIP L/14 checkpoint to an inference payload."""
    from attrivision.datasets.attribute_prompts import CategoryPromptMapper
    from attrivision.models.attrivision import AttriVision

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "attrivision_openclip_vit_l_14":
        raise ValueError("E1-A L/14 packaging requires an AttriVision ViT-L/14 checkpoint")
    if payload.get("model_name") != "ViT-L-14":
        raise ValueError("E1-A L/14 packaging requires OpenAI CLIP ViT-L/14")
    if payload.get("prompt_mode") != "category_complete":
        raise ValueError("E1-A L/14 packaging requires category_complete prompts")
    if int(payload.get("epoch", 0)) < 1:
        raise ValueError("E1-A L/14 packaging rejects an untrained checkpoint")

    attributes = list(payload["attribute_names"])
    mapper = CategoryPromptMapper(attributes)
    model = AttriVision(pretrained=None, model_name=payload["model_name"])
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=True)
    if missing or unexpected:
        raise ValueError(
            f"E1-A CLIP L/14 checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    with torch.inference_mode():
        text_features = model.encode_text(model.tokenize(mapper.prompts)).float().cpu()
    if tuple(text_features.shape) != (52, 768):
        raise ValueError(
            f"E1-A L/14 text features must have shape (52, 768), got {tuple(text_features.shape)}"
        )

    visual = {
        key.removeprefix("clip.visual."): value.detach().cpu()
        for key, value in payload["model_state_dict"].items()
        if key.startswith("clip.visual.")
    }
    if not visual:
        raise ValueError("E1-A L/14 checkpoint has no CLIP visual weights")
    training_config = payload.get("training_config", {})
    category_temperature = float(training_config.get("category_temperature", 0.01))
    if category_temperature <= 0:
        raise ValueError("E1-A L/14 category temperature must be positive")

    converted = {
        "format_version": 1,
        "architecture": "attrivision_submission_vit_l_14",
        "patch_size": 14,
        "quick_gelu": True,
        "visual_dim": 1024,
        "output_dim": 768,
        "visual_layers": 24,
        "visual_heads": 16,
        "visual_state_dict": visual,
        "text_features": text_features,
        "semantic_keys": list(mapper.keys),
        "attribute_names": attributes,
        "retrieval_scoring": "native52_category_nll",
        "category_temperature": category_temperature,
        "augmentation": "resize_pad_crop",
        "default_batch_size": int(training_config.get("eval_batch_size", 32)),
        # Codabench containers often have a small /dev/shm; load images inline.
        "default_num_workers": 0,
        "epoch": payload.get("epoch"),
        "metrics": payload.get("metrics", {}),
        "source_architecture": payload.get("architecture"),
        "source_experiment": "E1-A",
    }
    stream = BytesIO()
    torch.save(converted, stream)
    metadata = {
        "epoch": payload.get("epoch"),
        "model_kind": "attrivision_e1_a_shared_clip_l14",
        "metrics": payload.get("metrics", {}),
        "architecture": payload.get("architecture"),
        "packaged_architecture": converted["architecture"],
    }
    return stream.getvalue(), metadata


def _shared_e1_l14_submission_checkpoint(
    checkpoint: Path,
    expected_experiment: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Convert shared CLIP L/14 E1 joint/local checkpoints to a portable payload."""
    from attrivision.datasets.attribute_prompts import CategoryPromptMapper
    from attrivision.models.attrivision import AttriVision

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "attrivision_attrquery_shared_openclip_vit_l_14":
        raise ValueError("E1-B/E1-D packaging requires a shared CLIP ViT-L/14 checkpoint")
    experiment = payload.get("experiment")
    expected_stage = {
        "E1-B": "local",
        "E1-B2": "local",
        "E1-0-C": "joint",
        "E1-D": "full",
    }
    if experiment not in expected_stage or payload.get("model_stage") != expected_stage[experiment]:
        raise ValueError(
            "L/14 submission packaging supports E1-B/E1-B2 local-stage, "
            "E1-0-C joint-stage, or E1-D full-stage checkpoints"
        )
    if expected_experiment is not None and experiment != expected_experiment:
        raise ValueError(f"Expected {expected_experiment}, found {experiment!r}")
    if payload.get("model_name") != "ViT-L-14":
        raise ValueError("E1-B/E1-D submission packaging requires OpenAI CLIP ViT-L/14")
    if int(payload.get("epoch", 0)) < 1:
        raise ValueError("E1-B/E1-D submission packaging rejects an untrained checkpoint")

    attributes = list(payload["attribute_names"])
    mapper = CategoryPromptMapper(attributes)
    if list(payload.get("state_keys", [])) != list(mapper.keys):
        raise ValueError("L/14 shared E1 semantic state order differs from the submission mapper")
    if list(payload.get("state_prompt_texts", [])) != list(mapper.prompts):
        raise ValueError("L/14 shared E1 prompt order differs from the submission mapper")
    expected_groups = [list(group) for group in mapper.category_indices()]
    if payload.get("category_groups") != expected_groups:
        raise ValueError("L/14 shared E1 category partition differs from the Native52 mapper")

    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("L/14 shared E1 checkpoint has no model_state_dict")
    base_state = {
        key.removeprefix("base."): value.detach().cpu()
        for key, value in state.items()
        if key.startswith("base.")
    }
    visual_state = {
        key.removeprefix("base.clip.visual."): value.detach().cpu()
        for key, value in state.items()
        if key.startswith("base.clip.visual.")
    }
    local_state = {
        key.removeprefix("local_head."): value.detach().cpu()
        for key, value in state.items()
        if key.startswith("local_head.")
    }
    if not visual_state or not local_state:
        raise ValueError("L/14 shared E1 checkpoint lacks visual or local-head weights")

    base = AttriVision(pretrained=None, model_name=payload["model_name"])
    missing, unexpected = base.load_state_dict(base_state, strict=True)
    if missing or unexpected:
        raise ValueError(
            f"{experiment} CLIP L/14 checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    base.eval()
    with torch.inference_mode():
        text_features = base.encode_text(base.tokenize(mapper.prompts)).float().cpu()

    training_config = payload.get("training_config", {})
    category_temperature = float(training_config.get("category_temperature", 0.01))
    fusion_alpha = float(training_config.get("selection_alpha", 0.5))
    if category_temperature <= 0 or abs(fusion_alpha - 0.5) > 1e-9:
        raise ValueError("L/14 shared E1 submission requires category temperature > 0 and fusion alpha 0.5")
    converted = {
        "format_version": 1,
        "architecture": "attrivision_jointfce_submission_vit_l_14",
        "patch_size": 14,
        "quick_gelu": True,
        "visual_state_dict": visual_state,
        "local_head_state_dict": local_state,
        "text_features": text_features,
        "semantic_keys": list(mapper.keys),
        "attribute_names": attributes,
        "category_groups": expected_groups,
        "visual_dim": int(payload.get("visual_hidden_dim", 1024)),
        "text_dim": int(payload.get("joint_embedding_dim", 768)),
        "num_categories": len(expected_groups),
        "num_states": len(mapper.keys),
        "num_heads": int(training_config.get("num_heads", 8)),
        "dropout": float(training_config.get("dropout", 0.1)),
        "use_ffn": bool(training_config.get("use_ffn", True)),
        "visual_layers": 24,
        "visual_heads": 16,
        "retrieval_scoring": "native52_joint_fusion",
        "fusion_alpha": fusion_alpha,
        "category_temperature": category_temperature,
        "augmentation": "resize_pad_crop",
        "default_batch_size": int(training_config.get("eval_batch_size", 32)),
        # One loader worker overlaps image decode with CUDA inference. A single
        # prefetched batch keeps shared-memory use modest in Codabench Docker.
        "default_num_workers": 1,
        "default_prefetch_factor": 1,
        "epoch": payload.get("epoch"),
        "metrics": payload.get("metrics", {}),
        "source_architecture": payload.get("architecture"),
        "source_experiment": experiment,
    }
    stream = BytesIO()
    torch.save(converted, stream)
    metadata = {
        "epoch": payload.get("epoch"),
        "model_kind": f"attrivision_{experiment.lower().replace('-', '_')}_shared_clip_l14",
        "metrics": payload.get("metrics", {}),
        "architecture": payload.get("architecture"),
        "packaged_architecture": converted["architecture"],
        "attribute_names": attributes,
        "ensemble_config": {
            key: converted[key]
            for key in (
                "semantic_keys", "category_groups", "retrieval_scoring",
                "fusion_alpha", "category_temperature", "augmentation",
                "default_batch_size", "default_num_workers", "default_prefetch_factor",
            )
        },
    }
    return stream.getvalue(), metadata


def submission_checkpoint(
    checkpoint: Path,
    retrieval_scoring: str = "cosine_set",
    attribute_temperature: float | None = None,
    category_temperature: float = 0.01,
    augmentation: str = "center_crop",
    model: str = "auto",
) -> tuple[bytes | None, dict[str, Any]]:
    """Convert research AttriVision weights to a dependency-free inference payload."""
    metadata = load_checkpoint_metadata(checkpoint)
    architecture = metadata["architecture"]
    if model == "attrivision_e1_a_l14" or architecture == "attrivision_openclip_vit_l_14":
        if model not in {"auto", "attrivision_e1_a_l14"}:
            raise ValueError("The selected model preset does not accept an AttriVision A7 L/14 checkpoint")
        if architecture != "attrivision_openclip_vit_l_14":
            raise ValueError("The E1-A L/14 preset requires an AttriVision ViT-L/14 checkpoint")
        return _attrivision_a7_l14_submission_checkpoint(checkpoint)
    if model in {"attrivision_e1_b_l14", "attrivision_e1_b2_l14", "attrivision_e1_d"} or architecture == "attrivision_attrquery_shared_openclip_vit_l_14":
        if model not in {"auto", "attrivision_e1_b_l14", "attrivision_e1_b2_l14", "attrivision_e1_d"}:
            raise ValueError("The selected model preset does not accept a shared CLIP L/14 E1-B/E1-B2/E1-D checkpoint")
        if architecture != "attrivision_attrquery_shared_openclip_vit_l_14":
            raise ValueError("The selected E1 L/14 preset requires a shared CLIP ViT-L/14 checkpoint")
        expected_experiment = {
            "attrivision_e1_b_l14": "E1-B",
            "attrivision_e1_b2_l14": "E1-B2",
            "attrivision_e1_d": "E1-D",
        }.get(model)
        return _shared_e1_l14_submission_checkpoint(checkpoint, expected_experiment)
    if model in {"attrivision_e1_b", "attrivision_e1_b1", "attrivision_e1_b2", "attrivision_e1_c"} or architecture == "attrivision_attrquery_shared_openclip_vit_b_16":
        if model not in {"auto", "attrivision_e1_b", "attrivision_e1_b1", "attrivision_e1_b2", "attrivision_e1_c"}:
            raise ValueError("The selected model preset does not accept a shared E1 checkpoint")
        if architecture != "attrivision_attrquery_shared_openclip_vit_b_16":
            raise ValueError("The selected E1 preset requires a shared CLIP B/16 checkpoint")
        expected_experiment = {
            "attrivision_e1_b": "E1-B",
            "attrivision_e1_b1": "E1-B1",
            "attrivision_e1_b2": "E1-B2",
            "attrivision_e1_c": "E1-C",
        }.get(model)
        return _shared_e1_submission_checkpoint(checkpoint, expected_experiment)
    if model == "attrquery_jointfce" or architecture == "attrquery_fromscratch_jointfce":
        if model not in {"auto", "attrquery_jointfce"}:
            raise ValueError("The selected model preset does not accept a JointFCE checkpoint")
        if augmentation == "center_crop":
            # JointFCE validation and A7 comparison use resize_pad_crop.
            augmentation = "resize_pad_crop"
        if retrieval_scoring == "cosine_set":
            retrieval_scoring = "native52_joint_fusion"
        if retrieval_scoring not in {
            "native52_joint_fusion", "native52_global", "native52_local",
        }:
            raise ValueError("JointFCE packaging requires a Native52 JointFCE ranking mode")
        return _jointfce_submission_checkpoint(
            checkpoint, category_temperature, augmentation,
            retrieval_scoring=retrieval_scoring,
        )
    if model == "convnext" and architecture in {
        "attrivision_openclip_vit_b_32", "attrivision_submission_vit_b_32",
    }:
        raise ValueError("--model convnext requires a ConvNeXt checkpoint")
    if model == "attrivision_a7":
        if architecture != "attrivision_openclip_vit_b_32":
            raise ValueError("--model attrivision_a7 requires an AttriVision research checkpoint")
        retrieval_scoring = "native52_category_nll"
        category_temperature = 0.01
        augmentation = "resize_pad_crop"
    if architecture != "attrivision_openclip_vit_b_32":
        return None, metadata
    if retrieval_scoring not in {"cosine_set", "paired_l1", "native52_category_nll"}:
        raise ValueError(f"Unknown AttriVision retrieval scoring: {retrieval_scoring}")
    if augmentation not in {"center_crop", "resize_pad_crop"}:
        raise ValueError(f"Unknown AttriVision submission augmentation: {augmentation}")
    if category_temperature <= 0:
        raise ValueError("category_temperature must be positive")
    from attrivision.datasets.attribute_prompts import (
        CategoryPromptMapper, PaperAttributePromptMapper, prompt_pairs_for_attributes,
    )
    from attrivision.models.attrivision import AttriVision

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["model_name"].startswith("ViT-B-16"):
        packaged_architecture = "attrivision_submission_vit_b_16"
        patch_size = 16
    elif payload["model_name"].startswith("ViT-B-32"):
        packaged_architecture = "attrivision_submission_vit_b_32"
        patch_size = 32
    else:
        raise ValueError(
            "AttriVision submission packaging supports CLIP ViT-B/16 and ViT-B/32, "
            f"got {payload['model_name']!r}"
        )
    model = AttriVision(pretrained=None, model_name=payload["model_name"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    prompt_mode = payload.get("prompt_mode")
    if prompt_mode == "category_complete":
        mapper = CategoryPromptMapper(payload["attribute_names"])
        semantic_prompts = mapper.prompts
        semantic_keys = mapper.keys
    elif prompt_mode == "paper_binary":
        if retrieval_scoring != "paired_l1":
            raise ValueError(
                "paper_binary checkpoints must be packaged with --retrieval-scoring paired_l1"
            )
        mapper = PaperAttributePromptMapper(payload["attribute_names"])
        # The dependency-free adapter only needs semantic prototypes for the
        # cosine_set path. Keep an empty, shape-valid tensor for paired_l1.
        semantic_prompts = []
        semantic_keys = []
    else:
        raise ValueError(
            "AttriVision submission packaging supports category_complete or paper_binary"
        )
    if retrieval_scoring == "native52_category_nll" and prompt_mode != "category_complete":
        raise ValueError("native52_category_nll requires a category_complete AttriVision checkpoint")
    with torch.inference_mode():
        text_features = (
            model.encode_text(model.tokenize(semantic_prompts)).float().cpu()
            if semantic_prompts else torch.empty((0, 512), dtype=torch.float32)
        )
        if prompt_mode == "paper_binary":
            paired_prompts = mapper.prompts
        else:
            negative, positive = prompt_pairs_for_attributes(payload["attribute_names"])
            paired_prompts = [text for pair in zip(negative, positive) for text in pair]
        paired_text_features = (
            model.encode_text(model.tokenize(paired_prompts)).float().cpu()
            .reshape(len(payload["attribute_names"]), 2, -1)
        )
    if attribute_temperature is not None and attribute_temperature <= 0:
        raise ValueError("attribute_temperature must be positive")
    inverse_temperature = (
        1.0 / attribute_temperature
        if attribute_temperature is not None
        else float(model.logit_scale.detach().float().exp().clamp(max=100.0))
    )
    visual = {
        key.removeprefix("clip.visual."): value
        for key, value in payload["model_state_dict"].items()
        if key.startswith("clip.visual.")
    }
    converted = {
        "format_version": 1,
        "architecture": packaged_architecture,
        "patch_size": patch_size,
        "quick_gelu": payload["model_name"].endswith("-quickgelu"),
        "visual_state_dict": visual,
        "text_features": text_features,
        "semantic_keys": semantic_keys,
        "paired_text_features": paired_text_features,
        "inverse_temperature": inverse_temperature,
        "retrieval_scoring": retrieval_scoring,
        "category_temperature": float(category_temperature),
        "augmentation": augmentation,
        "attribute_names": payload["attribute_names"],
        "epoch": payload.get("epoch"),
        "metrics": payload.get("metrics", {}),
    }
    stream = BytesIO()
    torch.save(converted, stream)
    metadata["packaged_architecture"] = converted["architecture"]
    return stream.getvalue(), metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_output_path() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return REPOSITORY_ROOT / "submissions" / f"upar_task2_{timestamp}.zip"


def source_files() -> list[Path]:
    files = []
    for path in SUBMISSION_DIR.rglob("*"):
        relative = path.relative_to(SUBMISSION_DIR)
        if not path.is_file() or path.name in EXCLUDED_NAMES:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.as_posix() == "assets/model_best.pth":
            continue
        files.append(path)
    return sorted(files)


def build_archive(
    checkpoint: Path,
    output: Path,
    retrieval_scoring: str = "cosine_set",
    attribute_temperature: float | None = None,
    category_temperature: float = 0.01,
    augmentation: str = "center_crop",
    model: str = "auto",
) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    converted, metadata = submission_checkpoint(
        checkpoint, retrieval_scoring, attribute_temperature, category_temperature,
        augmentation, model,
    )
    with zipfile.ZipFile(temporary, "w") as archive:
        for path in source_files():
            relative = path.relative_to(SUBMISSION_DIR)
            archive.write(path, relative.as_posix(), compress_type=zipfile.ZIP_DEFLATED)
        if converted is None:
            archive.write(checkpoint, "assets/model_best.pth", compress_type=zipfile.ZIP_STORED)
        else:
            archive.writestr("assets/model_best.pth", converted, compress_type=zipfile.ZIP_STORED)

    with zipfile.ZipFile(temporary) as archive:
        names = set(archive.namelist())
        required = {"run.py", "metadata.yaml", "assets/model_best.pth"}
        missing = required - names
        if missing:
            raise RuntimeError(f"Submission archive is missing: {sorted(missing)}")
        if any(name.startswith("sample_code_submission/") for name in names):
            raise RuntimeError("Submission folder was nested instead of placing run.py at the root")
        archive.testzip()
        with archive.open("assets/model_best.pth") as checkpoint_stream:
            packaged_metadata = load_checkpoint_metadata(checkpoint_stream)

    temporary.replace(output)
    summary = {
        "archive": str(output.resolve()),
        "size_bytes": output.stat().st_size,
        "sha256": sha256(output),
        "checkpoint": str(checkpoint.resolve()),
        **metadata,
        "packaged_checkpoint": packaged_metadata,
    }
    print(json.dumps(summary, indent=2, default=str))


def build_ensemble_archives(
    checkpoints: list[Path],
    output_stem: str,
    output_dir: Path,
    methods: list[str],
) -> list[dict[str, Any]]:
    """Package shared E1-B L/14 checkpoints into score-averaging submissions."""
    if len(checkpoints) < 2:
        raise ValueError("An ensemble needs at least two checkpoints")
    if len(set(methods)) != len(methods) or not methods:
        raise ValueError("Ensemble methods must be a non-empty list without duplicates")
    allowed_methods = {"mean", "query_zscore_mean"}
    unknown_methods = set(methods) - allowed_methods
    if unknown_methods:
        raise ValueError(f"Unknown ensemble methods: {sorted(unknown_methods)}")
    if Path(output_stem).name != output_stem or output_stem in {"", ".", ".."}:
        raise ValueError("--ensemble-output-stem must be a simple filename stem")

    checkpoints = [path.resolve() for path in checkpoints]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("An ensemble cannot contain the same checkpoint more than once")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    staged_metadata: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix=".upar_ensemble_payloads_", dir=output_dir,
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        staged_payloads: list[Path] = []
        for index, checkpoint in enumerate(checkpoints, start=1):
            print(
                f"[packager] Converting ensemble checkpoint {index}/{len(checkpoints)}: "
                f"{checkpoint}",
                flush=True,
            )
            converted, metadata = _shared_e1_l14_submission_checkpoint(
                checkpoint, expected_experiment="E1-B",
            )
            stage_path = staging_dir / f"member_{index:02d}.pth"
            stage_path.write_bytes(converted)
            del converted
            config = metadata.get("ensemble_config")
            if not isinstance(config, dict):
                raise RuntimeError(f"Converted checkpoint has no ensemble config: {checkpoint}")
            staged_payloads.append(stage_path)
            staged_metadata.append({
                "path": str(checkpoint),
                "epoch": metadata.get("epoch"),
                "model_kind": metadata.get("model_kind"),
                "architecture": metadata.get("packaged_architecture"),
                "attribute_names": metadata.get("attribute_names"),
                "config": config,
            })

        reference_config = staged_metadata[0]["config"]
        reference_attributes = staged_metadata[0]["attribute_names"]
        for index, metadata in enumerate(staged_metadata[1:], start=2):
            if metadata["config"] != reference_config:
                raise ValueError(
                    f"Checkpoint {index} has incompatible inference settings or semantic ordering"
                )
            if metadata["attribute_names"] != reference_attributes:
                raise ValueError(f"Checkpoint {index} has a different attribute order")

        for method in methods:
            output = output_dir / f"{output_stem}_ensemble_{method}.zip"
            temporary_archive = output.with_suffix(output.suffix + ".tmp")
            if temporary_archive.exists():
                temporary_archive.unlink()
            member_entries = [
                {
                    "path": f"assets/ensemble_member_{index:02d}.pth",
                    "source_checkpoint": metadata["path"],
                    "epoch": metadata["epoch"],
                    "model_kind": metadata["model_kind"],
                    "architecture": metadata["architecture"],
                }
                for index, metadata in enumerate(staged_metadata, start=1)
            ]
            manifest = {
                "format_version": 1,
                "architecture": "attrivision_jointfce_ensemble_v1",
                "model_kind": "attrivision_e1_b_unseen_l14_ensemble",
                "ensemble_method": method,
                "members": member_entries,
                "attribute_names": reference_attributes,
                "shared_config": reference_config,
            }
            manifest_stream = BytesIO()
            torch.save(manifest, manifest_stream)
            human_manifest = {
                "ensemble_method": method,
                "member_count": len(member_entries),
                "members": member_entries,
                "score_definition": (
                    "Arithmetic mean of member distance matrices."
                    if method == "mean"
                    else "Per-query z-score each member over the gallery, then arithmetic mean."
                ),
            }

            print(
                f"[packager] Building {output.name}; method={method}; "
                f"members={len(member_entries)}",
                flush=True,
            )
            with zipfile.ZipFile(temporary_archive, "w", allowZip64=True) as archive:
                for path in source_files():
                    relative = path.relative_to(SUBMISSION_DIR)
                    archive.write(path, relative.as_posix(), compress_type=zipfile.ZIP_DEFLATED)
                archive.writestr(
                    "assets/model_best.pth",
                    manifest_stream.getvalue(),
                    compress_type=zipfile.ZIP_STORED,
                )
                archive.writestr(
                    "assets/ensemble_manifest.json",
                    json.dumps(human_manifest, indent=2).encode("utf-8"),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
                for entry, stage_path in zip(member_entries, staged_payloads):
                    archive.write(
                        stage_path, entry["path"], compress_type=zipfile.ZIP_STORED,
                    )

            with zipfile.ZipFile(temporary_archive) as archive:
                names = set(archive.namelist())
                required = {
                    "run.py", "metadata.yaml", "assets/model_best.pth",
                    "assets/ensemble_manifest.json",
                    *(entry["path"] for entry in member_entries),
                }
                missing = required - names
                if missing:
                    raise RuntimeError(f"Ensemble archive is missing: {sorted(missing)}")
                if any(name.startswith("sample_code_submission/") for name in names):
                    raise RuntimeError("Submission folder was nested instead of placing run.py at the root")
                corrupt = archive.testzip()
                if corrupt is not None:
                    raise RuntimeError(f"Ensemble archive has a corrupt member: {corrupt}")
                with archive.open("assets/model_best.pth") as manifest_file:
                    packaged_manifest = load_checkpoint_metadata(manifest_file)
                if packaged_manifest.get("architecture") != "attrivision_jointfce_ensemble_v1":
                    raise RuntimeError("Ensemble archive manifest failed metadata validation")

            temporary_archive.replace(output)
            summary = {
                "archive": str(output.resolve()),
                "size_bytes": output.stat().st_size,
                "sha256": sha256(output),
                "ensemble_method": method,
                "members": member_entries,
                "packaged_manifest": packaged_manifest,
            }
            summaries.append(summary)
            print(json.dumps(summary, indent=2, default=str), flush=True)

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a UPAR Task 2 Codabench submission zip")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--ensemble-checkpoint", action="append", type=Path,
        help="shared E1-B ViT-L/14 checkpoint to include in a score-level ensemble; repeat for each member",
    )
    parser.add_argument(
        "--ensemble-method", action="append",
        choices=("mean", "query_zscore_mean"),
        help="ensemble method; repeat to build multiple ZIPs (default: both methods)",
    )
    parser.add_argument(
        "--ensemble-output-stem",
        help="filename stem used for <stem>_ensemble_<method>.zip under submissions/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output zip path (default: submissions/upar_task2_YYYYMMDD_HHMMSS.zip)",
    )
    parser.add_argument(
        "--model", choices=("auto", "convnext", "attrivision_a7", "attrivision_e1_a_l14", "attrivision_e1_b", "attrivision_e1_b1", "attrivision_e1_b2", "attrivision_e1_c", "attrivision_e1_b_l14", "attrivision_e1_b2_l14", "attrivision_e1_d", "attrquery_jointfce"), default="auto",
        help="submission preset; auto preserves the checkpoint's architecture",
    )
    parser.add_argument(
        "--retrieval-scoring",
        choices=(
            "cosine_set", "paired_l1", "native52_category_nll",
            "native52_joint_fusion", "native52_global", "native52_local",
        ),
        default="cosine_set",
        help="Submission ranking method; JointFCE uses Native52 category scores",
    )
    parser.add_argument(
        "--attribute-temperature", type=float,
        help="paired_l1 softmax temperature T (default: checkpoint's learned CLIP temperature)",
    )
    parser.add_argument(
        "--category-temperature", type=float, default=0.01,
        help="native52_category_nll softmax temperature",
    )
    parser.add_argument(
        "--augmentation", choices=("center_crop", "resize_pad_crop"), default="center_crop",
        help="AttriVision submission evaluation geometry",
    )
    args = parser.parse_args()
    if args.ensemble_checkpoint:
        if args.checkpoint is not None or args.output is not None:
            parser.error("Use --ensemble-checkpoint with --ensemble-output-stem, not --checkpoint/--output")
        if not args.ensemble_output_stem:
            parser.error("--ensemble-output-stem is required with --ensemble-checkpoint")
        build_ensemble_archives(
            args.ensemble_checkpoint,
            args.ensemble_output_stem,
            REPOSITORY_ROOT / "submissions",
            args.ensemble_method or ["mean", "query_zscore_mean"],
        )
        return
    if args.checkpoint is None:
        parser.error("Either --checkpoint or repeated --ensemble-checkpoint values are required")
    output = args.output or default_output_path()
    if output.suffix.lower() != ".zip":
        raise ValueError("--output must end in .zip")
    build_archive(
        args.checkpoint.resolve(), output.resolve(),
        args.retrieval_scoring, args.attribute_temperature, args.category_temperature,
        args.augmentation, args.model,
    )


if __name__ == "__main__":
    main()
