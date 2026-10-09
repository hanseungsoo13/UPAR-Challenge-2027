"""Build a Codabench-ready Task 2 code-submission archive."""
from __future__ import annotations

import argparse
import hashlib
import json
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
    elif architecture == "attrivision_submission_vit_b_32":
        required = {"visual_state_dict", "text_features", "semantic_keys", "attribute_names"}
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
        "architecture": "attrivision_submission_vit_b_32",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a UPAR Task 2 Codabench submission zip")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="output zip path (default: submissions/upar_task2_YYYYMMDD_HHMMSS.zip)",
    )
    parser.add_argument(
        "--model", choices=("auto", "convnext", "attrivision_a7"), default="auto",
        help="submission preset; auto preserves the checkpoint's architecture",
    )
    parser.add_argument(
        "--retrieval-scoring",
        choices=("cosine_set", "paired_l1", "native52_category_nll"),
        default="cosine_set",
        help="AttriVision submission ranking method",
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
