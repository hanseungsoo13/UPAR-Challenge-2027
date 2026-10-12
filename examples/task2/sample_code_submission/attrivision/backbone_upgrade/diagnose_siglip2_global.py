"""Validate SigLIP2 global scoring and prepare the fused-selection D1 run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

SUBMISSION_DIR = Path(__file__).resolve().parents[2]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.backbone_upgrade.model import BackboneUpgradeModel  # noqa: E402
from attrivision.backbone_upgrade.pipeline import (  # noqa: E402
    CONFIG_DIR,
    _category_groups,
    _category_log_probs,
    _encode_gallery,
    _load_config,
    _load_payload,
    _raw_eval_transform,
    _retrieval_data,
    _score_views,
)
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.native52_retrieval_ablation import category_nll  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device  # noqa: E402
from upar.data import ImagePathDataset  # noqa: E402
from upar.retrieval import official_retrieval_metrics, reorder_columns  # noqa: E402


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _tokenizer_diagnostics(model: BackboneUpgradeModel) -> dict[str, Any]:
    backbone = model.local_backbone
    tokenizer = backbone.tokenizer
    prompts = list(model.mapper.prompts)
    encoded = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    embedding = backbone.vision.text_model.embeddings.token_embedding
    vocab_size = int(embedding.num_embeddings)
    input_ids = encoded["input_ids"]
    untruncated = tokenizer(prompts, padding=False, truncation=False)["input_ids"]
    text_config = backbone.vision.config.text_config

    tokenizer_special_ids = {
        name: getattr(tokenizer, f"{name}_token_id", None)
        for name in ("bos", "eos", "pad", "unk")
    }
    model_special_ids = {
        name: getattr(text_config, f"{name}_token_id", None)
        for name in ("bos", "eos", "pad")
    }

    def out_of_range(values: dict[str, int | None]) -> dict[str, int]:
        return {
            key: int(value)
            for key, value in values.items()
            if value is not None and not 0 <= int(value) < vocab_size
        }

    return {
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", vocab_size)),
        "embedding_vocab_size": vocab_size,
        "max_position_embeddings": int(getattr(text_config, "max_position_embeddings", -1)),
        "padding": "max_length",
        "max_length": 64,
        "prompt_count": len(prompts),
        "all_prompts_lowercase": all(prompt == prompt.lower() for prompt in prompts),
        "truncated_prompt_count": sum(len(ids) > 64 for ids in untruncated),
        "input_id_min": int(input_ids.min()),
        "input_id_max": int(input_ids.max()),
        "input_ids_in_embedding_range": bool(
            int(input_ids.min()) >= 0 and int(input_ids.max()) < vocab_size
        ),
        "tokenizer_special_ids": tokenizer_special_ids,
        "tokenizer_special_ids_out_of_range": out_of_range(tokenizer_special_ids),
        "model_config_special_ids": model_special_ids,
        "model_config_special_ids_out_of_range": out_of_range(model_special_ids),
    }


@torch.inference_mode()
def _reference_logit_parity(
    model: BackboneUpgradeModel,
    image_paths: list[str],
    roots: list[Path],
    device: torch.device,
    sample_count: int = 8,
) -> dict[str, Any]:
    backbone = model.local_backbone
    count = min(sample_count, len(image_paths))
    if count == 0:
        raise ValueError("Validation gallery is empty")
    dataset = ImagePathDataset(image_paths[:count], roots, _raw_eval_transform(224))
    images = next(iter(DataLoader(dataset, batch_size=count, shuffle=False))).to(device)
    tokens = backbone.tokenizer(
        model.mapper.prompts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    tokens = {name: value.to(device) for name, value in tokens.items()}
    pixel_values = backbone._normalize(images)

    reference = backbone.vision(pixel_values=pixel_values, **tokens)
    reference_logits = getattr(reference, "logits_per_image", None)
    if reference_logits is None:
        raise RuntimeError("The loaded SigLIP model did not return logits_per_image")
    adapter_logits = model(images)["global_scores"]
    difference = (adapter_logits.float() - reference_logits.float()).abs()
    max_abs_error = float(difference.max().item())
    return {
        "sample_images": count,
        "logit_shape": list(reference_logits.shape),
        "max_absolute_error": max_abs_error,
        "mean_absolute_error": float(difference.mean().item()),
        "tolerance": 1e-4,
        "passed": max_abs_error <= 1e-4,
    }


def _category_mapping_check(
    mapper: CategoryPromptMapper,
    queries: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    query_states = mapper.encode(torch.from_numpy(queries).float()).numpy()
    gallery_states = mapper.encode(torch.from_numpy(labels).float()).numpy()
    groups = mapper.category_indices()
    query_counts = np.stack([query_states[:, group].sum(axis=1) for group in groups], axis=1)
    gallery_counts = np.stack([gallery_states[:, group].sum(axis=1) for group in groups], axis=1)
    passed = bool(
        np.isfinite(query_states).all()
        and np.isfinite(gallery_states).all()
        and (query_counts > 0).all()
        and (gallery_counts > 0).all()
        and query_states.shape[1] == 52
        and gallery_states.shape[1] == 52
        and len(groups) == 12
    )
    return {
        "passed": passed,
        "state_count": int(query_states.shape[1]),
        "category_count": len(groups),
        "category_sizes": [len(group) for group in groups],
        "queries_with_empty_category": int((query_counts == 0).any(axis=1).sum()),
        "gallery_with_empty_category": int((gallery_counts == 0).any(axis=1).sum()),
    }


def _category_diagnostics(
    scores: np.ndarray,
    probabilities: np.ndarray,
    query_states: np.ndarray,
    gallery_states: np.ndarray,
    ids: np.ndarray,
    mapper: CategoryPromptMapper,
) -> dict[str, Any]:
    names = [item[0] for item in mapper._MULTI_GROUPS]
    names.extend(item[0] for item in mapper._BINARY_GROUPS)
    report: dict[str, Any] = {}
    for name, group in zip(names, _category_groups(mapper)):
        indices = np.asarray(group, dtype=np.int64)
        group_probabilities = probabilities[:, indices]
        group_queries = query_states[:, indices]
        group_labels = gallery_states[:, indices]
        distances = category_nll(
            group_queries,
            group_probabilities,
            [np.arange(len(indices), dtype=np.int64)],
        )
        metrics = official_retrieval_metrics(
            distances, group_queries, group_labels, ids,
        )

        predicted = scores[:, indices].argmax(axis=1)
        truth = group_labels > 0.5
        support = truth.sum(axis=0)
        confusion = np.zeros((len(indices), len(indices)), dtype=np.int64)
        for true_state in range(len(indices)):
            for predicted_state in range(len(indices)):
                confusion[true_state, predicted_state] = int(
                    (truth[:, true_state] & (predicted == predicted_state)).sum()
                )
        state_names = [mapper.keys[index] for index in indices]
        report[name] = {
            "exact_group_retrieval": {
                "rank1": float(metrics["Rank-1"]),
                "rank5": float(metrics["Rank-5"]),
                "mAP": float(metrics["mAP"]),
                "mADM": float(metrics["mADM"]),
            },
            "top1_in_any_positive_state": float(
                truth[np.arange(len(predicted)), predicted].mean()
            ),
            "state_support": {
                state_names[index]: int(support[index])
                for index in range(len(indices))
            },
            "state_recall": {
                state_names[index]: (
                    float(confusion[index, index] / support[index])
                    if support[index] else None
                )
                for index in range(len(indices))
            },
            "confusion_rows_true_states_columns_predicted_states": {
                "state_names": state_names,
                "counts": confusion.tolist(),
            },
        }
    return report


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="examples/task2/sample_code_submission/attrivision/backbone_upgrade/configs/e3a_siglip2_frozen.json",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/BackboneUpgrade/E3A_SigLIP2_Frozen/checkpoint_best.pth",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/BackboneUpgrade/D0_SigLIP2_GlobalScoring",
    )
    parser.add_argument("--device")
    parser.add_argument("--max-val-samples", type=int)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    if config["backbone"] != "siglip2_b16":
        raise ValueError("D0 expects a SigLIP2 configuration")
    if args.max_val_samples is not None:
        config["max_val_samples"] = args.max_val_samples
    device = choose_device(args.device or config.get("device", "auto"))
    checkpoint_path = _repo_path(args.checkpoint)
    payload = _load_payload(checkpoint_path, map_location="cpu")
    if payload["experiment"] != "E3A_SigLIP2_Frozen":
        raise ValueError("D0 checkpoint must come from E3A_SigLIP2_Frozen")
    if payload.get("model_id") != config["model_id"]:
        raise ValueError("D0 checkpoint and config use different SigLIP2 models")

    data_root = Path(config["data_root"])
    gt_path, table, queries_raw, ids, query_names = _retrieval_data(
        data_root, "val", config.get("max_val_samples"),
    )
    if list(payload["attribute_names"]) != list(table.attribute_names):
        raise ValueError("D0 checkpoint and validation data use different attribute orders")
    queries = reorder_columns(queries_raw, query_names, table.attribute_names)
    labels = table.labels
    model = BackboneUpgradeModel(
        experiment=config["experiment"],
        backbone=config["backbone"],
        model_id=config["model_id"],
        attribute_names=table.attribute_names,
        a7_checkpoint=None,
        trainable_vision_last_blocks=0,
        num_heads=int(config.get("num_heads", 8)),
        dropout=float(config.get("dropout", 0.1)),
        use_ffn=bool(config.get("use_ffn", True)),
    ).to(device)
    model.load_saveable_state_dict(payload["model_state_dict"])
    model.eval()

    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_report = _tokenizer_diagnostics(model)
    roots = [data_root, gt_path.parent, REPOSITORY_ROOT]
    parity_report = _reference_logit_parity(model, table.image_paths, roots, device)
    global_scores, local_scores = _encode_gallery(
        model,
        table.image_paths,
        roots,
        _raw_eval_transform(int(config["image_size"])),
        device,
        int(config["eval_batch_size"]),
        int(config["num_workers"]),
        bool(config.get("amp", True)),
        progress_every=25,
    )
    scores_path = output_dir / "val_gallery_scores.npz"
    np.savez_compressed(scores_path, global_scores=global_scores, local_scores=local_scores)

    mapper = CategoryPromptMapper(table.attribute_names)
    mapping_report = _category_mapping_check(mapper, queries, labels)
    query_states = mapper.encode(torch.from_numpy(queries).float()).numpy().astype(np.float32)
    gallery_states = mapper.encode(torch.from_numpy(labels).float()).numpy().astype(np.float32)
    modes = {"sigmoid_normalize": True, "logit_softmax": False}
    mode_results: dict[str, Any] = {}
    for mode in modes:
        mode_config = dict(config)
        mode_config["siglip2_global_scoring"] = mode
        mode_config["output_dir"] = str(output_dir)
        mode_results[mode] = _score_views(
            global_scores, local_scores, queries, labels, ids, mapper, mode_config,
        )

    recommended_mode = max(
        modes,
        key=lambda mode: (
            mode_results[mode]["views"]["fused"]["mADM"],
            mode_results[mode]["views"]["global"]["mADM"],
            mode == "sigmoid_normalize",
        ),
    )
    current_mode = config.get("siglip2_global_scoring", "sigmoid_normalize")
    recommended_log = _category_log_probs(
        global_scores,
        _category_groups(mapper),
        float(config.get("siglip2_global_temperature", 1.0)),
        sigmoid_alignment=modes[recommended_mode],
    )
    category_report = _category_diagnostics(
        global_scores,
        np.exp(recommended_log).astype(np.float32),
        query_states,
        gallery_states,
        ids,
        mapper,
    )

    checks = {
        "official_global_logits_match_adapter": parity_report["passed"],
        "prompt_input_ids_in_range": tokenizer_report["input_ids_in_embedding_range"],
        "prompts_lowercase": tokenizer_report["all_prompts_lowercase"],
        "expected_52_prompts": tokenizer_report["prompt_count"] == 52,
        "text_sequence_length_supported": tokenizer_report["max_position_embeddings"] >= 64,
        "prompts_not_truncated": tokenizer_report["truncated_prompt_count"] == 0,
        "category_mapping_valid": mapping_report["passed"],
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "status": "PASS" if passed else "FAIL",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "split": "val",
        "images": len(labels),
        "queries": len(queries),
        "checks": checks,
        "tokenizer": tokenizer_report,
        "reference_logit_parity": parity_report,
        "category_mapping": mapping_report,
        "scores_by_global_mode": {
            mode: {
                "global": metrics["views"]["global"],
                "local": metrics["views"]["local"],
                "fused": metrics["views"]["fused"],
                "selection_alpha": metrics["selection_alpha"],
            }
            for mode, metrics in mode_results.items()
        },
        "existing_e3a_selection_metric": "fused.mADM",
        "d1_training_required": bool(passed and recommended_mode != current_mode),
        "recommended_global_mode_by_fused_mADM": recommended_mode,
        "category_diagnostics_for_recommended_mode": category_report,
        "gallery_scores": str(scores_path),
    }

    report_path = output_dir / "diagnosis.json"
    _write_json(report_path, report)
    print(f"D0_STATUS={report['status']}")
    for mode in modes:
        metrics = mode_results[mode]["views"]
        print(
            f"mode={mode} global R1={100*metrics['global']['rank1']:.2f}% "
            f"mAP={100*metrics['global']['mAP']:.2f}% "
            f"mADM={100*metrics['global']['mADM']:.2f}% | "
            f"fused R1={100*metrics['fused']['rank1']:.2f}% "
            f"mAP={100*metrics['fused']['mAP']:.2f}% "
            f"mADM={100*metrics['fused']['mADM']:.2f}%"
        )
    print(f"recommended_global_mode={recommended_mode}")
    if tokenizer_report["model_config_special_ids_out_of_range"]:
        print(
            "WARNING: model config special token IDs exceed the text embedding "
            "vocabulary; actual prompt input IDs are reported in diagnosis.json."
        )
    print(f"diagnosis={report_path}")

    if passed and recommended_mode != current_mode:
        with (CONFIG_DIR / "d1_siglip2_fused.json").open(encoding="utf-8") as handle:
            d1_config = json.load(handle)
        d1_config["siglip2_global_scoring"] = recommended_mode
        d1_config["selection_alpha"] = float(config["selection_alpha"])
        d1_config["selection_metric"] = "fused.mADM"
        d1_config_path = output_dir / "d1_siglip2_fused.json"
        _write_json(d1_config_path, d1_config)
        print(f"D1_CONFIG={d1_config_path}")
    elif passed:
        report["reuse_checkpoint"] = str(checkpoint_path)
        _write_json(report_path, report)
        print("D1_NOT_REQUIRED=E3A already selected checkpoints by fused.mADM with this scoring mode")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
