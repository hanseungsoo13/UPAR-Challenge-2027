"""Native52 graded-ranking experiments on top of the A7 checkpoint.

This file is deliberately standalone.  The existing A7 trainer and checkpoint
are not modified.  The retrieval score used here is the same geometry as the
Native52 Category-NLL evaluator:

    image -> 52 cosine logits -> category-local softmax -> category NLL

The graded target is based on the original 40-bit Hamming distance.  G0 learns
only per-category score calibration.  G1 fine-tunes A7 with FCE plus the
graded listwise loss.  G2 adds raw 52-state logit distillation from a frozen
A7 teacher.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model, model_payload  # noqa: E402
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.datasets.upar_abpr import (  # noqa: E402
    AttriVisionDataset, PromptBatch, PromptCollator,
)
from attrivision.engine.evaluator_abpr import (  # noqa: E402
    _binary_metric, encode_gallery,
)
from attrivision.engine.trainer_attrivision import (  # noqa: E402
    _grad_scaler, _optimizer_groups, _optimizer_step, build_scheduler,
)
from attrivision.losses.focal_clip_loss import FocalCLIPLoss  # noqa: E402
from attrivision.transforms import build_eval_transform, build_train_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import AnnotationTable, find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import (  # noqa: E402
    autocast, load_retrieval_annotations, official_retrieval_metrics,
    reorder_columns, retrieval_metrics,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "a7_native52_graded.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _category_groups(mapper: CategoryPromptMapper) -> list[torch.Tensor]:
    return [torch.tensor(group, dtype=torch.long) for group in mapper.category_indices()]


class Native52Score(nn.Module):
    """Differentiable Native52 category-NLL score.

    ``score`` is higher for a better match.  For fixed G1/G2 this is exactly
    the existing evaluator: one shared temperature and equal category weights.
    G0 turns the temperature and category weights into the only trainable
    parameters, while using the resulting score again at evaluation time.
    """

    def __init__(self, groups: Sequence[Sequence[int]], temperature: float,
                 learnable: bool = False) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("Native52 temperature must be positive")
        self.groups = [tuple(int(index) for index in group) for group in groups]
        initial = math.log(float(temperature))
        temperature_parameter = nn.Parameter(
            torch.full((len(self.groups),), initial), requires_grad=learnable,
        )
        weight_parameter = nn.Parameter(
            torch.zeros(len(self.groups)), requires_grad=learnable,
        )
        self.log_temperature = temperature_parameter
        self.log_weight = weight_parameter
        self.learnable = bool(learnable)

    def temperatures(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(1.0e-4, 10.0)

    def weights(self) -> torch.Tensor:
        if self.learnable:
            return F.softmax(self.log_weight, dim=0) * len(self.groups)
        return torch.ones_like(self.log_weight)

    def score_from_logits(
        self, query_semantic: torch.Tensor, candidate_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[num_queries, num_candidates]`` score, higher is better."""
        if query_semantic.ndim != 2 or candidate_logits.ndim != 2:
            raise ValueError("query_semantic and candidate_logits must be matrices")
        if query_semantic.shape[1] != candidate_logits.shape[1]:
            raise ValueError("Query and candidate state widths differ")
        query_semantic = query_semantic.float()
        candidate_logits = candidate_logits.float()
        category_scores: list[torch.Tensor] = []
        temperatures = self.temperatures()
        for category_index, indices in enumerate(self.groups):
            index = torch.as_tensor(indices, device=candidate_logits.device)
            targets = query_semantic[:, index]
            counts = targets.sum(dim=1, keepdim=True)
            if (counts <= 0).any():
                raise ValueError("Every query must have an active state per category")
            targets = targets / counts
            log_prob = F.log_softmax(
                candidate_logits[:, index] / temperatures[category_index], dim=1,
            )
            category_scores.append(targets @ log_prob.T)
        return torch.stack(category_scores, dim=2).mul(self.weights()).sum(dim=2)

    def probabilities_from_logits(self, candidate_logits: torch.Tensor) -> torch.Tensor:
        """Return category-local probabilities with the active temperatures."""
        probabilities = torch.zeros_like(candidate_logits.float())
        temperatures = self.temperatures()
        for category_index, indices in enumerate(self.groups):
            index = torch.as_tensor(indices, device=candidate_logits.device)
            probabilities[:, index] = F.softmax(
                candidate_logits[:, index].float() / temperatures[category_index], dim=1,
            )
        return probabilities


def graded_target(query_labels: torch.Tensor, candidate_labels: torch.Tensor,
                  temperature: float) -> torch.Tensor:
    if query_labels.ndim != 2 or candidate_labels.ndim != 2:
        raise ValueError("graded labels must be matrices")
    if query_labels.shape[1] != candidate_labels.shape[1]:
        raise ValueError("graded query and candidate label widths differ")
    if temperature <= 0:
        raise ValueError("target temperature must be positive")
    distance = torch.abs(
        query_labels.float()[:, None, :] - candidate_labels.float()[None, :, :],
    ).sum(dim=2)
    return F.softmax(-distance / temperature, dim=1)


def graded_listwise_loss(score: torch.Tensor, query_labels: torch.Tensor,
                         candidate_labels: torch.Tensor, target_temperature: float,
                         score_temperature: float) -> torch.Tensor:
    target = graded_target(query_labels, candidate_labels, target_temperature).detach()
    predicted_log_probability = F.log_softmax(score / score_temperature, dim=1)
    return -(target * predicted_log_probability).sum(dim=1).mean()


def _annotation_views(data_root: Path, split: str, max_samples: int | None) -> tuple[Any, np.ndarray, np.ndarray, list[str]]:
    gt_path = find_annotation_file(data_root, split)
    table = read_gt_csv(gt_path)
    if max_samples is not None:
        count = min(int(max_samples), len(table.image_paths))
        table = AnnotationTable(
            table.image_paths[:count], table.labels[:count], table.attribute_names,
        )
        queries, ids = np.unique(table.labels, axis=0, return_inverse=True)
        query_names = table.attribute_names
    else:
        queries, ids, query_names = load_retrieval_annotations(gt_path.parent, table)
    return table, queries, ids, query_names


def _positive_state_indices(mapper: CategoryPromptMapper) -> list[int]:
    state_index = {key: index for index, key in enumerate(mapper.keys)}
    indices: list[int] = []
    for _, columns, state_keys, _ in mapper._MULTI_GROUPS:
        indices.extend(state_index[key] for key in state_keys)
        if len(indices) > 40:
            break
    # The public evaluator's mapping is clearer and avoids relying on the
    # ordering of the 52 states for the 40 official attributes.
    positive: dict[str, str] = {}
    for _, columns, state_keys, _ in mapper._MULTI_GROUPS:
        positive.update(zip(columns, state_keys))
    for column, positive_key, _ in mapper._BINARY_GROUPS:
        positive[column] = positive_key
    return [state_index[positive[name]] for name in mapper.attribute_names]


def _sampled_hamming_ranking(
    distances: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray,
    max_pairs_per_query: int, seed: int,
) -> dict[str, float]:
    """Estimate pairwise ranking accuracy grouped by lower Hamming distance."""
    rng = np.random.default_rng(seed)
    correct = np.zeros(41, dtype=np.float64)
    total = np.zeros(41, dtype=np.float64)
    for query_index, row in enumerate(distances):
        hamming = np.count_nonzero(
            gallery_labels != query_labels[query_index][None, :], axis=1,
        )
        if len(row) < 2:
            continue
        pair_count = min(int(max_pairs_per_query), len(row) * 2)
        left = rng.integers(0, len(row), pair_count)
        right = rng.integers(0, len(row), pair_count)
        valid = hamming[left] != hamming[right]
        if not valid.any():
            continue
        left, right = left[valid], right[valid]
        lower = np.minimum(hamming[left], hamming[right])
        lower_is_left = hamming[left] < hamming[right]
        better_score = np.where(lower_is_left, row[left] < row[right], row[right] < row[left])
        for distance in np.unique(lower):
            mask = lower == distance
            total[distance] += float(mask.sum())
            correct[distance] += float(better_score[mask].sum())
    result = {
        f"hamming_pairwise_accuracy_d{distance}": float(correct[distance] / total[distance])
        if total[distance] else float("nan")
        for distance in range(41)
    }
    result["hamming_pairwise_accuracy"] = float(correct.sum() / max(total.sum(), 1.0))
    result["hamming_pair_count"] = float(total.sum())
    return result


@torch.inference_mode()
def evaluate_native52_score(
    model: Any, mapper: CategoryPromptMapper, score_module: Native52Score,
    data_root: Path, device: torch.device, config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the exact score used by the graded-ranking training loss."""
    table, queries_raw, ids, query_names = _annotation_views(
        data_root, "val", config.get("max_val_samples"),
    )
    queries = reorder_columns(queries_raw, query_names, mapper.attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, mapper.attribute_names)
    transform = build_eval_transform(config["image_size"], config.get("augmentation", "resize_pad_crop"))
    gallery = encode_gallery(
        model, table.image_paths, [data_root, data_root / "annotations", REPOSITORY_ROOT],
        transform, device, config["eval_batch_size"], config["num_workers"], config["amp"],
    )
    model.eval()
    with autocast(device, config["amp"]):
        state_features = model.encode_text(model.tokenize(mapper.prompts).to(device)).float()
    query_semantic = mapper.encode(torch.from_numpy(queries).float()).to(device)
    distances = np.empty((len(queries), len(gallery)), dtype=np.float32)
    predictions = np.empty((len(gallery), 40), dtype=np.float32)
    positive_indices = _positive_state_indices(mapper)
    score_module = score_module.to(device)
    score_module.eval()
    chunk_size = int(config.get("score_gallery_chunk", 1024))
    for start in range(0, len(gallery), chunk_size):
        stop = min(start + chunk_size, len(gallery))
        candidate = gallery[start:stop].to(device, non_blocking=True)
        raw_logits = candidate.float() @ state_features.T
        scores = score_module.score_from_logits(query_semantic, raw_logits)
        probabilities = score_module.probabilities_from_logits(raw_logits)
        distances[:, start:stop] = (-scores).cpu().numpy().astype(np.float32, copy=False)
        predictions[start:stop] = probabilities[:, positive_indices].cpu().numpy().astype(
            np.float32, copy=False,
        )
    official = official_retrieval_metrics(distances, queries, labels, ids)
    per_attribute = [
        _binary_metric(labels[:, index], predictions[:, index])
        for index in range(labels.shape[1])
    ]
    result: dict[str, Any] = {
        "images": float(len(table.image_paths)),
        "queries": float(len(queries)),
        "rank1": float(official["Rank-1"]),
        "rank5": float(official["Rank-5"]),
        "rank10": float(official["Rank-10"]),
        "map": float(official["mAP"]),
        "mADM": float(official["mADM"]),
        "mINP": float(official["mINP"]),
        "macro_auroc": float(np.nanmean([item["auroc"] for item in per_attribute])),
        "macro_ap": float(np.nanmean([item["ap"] for item in per_attribute])),
        "macro_f1": float(np.nanmean([item["f1"] for item in per_attribute])),
        "native52_score_mean": float((-distances).mean()),
        "native52_score_std": float((-distances).std()),
        "native52_temperature_mean": float(score_module.temperatures().mean().cpu()),
        "native52_category_weight_min": float(score_module.weights().min().cpu()),
        "native52_category_weight_max": float(score_module.weights().max().cpu()),
    }
    result.update(_sampled_hamming_ranking(
        distances, queries, labels, int(config["hamming_pairs_per_query"]), int(config["seed"]),
    ))
    # Keep the matrices only when requested by a smoke/debug run. Full
    # matrices are intentionally not serialized with every epoch.
    if config.get("return_eval_arrays", False):
        result["_distances"] = distances
        result["_queries"] = queries
        result["_labels"] = labels
    return result


@torch.no_grad()
def _teacher_state_features(model: Any, mapper: CategoryPromptMapper,
                            state_tokens: torch.Tensor, device: torch.device,
                            amp: bool) -> torch.Tensor:
    model.eval()
    with autocast(device, amp):
        return model.encode_text(state_tokens).float()


def _train_g0_epoch(
    model: Any, loader: DataLoader, score_module: Native52Score,
    state_features: torch.Tensor, mapper: CategoryPromptMapper,
    optimizer: torch.optim.Optimizer, scheduler: Any, device: torch.device,
    config: dict[str, Any], scaler: Any,
) -> dict[str, float]:
    model.eval()
    score_module.train()
    total_loss = 0.0
    samples = 0
    batches = 0
    max_grad = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        semantic = mapper.encode(labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast(device, config["amp"]):
            features = model.encode_image(images).float()
            raw_logits = features @ state_features.T
        score = score_module.score_from_logits(semantic, raw_logits)
        loss = graded_listwise_loss(
            score, labels, labels, config["target_temperature"], config["score_temperature"],
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(score_module.parameters(), config["grad_clip"])
        if not torch.isfinite(grad):
            raise RuntimeError(f"G0 calibrator gradient is non-finite: {float(grad)}")
        max_grad = max(max_grad, float(grad))
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        total_loss += float(loss.detach()) * count
        samples += count
        batches += 1
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("G0 encoder gradient detected despite full freeze")
    return {
        "total_loss": total_loss / max(samples, 1),
        "graded_loss": total_loss / max(samples, 1),
        "fce_loss": 0.0,
        "distill_loss": 0.0,
        "train_batches": float(batches),
        "average_batch_size": samples / max(batches, 1),
        "encoder_grad_max": 0.0,
        "calibrator_grad_norm": max_grad,
    }


def _train_finetune_epoch(
    model: Any, loader: DataLoader, score_module: Native52Score,
    state_tokens: torch.Tensor, mapper: CategoryPromptMapper,
    optimizer: torch.optim.Optimizer, scheduler: Any, criterion: FocalCLIPLoss,
    teacher: Any | None, teacher_state_features: torch.Tensor | None,
    device: torch.device, config: dict[str, Any], scaler: Any,
    lambda_graded: float, lambda_distill: float,
) -> dict[str, float]:
    model.train()
    score_module.eval()
    totals = {"total_loss": 0.0, "graded_loss": 0.0, "fce_loss": 0.0,
              "distill_loss": 0.0, "samples": 0.0, "batches": 0.0}
    first_score_grad = 0.0
    encoder_grad_max = 0.0
    for batch in loader:
        if not isinstance(batch, PromptBatch):
            raise TypeError("A7 PromptCollator must return PromptBatch")
        images = batch.images.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        semantic = batch.semantic_labels.to(device, non_blocking=True)
        tokens = batch.tokens.to(device, non_blocking=True)
        selected = batch.selected_semantics.to(device, non_blocking=True)
        owners = batch.text_owners.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, config["amp"]):
            image_features = model.encode_image(images)
            text_features = model.encode_text(tokens)
            scale = model.logit_scale.exp().clamp(max=100.0)
            sampled_logits = scale * image_features @ text_features.T
            fce_output = criterion(sampled_logits, semantic, selected, owners)
            state_features = model.encode_text(state_tokens)
            raw_logits = image_features.float() @ state_features.float().T
            raw_logits.retain_grad()
            score = score_module.score_from_logits(semantic, raw_logits)
            graded = graded_listwise_loss(
                score, labels, labels, config["target_temperature"], config["score_temperature"],
            )
            if teacher is not None:
                with torch.no_grad(), autocast(device, config["amp"]):
                    teacher_features = teacher.encode_image(images)
                    teacher_logits = teacher_features.float() @ teacher_state_features.T
                distill = F.mse_loss(raw_logits, teacher_logits)
            else:
                distill = raw_logits.new_zeros(())
            total = fce_output.loss + lambda_graded * graded + lambda_distill * distill
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        if raw_logits.grad is not None:
            first_score_grad = max(first_score_grad, float(raw_logits.grad.detach().abs().max()))
        for parameter in model.parameters():
            if parameter.grad is not None:
                encoder_grad_max = max(encoder_grad_max, float(parameter.grad.detach().abs().max()))
        if not math.isfinite(first_score_grad) or not math.isfinite(encoder_grad_max):
            raise RuntimeError(
                "Native52 graded path produced a non-finite gradient: "
                f"score={first_score_grad}, encoder={encoder_grad_max}"
            )
        nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        totals["total_loss"] += float(total.detach()) * count
        totals["graded_loss"] += float(graded.detach()) * count
        totals["fce_loss"] += float(fce_output.loss.detach()) * count
        totals["distill_loss"] += float(distill.detach()) * count
        totals["samples"] += count
        totals["batches"] += 1
    samples = max(totals.pop("samples"), 1.0)
    batches = max(totals.pop("batches"), 1.0)
    result = {key: value / samples for key, value in totals.items()}
    result["train_batches"] = batches
    result["average_batch_size"] = samples / batches
    result["native_score_grad_max"] = first_score_grad
    result["encoder_grad_max"] = encoder_grad_max
    return result


def _save_experiment_checkpoint(
    path: Path, model: Any, mapper: CategoryPromptMapper, epoch: int,
    metrics: dict[str, Any], score_module: Native52Score, config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None, scheduler: Any | None = None,
    scaler: Any | None = None,
) -> None:
    payload = model_payload(
        model, mapper.attribute_names, epoch, metrics, "category_complete",
        {"training_objective": config["experiment"], **config},
    )
    payload["native52_score_state_dict"] = {
        key: value.detach().cpu() for key, value in score_module.state_dict().items()
    }
    payload["native52_score_config"] = {
        "temperature": config["category_temperature"],
        "groups": score_module.groups,
        "learnable": score_module.learnable,
    }
    if optimizer is not None and scheduler is not None and scaler is not None:
        payload["training_state"] = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
    _atomic_torch_save(payload, path)


def _build_train_loaders(config: dict[str, Any], model: Any, table: Any,
                         train_roots: list[Path]) -> tuple[DataLoader, DataLoader]:
    dataset = AttriVisionDataset(
        table, train_roots,
        build_train_transform(config["image_size"], config["rotation"], "resize_pad_crop"),
        config.get("max_train_samples"),
    )
    plain_loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        drop_last=len(dataset) >= config["batch_size"],
        num_workers=config["num_workers"], pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
    )
    collator = PromptCollator(
        table.attribute_names, model.tokenizer, text_sampling="single",
        multi_attributes=1, prompt_mode="category_complete", stochastic=True,
    )
    fce_loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"] + 1),
        drop_last=len(dataset) >= config["batch_size"],
        num_workers=config["num_workers"], pin_memory=True,
        persistent_workers=config["num_workers"] > 0, collate_fn=collator,
    )
    return plain_loader, fce_loader


def _resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    for key in (
        "checkpoint", "data_root", "output_root", "device", "epochs", "batch_size",
        "eval_batch_size", "num_workers", "max_train_samples", "max_val_samples",
        "seed", "experiment", "lambda_graded", "lambda_distill",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.experiment == "G1" and args.lambda_graded is None:
        raise ValueError("G1 requires --lambda-graded 0.01, 0.05, or 0.1")
    if args.experiment is not None:
        config["experiment"] = args.experiment
    config["experiment"] = str(config.get("experiment", "G1")).upper()
    if config["experiment"] not in {"G0", "G1", "G2"}:
        raise ValueError("experiment must be G0, G1, or G2")
    config["checkpoint"] = str(Path(config["checkpoint"]).resolve())
    config["data_root"] = str(Path(config["data_root"]).resolve())
    config["output_root"] = str(Path(config["output_root"]).resolve())
    config["device"] = str(config.get("device", "auto"))
    config["amp"] = bool(config.get("amp", True))
    config["lambda_graded"] = float(config.get("lambda_graded", 0.05))
    config["lambda_distill"] = float(config.get("lambda_distill", 0.1))
    if config["experiment"] == "G0":
        config["lambda_graded"] = 1.0
        config["lambda_distill"] = 0.0
    if config["experiment"] == "G1":
        config["lambda_distill"] = 0.0
    if config["lambda_graded"] < 0 or config["lambda_distill"] < 0:
        raise ValueError("loss weights cannot be negative")
    return config


def _output_dir(config: dict[str, Any]) -> Path:
    experiment = config["experiment"]
    if experiment == "G1":
        name = f"G1_lambda_{config['lambda_graded']:g}"
    elif experiment == "G2":
        name = f"G2_lambda_{config['lambda_graded']:g}_distill_{config['lambda_distill']:g}"
    else:
        name = "G0_frozen_calibration"
    return Path(config["output_root"]) / name


def _write_comparison_index(root: Path, baseline: dict[str, Any]) -> None:
    """Collect A7 and every completed graded run under one output root."""
    rows: list[dict[str, Any]] = [{
        "experiment": "A7",
        "checkpoint": "outputs/attrivision_ablation_mADM/A7/checkpoint_best.pth",
        "best_epoch": 8,
        "rank1": baseline.get("rank1"),
        "map": baseline.get("map"),
        "mADM": baseline.get("mADM"),
        "macro_auroc": baseline.get("macro_auroc"),
        "macro_ap": baseline.get("macro_ap"),
        "macro_f1": baseline.get("macro_f1"),
        "hamming_pairwise_accuracy": "",
        "delta_mADM": 0.0,
    }]
    for result_path in sorted(root.glob("*_result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            metrics = result.get("best_metrics", {})
            experiment_name = str(result.get("experiment", result_path.stem))
            if experiment_name == "G1":
                experiment_name = f"G1_lambda_{metrics.get('lambda_graded', '?')}"
            elif experiment_name == "G2":
                experiment_name = (
                    f"G2_lambda_{metrics.get('lambda_graded', '?')}"
                    f"_distill_{metrics.get('lambda_distill', '?')}"
                )
            rows.append({
                "experiment": experiment_name,
                "checkpoint": result.get("checkpoint", ""),
                "best_epoch": result.get("best_epoch", 0),
                "rank1": metrics.get("rank1"), "map": metrics.get("map"),
                "mADM": result.get("best_mADM"),
                "last_mADM": result.get("last_metrics", {}).get("mADM"),
                "last_rank1": result.get("last_metrics", {}).get("rank1"),
                "last_map": result.get("last_metrics", {}).get("map"),
                "macro_auroc": metrics.get("macro_auroc"),
                "macro_ap": metrics.get("macro_ap"),
                "macro_f1": metrics.get("macro_f1"),
                "hamming_pairwise_accuracy": metrics.get("hamming_pairwise_accuracy"),
                "delta_mADM": result.get("delta_mADM"),
            })
        except (OSError, json.JSONDecodeError):
            continue
    _write_csv(root / "comparison.csv", rows)


def _load_or_evaluate_baseline(
    config: dict[str, Any], mapper: CategoryPromptMapper, device: torch.device,
    train_attribute_names: Sequence[str],
) -> dict[str, Any]:
    root = Path(config["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "baseline_native52.json"
    expected = {
        "checkpoint": config["checkpoint"],
        "max_val_samples": config.get("max_val_samples"),
    }
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("_cache_key") == expected:
                return cached["metrics"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    model, _ = load_model(config["checkpoint"], device)
    score = Native52Score(mapper.category_indices(), config["category_temperature"], False).to(device)
    metrics = evaluate_native52_score(
        model, mapper, score, Path(config["data_root"]), device, config,
    )
    _write_json(path, {"_cache_key": expected, "metrics": metrics})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def _run_training(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(config["seed"], config.get("deterministic", False))
    device = choose_device(config["device"])
    data_root = Path(config["data_root"])
    train_table = read_gt_csv(find_annotation_file(data_root, "train"))
    base_model, base_payload = load_model(config["checkpoint"], device)
    mapper = CategoryPromptMapper(train_table.attribute_names)
    baseline = _load_or_evaluate_baseline(config, mapper, device, train_table.attribute_names)
    output_dir = _output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {**config, "output_dir": str(output_dir)}
    _write_json(output_dir / "config.json", run_config)
    print(
        f"A7 Native52 graded ranking: {config['experiment']}\n"
        f"Checkpoint: {config['checkpoint']}\nDevice: {device}\n"
        f"Baseline Native52: Rank-1={100 * baseline['rank1']:.2f}%, "
        f"mAP={100 * baseline['map']:.2f}%, mADM={100 * baseline['mADM']:.2f}%",
        flush=True,
    )
    if config["experiment"] == "G0":
        model = base_model
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        plain_loader, _ = _build_train_loaders(config, model, train_table, [data_root, REPOSITORY_ROOT])
    else:
        model = base_model
        _, fce_loader = _build_train_loaders(config, model, train_table, [data_root, REPOSITORY_ROOT])
    score_module = Native52Score(
        mapper.category_indices(), config["category_temperature"], config["experiment"] == "G0",
    ).to(device)
    state_tokens = model.tokenize(mapper.prompts).to(device)
    criterion = FocalCLIPLoss(
        "focal_clip", "multi_positive", config["focal_alpha"], config["focal_gamma"],
    )
    teacher = None
    teacher_state_features = None
    if config["experiment"] == "G2":
        teacher, _ = load_model(config["checkpoint"], device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher_state_features = _teacher_state_features(
            teacher, mapper, state_tokens, device, config["amp"],
        )
    if config["experiment"] == "G0":
        train_parameters = list(score_module.parameters())
        optimizer = torch.optim.AdamW(
            train_parameters, lr=config["calibration_learning_rate"],
            weight_decay=config["calibration_weight_decay"],
        )
        learning_rate = config["calibration_learning_rate"]
        loader = plain_loader
    else:
        optimizer = torch.optim.AdamW(
            _optimizer_groups(model, config["weight_decay"], config["learning_rate"]),
        )
        learning_rate = config["learning_rate"]
        loader = fce_loader
    scheduler = build_scheduler(
        optimizer, len(loader), config["epochs"], config["warmup_epochs"],
        learning_rate, config["min_learning_rate"],
    )
    scaler = _grad_scaler(config["amp"] and device.type == "cuda")
    best_path = output_dir / "checkpoint_best.pth"
    last_path = output_dir / "checkpoint_last.pth"
    best_score = float(baseline["mADM"])
    best_epoch = 0
    best_metrics: dict[str, Any] = {**baseline, "epoch": 0}
    _save_experiment_checkpoint(
        best_path, model, mapper, 0, best_metrics, score_module, run_config,
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        started = time.perf_counter()
        if config["experiment"] == "G0":
            losses = _train_g0_epoch(
                model, loader, score_module, _teacher_state_features(
                    model, mapper, state_tokens, device, config["amp"],
                ), mapper, optimizer, scheduler, device, config, scaler,
            )
        else:
            losses = _train_finetune_epoch(
                model, loader, score_module, state_tokens, mapper, optimizer, scheduler,
                criterion, teacher, teacher_state_features, device, config, scaler,
                config["lambda_graded"], config["lambda_distill"],
            )
        metrics = evaluate_native52_score(
            model, mapper, score_module, data_root, device, config,
        )
        row = {
            "epoch": epoch, "experiment": config["experiment"],
            "lambda_graded": config["lambda_graded"],
            "lambda_distill": config["lambda_distill"], **losses, **metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        _write_csv(output_dir / "metrics.csv", history)
        with (output_dir / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=float) + "\n")
        if float(metrics["mADM"]) > best_score:
            best_score = float(metrics["mADM"])
            best_epoch = epoch
            best_metrics = dict(row)
            _save_experiment_checkpoint(
                best_path, model, mapper, epoch, row, score_module, run_config,
            )
        _save_experiment_checkpoint(
            last_path, model, mapper, epoch, row, score_module, run_config,
            optimizer, scheduler, scaler,
        )
        print(
            f"{config['experiment']} epoch {epoch:03d}/{config['epochs']}: "
            f"total={losses['total_loss']:.6f}, graded={losses['graded_loss']:.6f}, "
            f"fce={losses['fce_loss']:.6f}, distill={losses['distill_loss']:.6f}, "
            f"Rank-1={100 * metrics['rank1']:.2f}%, mAP={100 * metrics['map']:.2f}%, "
            f"mADM={100 * metrics['mADM']:.2f}%, "
            f"HammingPair={100 * metrics['hamming_pairwise_accuracy']:.2f}%",
            flush=True,
        )
        if config["experiment"] == "G0" and any(
            parameter.grad is not None for parameter in model.parameters()
        ):
            raise RuntimeError("G0 encoder gradient detected after optimizer step")
    result = {
        "experiment": config["experiment"],
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_mADM": best_score,
        "best_metrics": best_metrics,
        "last_metrics": history[-1] if history else best_metrics,
        "baseline_metrics": baseline,
        "delta_mADM": best_score - float(baseline["mADM"]),
    }
    _write_json(output_dir / "validation_metrics.json", result)
    result_name = f"{output_dir.name}_result.json"
    _write_json(Path(config["output_root"]) / result_name, result)
    _write_comparison_index(Path(config["output_root"]), baseline)
    del model, base_model, teacher, optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _smoke() -> None:
    torch.manual_seed(7)
    mapper = CategoryPromptMapper([
        "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
        "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
        "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue",
        "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
        "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple",
        "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
        "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black",
        "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
        "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
        "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
        "LowerBody-Color-Yellow", "LowerBody-Color-Other", "LowerBody-Type-Trousers&Shorts",
        "LowerBody-Type-Skirt&Dress", "Accessory-Backpack", "Accessory-Bag",
        "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat",
    ])
    score_module = Native52Score(mapper.category_indices(), 0.01, True)
    labels = torch.randint(0, 2, (8, 40)).float()
    semantic = mapper.encode(labels)
    image = F.normalize(torch.randn(8, 16), dim=-1)
    states = F.normalize(torch.randn(52, 16), dim=-1)
    logits = image @ states.T
    score = score_module.score_from_logits(semantic, logits)
    loss = graded_listwise_loss(score, labels, labels, 2.0, 0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert score_module.log_temperature.grad is not None
    assert score_module.log_weight.grad is not None
    print(
        f"Native52 graded smoke passed: loss={float(loss):.6f}, "
        f"score_shape={tuple(score.shape)}, calibration_grad=True",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone A7 Native52 graded-ranking experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=("train", "smoke"), default="train")
    parser.add_argument("--experiment", choices=("G0", "G1", "G2"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lambda-graded", type=float)
    parser.add_argument("--lambda-distill", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "smoke":
        _smoke()
        return
    config = _resolved_config(args)
    if config["epochs"] <= 0 or config["batch_size"] <= 0 or config["eval_batch_size"] <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    _run_training(config)


if __name__ == "__main__":
    main()
