"""A7 graded-ranking ablation with the Native52 inference score.

The existing A7 code and checkpoint are intentionally left untouched.  This
runner defines the requested comparison as:

* G0: the existing A7 FCE checkpoint, used as the baseline reference;
* G1: frozen A7 plus learnable Native52 score calibration;
* G2: A7 FCE plus graded pairwise ranking fine-tuning;
* G3: A7 FCE plus graded listwise KL fine-tuning.

The ranking score is the same score used at inference: category-local
52-state softmax followed by category NLL.  The ranking target follows the
official mADM agreement construction: exact attribute equality is relevance,
and lower agreement receives lower normalized relevance.  No joint-query
embedding is constructed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    AttriVisionDataset,
    PromptBatch,
    PromptCollator,
)
from attrivision.engine.evaluator_abpr import (  # noqa: E402
    _binary_metric,
    encode_gallery,
)
from attrivision.engine.trainer_attrivision import (  # noqa: E402
    _grad_scaler,
    _optimizer_groups,
    _optimizer_step,
    build_scheduler,
    evaluate_paper_fce,
)
from attrivision.losses.focal_clip_loss import FocalCLIPLoss  # noqa: E402
from attrivision.transforms import build_eval_transform, build_train_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import AnnotationTable, find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import (  # noqa: E402
    autocast,
    load_retrieval_annotations,
    official_retrieval_metrics,
    reorder_columns,
)


DEFAULT_CHECKPOINT = Path("outputs/attrivision_ablation_mADM/A7/checkpoint_best.pth")
DEFAULT_OUTPUT_ROOT = Path("outputs/attrivision_graded_ranking_15ep")
LAMBDA_VALUES = (0.01, 0.05, 0.1)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=float, allow_nan=True)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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


class Native52Score(nn.Module):
    """Differentiable version of the deployed Native52 Category-NLL score."""

    def __init__(self, groups: Sequence[Sequence[int]], temperature: float,
                 learnable: bool = False) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("Native52 temperature must be positive")
        self.groups = [tuple(int(index) for index in group) for group in groups]
        self.learnable = bool(learnable)
        self.log_temperature = nn.Parameter(
            torch.full((len(self.groups),), math.log(float(temperature))),
            requires_grad=self.learnable,
        )
        self.log_weight = nn.Parameter(
            torch.zeros(len(self.groups)), requires_grad=self.learnable,
        )

    def temperatures(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(1.0e-4, 10.0)

    def weights(self) -> torch.Tensor:
        if not self.learnable:
            return torch.ones_like(self.log_weight)
        # Mean weight is one, so calibration does not silently rescale the
        # Native52 score by the number of categories.
        return F.softmax(self.log_weight, dim=0) * len(self.groups)

    def score_from_logits(self, query_semantic: torch.Tensor,
                          candidate_logits: torch.Tensor) -> torch.Tensor:
        """Return higher-is-better scores with shape [queries, candidates]."""
        if query_semantic.ndim != 2 or candidate_logits.ndim != 2:
            raise ValueError("Native52 inputs must be matrices")
        if query_semantic.shape[1] != candidate_logits.shape[1]:
            raise ValueError("Native52 query/candidate state widths differ")
        targets = query_semantic.float()
        logits = candidate_logits.float()
        category_scores: list[torch.Tensor] = []
        temperatures = self.temperatures()
        for category_index, indices in enumerate(self.groups):
            index = torch.as_tensor(indices, device=logits.device)
            category_target = targets[:, index]
            counts = category_target.sum(dim=1, keepdim=True)
            if (counts <= 0).any():
                raise ValueError("Every query must activate a Native52 category state")
            category_target = category_target / counts
            log_probability = F.log_softmax(
                logits[:, index] / temperatures[category_index], dim=1,
            )
            # [Q,S] @ [G,S].T = expected log probability for this category.
            category_scores.append(category_target @ log_probability.T)
        stacked = torch.stack(category_scores, dim=2)
        return (stacked * self.weights()).mean(dim=2)

    def probabilities_from_logits(self, candidate_logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.zeros_like(candidate_logits.float())
        temperatures = self.temperatures()
        for category_index, indices in enumerate(self.groups):
            index = torch.as_tensor(indices, device=candidate_logits.device)
            probabilities[:, index] = F.softmax(
                candidate_logits[:, index].float() / temperatures[category_index], dim=1,
            )
        return probabilities


def official_normalized_relevance(query_labels: torch.Tensor,
                                  candidate_labels: torch.Tensor) -> torch.Tensor:
    """Return the official mADM normalized attribute agreement grades.

    ``official_retrieval_metrics`` defines agreement as one minus normalized
    attribute Hamming distance, exact equality as relevant, and accumulates
    normalized agreement at the ranks of those exact matches.  This function
    uses the same normalization as the official ADM calculation for training
    targets.  The candidate set is the current batch during training.
    """
    if query_labels.ndim != 2 or candidate_labels.ndim != 2:
        raise ValueError("labels must be [N,A] matrices")
    if query_labels.shape[1] != candidate_labels.shape[1]:
        raise ValueError("query/candidate attribute widths differ")
    agreement = 1.0 - torch.abs(
        query_labels.float()[:, None, :] - candidate_labels.float()[None, :, :],
    ).sum(dim=2) / float(query_labels.shape[1])
    mean = agreement.mean(dim=1, keepdim=True)
    maximum = agreement.max(dim=1, keepdim=True).values
    span = maximum - mean
    normalized = torch.where(
        span > 1.0e-12,
        (agreement - mean) / span,
        torch.ones_like(agreement),
    )
    return normalized.clamp_min(0.0)


def graded_pairwise_loss(scores: torch.Tensor, query_labels: torch.Tensor,
                         candidate_labels: torch.Tensor,
                         pair_temperature: float = 0.1) -> tuple[torch.Tensor, int]:
    """Graded Bradley-Terry loss, weighted by the official grade gap."""
    if pair_temperature <= 0:
        raise ValueError("pair temperature must be positive")
    grades = official_normalized_relevance(query_labels, candidate_labels).detach()
    if scores.shape != grades.shape:
        raise ValueError(f"scores {tuple(scores.shape)} do not match grades {tuple(grades.shape)}")
    score_gap = scores[:, :, None] - scores[:, None, :]
    grade_gap = grades[:, :, None] - grades[:, None, :]
    mask = grade_gap > 1.0e-8
    pair_count = int(mask.sum().detach().cpu())
    if pair_count == 0:
        return scores.sum() * 0.0, 0
    weighted = F.softplus(-score_gap / pair_temperature) * grade_gap
    return weighted[mask].mean(), pair_count


def graded_listwise_kl(scores: torch.Tensor, query_labels: torch.Tensor,
                       candidate_labels: torch.Tensor,
                       target_temperature: float = 0.1,
                       score_temperature: float = 0.1) -> tuple[torch.Tensor, int]:
    """Listwise KL from official mADM relevance grades to Native52 scores."""
    if target_temperature <= 0 or score_temperature <= 0:
        raise ValueError("listwise temperatures must be positive")
    grades = official_normalized_relevance(query_labels, candidate_labels).detach()
    normalizer = grades.sum(dim=1, keepdim=True)
    target = grades / normalizer.clamp_min(1.0e-12)
    log_probability = F.log_softmax(scores / score_temperature, dim=1)
    loss = F.kl_div(log_probability, target, reduction="batchmean", log_target=False)
    return loss, int((grades > 1.0e-8).sum().detach().cpu())


def _annotation_views(data_root: Path, max_samples: int | None) -> tuple[Any, np.ndarray, np.ndarray, list[str]]:
    gt_path = find_annotation_file(data_root, "val")
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
    positive: dict[str, str] = {}
    for _, columns, state_keys, _ in mapper._MULTI_GROUPS:
        positive.update(zip(columns, state_keys))
    for column, positive_key, _ in mapper._BINARY_GROUPS:
        positive[column] = positive_key
    return [state_index[positive[name]] for name in mapper.attribute_names]


def _sampled_distance_ranking(
    distances: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray,
    max_pairs_per_query: int, seed: int,
) -> dict[str, float]:
    """Measure lower attribute Hamming distance beating a larger one."""
    rng = np.random.default_rng(seed)
    correct = np.zeros(41, dtype=np.float64)
    total = np.zeros(41, dtype=np.float64)
    for query_index, row in enumerate(distances):
        hamming = np.count_nonzero(
            gallery_labels != query_labels[query_index][None, :], axis=1,
        )
        pair_count = min(int(max_pairs_per_query), max(len(row) * 2, 2))
        left = rng.integers(0, len(row), pair_count)
        right = rng.integers(0, len(row), pair_count)
        valid = hamming[left] != hamming[right]
        if not valid.any():
            continue
        left, right = left[valid], right[valid]
        lower = np.minimum(hamming[left], hamming[right])
        lower_left = hamming[left] < hamming[right]
        better = np.where(lower_left, row[left] < row[right], row[right] < row[left])
        for distance in np.unique(lower):
            mask = lower == distance
            total[distance] += float(mask.sum())
            correct[distance] += float(better[mask].sum())
    result = {
        f"attribute_distance_accuracy_d{distance}": (
            float(correct[distance] / total[distance]) if total[distance] else float("nan")
        )
        for distance in range(41)
    }
    result.update({
        "attribute_distance_accuracy": float(correct.sum() / max(total.sum(), 1.0)),
        "attribute_distance_pair_count": float(total.sum()),
    })
    return result


@torch.inference_mode()
def evaluate_native52_score(
    model: Any, mapper: CategoryPromptMapper, score_module: Native52Score,
    data_root: Path, device: torch.device, config: dict[str, Any],
) -> dict[str, Any]:
    table, queries_raw, ids, query_names = _annotation_views(
        data_root, config.get("max_val_samples"),
    )
    queries = reorder_columns(queries_raw, query_names, mapper.attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, mapper.attribute_names)
    gt_path = find_annotation_file(data_root, "val")
    transform = build_eval_transform(config["image_size"], config["augmentation"])
    gallery = encode_gallery(
        model, table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
        transform, device, config["eval_batch_size"], config["num_workers"], config["amp"],
    )
    with autocast(device, config["amp"]):
        state_features = model.encode_text(model.tokenize(mapper.prompts).to(device)).float()
    query_semantic = mapper.encode(torch.from_numpy(queries).float()).to(device)
    score_module = score_module.to(device)
    score_module.eval()
    distances = np.empty((len(queries), len(gallery)), dtype=np.float32)
    predictions = np.empty((len(gallery), len(mapper.attribute_names)), dtype=np.float32)
    positive_indices = _positive_state_indices(mapper)
    chunk_size = int(config["score_gallery_chunk"])
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
        "retrieval_scoring": "native52_category_nll",
        "native52_temperature_mean": float(score_module.temperatures().mean().cpu()),
        "native52_category_weight_min": float(score_module.weights().min().cpu()),
        "native52_category_weight_max": float(score_module.weights().max().cpu()),
    }
    result.update(_sampled_distance_ranking(
        distances, queries, labels, int(config["ranking_pairs_per_query"]), int(config["seed"]),
    ))
    return result


def _make_loaders(config: dict[str, Any], model: Any, table: Any,
                  data_root: Path) -> tuple[DataLoader, DataLoader, DataLoader]:
    gt_path = find_annotation_file(data_root, "train")
    dataset = AttriVisionDataset(
        table, [data_root, gt_path.parent, REPOSITORY_ROOT],
        build_train_transform(config["image_size"], config["rotation"], config["augmentation"]),
        config.get("max_train_samples"),
    )
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        drop_last=len(dataset) >= config["batch_size"],
        num_workers=config["num_workers"], pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
    )
    collator = PromptCollator(
        table.attribute_names, model.tokenizer, text_sampling="single",
        multi_attributes=3, prompt_mode="category_complete", stochastic=True,
    )
    fce_loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"] + 1),
        drop_last=len(dataset) >= config["batch_size"],
        num_workers=config["num_workers"], pin_memory=True,
        persistent_workers=config["num_workers"] > 0, collate_fn=collator,
    )
    val_gt = find_annotation_file(data_root, "val")
    val_table = read_gt_csv(val_gt)
    val_dataset = AttriVisionDataset(
        val_table, [data_root, val_gt.parent, REPOSITORY_ROOT],
        build_eval_transform(config["image_size"], config["augmentation"]),
        config.get("max_val_samples"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["eval_batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
        collate_fn=PromptCollator(
            val_table.attribute_names, model.tokenizer, text_sampling="single",
            multi_attributes=3, prompt_mode="category_complete", stochastic=False,
        ),
    )
    return loader, fce_loader, val_loader


def _train_g1_epoch(
    model: Any, loader: DataLoader, score_module: Native52Score,
    state_features: torch.Tensor, mapper: CategoryPromptMapper,
    optimizer: torch.optim.Optimizer, scheduler: Any, device: torch.device,
    config: dict[str, Any], scaler: Any,
) -> dict[str, float]:
    model.eval()
    score_module.train()
    totals = {"total_loss": 0.0, "graded_loss": 0.0, "pairs": 0.0,
              "samples": 0.0, "batches": 0.0, "zero_pair_batches": 0.0}
    max_calibrator_grad = 0.0
    first_audit: dict[str, Any] = {}
    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        semantic = mapper.encode(labels)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast(device, config["amp"]):
            features = model.encode_image(images).float()
        raw_logits = features @ state_features.T
        scores = score_module.score_from_logits(semantic, raw_logits)
        graded, pair_count = graded_pairwise_loss(
            scores, labels, labels, config["pair_temperature"],
        )
        total = config["lambda_graded"] * graded
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(score_module.parameters(), config["grad_clip"])
        if not torch.isfinite(grad_norm):
            raise RuntimeError("G1 calibrator gradient is non-finite")
        max_calibrator_grad = max(
            max_calibrator_grad,
            max(float(parameter.grad.detach().abs().max()) for parameter in score_module.parameters()
                if parameter.grad is not None),
        )
        if batch_index == 0:
            if pair_count <= 0 or not torch.isfinite(graded):
                raise RuntimeError("G1 first batch has no valid graded positive/negative pairs")
            first_audit = {
                "pair_count_first_batch": pair_count,
                "graded_loss_first_batch": float(graded.detach()),
                "calibrator_grad_max_first_batch": max_calibrator_grad,
                "encoder_frozen": True,
                "native52_score_path": True,
            }
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        totals["total_loss"] += float(total.detach()) * count
        totals["graded_loss"] += float(graded.detach()) * count
        totals["pairs"] += pair_count
        totals["samples"] += count
        totals["batches"] += 1
        totals["zero_pair_batches"] += int(pair_count == 0)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("G1 encoder gradient detected despite full freeze")
    samples = max(totals.pop("samples"), 1.0)
    batches = max(totals.pop("batches"), 1.0)
    result = {key: value / samples for key, value in totals.items() if key not in {"pairs", "zero_pair_batches"}}
    result["pairs_per_sample"] = totals["pairs"] / samples
    result["zero_pair_batches"] = totals["zero_pair_batches"]
    result["train_batches"] = batches
    result["average_batch_size"] = samples / batches
    result["calibrator_grad_max"] = max_calibrator_grad
    result.update({f"audit_{key}": value for key, value in first_audit.items()})
    return result


def _train_finetune_epoch(
    model: Any, loader: DataLoader, score_module: Native52Score,
    state_tokens: torch.Tensor, mapper: CategoryPromptMapper,
    optimizer: torch.optim.Optimizer, scheduler: Any, criterion: FocalCLIPLoss,
    device: torch.device, config: dict[str, Any], scaler: Any,
    ranking_kind: str,
) -> dict[str, float]:
    model.train()
    score_module.eval()
    totals = {"total_loss": 0.0, "graded_loss": 0.0, "fce_loss": 0.0,
              "pairs": 0.0, "samples": 0.0, "batches": 0.0,
              "zero_pair_batches": 0.0}
    first_audit: dict[str, Any] = {}
    max_encoder_grad = 0.0
    max_score_grad = 0.0
    for batch_index, batch in enumerate(loader):
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
            image_features, text_features, logits = model(images, tokens)
            fce_output = criterion(logits, semantic, selected, owners)
            # The deployed Native52 temperature is 0.01.  Keeping this
            # branch in fp16 can overflow its ranking gradients under AMP,
            # even though the FCE branch remains numerically stable.  Casts
            # after an fp16 image forward are not sufficient: the backward
            # pass can still traverse fp16 activations.  Recompute the same
            # image embedding in FP32 for the ranking branch.  This is still
            # the deployed Native52 score, with no joint-query embedding.
            with autocast(device, False):
                ranking_image_features = model.encode_image(images).float()
                state_features = model.encode_text(state_tokens).float()
                raw_logits = ranking_image_features @ state_features.T
            raw_logits.retain_grad()
            scores = score_module.score_from_logits(semantic, raw_logits)
            if ranking_kind == "pairwise":
                graded, pair_count = graded_pairwise_loss(
                    scores, labels, labels, config["pair_temperature"],
                )
            elif ranking_kind == "listwise":
                graded, pair_count = graded_listwise_kl(
                    scores, labels, labels, config["target_temperature"],
                    config["score_temperature"],
                )
            else:
                raise ValueError(f"unknown ranking kind {ranking_kind}")
            total = fce_output.loss + config["lambda_graded"] * graded
        positive_count = int(criterion.positive_mask(
            logits, semantic, selected, owners,
        ).sum().detach().cpu())
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        score_grad = raw_logits.grad
        if score_grad is None or not torch.isfinite(score_grad).all():
            raise RuntimeError("Native52 score has no finite gradient path")
        max_score_grad = max(max_score_grad, float(score_grad.detach().abs().max()))
        if config["grad_clip"] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        for parameter in model.parameters():
            if parameter.grad is not None:
                value = float(parameter.grad.detach().abs().max())
                if not math.isfinite(value):
                    raise RuntimeError("A7 encoder gradient is non-finite")
                max_encoder_grad = max(max_encoder_grad, value)
        if batch_index == 0:
            if pair_count <= 0:
                raise RuntimeError("First batch has no valid graded positive/negative pairs")
            if positive_count <= 0:
                raise RuntimeError("FCE first batch has no positive target")
            first_audit = {
                "pair_count_first_batch": pair_count,
                "fce_positive_count_first_batch": positive_count,
                "native52_score_grad_max_first_batch": max_score_grad,
                "encoder_grad_max_first_batch": max_encoder_grad,
                "encoder_frozen": False,
                "native52_score_path": True,
            }
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        totals["total_loss"] += float(total.detach()) * count
        totals["graded_loss"] += float(graded.detach()) * count
        totals["fce_loss"] += float(fce_output.loss.detach()) * count
        totals["pairs"] += pair_count
        totals["samples"] += count
        totals["batches"] += 1
        totals["zero_pair_batches"] += int(pair_count == 0)
    samples = max(totals.pop("samples"), 1.0)
    batches = max(totals.pop("batches"), 1.0)
    result = {key: value / samples for key, value in totals.items()
              if key not in {"pairs", "zero_pair_batches"}}
    result["pairs_per_sample"] = totals["pairs"] / samples
    result["zero_pair_batches"] = totals["zero_pair_batches"]
    result["train_batches"] = batches
    result["average_batch_size"] = samples / batches
    result["native52_score_grad_max"] = max_score_grad
    result["encoder_grad_max"] = max_encoder_grad
    result.update({f"audit_{key}": value for key, value in first_audit.items()})
    return result


def _save_checkpoint(path: Path, model: Any, mapper: CategoryPromptMapper,
                     epoch: int, metrics: dict[str, Any], score_module: Native52Score,
                     config: dict[str, Any], optimizer: Any = None,
                     scheduler: Any = None, scaler: Any = None) -> None:
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
    if optimizer is not None:
        payload["training_state"] = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
    _atomic_torch_save(payload, path)


def _base_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "device": args.device,
        "experiment": args.experiment,
        "lambda_graded": float(args.lambda_graded),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "num_workers": int(args.num_workers),
        "image_size": 224,
        "rotation": 10.0,
        "augmentation": "resize_pad_crop",
        "learning_rate": 1.0e-5,
        "calibration_learning_rate": 1.0e-2,
        "weight_decay": 0.01,
        "calibration_weight_decay": 0.0,
        "warmup_epochs": 5,
        "min_learning_rate": 1.0e-7,
        "grad_clip": 1.0,
        "category_temperature": 0.01,
        "pair_temperature": 0.1,
        "target_temperature": 0.1,
        "score_temperature": 0.1,
        "score_gallery_chunk": 1024,
        "ranking_pairs_per_query": 512,
        "seed": int(args.seed),
        "amp": True,
        "deterministic": False,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "selection_metric": "mADM",
        "training_reference": "A7 FCE checkpoint, epoch 8, Native52 Category-NLL",
        "official_relevance": (
            "Exact 40-bit equality is relevant; target grade is official normalized "
            "mean attribute agreement used by mADM"
        ),
    }
    if config["experiment"] not in {"G1", "G2", "G3"}:
        raise ValueError("This runner executes G1, G2, or G3; G0 is the A7 reference")
    if config["lambda_graded"] not in LAMBDA_VALUES:
        raise ValueError(f"lambda-graded must be one of {LAMBDA_VALUES}")
    return config


def _run_baseline(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["output_root"])
    path = root / "baseline_native52.json"
    expected = {"checkpoint": config["checkpoint"], "max_val_samples": config["max_val_samples"]}
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == expected:
                return cached["metrics"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    device = choose_device(config["device"])
    data_root = Path(config["data_root"])
    table = read_gt_csv(find_annotation_file(data_root, "train"))
    model, _ = load_model(config["checkpoint"], device)
    mapper = CategoryPromptMapper(table.attribute_names)
    score = Native52Score(mapper.category_indices(), config["category_temperature"], False).to(device)
    metrics = evaluate_native52_score(model, mapper, score, data_root, device, config)
    _write_json(path, {"cache_key": expected, "metrics": metrics})
    return metrics


def run_experiment(args: argparse.Namespace) -> None:
    config = _base_config(args)
    set_seed(config["seed"], config["deterministic"])
    device = choose_device(config["device"])
    data_root = Path(config["data_root"])
    output_dir = Path(config["output_root"]) / f"{config['experiment']}_lambda_{config['lambda_graded']:g}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)
    _write_json(output_dir / "config.json", config)
    baseline = _run_baseline(config)
    train_table = read_gt_csv(find_annotation_file(data_root, "train"))
    model, _ = load_model(config["checkpoint"], device)
    mapper = CategoryPromptMapper(train_table.attribute_names)
    plain_loader, fce_loader, val_loader = _make_loaders(config, model, train_table, data_root)
    score_module = Native52Score(
        mapper.category_indices(), config["category_temperature"], config["experiment"] == "G1",
    ).to(device)
    state_tokens = model.tokenize(mapper.prompts).to(device)
    criterion = FocalCLIPLoss("focal_clip", "multi_positive", 1.0, 2.0)
    if config["experiment"] == "G1":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        train_parameters = list(score_module.parameters())
        optimizer = torch.optim.AdamW(
            train_parameters, lr=config["calibration_learning_rate"],
            weight_decay=config["calibration_weight_decay"],
        )
        loader = plain_loader
        learning_rate = config["calibration_learning_rate"]
    else:
        # A7 itself is fully fine-tuned; G2/G3 preserve that trainability.
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            _optimizer_groups(model, config["weight_decay"], config["learning_rate"]),
        )
        loader = fce_loader
        learning_rate = config["learning_rate"]
    scheduler = build_scheduler(
        optimizer, len(loader), config["epochs"], config["warmup_epochs"],
        learning_rate, config["min_learning_rate"],
    )
    scaler = _grad_scaler(config["amp"] and device.type == "cuda")
    best_path = output_dir / "checkpoint_best.pth"
    last_path = output_dir / "checkpoint_last.pth"
    best_score = float(baseline["mADM"])
    best_epoch = 0
    best_metrics: dict[str, Any] = {**baseline, "epoch": 0, "experiment": config["experiment"],
                                    "lambda_graded": config["lambda_graded"]}
    _save_checkpoint(best_path, model, mapper, 0, best_metrics, score_module, config)
    history: list[dict[str, Any]] = []
    print(
        f"[{config['experiment']} lambda={config['lambda_graded']:g}] device={device} "
        f"baseline mADM={baseline['mADM']:.6f} epochs={config['epochs']}", flush=True,
    )
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        if config["experiment"] == "G1":
            losses = _train_g1_epoch(
                model, loader, score_module,
                model.encode_text(state_tokens).float().detach(), mapper,
                optimizer, scheduler, device, config, scaler,
            )
        else:
            losses = _train_finetune_epoch(
                model, loader, score_module, state_tokens, mapper, optimizer, scheduler,
                criterion, device, config, scaler,
                "pairwise" if config["experiment"] == "G2" else "listwise",
            )
        metrics = evaluate_native52_score(model, mapper, score_module, data_root, device, config)
        with torch.no_grad():
            fce_metrics = evaluate_paper_fce(
                model, val_loader, criterion, device, config["amp"],
            )
        row = {
            "epoch": epoch,
            "experiment": config["experiment"],
            "lambda_graded": config["lambda_graded"],
            "elapsed_seconds": time.perf_counter() - started,
            **losses,
            **fce_metrics,
            **metrics,
        }
        history.append(row)
        _write_csv(output_dir / "metrics.csv", history)
        if float(metrics["mADM"]) > best_score:
            best_score = float(metrics["mADM"])
            best_epoch = epoch
            best_metrics = row
            _save_checkpoint(best_path, model, mapper, epoch, row, score_module, config)
        _save_checkpoint(last_path, model, mapper, epoch, row, score_module, config,
                         optimizer, scheduler, scaler)
        print(
            f"[{config['experiment']} lambda={config['lambda_graded']:g}] epoch={epoch:02d} "
            f"loss={losses['total_loss']:.5f} rank={metrics['attribute_distance_accuracy']:.4f} "
            f"R1={metrics['rank1']:.4f} mAP={metrics['map']:.4f} "
            f"mADM={metrics['mADM']:.6f} best={best_score:.6f}", flush=True,
        )
    audit_source = next(
        (row for row in history if any(key.startswith("audit_") for key in row)),
        history[-1] if history else best_metrics,
    )
    result = {
        "experiment": config["experiment"],
        "lambda_graded": config["lambda_graded"],
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_mADM": best_score,
        "delta_mADM": best_score - float(baseline["mADM"]),
        "baseline_metrics": baseline,
        "best_metrics": best_metrics,
        "last_metrics": history[-1] if history else best_metrics,
        "gradient_audit": {
            key: audit_source.get(key)
            for key in audit_source
            if key.startswith("audit_")
        },
    }
    _write_json(output_dir / "result.json", result)
    print(
        f"[{config['experiment']} lambda={config['lambda_graded']:g}] DONE "
        f"best_epoch={best_epoch} best_mADM={best_score:.6f} "
        f"delta={result['delta_mADM']:+.6f}", flush=True,
    )


def write_summary(root: Path) -> None:
    baseline_path = root / "baseline_native52.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"]
    rows: list[dict[str, Any]] = [{
        "method": "G0_A7_baseline",
        "experiment": "G0",
        "lambda_graded": "",
        "best_epoch": 0,
        "rank1": baseline.get("rank1"),
        "rank5": baseline.get("rank5"),
        "rank10": baseline.get("rank10"),
        "mAP": baseline.get("map"),
        "mADM": baseline.get("mADM"),
        "delta_mADM": 0.0,
        "attribute_distance_accuracy": baseline.get("attribute_distance_accuracy"),
        "macro_auroc": baseline.get("macro_auroc"),
        "macro_ap": baseline.get("macro_ap"),
        "macro_f1": baseline.get("macro_f1"),
        "fce_loss": None,
    }]
    distance_rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("G[123]_lambda_*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["best_metrics"]
        rows.append({
            "method": path.parent.name,
            "experiment": result["experiment"],
            "lambda_graded": result["lambda_graded"],
            "best_epoch": result["best_epoch"],
            "rank1": metrics.get("rank1"), "rank5": metrics.get("rank5"),
            "rank10": metrics.get("rank10"), "mAP": metrics.get("map"),
            "mADM": result["best_mADM"], "delta_mADM": result["delta_mADM"],
            "attribute_distance_accuracy": metrics.get("attribute_distance_accuracy"),
            "macro_auroc": metrics.get("macro_auroc"),
            "macro_ap": metrics.get("macro_ap"),
            "macro_f1": metrics.get("macro_f1"),
            "fce_loss": metrics.get("fce_loss"),
        })
        for distance in range(41):
            distance_rows.append({
                "method": path.parent.name,
                "experiment": result["experiment"],
                "lambda_graded": result["lambda_graded"],
                "attribute_distance": distance,
                "ranking_accuracy": metrics.get(f"attribute_distance_accuracy_d{distance}"),
            })
    for distance in range(41):
        distance_rows.append({
            "method": "G0_A7_baseline", "experiment": "G0", "lambda_graded": "",
            "attribute_distance": distance,
            "ranking_accuracy": baseline.get(f"attribute_distance_accuracy_d{distance}"),
        })
    _write_csv(root / "comparison.csv", rows)
    _write_csv(root / "attribute_distance_ranking.csv", distance_rows)
    best = max(rows[1:], key=lambda row: float(row["mADM"])) if len(rows) > 1 else rows[0]
    markdown = [
        "# A7 Graded Ranking Ablation (15 epochs)", "",
        "Official mADM uses exact 40-bit equality as relevance and accumulates normalized mean attribute agreement. ",
        "The training target uses that same normalized agreement; the differentiable score is Native52 category-local NLL.",
        "", "| Method | λ | Best epoch | Rank-1 | mAP | mADM | ΔmADM | Distance rank acc. | FCE loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['method']} | {row['lambda_graded'] or '-'} | {row['best_epoch']} | "
            f"{100*float(row['rank1']):.3f}% | {100*float(row['mAP']):.3f}% | "
            f"{100*float(row['mADM']):.3f}% | {100*float(row['delta_mADM']):+.3f} pp | "
            f"{100*float(row['attribute_distance_accuracy']):.3f}% | "
            f"{row['fce_loss'] if row['fce_loss'] is not None else '-'} |"
        )
    markdown.extend([
        "", f"Best mADM result: **{best['method']}** (mADM={100*float(best['mADM']):.3f}%).", "",
        "Files: `comparison.csv`, `attribute_distance_ranking.csv`, and one `metrics.csv` per run.",
    ])
    (root / "comparison.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print((root / "comparison.md").resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A7 Native52 graded-ranking ablation")
    parser.add_argument("--mode", choices=("train", "baseline", "summary"), default="train")
    parser.add_argument("--reuse-a7-validation", action="store_true")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--experiment", choices=("G1", "G2", "G3"))
    parser.add_argument("--lambda-graded", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "summary":
        write_summary(Path(args.output_root).resolve())
        return
    if args.mode == "baseline":
        if args.reuse_a7_validation:
            source = Path(args.checkpoint).resolve().parent / "validation_metrics.json"
            if not source.is_file():
                raise FileNotFoundError(source)
            original = json.loads(source.read_text(encoding="utf-8"))
            metrics = {
                "images": original.get("images"),
                "queries": original.get("queries"),
                "rank1": original.get("rank1"),
                "rank5": original.get("rank5"),
                "rank10": original.get("rank10"),
                "map": original.get("map"),
                "mADM": original.get("mADM"),
                "mINP": original.get("mINP"),
                "macro_auroc": original.get("macro_auroc"),
                "macro_ap": original.get("macro_ap"),
                "macro_f1": float("nan"),
                "retrieval_scoring": "native52_category_nll",
                "native52_temperature_mean": 0.01,
                "native52_category_weight_min": 1.0,
                "native52_category_weight_max": 1.0,
                "attribute_distance_accuracy": float("nan"),
                "attribute_distance_pair_count": 0.0,
            }
            for distance in range(41):
                metrics[f"attribute_distance_accuracy_d{distance}"] = float("nan")
            cache_key = {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "max_val_samples": args.max_val_samples,
            }
            _write_json(Path(args.output_root).resolve() / "baseline_native52.json", {
                "cache_key": cache_key, "metrics": metrics,
                "source": str(source), "distance_ranking": "not_recomputed",
            })
            print(
                f"G0 A7 baseline reused: Rank-1={metrics['rank1']:.6f} "
                f"mAP={metrics['map']:.6f} mADM={metrics['mADM']:.6f}",
                flush=True,
            )
            return
        config = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data_root": str(Path(args.data_root).resolve()),
            "output_root": str(Path(args.output_root).resolve()),
            "device": args.device,
            "max_val_samples": args.max_val_samples,
            "image_size": 224,
            "augmentation": "resize_pad_crop",
            "eval_batch_size": args.eval_batch_size,
            "num_workers": args.num_workers,
            "amp": True,
            "category_temperature": 0.01,
            "score_gallery_chunk": 1024,
            "ranking_pairs_per_query": 512,
            "seed": args.seed,
        }
        metrics = _run_baseline(config)
        print(
            f"G0 A7 baseline: Rank-1={metrics['rank1']:.6f} "
            f"mAP={metrics['map']:.6f} mADM={metrics['mADM']:.6f}",
            flush=True,
        )
        return
    if args.experiment is None:
        raise SystemExit("--experiment is required for --mode train")
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("epochs and batch sizes must be positive")
    run_experiment(args)


if __name__ == "__main__":
    main()
