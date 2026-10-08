"""Training loop and local Task 2 evaluation."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import load_training_checkpoint, save_checkpoint, save_training_checkpoint
from .config import NUM_ATTRIBUTES, REPOSITORY_ROOT, PreprocessingConfig, choose_device, set_seed
from .data import AnnotationTable, AttributeDataset, build_eval_transform, build_train_transform, find_annotation_file, read_gt_csv
from .modeling import ModelEMA, UPARModel, WeightedBCELoss
from .retrieval import (
    autocast, infer_probabilities, l1_attribute_distances,
    load_retrieval_annotations, official_retrieval_metrics, reorder_columns,
    retrieval_metrics,
)
from .tracking import MetricsCSV, TrainingLogger


def evaluate_model(model: nn.Module, model_attribute_names: Sequence[str], data_root: Path,
                   preprocessing: PreprocessingConfig, device: torch.device, batch_size: int,
                   num_workers: int, amp: bool, max_val_samples: int | None = None,
                   query_chunk_size: int = 256, cached_probs: np.ndarray | None = None) -> dict[str, float]:
    gt_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(gt_path)
    if max_val_samples is not None:
        count = min(max_val_samples, len(table.image_paths))
        table = AnnotationTable(table.image_paths[:count], table.labels[:count], table.attribute_names)
        queries, ids = np.unique(table.labels, axis=0, return_inverse=True)
        query_names = table.attribute_names
    else:
        queries, ids, query_names = load_retrieval_annotations(gt_path.parent, table)

    probabilities = cached_probs
    if probabilities is None:
        probabilities = infer_probabilities(
            model,
            table.image_paths,
            [data_root, gt_path.parent, REPOSITORY_ROOT],
            build_eval_transform(preprocessing),
            device,
            batch_size,
            num_workers,
            amp,
        )
    if probabilities.shape != (len(table.image_paths), NUM_ATTRIBUTES):
        raise ValueError("Cached validation probabilities do not match the gallery")
    queries = reorder_columns(queries, query_names, model_attribute_names)
    labels = reorder_columns(table.labels, table.attribute_names, model_attribute_names)
    distances = l1_attribute_distances(queries, probabilities, query_chunk_size)
    rank1, mean_ap = retrieval_metrics(distances, ids)
    official = official_retrieval_metrics(distances, queries, labels, ids)
    return {
        "images": float(len(table.image_paths)),
        "queries": float(len(queries)),
        "rank1": rank1,
        "map": mean_ap,
        "madm": official["mADM"],
        "rank5": official["Rank-5"],
        "rank10": official["Rank-10"],
        "minp": official["mINP"],
    }


def print_evaluation(metrics: dict[str, float], checkpoint: str) -> None:
    print("=" * 44)
    print("UPAR Task2 Validation")
    print("=" * 44)
    print(f"Images       : {int(metrics['images'])}")
    print(f"Queries      : {int(metrics['queries'])}")
    print(f"Checkpoint   : {checkpoint}")
    print(f"Rank-1       : {100 * metrics['rank1']:.2f} %")
    print(f"mAP          : {100 * metrics['map']:.2f} %")
    print(f"mADM         : {100 * metrics['madm']:.2f} %")
    print(f"Rank-5/10    : {100 * metrics['rank5']:.2f} % / {100 * metrics['rank10']:.2f} %")
    print(f"mINP         : {100 * metrics['minp']:.2f} %")
    print("=" * 44)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, scaler: Any, device: torch.device, amp: bool,
                    grad_clip: float, ema: ModelEMA | None) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            loss = criterion(model(images), targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        total_loss += float(loss.detach()) * len(images)
        total_samples += len(images)
    return total_loss / max(total_samples, 1)


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed, args.deterministic)
    device = choose_device(args.device)
    data_root = Path(args.data_root).resolve()
    train_gt = find_annotation_file(data_root, "train")
    table = read_gt_csv(train_gt)
    positive_ratios = table.labels.mean(axis=0, dtype=np.float64).astype(np.float32)
    resize_size = args.resize_size or int(round(args.image_size * 232 / 224))
    resize_width = args.resize_width
    if args.image_width is not None and resize_width is None:
        resize_width = args.image_width
    preprocessing = PreprocessingConfig(
        image_size=args.image_size,
        resize_size=resize_size,
        image_width=args.image_width,
        resize_width=resize_width,
        crop_policy=args.crop_policy,
        augmix=args.augmix,
    )
    dataset = AttributeDataset(
        table,
        [data_root, train_gt.parent, REPOSITORY_ROOT],
        build_train_transform(preprocessing),
        args.max_train_samples,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=loader_generator,
    )
    model = UPARModel(
        dropout=args.dropout,
        pretrained=not args.no_pretrained and not args.resume,
    ).to(device)
    ema = ModelEMA(model, args.ema_decay) if args.ema else None
    criterion = WeightedBCELoss(
        positive_ratios,
        args.label_smoothing,
        not args.unweighted_bce,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )
    scaler = make_grad_scaler(args.amp and device.type == "cuda")
    output_dir = Path(args.output_dir)
    best_path = output_dir / "best.pth"
    last_path = output_dir / "last.pth"
    resume_path = Path(args.resume) if args.resume else None
    logger = TrainingLogger(output_dir / "train.log", resume=resume_path is not None)
    metrics_csv = MetricsCSV(output_dir / "metrics.csv", resume=resume_path is not None)
    best_score = -math.inf
    best_epoch = 0
    evaluations_without_improvement = 0
    start_epoch = 1

    if resume_path is not None:
        checkpoint = load_training_checkpoint(
            resume_path, model, ema.module if ema is not None else None,
            optimizer, scheduler, scaler, loader_generator,
        )
        if list(checkpoint["attribute_names"]) != table.attribute_names:
            raise ValueError("Resume checkpoint attributes do not match the training data")
        state = checkpoint["training_state"]
        best_score = float(state.get("best_score", state["best_map"]))
        best_epoch = int(state["best_epoch"])
        evaluations_without_improvement = int(state["evaluations_without_improvement"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if not best_path.is_file():
            raise FileNotFoundError(
                f"Best checkpoint is missing: {best_path}. Resume with the original output directory."
            )

    logger.log(json.dumps({**vars(args), "device_resolved": str(device)}, indent=2, default=str))
    logger.log(f"Training samples: {len(dataset)}; attributes: {len(table.attribute_names)}")
    if resume_path is not None:
        logger.log(
            f"Resumed from {resume_path} at epoch {start_epoch}; "
            f"best {args.selection_metric}={100 * best_score:.2f}% (epoch {best_epoch})"
        )
    if start_epoch > args.epochs:
        logger.log(f"Checkpoint already reached epoch {start_epoch - 1}; target is {args.epochs}.")
        logger.close()
        metrics_csv.close()
        return best_path

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        loss = train_one_epoch(
            model, loader, optimizer, criterion, scaler, device,
            args.amp, args.grad_clip, ema,
        )
        should_evaluate = epoch % args.retrieval_interval == 0 or epoch == args.epochs
        epoch_results: dict[str, dict[str, float]] = {}
        should_stop = False
        if should_evaluate:
            candidates = [("model", model)]
            if ema is not None:
                candidates.append(("ema", ema.module))
            epoch_best = -math.inf
            improved = False
            for kind, candidate in candidates:
                metrics = evaluate_model(
                    candidate, table.attribute_names, data_root, preprocessing, device,
                    args.eval_batch_size, args.num_workers, args.amp,
                    args.max_val_samples, args.query_chunk_size,
                )
                epoch_results[kind] = metrics
                epoch_best = max(epoch_best, metrics[args.selection_metric])
                logger.log(
                    f"Epoch {epoch:03d}/{args.epochs} [{kind}]: loss={loss:.6f}, "
                    f"Rank-1={100 * metrics['rank1']:.2f}%, "
                    f"mAP={100 * metrics['map']:.2f}%, "
                    f"mADM={100 * metrics['madm']:.2f}% "
                    f"({time.time() - started:.1f}s)"
                )
                if metrics[args.selection_metric] > best_score:
                    best_score = metrics[args.selection_metric]
                    best_epoch = epoch
                    improved = True
                    save_checkpoint(
                        best_path, candidate, table.attribute_names, preprocessing,
                        positive_ratios, epoch,
                        {
                            "train_loss": loss,
                            "rank1": metrics["rank1"],
                            "map": metrics["map"],
                            "madm": metrics["madm"],
                            "rank5": metrics["rank5"],
                            "rank10": metrics["rank10"],
                            "minp": metrics["minp"],
                            "selection_metric": args.selection_metric,
                        },
                        kind, args.dropout,
                    )
                    logger.log(f"Saved new best {kind} checkpoint to {best_path}")
            scheduler.step(epoch_best)
            if improved:
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
                logger.log(
                    "Early stopping: "
                    f"{evaluations_without_improvement}/{args.early_stopping_patience} "
                    f"evaluations without {args.selection_metric} improvement "
                    f"(best epoch: {best_epoch})"
                )
                should_stop = evaluations_without_improvement >= args.early_stopping_patience
        else:
            logger.log(
                f"Epoch {epoch:03d}/{args.epochs}: loss={loss:.6f} "
                f"({time.time() - started:.1f}s)"
            )

        elapsed = time.time() - started
        learning_rate = float(optimizer.param_groups[0]["lr"])
        latest_metrics = {"train_loss": loss}
        for kind, metrics in epoch_results.items():
            latest_metrics[f"{kind}_rank1"] = metrics["rank1"]
            latest_metrics[f"{kind}_map"] = metrics["map"]
            latest_metrics[f"{kind}_madm"] = metrics["madm"]
        metrics_csv.write({
            "epoch": epoch,
            "train_loss": loss,
            "learning_rate": learning_rate,
            "model_rank1": epoch_results.get("model", {}).get("rank1", ""),
            "model_map": epoch_results.get("model", {}).get("map", ""),
            "model_madm": epoch_results.get("model", {}).get("madm", ""),
            "ema_rank1": epoch_results.get("ema", {}).get("rank1", ""),
            "ema_map": epoch_results.get("ema", {}).get("map", ""),
            "ema_madm": epoch_results.get("ema", {}).get("madm", ""),
            "best_map": best_score,
            "best_score": best_score,
            "selection_metric": args.selection_metric,
            "best_epoch": best_epoch,
            "evaluations_without_improvement": evaluations_without_improvement,
            "elapsed_seconds": elapsed,
        })
        save_training_checkpoint(
            last_path, model, ema.module if ema is not None else None,
            optimizer, scheduler, scaler, loader_generator,
            table.attribute_names, preprocessing, positive_ratios, epoch,
            latest_metrics, best_score, best_epoch, evaluations_without_improvement,
            args.dropout, args.selection_metric,
        )
        logger.log(f"Saved latest training state to {last_path}")
        if should_stop:
            logger.log(
                f"Stopped early at epoch {epoch}; best {args.selection_metric} was "
                f"{100 * best_score:.2f}% at epoch {best_epoch}."
            )
            break
    logger.close()
    metrics_csv.close()
    if not best_path.is_file():
        raise RuntimeError("Training ended without a retrieval-selected checkpoint")
    return best_path
