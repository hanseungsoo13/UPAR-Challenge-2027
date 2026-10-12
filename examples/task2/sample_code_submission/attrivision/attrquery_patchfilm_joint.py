"""Training and Category-NLL evaluation for PatchFiLM-JointFromCLIP.

This experiment is initialized only from original pretrained OpenCLIP. It
reuses FromScratchJointFCE's data, loss, optimizer, scheduler, seed, and
validation protocol while adding identity-initialized PatchFiLM to value
aggregation.
"""
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

from .attrquery_fromscratch_jointfce import (
    atomic_json,
    best_fusion_row,
    evaluate_model,
    protocol_validation,
    restore_rng,
    rng_state,
    save_gallery_scores,
    valid_category_mask,
    write_rows_csv,
)
from .datasets.attribute_prompts import CategoryPromptMapper
from .datasets.upar_abpr import PromptBatch
from .engine.trainer_attrivision import _grad_scaler, _optimizer_step, build_scheduler
from .losses.category_local_fce import CategoryLocalFCELoss
from .losses.focal_clip_loss import FocalCLIPLoss
from .models.attrquery_patchfilm_joint import (
    ARCHITECTURE_NAME,
    AttrQueryPatchFiLMJointFromCLIP,
    build_model_from_pretrained,
    checkpoint_payload,
    load_experiment_model,
    save_checkpoint,
)
from .train_attrquery_fromscratch_jointfce import (
    _build_loader,
    _effective_local_weight,
    _gradient_norm,
    _loss_gradient_norm,
    _optimizer_groups,
)
from upar.config import REPOSITORY_ROOT, choose_device, set_seed
from upar.data import find_annotation_file, read_gt_csv
from upar.retrieval import autocast


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "attrquery_patchfilm_joint.json"
DEFAULT_OUTPUT = "outputs/AttrQuery-PatchFiLM-JointFromCLIP"


def _resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    for key, value in vars(args).items():
        if key in {"config", "debug"} or value is None:
            continue
        config[key] = value
    # The named controlled condition is fixed to JointFCE-Lambda050.
    config["local_loss_weight"] = 0.5
    config["local_loss_warmup_epochs"] = 0
    config["condition_name"] = "JointFCE-Lambda050+PatchFiLM"
    config["output_dir"] = str(Path(config.get("output_dir") or DEFAULT_OUTPUT).resolve())
    config["data_root"] = str(Path(config["data_root"]).resolve())
    config["amp"] = bool(config.get("amp", True))
    config["freeze_text_encoder"] = bool(config.get("freeze_text_encoder", False))
    config["freeze_logit_scale"] = bool(config.get("freeze_logit_scale", False))
    if config.get("pretrained_tag") in {None, "", "none", "None"}:
        raise ValueError("PatchFiLM-JointFromCLIP requires original pretrained CLIP weights")
    if int(config["bottleneck_ratio"]) != 4 or float(config["patchfilm_delta"]) != 0.1:
        raise ValueError("This experiment fixes PatchFiLM bottleneck_ratio=4 and delta=0.1")
    return config


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        **config,
        "experiment": "AttrQuery-PatchFiLM-JointFromCLIP",
        "architecture": ARCHITECTURE_NAME,
        "initialization_source": "original_pretrained_clip",
        "a7_checkpoint_used": False,
        "attrquery_checkpoint_used": False,
        "jointfce_reference_condition": "JointFCE-Lambda050",
        "jointfce_shared_initialization": "same_constructor_order_and_seed",
        "patchfilm_extra_rng_restored": True,
        "training_objective": "paper_fce_plus_0.5_category_local_fce",
        "global_objective": "paper_fce",
        "local_objective": "category_local_fce",
        "prompt_mode": "category_complete",
        "text_sampling": "single",
        "contrastive_target": "multi_positive",
        "multi_label": False,
        "category_ce": False,
        "validation_protocol": "native52_category_nll",
        "selection_alpha": 0.5,
        "selection_metric": "mADM",
        "fusion_space": "category_log_probability",
        "patchfilm": {
            "bottleneck_ratio": 4,
            "delta": 0.1,
            "category_chunk_size": int(config["patchfilm_category_chunk_size"]),
            "identity_initialization": True,
            "value_modulation_before_aggregation": True,
        },
    }


def _resume_state(
    path: Path,
    model: AttrQueryPatchFiLMJointFromCLIP,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
) -> tuple[dict[str, Any], float, int, int]:
    resumed_model, payload = load_experiment_model(str(path), device)
    if payload["attribute_names"] != model.attribute_names:
        raise ValueError("Resume checkpoint attribute order differs from training annotations")
    saved_config = payload.get("training_config", {})
    if saved_config.get("condition_name") != "JointFCE-Lambda050+PatchFiLM":
        raise ValueError("Resume checkpoint is not the controlled Lambda050 PatchFiLM condition")
    model.load_state_dict(resumed_model.state_dict(), strict=True)
    state = payload.get("training_state")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not resumable: {path}")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    if state.get("rng") is not None:
        restore_rng(state["rng"])
    return payload, float(state["best_score"]), int(state["best_epoch"]), int(state["stale_evaluations"])


def _train_one_epoch(
    model: AttrQueryPatchFiLMJointFromCLIP,
    loader: DataLoader,
    mapper: CategoryPromptMapper,
    state_tokens: torch.Tensor,
    global_criterion: FocalCLIPLoss,
    local_criterion: CategoryLocalFCELoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    model.assert_trainability()
    local_weight = _effective_local_weight(config, epoch)
    groups = mapper.category_indices()
    diag_interval = int(config.get("gradient_diagnostics_interval", 100))
    visual_parameters = [p for p in model.base.clip.visual.parameters() if p.requires_grad]
    text_parameters = [
        p for name, p in model.base.clip.named_parameters()
        if not name.startswith("visual.") and p.requires_grad
    ]
    patchfilm_parameters = model.patchfilm_parameters()
    totals = {
        "total_loss": 0.0, "global_loss": 0.0, "local_loss": 0.0,
        "effective_local_weight": 0.0,
        "global_backbone_grad_norm": 0.0, "local_backbone_grad_norm": 0.0,
        "backbone_grad_norm": 0.0, "local_head_grad_norm": 0.0,
        "patchfilm_grad_norm": 0.0, "patchfilm_gradient_tensor_coverage": 0.0,
        "patchfilm_nonzero_gradient_parameter_fraction": 0.0,
        "text_grad_norm": 0.0, "gradient_diagnostics": 0.0,
        "flow_local_head": 0.0, "flow_patchfilm": 0.0,
        "flow_visual_backbone": 0.0, "flow_text_encoder": 0.0,
        "samples": 0.0, "batches": 0.0,
    }
    for batch_index, batch in enumerate(loader, start=1):
        if not isinstance(batch, PromptBatch):
            raise TypeError("PromptCollator must return PromptBatch")
        images = batch.images.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        semantic_labels = batch.semantic_labels.to(device, non_blocking=True)
        tokens = batch.tokens.to(device, non_blocking=True)
        selected = batch.selected_semantics.to(device, non_blocking=True)
        owners = batch.text_owners.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, bool(config["amp"])):
            image_features, local_features = model.joint_features(images)
            all_text = model.encode_text(torch.cat([tokens, state_tokens], dim=0))
            global_logits = model.global_logits_from_features(image_features, all_text[:len(tokens)])
            global_output = global_criterion(global_logits, semantic_labels, selected, owners)
            encoded_labels = mapper.encode(labels)
            local_output = local_criterion(
                local_features, all_text[len(tokens):], encoded_labels,
                valid_category_mask(encoded_labels, groups),
            )
            total = global_output.loss + local_weight * local_output.loss

        diagnostic = diag_interval > 0 and (batch_index == 1 or batch_index % diag_interval == 0)
        if diagnostic:
            totals["global_backbone_grad_norm"] += _loss_gradient_norm(global_output.loss, visual_parameters)
            totals["local_backbone_grad_norm"] += _loss_gradient_norm(local_output.loss, visual_parameters)
            totals["gradient_diagnostics"] += 1.0
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        if float(config["grad_clip"]) > 0:
            nn.utils.clip_grad_norm_(model.trainable_parameters(), float(config["grad_clip"]))
        flags = model.gradient_flow_flags()
        if batch_index == 1:
            required = ("local_head", "patchfilm", "visual_backbone")
            if not all(flags[name] for name in required):
                raise RuntimeError(f"Gradient flow check failed on first batch: {flags}")
            missing = [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad and name.startswith("local_head.patch_film.")
                and parameter.grad is None
            ]
            if missing:
                raise RuntimeError(f"PatchFiLM parameters without gradients: {missing}")
            if not config["freeze_text_encoder"] and not flags["text_encoder"]:
                raise RuntimeError(f"JointFCE-compatible text gradient flow check failed: {flags}")

        gradients = [p.grad for p in patchfilm_parameters]
        coverage = sum(gradient is not None for gradient in gradients) / max(len(gradients), 1)
        nonzero = sum(
            gradient is not None and bool(torch.count_nonzero(gradient.detach()).item())
            for gradient in gradients
        ) / max(len(gradients), 1)
        if coverage != 1.0:
            raise RuntimeError(f"PatchFiLM gradient tensor coverage is {coverage:.3f}, expected 1.0")
        totals["backbone_grad_norm"] += _gradient_norm(visual_parameters)
        totals["local_head_grad_norm"] += _gradient_norm(list(model.local_head.parameters()))
        totals["patchfilm_grad_norm"] += _gradient_norm(patchfilm_parameters)
        totals["patchfilm_gradient_tensor_coverage"] += coverage
        totals["patchfilm_nonzero_gradient_parameter_fraction"] += nonzero
        totals["text_grad_norm"] += _gradient_norm(text_parameters)
        totals["flow_local_head"] += float(flags["local_head"])
        totals["flow_patchfilm"] += float(flags["patchfilm"])
        totals["flow_visual_backbone"] += float(flags["visual_backbone"])
        totals["flow_text_encoder"] += float(flags["text_encoder"])
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        totals["total_loss"] += float(total.detach()) * count
        totals["global_loss"] += float(global_output.loss.detach()) * count
        totals["local_loss"] += float(local_output.loss.detach()) * count
        totals["effective_local_weight"] += local_weight * count
        totals["samples"] += count
        totals["batches"] += 1.0

    samples = max(totals.pop("samples"), 1.0)
    batches = max(totals.pop("batches"), 1.0)
    diagnostics = max(totals["gradient_diagnostics"], 1.0)
    result = {key: value / samples for key, value in totals.items()}
    for key in (
        "global_backbone_grad_norm", "local_backbone_grad_norm",
    ):
        result[key] = totals[key] / diagnostics
    for key in (
        "backbone_grad_norm", "local_head_grad_norm", "patchfilm_grad_norm",
        "patchfilm_gradient_tensor_coverage", "patchfilm_nonzero_gradient_parameter_fraction",
        "text_grad_norm", "flow_local_head", "flow_patchfilm",
        "flow_visual_backbone", "flow_text_encoder",
    ):
        result[key] = totals[key] / batches
    result["gradient_diagnostics"] = totals["gradient_diagnostics"]
    result["train_batches"] = batches
    result["average_batch_size"] = samples / batches
    result["local_loss_weight"] = local_weight
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train AttrQuery-PatchFiLM-JointFromCLIP from original pretrained CLIP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--head-learning-rate", type=float)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze-text-encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze-logit-scale", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--gradient-diagnostics-interval", type=int)
    parser.add_argument("--debug", action="store_true")
    return parser


def train_main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _resolved_config(args)
    if args.debug:
        config["epochs"] = min(int(config["epochs"]), 1)
        config["max_train_samples"] = config.get("max_train_samples") or 64
        config["max_val_samples"] = config.get("max_val_samples") or 64
    device = choose_device(config.get("device", "auto"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "config.json", config)
    set_seed(config["seed"], bool(config.get("deterministic", False)))

    train_gt = find_annotation_file(Path(config["data_root"]), "train")
    train_table = read_gt_csv(train_gt)
    config["train_annotation_parent"] = str(train_gt.parent)
    mapper = CategoryPromptMapper(train_table.attribute_names)
    training_config = _training_config(config)
    protocol = protocol_validation(training_config, mapper)
    atomic_json(output_dir / "protocol_validation.json", protocol)

    identity_difference = None
    if config.get("resume"):
        model, payload = load_experiment_model(str(config["resume"]), device)
        if payload["attribute_names"] != train_table.attribute_names:
            raise ValueError("Resume checkpoint attribute order differs from training annotations")
        if payload.get("training_config", {}).get("condition_name") != config["condition_name"]:
            raise ValueError("Resume checkpoint uses a different experiment condition")
    else:
        model, identity_difference = build_model_from_pretrained(
            train_table.attribute_names, device,
            pretrained_tag=config["pretrained_tag"], model_name=config["model_name"],
            num_heads=config["num_heads"], dropout=config["dropout"],
            use_ffn=config["use_ffn"],
            freeze_text_encoder=config["freeze_text_encoder"],
            freeze_logit_scale=config["freeze_logit_scale"],
            bottleneck_ratio=config["bottleneck_ratio"],
            patchfilm_delta=config["patchfilm_delta"],
            patchfilm_category_chunk_size=config["patchfilm_category_chunk_size"],
        )
    counts = model.parameter_counts()
    parameter_record = {
        **counts,
        "patchfilm_parameter_names": [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("local_head.patch_film.")
        ],
        "all_trainable_parameters": sum(p.numel() for p in model.trainable_parameters()),
        "identity_initialization_max_abs_output_difference": identity_difference,
    }
    atomic_json(output_dir / "parameter_counts.json", parameter_record)
    atomic_json(output_dir / "initialization_validation.json", {
        "initialization_source": "original_pretrained_clip",
        "a7_checkpoint_used": False,
        "attrquery_checkpoint_used": False,
        "shared_jointfce_modules_initialized_first": True,
        "patchfilm_identity_max_abs_output_difference": identity_difference,
        "patchfilm_bottleneck_ratio": model.bottleneck_ratio,
        "patchfilm_delta": model.patchfilm_delta,
        "patchfilm_rng_restored_after_initialization": True,
    })
    print(
        f"AttrQuery-PatchFiLM-JointFromCLIP condition=JointFCE-Lambda050; "
        f"device={device}; pretrained={config['pretrained_tag']}; parameters={parameter_record}",
        flush=True,
    )

    # The data pipeline and all sampling options match JointFCE-Lambda050.
    loader = _build_loader(config, train_table, model, Path(config["data_root"]), device)
    state_tokens = model.base.tokenize(mapper.prompts).to(device)
    global_criterion = FocalCLIPLoss(
        "focal_clip", "multi_positive", config["focal_alpha"], config["focal_gamma"],
    )
    local_criterion = CategoryLocalFCELoss(
        mapper.category_indices(), config["category_temperature"], config["focal_gamma"],
    ).to(device)
    optimizer = torch.optim.AdamW(_optimizer_groups(model, config))
    scheduler = build_scheduler(
        optimizer, len(loader), config["epochs"], config["warmup_epochs"],
        config["learning_rate"], config["min_learning_rate"],
    )
    scaler = _grad_scaler(bool(config["amp"]) and device.type == "cuda")
    best_path = output_dir / "checkpoint_best.pth"
    last_path = output_dir / "checkpoint_last.pth"
    log_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    best_score, best_epoch, stale, start_epoch = -math.inf, 0, 0, 1
    if config.get("resume"):
        payload, best_score, best_epoch, stale = _resume_state(
            Path(config["resume"]), model, optimizer, scheduler, scaler, device,
        )
        start_epoch = int(payload["epoch"]) + 1

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        started = time.perf_counter()
        train_metrics = _train_one_epoch(
            model, loader, mapper, state_tokens, global_criterion, local_criterion,
            optimizer, scheduler, scaler, config, device, epoch,
        )
        rows, details = evaluate_model(
            model, train_table.attribute_names, Path(config["data_root"]), device,
            config["eval_batch_size"], config["num_workers"], config["amp"],
            config["category_temperature"], config.get("max_val_samples"),
            alphas=(0.5,), progress_every=config.get("progress_every", 0),
        )
        fusion = best_fusion_row(rows, 0.5)
        row: dict[str, Any] = {
            "epoch": epoch, "condition": config["condition_name"],
            "elapsed_sec": time.perf_counter() - started, **train_metrics,
            "global_rank1": next(item["rank1"] for item in rows if item["score_type"] == "global"),
            "global_mAP": next(item["mAP"] for item in rows if item["score_type"] == "global"),
            "global_mADM": next(item["mADM"] for item in rows if item["score_type"] == "global"),
            "global_mINP": next(item["mINP"] for item in rows if item["score_type"] == "global"),
            "local_rank1": next(item["rank1"] for item in rows if item["score_type"] == "local"),
            "local_mAP": next(item["mAP"] for item in rows if item["score_type"] == "local"),
            "local_mADM": next(item["mADM"] for item in rows if item["score_type"] == "local"),
            "local_mINP": next(item["mINP"] for item in rows if item["score_type"] == "local"),
            "fusion_rank1": fusion["rank1"], "fusion_mAP": fusion["mAP"],
            "fusion_mADM": fusion["mADM"], "fusion_mINP": fusion["mINP"],
            "selection_alpha": 0.5,
        }
        log_rows.append(row)
        validation_rows.extend({"epoch": epoch, "condition": config["condition_name"], **item} for item in rows)
        write_rows_csv(output_dir / "training_metrics.csv", log_rows)
        write_rows_csv(output_dir / "validation_metrics.csv", validation_rows)
        score = float(fusion["mADM"])
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            save_checkpoint(
                best_path,
                checkpoint_payload(model, epoch, row, training_config,
                                   best_score=best_score, best_epoch=best_epoch),
            )
            save_gallery_scores(output_dir / "best_validation_scores.npz", (
                np.asarray(details["global_scores"]), np.asarray(details["local_scores"]),
            ))
            np.savez_compressed(output_dir / "best_fusion_scores.npz", **details["fusion_scores"])
        else:
            stale += 1
        save_checkpoint(
            last_path,
            checkpoint_payload(
                model, epoch, row, training_config, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, best_score=best_score,
                best_epoch=best_epoch, stale_evaluations=stale, rng_state=rng_state(),
            ),
        )
        print(
            f"epoch={epoch:03d}/{config['epochs']}: total={train_metrics['total_loss']:.6f}, "
            f"global={train_metrics['global_loss']:.6f}, local={train_metrics['local_loss']:.6f}, "
            f"patchfilm_grad_norm={train_metrics['patchfilm_grad_norm']:.6g}, "
            f"fusion@0.5 mADM={100*fusion['mADM']:.2f}%, time={row['elapsed_sec']:.1f}s",
            flush=True,
        )
        if epoch >= int(config["minimum_training_epochs"]) and stale >= int(config["early_stopping_patience"]):
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
            break

    atomic_json(output_dir / "training_summary.json", {
        "experiment": "AttrQuery-PatchFiLM-JointFromCLIP",
        "condition": config["condition_name"], "best_epoch": best_epoch,
        "best_mADM": best_score, "selection_alpha": 0.5,
        "best_checkpoint": str(best_path),
        "initialization": "original_pretrained_clip",
        "a7_checkpoint_used": False, "attrquery_checkpoint_used": False,
        "parameter_counts": parameter_record,
    })


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PatchFiLM-JointFromCLIP, optionally beside JointFCE-Lambda050",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="PatchFiLM checkpoint_best.pth")
    parser.add_argument("--baseline-checkpoint", help="Optional original JointFCE-Lambda050 checkpoint")
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", default=f"{DEFAULT_OUTPUT}/evaluation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--category-temperature", type=float, default=0.01)
    parser.add_argument("--fixed-alpha", type=float, default=0.5)
    parser.add_argument("--alpha-grid", type=str, default="0,0.1,0.25,0.5,0.75,0.9,1")
    parser.add_argument("--tuning-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--attention-samples", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def eval_main(argv: Sequence[str] | None = None) -> None:
    from .attrquery_fromscratch_jointfce import (
        category_names as get_category_names, encode_gallery_scores, retrieval_views,
    )
    from .attrquery_attribute_adapter import (
        _category_logs, _category_nll_distances, _category_rows,
        _fuse_logs, _query_split, _subset_metrics, _write_csv,
        _atomic_json,
    )
    from .models.attrquery_fromscratch_jointfce import load_experiment_model as load_baseline
    from .models.attrquery_fromscratch_jointfce import validate_mapper
    from .transforms import build_eval_transform
    from upar.retrieval import autocast as retrieval_autocast, reorder_columns

    args = build_eval_parser().parse_args(argv)
    if args.eval_batch_size <= 0 or args.num_workers < 0 or args.ece_bins <= 0:
        raise ValueError("eval-batch-size/ece-bins must be positive and num-workers non-negative")
    if not math.isclose(args.category_temperature, 0.01) or not math.isclose(args.fixed_alpha, 0.5):
        raise ValueError("This comparison fixes category_temperature=0.01 and fixed_alpha=0.5")
    alphas = sorted({float(value.strip()) for value in args.alpha_grid.split(",") if value.strip()})
    if not alphas or any(value < 0 or value > 1 for value in alphas):
        raise ValueError("alpha-grid must contain comma-separated values in [0,1]")
    if not 0 < args.tuning_fraction < 1 or args.attention_samples < 0:
        raise ValueError("tuning-fraction must be in (0,1) and attention-samples non-negative")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    data_root = Path(args.data_root).resolve()
    gt_path, table, queries_raw, ids, query_names = retrieval_views(
        data_root, "val", args.max_val_samples,
    )
    attributes = list(table.attribute_names)
    mapper = CategoryPromptMapper(attributes)
    validate_mapper(mapper)
    queries = reorder_columns(queries_raw, query_names, attributes)
    labels = table.labels.astype(np.float32, copy=False)
    query_targets = mapper.encode(torch.from_numpy(queries).float()).numpy().astype(np.float32)
    target52 = mapper.encode(torch.from_numpy(labels).float()).numpy().astype(np.float32)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    category_names = get_category_names(mapper)
    tuning_indices, evaluation_indices = _query_split(len(queries), args.split_seed, args.tuning_fraction)
    tuning_ids = set(tuning_indices.tolist())
    _write_csv(output_dir / "query_split.csv", [
        {"query_index": int(index), "split": "tuning" if index in tuning_ids else "evaluation"}
        for index in range(len(queries))
    ])
    protocol = {
        "experiment": "AttrQuery-PatchFiLM-JointFromCLIP comparison",
        "validation_protocol": "native52_category_nll",
        "states": 52, "categories": 12,
        "category_temperature": args.category_temperature,
        "fixed_fusion_alpha": args.fixed_alpha,
        "checkpoint_selection": "training_validation_full_gallery_alpha_0.5_mADM",
        "posthoc_alpha_selection": "tuning_query_split_mADM",
        "evaluation_split_alpha": "fixed_to_tuning_selected_value",
        "split_seed": args.split_seed, "tuning_fraction": args.tuning_fraction,
        "tuning_queries": int(len(tuning_indices)),
        "evaluation_queries": int(len(evaluation_indices)),
        "gallery_images": int(len(table.image_paths)),
        "patchfilm_checkpoint": str(Path(args.checkpoint).resolve()),
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()) if args.baseline_checkpoint else None,
    }
    _atomic_json(output_dir / "protocol.json", protocol)

    checkpoints: list[tuple[str, Path, bool]] = [
        ("PatchFiLM-JointFromCLIP", Path(args.checkpoint).resolve(), True),
    ]
    if args.baseline_checkpoint:
        checkpoints.insert(0, ("JointFCE-Lambda050-NoAdapter", Path(args.baseline_checkpoint).resolve(), False))

    score_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, checkpoint, is_patchfilm in checkpoints:
        if is_patchfilm:
            model, payload = load_experiment_model(str(checkpoint), device)
        else:
            model, payload = load_baseline(checkpoint, device)
        if list(payload.get("attribute_names", [])) != attributes:
            raise ValueError(f"{name} attribute order does not match the validation annotations")
        if not is_patchfilm and payload.get("training_config", {}).get("local_loss_weight") != 0.5:
            raise ValueError("Baseline checkpoint must be JointFCE-Lambda050 (local loss weight 0.5)")
        model.eval()
        with torch.inference_mode(), retrieval_autocast(device, bool(args.amp)):
            state_features = model.encode_text(model.base.tokenize(mapper.prompts).to(device)).float()
        scores = encode_gallery_scores(
            model, table.image_paths, [data_root, gt_path.parent, REPOSITORY_ROOT],
            build_eval_transform(224, "resize_pad_crop"), state_features,
            device, args.eval_batch_size, args.num_workers, bool(args.amp), args.progress_every,
        )
        score_cache[name] = scores
        np.savez_compressed(output_dir / f"gallery_scores_{'patchfilm' if is_patchfilm else 'jointfce'}.npz",
                            global_scores=scores[0], local_scores=scores[1])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    for name, _checkpoint, _is_patchfilm in checkpoints:
        global_scores, local_scores = score_cache[name]
        global_logs = _category_logs(global_scores, groups, args.category_temperature)
        local_logs = _category_logs(local_scores, groups, args.category_temperature)
        fixed_logs = _fuse_logs(global_logs, local_logs, args.fixed_alpha)
        views = {
            "Global-only": global_logs,
            "Local-only": local_logs,
            f"Global+Local(alpha={args.fixed_alpha:g})": fixed_logs,
        }
        gallery_tuning = np.flatnonzero(np.isin(ids, tuning_indices))
        gallery_evaluation = np.flatnonzero(np.isin(ids, evaluation_indices))
        for view_name, logs in views.items():
            distances = _category_nll_distances(query_targets, logs, groups)
            for split_name, indices in (("tuning", tuning_indices), ("evaluation", evaluation_indices)):
                metrics_rows.append({
                    "architecture": name, "method": view_name, "split": split_name,
                    **_subset_metrics(distances, indices, queries, labels, ids),
                })
        # Category accuracy/NLL/ECE for all three views, separately on images
        # belonging to tuning and held-out evaluation queries.
        category_rows.extend(
            {**row, "split": "tuning"}
            for row in _category_rows(
                global_logs[gallery_tuning], local_logs[gallery_tuning], target52[gallery_tuning],
                groups, category_names, args.ece_bins, name, args.fixed_alpha,
            )
        )
        category_rows.extend(
            {**row, "split": "evaluation"}
            for row in _category_rows(
                global_logs[gallery_evaluation], local_logs[gallery_evaluation], target52[gallery_evaluation],
                groups, category_names, args.ece_bins, name, args.fixed_alpha,
            )
        )

        distances_by_alpha: dict[float, np.ndarray] = {}
        for alpha in alphas:
            fused = _fuse_logs(global_logs, local_logs, alpha)
            distances = _category_nll_distances(query_targets, fused, groups)
            distances_by_alpha[alpha] = distances
            sweep_rows.append({
                "architecture": name, "scope": "tuning", "alpha": alpha,
                **_subset_metrics(distances, tuning_indices, queries, labels, ids),
            })
        best_alpha = max(
            alphas,
            key=lambda alpha: (
                float(_subset_metrics(distances_by_alpha[alpha], tuning_indices, queries, labels, ids)["mADM"]),
                -alpha,
            ),
        )
        sweep_rows.append({
            "architecture": name, "scope": "evaluation_tuning_selected_only",
            "alpha": best_alpha,
            **_subset_metrics(distances_by_alpha[best_alpha], evaluation_indices, queries, labels, ids),
        })
        metrics_rows.append({
            "architecture": name, "method": "Global+Local(tuning-selected alpha)",
            "split": "evaluation", "alpha": best_alpha,
            **_subset_metrics(distances_by_alpha[best_alpha], evaluation_indices, queries, labels, ids),
        })
        category_rows.extend(
            {**row, "split": "tuning_selected_alpha", "selected_alpha": best_alpha}
            for row in _category_rows(
                global_logs[gallery_tuning], local_logs[gallery_tuning], target52[gallery_tuning],
                groups, category_names, args.ece_bins, name, best_alpha,
            )
        )
        category_rows.extend(
            {**row, "split": "evaluation_tuning_selected_alpha", "selected_alpha": best_alpha}
            for row in _category_rows(
                global_logs[gallery_evaluation], local_logs[gallery_evaluation], target52[gallery_evaluation],
                groups, category_names, args.ece_bins, name, best_alpha,
            )
        )
        protocol.setdefault("selected_alphas", {})[name] = best_alpha

    _atomic_json(output_dir / "protocol.json", protocol)
    _write_csv(output_dir / "metrics_summary.csv", metrics_rows)
    _write_csv(output_dir / "category_metrics.csv", category_rows)
    _write_csv(output_dir / "fusion_alpha_sweep.csv", sweep_rows)
    baseline_name = next((name for name, _path, is_patchfilm in checkpoints if not is_patchfilm), None)
    patchfilm_name = next((name for name, _path, is_patchfilm in checkpoints if is_patchfilm), None)
    if baseline_name and patchfilm_name:
        comparison_rows: list[dict[str, Any]] = []
        metric_keys = ("Rank-1", "mAP", "mADM", "mINP", "Rank-5", "Rank-10")
        for baseline_row in metrics_rows:
            if baseline_row["architecture"] != baseline_name:
                continue
            patch_row = next((
                row for row in metrics_rows
                if row["architecture"] == patchfilm_name
                and row["method"] == baseline_row["method"]
                and row["split"] == baseline_row["split"]
            ), None)
            if patch_row is None:
                continue
            comparison_rows.append({
                "method": baseline_row["method"], "split": baseline_row["split"],
                "baseline_alpha": baseline_row.get("alpha", args.fixed_alpha),
                "patchfilm_alpha": patch_row.get("alpha", args.fixed_alpha),
                **{f"baseline_{key}": baseline_row[key] for key in metric_keys if key in baseline_row},
                **{f"patchfilm_{key}": patch_row[key] for key in metric_keys if key in patch_row},
                **{
                    f"delta_{key}": float(patch_row[key]) - float(baseline_row[key])
                    for key in metric_keys if key in baseline_row and key in patch_row
                },
            })
        _write_csv(output_dir / "model_comparison.csv", comparison_rows)

    if args.attention_samples:
        _save_attention_diagnostics(
            checkpoints, table, [data_root, gt_path.parent, REPOSITORY_ROOT],
            device, bool(args.amp), output_dir / "attention_analysis", args.attention_samples,
        )
    print({
        "output_dir": str(output_dir),
        "evaluation_metrics": [row for row in metrics_rows if row["split"] == "evaluation"],
        "selected_alphas": protocol.get("selected_alphas", {}),
    }, flush=True)


def _save_attention_diagnostics(
    checkpoints: Sequence[tuple[str, Path, bool]],
    table: Any,
    roots: Sequence[Path],
    device: torch.device,
    amp: bool,
    output_dir: Path,
    sample_count: int,
) -> None:
    """Save paired attention arrays and PatchFiLM gain/modulation summaries."""
    from PIL import Image
    from upar.data import resolve_image_path
    from upar.retrieval import autocast
    from .models.attrquery_fromscratch_jointfce import load_experiment_model as load_baseline
    from .models.attrquery_patchfilm_joint import load_experiment_model as load_patchfilm
    from .transforms import build_eval_transform

    count = min(sample_count, len(table.image_paths))
    indices = np.unique(np.linspace(0, max(len(table.image_paths) - 1, 0), count, dtype=np.int64))
    summary_rows: list[dict[str, Any]] = []
    per_model: dict[str, dict[int, tuple[np.ndarray, np.ndarray | None, np.ndarray]]] = {}
    category_name_by_index: dict[int, str] = {}
    for name, checkpoint, is_patchfilm in checkpoints:
        model, _payload = (
            load_patchfilm(str(checkpoint), device) if is_patchfilm else load_baseline(checkpoint, device)
        )
        if is_patchfilm:
            category_name_by_index = {
                index: name for index, name in enumerate(model.category_names)
            }
        model.eval()
        transform = build_eval_transform(224, "resize_pad_crop")
        per_model[name] = {}
        target = output_dir / name.replace("/", "_")
        target.mkdir(parents=True, exist_ok=True)
        for index in indices:
            image_path = resolve_image_path(table.image_paths[int(index)], roots)
            with Image.open(image_path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            with torch.inference_mode(), autocast(device, amp):
                if is_patchfilm:
                    _global, local, attention, grid, gain = model.joint_features_with_patch_diagnostics(tensor)
                    modulation = 1.0 + model.patchfilm_delta * torch.tanh(gain)
                    gain_np = gain[0].float().cpu().numpy()
                    modulation_np = modulation[0].float().cpu().numpy()
                    local_np = local[0].float().cpu().numpy()
                else:
                    _global, local, attention, grid = model.joint_features_with_attention(tensor)
                    gain_np = None
                    modulation_np = None
                    local_np = local[0].float().cpu().numpy()
                attention_np = attention[0].float().cpu().numpy()
            per_model[name][int(index)] = (attention_np, gain_np, local_np)
            sample_dir = target / f"sample_{int(index):05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.save(sample_dir / "attention_heads.npy", attention_np)
            np.save(sample_dir / "attention_mean.npy", attention_np.mean(axis=0))
            if gain_np is not None and modulation_np is not None:
                np.save(sample_dir / "patchfilm_gain_raw.npy", gain_np)
                np.save(sample_dir / "patchfilm_modulation.npy", modulation_np)
                delta = modulation_np - 1.0
                np.save(sample_dir / "patchfilm_mean_abs_modulation_by_patch.npy", np.abs(delta).mean(axis=(0, 3)))
                for category_index in range(modulation_np.shape[1]):
                    values = modulation_np[:, category_index]
                    summary_rows.append({
                        "image_index": int(index), "architecture": name,
                        "category_index": category_index, "category": model.category_names[category_index],
                        "grid_h": grid[0], "grid_w": grid[1],
                        "modulation_mean": float(values.mean()),
                        "modulation_std": float(values.std()),
                        "mean_abs_delta_from_identity": float(np.abs(values - 1.0).mean()),
                        "modulation_p05": float(np.quantile(values, 0.05)),
                        "modulation_p95": float(np.quantile(values, 0.95)),
                    })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison_rows: list[dict[str, Any]] = []
    baseline_name = next((name for name, _path, patch in checkpoints if not patch), None)
    patch_name = next((name for name, _path, patch in checkpoints if patch), None)
    if baseline_name and patch_name:
        for index in indices:
            baseline_attention, _, baseline_local = per_model[baseline_name][int(index)]
            patch_attention, gain, patch_local = per_model[patch_name][int(index)]
            for category_index in range(patch_attention.shape[1]):
                for head_index in range(patch_attention.shape[0]):
                    a = baseline_attention[head_index, category_index].astype(np.float64)
                    b = patch_attention[head_index, category_index].astype(np.float64)
                    a /= max(float(a.sum()), 1e-12)
                    b /= max(float(b.sum()), 1e-12)
                    midpoint = 0.5 * (a + b)
                    js = 0.5 * np.sum(a * np.log(np.maximum(a, 1e-12) / np.maximum(midpoint, 1e-12)))
                    js += 0.5 * np.sum(b * np.log(np.maximum(b, 1e-12) / np.maximum(midpoint, 1e-12)))
                    cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
                    row = {
                        "image_index": int(index), "category_index": category_index,
                        "category": category_name_by_index.get(category_index, str(category_index)),
                        "head": head_index,
                        "attention_cosine": cosine, "attention_js_divergence": float(js),
                        "local_feature_cosine": float(
                            np.dot(baseline_local[category_index], patch_local[category_index]) /
                            max(np.linalg.norm(baseline_local[category_index]) * np.linalg.norm(patch_local[category_index]), 1e-12)
                        ),
                    }
                    if gain is not None:
                        row["patchfilm_gain_mean_abs"] = float(np.abs(gain[head_index, category_index]).mean())
                        row["patchfilm_modulation_mean_abs_delta"] = float(
                            np.abs(1.0 + 0.1 * np.tanh(gain[head_index, category_index]) - 1.0).mean()
                        )
                    comparison_rows.append(row)
    if summary_rows:
        from .attrquery_fromscratch_jointfce import write_rows_csv
        write_rows_csv(output_dir / "patchfilm_modulation_summary.csv", summary_rows)
    if comparison_rows:
        from .attrquery_fromscratch_jointfce import write_rows_csv
        write_rows_csv(output_dir / "attention_comparison.csv", comparison_rows)


if __name__ == "__main__":
    train_main()
