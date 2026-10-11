"""Train AttrQuery-FromScratchJointFCE.

Each condition starts from the original pretrained OpenCLIP weights.  The
script never accepts an A7/AttrQuery initialization checkpoint; ``--resume``
is limited to checkpoints produced by this script itself.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.attrquery_fromscratch_jointfce import (  # noqa: E402
    ALPHAS,
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
from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.datasets.upar_abpr import AttriVisionDataset, PromptBatch, PromptCollator  # noqa: E402
from attrivision.engine.trainer_attrivision import (  # noqa: E402
    _grad_scaler,
    _optimizer_step,
    build_scheduler,
)
from attrivision.losses.category_local_fce import CategoryLocalFCELoss  # noqa: E402
from attrivision.losses.focal_clip_loss import FocalCLIPLoss  # noqa: E402
from attrivision.models.attrquery_fromscratch_jointfce import (  # noqa: E402
    AttrQueryFromScratchJointFCE,
    build_model_from_pretrained,
    checkpoint_payload,
    load_experiment_model,
    save_checkpoint,
)
from attrivision.transforms import build_train_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import autocast  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "attrquery_fromscratch_jointfce.json"
CONDITIONS = {
    (0.25, 0): "JointFCE-Lambda025",
    (0.5, 0): "JointFCE-Lambda050",
    (0.25, 5): "JointFCE-Warmup",
}


def _condition_name(weight: float, warmup: int) -> str:
    key = (round(float(weight), 2), int(warmup))
    if key not in CONDITIONS:
        if key[0] in {0.1, 0.25, 0.5} and key[1] in {0, 5}:
            suffix = "" if key[1] == 0 else "-Warmup5"
            return f"JointFCE-Lambda{int(round(key[0] * 100)):03d}{suffix}"
        raise ValueError(f"Unsupported local-loss condition: weight={weight}, warmup={warmup}")
    return CONDITIONS[key]


def _optimizer_groups(model: AttrQueryFromScratchJointFCE, config: dict[str, Any]) -> list[dict[str, Any]]:
    base_lr = float(config["learning_rate"])
    head_lr = float(config["head_learning_rate"])
    decay = float(config["weight_decay"])
    visual: list[nn.Parameter] = []
    text: list[nn.Parameter] = []
    auxiliary: list[nn.Parameter] = []
    head: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.local_head.parameters():
        if parameter.requires_grad:
            head.append(parameter)
            seen.add(id(parameter))
    for name, parameter in model.base.clip.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        (visual if name.startswith("visual.") else text).append(parameter)
        seen.add(id(parameter))
    for parameter in model.base.binary_head.parameters():
        if parameter.requires_grad and id(parameter) not in seen:
            auxiliary.append(parameter)
            seen.add(id(parameter))
    groups = []
    if visual:
        groups.append({"params": visual, "lr": base_lr, "weight_decay": decay, "name": "visual_backbone"})
    if text:
        groups.append({"params": text, "lr": base_lr, "weight_decay": decay, "name": "text_encoder"})
    if auxiliary:
        groups.append({"params": auxiliary, "lr": base_lr, "weight_decay": decay, "name": "a7_binary_head_unused"})
    if head:
        groups.append({"params": head, "lr": head_lr, "weight_decay": decay, "name": "category_query_head"})
    if not groups:
        raise RuntimeError("No trainable parameters were found")
    return groups


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    squared = [parameter.grad.detach().float().pow(2).sum() for parameter in parameters if parameter.grad is not None]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


def _loss_gradient_norm(loss: torch.Tensor, parameters: list[nn.Parameter]) -> float:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True,
    )
    squared = [gradient.detach().float().pow(2).sum() for gradient in gradients if gradient is not None]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


def _effective_local_weight(config: dict[str, Any], epoch: int) -> float:
    target = float(config["local_loss_weight"])
    warmup = int(config["local_loss_warmup_epochs"])
    if warmup <= 0:
        return target
    return target * min(max(epoch, 0) / warmup, 1.0)


def _train_one_epoch(
    model: AttrQueryFromScratchJointFCE,
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
    visual_parameters = [parameter for parameter in model.base.clip.visual.parameters() if parameter.requires_grad]
    text_parameters = [
        parameter for name, parameter in model.base.clip.named_parameters()
        if not name.startswith("visual.") and parameter.requires_grad
    ]
    totals = {
        "total_loss": 0.0,
        "global_loss": 0.0,
        "local_loss": 0.0,
        "effective_local_weight": 0.0,
        "global_backbone_grad_norm": 0.0,
        "local_backbone_grad_norm": 0.0,
        "backbone_grad_norm": 0.0,
        "local_head_grad_norm": 0.0,
        "text_grad_norm": 0.0,
        "gradient_diagnostics": 0.0,
        "flow_local_head": 0.0,
        "flow_visual_backbone": 0.0,
        "flow_text_encoder": 0.0,
        "samples": 0.0,
        "batches": 0.0,
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
            all_tokens = torch.cat([tokens, state_tokens], dim=0)
            all_text_features = model.encode_text(all_tokens)
            sampled_text = all_text_features[:len(tokens)]
            state_features = all_text_features[len(tokens):]
            global_logits = model.global_logits_from_features(image_features, sampled_text)
            global_output = global_criterion(global_logits, semantic_labels, selected, owners)
            local_output = local_criterion(
                local_features, state_features, mapper.encode(labels),
                valid_category_mask(mapper.encode(labels), groups),
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
            required = ("local_head", "visual_backbone")
            if not all(flags[name] for name in required):
                raise RuntimeError(f"Gradient flow check failed on first batch: {flags}")
            if not config["freeze_text_encoder"] and not flags["text_encoder"]:
                raise RuntimeError(f"A7-compatible text gradient flow check failed: {flags}")
        totals["backbone_grad_norm"] += _gradient_norm(visual_parameters)
        totals["local_head_grad_norm"] += _gradient_norm(list(model.local_head.parameters()))
        totals["text_grad_norm"] += _gradient_norm(text_parameters)
        totals["flow_local_head"] += float(flags["local_head"])
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
    result["global_backbone_grad_norm"] = totals["global_backbone_grad_norm"] / diagnostics
    result["local_backbone_grad_norm"] = totals["local_backbone_grad_norm"] / diagnostics
    for key in (
        "backbone_grad_norm", "local_head_grad_norm", "text_grad_norm",
        "flow_local_head", "flow_visual_backbone", "flow_text_encoder",
    ):
        result[key] = totals[key] / batches
    result["gradient_diagnostics"] = totals["gradient_diagnostics"]
    result["train_batches"] = batches
    result["average_batch_size"] = samples / batches
    result["local_loss_weight"] = local_weight
    return result


def _build_loader(config: dict[str, Any], table: Any, model: AttrQueryFromScratchJointFCE, data_root: Path, device: torch.device) -> DataLoader:
    dataset = AttriVisionDataset(
        table, [data_root, Path(config["train_annotation_parent"]), REPOSITORY_ROOT],
        build_train_transform(config["image_size"], config["rotation"], config["augmentation"]),
        config.get("max_train_samples"),
    )
    collator = PromptCollator(
        table.attribute_names, model.base.tokenizer,
        text_sampling="single", multi_attributes=1,
        prompt_mode="category_complete", unique_prompts=False, stochastic=True,
    )
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        drop_last=len(dataset) >= config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["num_workers"] > 0,
        collate_fn=collator,
    )


def _resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config[key] = value
    config["local_loss_weight"] = float(config["local_loss_weight"])
    config["local_loss_warmup_epochs"] = int(config["local_loss_warmup_epochs"])
    config["condition_name"] = _condition_name(
        config["local_loss_weight"], config["local_loss_warmup_epochs"],
    )
    if args.output_dir is None:
        config["output_dir"] = str(Path(config["output_root"]) / config["condition_name"])
    else:
        config["output_dir"] = str(Path(args.output_dir))
    config["data_root"] = str(Path(config["data_root"]).resolve())
    config["output_dir"] = str(Path(config["output_dir"]).resolve())
    config["amp"] = bool(config.get("amp", True))
    if config["local_loss_weight"] not in {0.1, 0.25, 0.5}:
        raise ValueError("local-loss-weight must be one of 0.1, 0.25, 0.5")
    if config["local_loss_warmup_epochs"] not in {0, 5}:
        raise ValueError("local-loss-warmup-epochs must be 0 or 5")
    if config["pretrained_tag"] in {None, "", "none", "None"}:
        raise ValueError("This experiment requires original pretrained CLIP weights")
    return config


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        **config,
        "architecture": "attrquery_fromscratch_jointfce",
        "initialization_source": "original_pretrained_clip",
        "a7_checkpoint_used": False,
        "attrquery_checkpoint_used": False,
        "training_objective": "paper_fce_plus_category_local_fce",
        "global_objective": "paper_fce",
        "prompt_mode": "category_complete",
        "text_sampling": "single",
        "contrastive_target": "multi_positive",
        "multi_label": False,
        "category_ce": False,
        "validation_protocol": "native52_category_nll",
        "selection_alpha": 0.5,
        "selection_metric": "mADM",
        "fusion_space": "category_log_probability",
    }


def _resume_state(
    path: Path,
    model: AttrQueryFromScratchJointFCE,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
) -> tuple[dict[str, Any], float, int, int]:
    resumed_model, payload = load_experiment_model(path, device)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train AttrQuery-FromScratchJointFCE from original pretrained CLIP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--local-loss-weight", type=float, choices=(0.1, 0.25, 0.5))
    parser.add_argument("--local-loss-warmup-epochs", type=int, choices=(0, 5))
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
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


def train_main() -> None:
    args = build_parser().parse_args()
    config = _resolved_config(args)
    if args.debug:
        config["epochs"] = min(int(config["epochs"]), 1)
        config["max_train_samples"] = config.get("max_train_samples") or 64
        config["max_val_samples"] = config.get("max_val_samples") or 64
    device = choose_device(config["device"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "config.json", config)
    set_seed(config["seed"], bool(config.get("deterministic", False)))

    train_gt = find_annotation_file(Path(config["data_root"]), "train")
    train_table = read_gt_csv(train_gt)
    config["train_annotation_parent"] = str(train_gt.parent)
    mapper = CategoryPromptMapper(train_table.attribute_names)
    protocol = protocol_validation(_training_config(config), mapper)
    atomic_json(output_dir / "protocol_validation.json", protocol)

    if config.get("resume"):
        model, payload = load_experiment_model(config["resume"], device)
        if payload["attribute_names"] != train_table.attribute_names:
            raise ValueError("Resume checkpoint attribute order differs from training annotations")
        saved_config = payload.get("training_config", {})
        if saved_config.get("condition_name") != config["condition_name"]:
            raise ValueError(
                "Resume checkpoint condition differs from the requested condition: "
                f"checkpoint={saved_config.get('condition_name')!r}, "
                f"requested={config['condition_name']!r}"
            )
    else:
        model = build_model_from_pretrained(
            train_table.attribute_names, device,
            pretrained_tag=config["pretrained_tag"],
            model_name=config["model_name"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            use_ffn=config["use_ffn"],
            freeze_text_encoder=config["freeze_text_encoder"],
            freeze_logit_scale=config["freeze_logit_scale"],
        )
        payload = None
    print(
        f"AttrQuery-FromScratchJointFCE condition={config['condition_name']}; "
        f"device={device}; pretrained={config['pretrained_tag']}; "
        f"parameters={model.parameter_counts()}", flush=True,
    )
    atomic_json(output_dir / "parameter_counts.json", model.parameter_counts())

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
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    start_epoch = 1
    if config.get("resume"):
        payload, best_score, best_epoch, stale = _resume_state(
            Path(config["resume"]), model, optimizer, scheduler, scaler, device,
        )
        start_epoch = int(payload["epoch"]) + 1

    training_config = _training_config(config)
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
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "condition": config["condition_name"],
            "elapsed_sec": elapsed,
            **train_metrics,
            "global_rank1": next(item["rank1"] for item in rows if item["score_type"] == "global"),
            "global_mAP": next(item["mAP"] for item in rows if item["score_type"] == "global"),
            "global_mADM": next(item["mADM"] for item in rows if item["score_type"] == "global"),
            "global_mINP": next(item["mINP"] for item in rows if item["score_type"] == "global"),
            "local_rank1": next(item["rank1"] for item in rows if item["score_type"] == "local"),
            "local_mAP": next(item["mAP"] for item in rows if item["score_type"] == "local"),
            "local_mADM": next(item["mADM"] for item in rows if item["score_type"] == "local"),
            "local_mINP": next(item["mINP"] for item in rows if item["score_type"] == "local"),
            "fusion_rank1": fusion["rank1"],
            "fusion_mAP": fusion["mAP"],
            "fusion_mADM": fusion["mADM"],
            "fusion_mINP": fusion["mINP"],
            "selection_alpha": 0.5,
        }
        log_rows.append(row)
        for item in rows:
            validation_rows.append({"epoch": epoch, "condition": config["condition_name"], **item})
        write_rows_csv(output_dir / "training_metrics.csv", log_rows)
        write_rows_csv(output_dir / "validation_metrics.csv", validation_rows)
        score = float(fusion["mADM"])
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                best_path,
                checkpoint_payload(model, epoch, row, training_config, best_score=best_score, best_epoch=best_epoch),
            )
            save_gallery_scores(output_dir / "best_validation_scores.npz", (
                np.asarray(details["global_scores"]),
                np.asarray(details["local_scores"]),
            ))
            np.savez_compressed(
                output_dir / "best_fusion_scores.npz", **details["fusion_scores"],
            )
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
            f"global={train_metrics['global_loss']:.6f}, "
            f"local={train_metrics['local_loss']:.6f}, "
            f"fusion@0.5 R1={100*fusion['rank1']:.2f}%, "
            f"mAP={100*fusion['mAP']:.2f}%, mADM={100*fusion['mADM']:.2f}%, "
            f"mINP={100*fusion['mINP']:.2f}%, time={elapsed:.1f}s", flush=True,
        )
        if (
            epoch >= int(config["minimum_training_epochs"])
            and stale >= int(config["early_stopping_patience"])
        ):
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
            break
    atomic_json(output_dir / "training_summary.json", {
        "condition": config["condition_name"],
        "best_epoch": best_epoch,
        "best_mADM": best_score,
        "best_checkpoint": str(best_path),
        "initialization": "original_pretrained_clip",
        "a7_checkpoint_used": False,
        "attrquery_checkpoint_used": False,
        "parameter_counts": model.parameter_counts(),
    })


if __name__ == "__main__":
    train_main()
