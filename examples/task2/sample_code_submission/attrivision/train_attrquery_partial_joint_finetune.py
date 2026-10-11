"""Train AttrQuery-PartialJointFinetune from an existing A7 checkpoint.

This script intentionally starts every condition from the same A7 best
checkpoint and initializes the category-local query/cross-attention/projection
modules from the same seed.  It never loads an AttrQuery-FrozenCLIP-LocalFCE
checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.datasets.attribute_prompts import CategoryPromptMapper  # noqa: E402
from attrivision.datasets.upar_abpr import AttriVisionDataset, PromptBatch, PromptCollator  # noqa: E402
from attrivision.engine.trainer_attrivision import (  # noqa: E402
    _grad_scaler, _optimizer_step, build_scheduler,
)
from attrivision.eval_attrquery_partial_joint_finetune import (  # noqa: E402
    _retrieval_views, _score_model,
)
from attrivision.losses.category_local_fce import CategoryLocalFCELoss  # noqa: E402
from attrivision.losses.focal_clip_loss import FocalCLIPLoss  # noqa: E402
from attrivision.models.attrquery_partial_joint_finetune import (  # noqa: E402
    AttrQueryPartialJointFinetune,
    build_model_from_a7,
    load_experiment_model,
    save_checkpoint,
)
from attrivision.transforms import build_eval_transform, build_train_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device, set_seed  # noqa: E402
from upar.data import find_annotation_file, read_gt_csv  # noqa: E402
from upar.retrieval import autocast  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "attrquery_partial_joint_finetune.json"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def _valid_category_mask(
    semantic_targets: torch.Tensor,
    groups: Sequence[Sequence[int]],
) -> torch.Tensor:
    masks = []
    for group in groups:
        indices = torch.as_tensor(group, device=semantic_targets.device, dtype=torch.long)
        masks.append(semantic_targets[:, indices].sum(dim=1) > 0)
    return torch.stack(masks, dim=1)


def _squared_norm(values: Sequence[torch.Tensor | None]) -> float:
    total = 0.0
    for value in values:
        if value is not None:
            total += float(value.detach().float().pow(2).sum())
    return math.sqrt(total)


def _gradient_norms(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[nn.Parameter],
    local_ids: set[int],
    visual_ids: set[int],
) -> dict[str, float]:
    local = [gradient for gradient, parameter in zip(gradients, parameters) if id(parameter) in local_ids]
    visual = [gradient for gradient, parameter in zip(gradients, parameters) if id(parameter) in visual_ids]
    return {
        "all": _squared_norm(gradients),
        "local_head": _squared_norm(local),
        "visual_backbone": _squared_norm(visual),
    }


def _safe_component_grads(
    loss: torch.Tensor,
    parameters: Sequence[nn.Parameter],
) -> list[torch.Tensor | None]:
    if not loss.requires_grad:
        return [None] * len(parameters)
    return list(torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True))


def _train_one_epoch(
    model: AttrQueryPartialJointFinetune,
    loader: DataLoader,
    mapper: CategoryPromptMapper,
    global_criterion: FocalCLIPLoss,
    local_criterion: CategoryLocalFCELoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    device: torch.device,
    amp: bool,
    grad_clip: float,
    global_loss_weight: float,
    local_loss_weight: float,
    gradient_diagnostics_interval: int,
) -> dict[str, float | bool]:
    model.train()
    model.assert_trainability()
    trainable = model.trainable_parameters()
    local_ids = {id(parameter) for parameter in model.local_head.parameters()}
    visual_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("base.clip.visual.") and parameter.requires_grad
    }
    totals: dict[str, float] = {
        "total_loss": 0.0,
        "global_loss": 0.0,
        "local_loss": 0.0,
        "weighted_global_loss": 0.0,
        "weighted_local_loss": 0.0,
        "global_i2t": 0.0,
        "global_t2i": 0.0,
        "local_valid_pairs": 0.0,
        "samples": 0.0,
        "batches": 0.0,
        "gradient_diagnostic_batches": 0.0,
        "global_grad_norm": 0.0,
        "global_grad_norm_local_head": 0.0,
        "global_grad_norm_visual_backbone": 0.0,
        "local_grad_norm": 0.0,
        "local_grad_norm_local_head": 0.0,
        "local_grad_norm_visual_backbone": 0.0,
        "total_grad_norm": 0.0,
        "total_grad_norm_local_head": 0.0,
        "total_grad_norm_visual_backbone": 0.0,
    }
    first_flow: dict[str, bool] | None = None

    for batch_index, batch in enumerate(loader, start=1):
        if not isinstance(batch, PromptBatch):
            raise TypeError("PromptCollator must return PromptBatch")
        images = batch.images.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        semantic_targets = batch.semantic_labels.to(device, non_blocking=True)
        tokens = batch.tokens.to(device, non_blocking=True)
        selected = batch.selected_semantics.to(device, non_blocking=True)
        text_owners = batch.text_owners.to(device, non_blocking=True)
        valid_categories = _valid_category_mask(semantic_targets, mapper.category_indices())
        optimizer.zero_grad(set_to_none=True)

        with autocast(device, amp):
            _, _, global_logits = model.global_fce_outputs(images, tokens)
            global_output = global_criterion(
                global_logits, semantic_targets, selected, text_owners,
            )
            local_features = model.local_features(images)
            local_output = local_criterion(
                local_features, model.text_prototypes,
                semantic_targets, valid_categories,
            )
            global_loss = global_output.loss
            local_loss = local_output.loss
            total = global_loss_weight * global_loss + local_loss_weight * local_loss

        if gradient_diagnostics_interval > 0 and batch_index % gradient_diagnostics_interval == 0:
            global_grads = _safe_component_grads(global_loss, trainable)
            local_grads = _safe_component_grads(local_loss, trainable)
            global_norms = _gradient_norms(global_grads, trainable, local_ids, visual_ids)
            local_norms = _gradient_norms(local_grads, trainable, local_ids, visual_ids)
            totals["global_grad_norm"] += global_norms["all"]
            totals["global_grad_norm_local_head"] += global_norms["local_head"]
            totals["global_grad_norm_visual_backbone"] += global_norms["visual_backbone"]
            totals["local_grad_norm"] += local_norms["all"]
            totals["local_grad_norm_local_head"] += local_norms["local_head"]
            totals["local_grad_norm_visual_backbone"] += local_norms["visual_backbone"]
            totals["gradient_diagnostic_batches"] += 1.0

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        total_gradients = [parameter.grad for parameter in trainable]
        total_norms = _gradient_norms(total_gradients, trainable, local_ids, visual_ids)
        totals["total_grad_norm"] += total_norms["all"]
        totals["total_grad_norm_local_head"] += total_norms["local_head"]
        totals["total_grad_norm_visual_backbone"] += total_norms["visual_backbone"]
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable, grad_clip)
        _optimizer_step(scaler, optimizer, scheduler)

        if first_flow is None:
            first_flow = {
                "global_loss_requires_grad": bool(global_loss.requires_grad),
                "local_loss_requires_grad": bool(local_loss.requires_grad),
                "total_loss_requires_grad": bool(total.requires_grad),
                "global_grad_reaches_visual": global_norms["visual_backbone"] > 0.0 if gradient_diagnostics_interval > 0 and batch_index % gradient_diagnostics_interval == 0 else False,
                "global_grad_reaches_local_head": global_norms["local_head"] > 0.0 if gradient_diagnostics_interval > 0 and batch_index % gradient_diagnostics_interval == 0 else False,
                "local_grad_reaches_visual": local_norms["visual_backbone"] > 0.0 if gradient_diagnostics_interval > 0 and batch_index % gradient_diagnostics_interval == 0 else False,
                "local_grad_reaches_local_head": local_norms["local_head"] > 0.0 if gradient_diagnostics_interval > 0 and batch_index % gradient_diagnostics_interval == 0 else False,
            }

        batch_size = len(images)
        valid_pairs = float(local_output.valid_counts.sum().detach())
        totals["total_loss"] += float(total.detach()) * batch_size
        totals["global_loss"] += float(global_loss.detach()) * batch_size
        totals["local_loss"] += float(local_loss.detach()) * max(valid_pairs, 1.0)
        totals["weighted_global_loss"] += float(global_loss.detach()) * global_loss_weight * batch_size
        totals["weighted_local_loss"] += float(local_loss.detach()) * local_loss_weight * max(valid_pairs, 1.0)
        totals["global_i2t"] += float(global_output.i2t.detach()) * batch_size
        totals["global_t2i"] += float(global_output.t2i.detach()) * batch_size
        totals["local_valid_pairs"] += valid_pairs
        totals["samples"] += batch_size
        totals["batches"] += 1.0

    samples = max(totals.pop("samples"), 1.0)
    batches = max(totals.pop("batches"), 1.0)
    valid_pairs = max(totals["local_valid_pairs"], 1.0)
    diagnostic_batches = max(totals["gradient_diagnostic_batches"], 1.0)
    result: dict[str, float | bool] = {
        "total_loss": totals.pop("total_loss") / samples,
        "global_loss": totals.pop("global_loss") / samples,
        "local_loss": totals.pop("local_loss") / valid_pairs,
        "weighted_global_loss": totals.pop("weighted_global_loss") / samples,
        "weighted_local_loss": totals.pop("weighted_local_loss") / valid_pairs,
        "global_i2t": totals.pop("global_i2t") / samples,
        "global_t2i": totals.pop("global_t2i") / samples,
        "local_valid_pairs": totals.pop("local_valid_pairs"),
        "train_batches": batches,
        "average_batch_size": samples / batches,
        "gradient_diagnostic_batches": totals.pop("gradient_diagnostic_batches"),
    }
    for key, value in list(totals.items()):
        result[key] = value / diagnostic_batches
    result.update(first_flow or {})
    return result


def _optimizer_groups(
    model: AttrQueryPartialJointFinetune,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = backbone_lr if name.startswith("base.clip.visual.") else head_lr
        use_decay = parameter.ndim >= 2 and not name.endswith("logit_scale")
        grouped.setdefault((lr, use_decay), []).append(parameter)
    return [
        {
            "params": parameters,
            "lr": lr,
            "weight_decay": weight_decay if use_decay else 0.0,
        }
        for (lr, use_decay), parameters in sorted(grouped.items(), key=lambda item: item[0])
    ]


def _condition_name(unfreeze_last_blocks: int) -> str:
    return {
        0: "FrozenBackbone-Control",
        1: "PartialFinetune-Last1",
        2: "PartialFinetune-Last2",
    }[unfreeze_last_blocks]


def _validate_config(config: dict[str, Any]) -> None:
    if config["image_size"] != 224:
        raise ValueError("ViT-B/32 requires image_size=224")
    if config["unfreeze_last_blocks"] not in {0, 1, 2}:
        raise ValueError("unfreeze_last_blocks must be 0, 1, or 2")
    positive = (
        "epochs", "batch_size", "eval_batch_size", "head_lr", "backbone_lr",
        "category_temperature", "num_heads",
    )
    if any(float(config[key]) <= 0 for key in positive):
        raise ValueError(f"Invalid non-positive training setting in {positive}")
    if config["num_workers"] < 0 or config["weight_decay"] < 0 or config["grad_clip"] < 0:
        raise ValueError("num_workers, weight_decay, and grad_clip must be non-negative")
    if config["warmup_epochs"] < 0 or config["warmup_epochs"] >= config["epochs"]:
        raise ValueError("warmup_epochs must be smaller than epochs")
    if config["global_loss_weight"] < 0 or config["local_loss_weight"] < 0:
        raise ValueError("loss weights must be non-negative")
    if config["global_loss_weight"] + config["local_loss_weight"] <= 0:
        raise ValueError("at least one joint loss weight must be positive")
    if config["focal_alpha"] < 0 or config["focal_gamma"] < 0:
        raise ValueError("focal alpha/gamma must be non-negative")
    if not 0.0 <= config["selection_alpha"] <= 1.0:
        raise ValueError("selection_alpha must be in [0,1]")
    if config["gradient_diagnostics_interval"] < 0:
        raise ValueError("gradient_diagnostics_interval must be non-negative")


def _resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        config[key] = value
    config["a7_checkpoint"] = str(Path(config["a7_checkpoint"]).resolve())
    config["data_root"] = str(Path(config["data_root"]).resolve())
    config["output_root"] = str(Path(config["output_root"]).resolve())
    if config.get("output_dir"):
        config["output_dir"] = str(Path(config["output_dir"]).resolve())
    config["amp"] = bool(config.get("amp", True))
    config["deterministic"] = bool(config.get("deterministic", False))
    _validate_config(config)
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train AttrQuery-PartialJointFinetune from A7",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--a7-checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--unfreeze-last-blocks", type=int, choices=(0, 1, 2))
    parser.add_argument("--unfreeze-vision-ln-proj", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--rotation", type=float)
    parser.add_argument("--global-loss-weight", type=float)
    parser.add_argument("--local-loss-weight", type=float)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--backbone-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--category-temperature", type=float)
    parser.add_argument("--focal-alpha", type=float)
    parser.add_argument("--focal-gamma", type=float)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--use-ffn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--retrieval-interval", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--selection-alpha", type=float)
    parser.add_argument("--gradient-diagnostics-interval", type=int)
    return parser


def _initialization_metadata(
    a7_payload: dict[str, Any],
    a7_checkpoint: Path,
    model: AttrQueryPartialJointFinetune,
) -> dict[str, Any]:
    training_config = a7_payload.get("training_config", {})
    expected = {
        "architecture": "attrivision_openclip_vit_b_32",
        "prompt_mode": "category_complete",
        "training_objective": "paper_fce",
        "text_sampling": "single",
        "contrastive_target": "multi_positive",
    }
    mismatches = {
        key: {"expected": value, "actual": a7_payload.get(key, training_config.get(key))}
        for key, value in expected.items()
        if (a7_payload.get(key, training_config.get(key)) is not None
            and a7_payload.get(key, training_config.get(key)) != value)
    }
    return {
        "source": "A7 best checkpoint only",
        "a7_checkpoint": str(a7_checkpoint.resolve()),
        "a7_architecture": a7_payload.get("architecture"),
        "a7_prompt_mode": a7_payload.get("prompt_mode"),
        "a7_training_config": training_config,
        "verification_mismatches": mismatches,
        "attrquery_frozenclip_checkpoint_loaded": False,
        "new_local_head_initialized_from_seed": True,
        "parameter_counts": model.parameter_counts(),
        "trainable_parameter_names": model.trainable_parameter_names(),
    }


def _evaluate_epoch(
    model: AttrQueryPartialJointFinetune,
    config: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    data_root = Path(config["data_root"])
    views = _retrieval_views(data_root, config.get("max_val_samples"))
    mapper = CategoryPromptMapper(views.table.attribute_names)
    groups = [np.asarray(group, dtype=np.int64) for group in mapper.category_indices()]
    epoch_cache = output_dir / "validation_cache" / f"epoch_{epoch:03d}"
    rows = _score_model(
        _condition_name(int(config["unfreeze_last_blocks"])), model, views, mapper, groups,
        data_root, epoch_cache, build_eval_transform(config["image_size"], "resize_pad_crop"),
        device, config["eval_batch_size"], config["num_workers"], config["amp"],
        config["category_temperature"], [config["selection_alpha"], 0.0, 1.0, 0.5], "partial",
    )
    selected_alpha = float(config["selection_alpha"])
    selected = next(
        row for row in rows
        if row["score_type"] == "fusion" and float(row["alpha"]) == selected_alpha
    )
    global_row = next(row for row in rows if row["score_type"] == "global-only")
    local_row = next(row for row in rows if row["score_type"] == "local-only")
    return {
        "selection_alpha": selected_alpha,
        "rank1": selected["rank1"], "rank5": selected["rank5"],
        "rank10": selected["rank10"], "map": selected["mAP"],
        "mADM": selected["mADM"], "mINP": selected["mINP"],
        "global_rank1": global_row["rank1"], "global_map": global_row["mAP"],
        "global_mADM": global_row["mADM"], "global_mINP": global_row["mINP"],
        "local_rank1": local_row["rank1"], "local_map": local_row["mAP"],
        "local_mADM": local_row["mADM"], "local_mINP": local_row["mINP"],
        "fusion_rows": rows,
    }


def train_main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = _resolve_config(args)
    set_seed(int(config["seed"]), bool(config["deterministic"]))
    device = choose_device(config.get("device", "auto"))
    data_root = Path(config["data_root"])
    train_gt = find_annotation_file(data_root, "train")
    train_table = read_gt_csv(train_gt)
    mapper = CategoryPromptMapper(train_table.attribute_names)
    condition = _condition_name(int(config["unfreeze_last_blocks"]))
    output_dir = Path(config.get("output_dir") or Path(config["output_root"]) / condition).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)
    config["condition"] = condition
    _atomic_json(output_dir / "config.json", config)

    resume_payload: dict[str, Any] | None = None
    a7_payload: dict[str, Any] | None = None
    if config.get("resume"):
        resume_path = Path(config["resume"])
        if str(resume_path) == "__LAST__":
            resume_path = output_dir / "checkpoint_last.pth"
        model, resume_payload = load_experiment_model(resume_path, device)
        if list(resume_payload["attribute_names"]) != list(train_table.attribute_names):
            raise ValueError("Resume checkpoint attribute order differs from training annotations")
        saved_config = resume_payload["training_config"]
        for key in (
            "unfreeze_last_blocks", "unfreeze_vision_ln_proj", "global_loss_weight",
            "local_loss_weight", "head_lr", "backbone_lr", "category_temperature",
        ):
            if key in saved_config and config[key] != saved_config[key]:
                raise ValueError(f"Resume configuration mismatch for {key}: {config[key]} != {saved_config[key]}")
        a7_checkpoint = Path(resume_payload["base_checkpoint"]).resolve()
    else:
        a7_checkpoint = Path(config["a7_checkpoint"]).resolve()
        if not a7_checkpoint.is_file():
            raise FileNotFoundError(f"A7 checkpoint not found: {a7_checkpoint}")
        model, a7_payload = build_model_from_a7(
            a7_checkpoint, train_table.attribute_names, device, config["amp"],
            int(config["unfreeze_last_blocks"]), bool(config["unfreeze_vision_ln_proj"]),
            int(config["num_heads"]), float(config["dropout"]), bool(config["use_ffn"]),
        )
        if a7_payload.get("architecture") != "attrivision_openclip_vit_b_32":
            raise ValueError("The initialization checkpoint is not an AttriVision A7 checkpoint")
        if a7_payload.get("prompt_mode") != "category_complete":
            raise ValueError("The initialization checkpoint must use category_complete")
        if a7_payload.get("training_config", {}).get("training_objective") not in {None, "paper_fce"}:
            raise ValueError("The initialization checkpoint is not the requested A7 paper_fce run")

    model.assert_trainability()
    counts = model.parameter_counts()
    initialization = _initialization_metadata(
        a7_payload or {
            "architecture": "attrivision_openclip_vit_b_32",
            "training_config": {"training_objective": "paper_fce"},
            "prompt_mode": "category_complete",
        },
        a7_checkpoint,
        model,
    )
    _atomic_json(output_dir / "initialization_verification.json", initialization)
    _atomic_json(output_dir / "parameter_counts.json", counts)
    print(
        f"AttrQuery-PartialJointFinetune condition={condition}; device={device}; "
        f"parameters total={counts['total']:,}, frozen={counts['frozen']:,}, "
        f"trainable={counts['trainable']:,}, visual_trainable={counts['visual_backbone_trainable']:,}, "
        f"local_head={counts['local_head_trainable']:,}", flush=True,
    )

    dataset = AttriVisionDataset(
        train_table, [data_root, train_gt.parent, REPOSITORY_ROOT],
        build_train_transform(config["image_size"], config["rotation"], "resize_pad_crop"),
        config.get("max_train_samples"),
    )
    collator = PromptCollator(
        train_table.attribute_names, model.base.tokenizer,
        text_sampling="single", multi_attributes=1,
        prompt_mode="category_complete", unique_prompts=False, stochastic=True,
    )
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        drop_last=len(dataset) >= config["batch_size"],
        num_workers=config["num_workers"], pin_memory=device.type == "cuda",
        persistent_workers=config["num_workers"] > 0, collate_fn=collator,
    )
    semantic_labels = mapper.encode(torch.from_numpy(dataset.labels.copy()))
    global_criterion = FocalCLIPLoss(
        "focal_clip", "multi_positive", config["focal_alpha"], config["focal_gamma"],
    )
    local_criterion = CategoryLocalFCELoss(
        mapper.category_indices(), config["category_temperature"], config["focal_gamma"],
    )
    trainable = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, config["head_lr"], config["backbone_lr"], config["weight_decay"]),
    )
    visual_parameter_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("base.clip.visual.") and parameter.requires_grad
    }
    scheduler = build_scheduler(
        optimizer, max(len(loader), 1), config["epochs"], config["warmup_epochs"],
        config["head_lr"], config["min_learning_rate"],
    )
    scaler = _grad_scaler(config["amp"] and device.type == "cuda")
    best_path = output_dir / "checkpoint_best.pth"
    last_path = output_dir / "checkpoint_last.pth"
    training_config = {
        **config,
        "architecture": "AttrQuery-PartialJointFinetune",
        "base_checkpoint": str(a7_checkpoint),
        "prompt_mode": "category_complete",
        "vocabulary_states": 52,
        "category_count": 12,
        "mixed_state": False,
        "multi_label": False,
        "text_sampling": "single",
        "contrastive_target": "multi_positive",
        "global_loss": "paper_fce via existing FocalCLIPLoss",
        "local_loss": "category_local_fce via existing CategoryLocalFCELoss",
        "retrieval_protocol": "native52_category_nll",
        "selection_metric": "mADM",
        "selection_alpha": config["selection_alpha"],
        "layernorm_projection_policy": (
            "visual.ln_post and visual.proj are trainable when unfreeze_last_blocks>0"
            if config["unfreeze_vision_ln_proj"] else "visual.ln_post and visual.proj remain frozen"
        ),
    }
    start_epoch = 1
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    if resume_payload is not None:
        state = resume_payload.get("training_state")
        if not isinstance(state, dict):
            raise ValueError("Resume checkpoint has no training state")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        _restore_rng(state)
        start_epoch = int(resume_payload["epoch"]) + 1
        best_score = float(state.get("best_score", -math.inf))
        best_epoch = int(state.get("best_epoch", 0))
        stale = int(state.get("stale_evaluations", 0))

    for epoch in range(start_epoch, config["epochs"] + 1):
        started = time.perf_counter()
        losses = _train_one_epoch(
            model, loader, mapper, global_criterion, local_criterion,
            optimizer, scheduler, scaler, device, config["amp"], config["grad_clip"],
            config["global_loss_weight"], config["local_loss_weight"],
            config["gradient_diagnostics_interval"],
        )
        validation = {}
        if epoch % config["retrieval_interval"] == 0 or epoch == config["epochs"]:
            validation = _evaluate_epoch(model, config, output_dir, device, epoch)
        backbone_lrs = [
            group["lr"] for group in optimizer.param_groups
            if any(id(parameter) in visual_parameter_ids for parameter in group["params"])
        ]
        head_lrs = [
            group["lr"] for group in optimizer.param_groups
            if not any(id(parameter) in visual_parameter_ids for parameter in group["params"])
        ]
        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate_head": max(head_lrs) if head_lrs else 0.0,
            "learning_rate_backbone": max(backbone_lrs) if backbone_lrs else 0.0,
            "elapsed_seconds": time.perf_counter() - started,
            **losses,
            **{key: value for key, value in validation.items() if key != "fusion_rows"},
        }
        history.append(row)
        _write_rows(output_dir / "training_metrics.csv", history)
        _atomic_json(output_dir / "metrics_latest.json", row)
        current_score = float(validation.get("mADM", -math.inf))
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                best_path, model, epoch, row, training_config, str(a7_checkpoint),
            )
        else:
            stale += 1
        save_checkpoint(
            last_path, model, epoch, row, training_config, str(a7_checkpoint),
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            best_score=best_score, best_epoch=best_epoch, stale_evaluations=stale,
        )
        _atomic_json(output_dir / "initialization_verification.json", {
            **initialization,
            "best_epoch": best_epoch,
            "best_mADM": best_score,
            "last_epoch": epoch,
        })
        print(
            f"{condition} epoch={epoch:03d}/{config['epochs']}: "
            f"total={float(row['total_loss']):.6f}, "
            f"global={float(row['global_loss']):.6f}, local={float(row['local_loss']):.6f}, "
            f"alpha={config['selection_alpha']:.2f}, "
            f"R1={100 * float(row.get('rank1', float('nan'))):.2f}%, "
            f"mAP={100 * float(row.get('map', float('nan'))):.2f}%, "
            f"mADM={100 * float(row.get('mADM', float('nan'))):.2f}%",
            flush=True,
        )
        if (
            epoch >= config.get("minimum_training_epochs", 1)
            and stale >= config["early_stopping_patience"]
        ):
            break

    if not best_path.is_file():
        raise RuntimeError("Training finished without a validation-selected checkpoint")
    _atomic_json(output_dir / "training_summary.json", {
        "condition": condition,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
        "best_epoch": best_epoch,
        "best_mADM": best_score,
        "parameter_counts": counts,
        "initialization": initialization,
    })
    print(f"Best checkpoint: {best_path} (epoch={best_epoch}, mADM={best_score:.6f})")


if __name__ == "__main__":
    train_main()
