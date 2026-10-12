"""Training, checkpointing, and evaluation for shared CLIP ViT-B/16 E1."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..checkpoint import load_model as load_attrivision_checkpoint
from ..datasets.attribute_prompts import CategoryPromptMapper
from ..datasets.upar_abpr import AttriVisionDataset, PromptBatch, PromptCollator
from ..engine.evaluator_abpr import evaluate_native52_category_nll
from ..engine.trainer_attrivision import (
    _grad_scaler,
    _optimizer_step,
    build_scheduler,
)
from ..losses.category_local_fce import CategoryLocalFCELoss
from ..losses.focal_clip_loss import FocalCLIPLoss
from ..models.attrivision import AttriVision
from ..transforms import build_eval_transform, build_train_transform
from ..attrquery_fromscratch_jointfce import (
    category_names,
    evaluate_model,
    restore_rng,
    rng_state,
    valid_category_mask,
)
from upar.config import REPOSITORY_ROOT, choose_device, set_seed
from upar.data import find_annotation_file, read_gt_csv
from upar.retrieval import autocast
from .model import ARCHITECTURE, MODEL_NAME, SharedCLIPB16AttrQuery


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=_json_default)
    temporary.replace(path)


def write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _load_payload(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a mapping: {source}")
    return payload


def _load_a7_base(
    base: AttriVision,
    path: str | Path,
    expected_attributes: Sequence[str],
) -> dict[str, Any]:
    payload = _load_payload(path)
    if payload.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"E1-B requires an E1-A CLIP B/16 checkpoint; found model_name="
            f"{payload.get('model_name')!r}. B/32 weights cannot initialize B/16."
        )
    if payload.get("prompt_mode") != "category_complete":
        raise ValueError("E1-B requires the E1-A category_complete 52-state checkpoint")
    training_config = payload.get("training_config", {})
    if training_config.get("training_objective") != "paper_fce":
        raise ValueError("E1-B requires the A7 paper_fce objective used by E1-A")
    sidecar_path = Path(path).resolve().parent / "experiment_config.json"
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            "E1-B requires the E1-A experiment_config.json beside checkpoint_best.pth"
        )
    with sidecar_path.open(encoding="utf-8") as handle:
        sidecar = json.load(handle)
    if sidecar.get("experiment") != "E1-A" or sidecar.get("validation_protocol") != "native52_category_nll":
        raise ValueError("E1-B requires the Native52 Category-NLL E1-A protocol")
    if list(payload.get("attribute_names", [])) != list(expected_attributes):
        raise ValueError("E1-A checkpoint attribute order differs from the training annotations")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("E1-A checkpoint has no model_state_dict")
    missing, unexpected = base.load_state_dict(state, strict=False)
    allowed_missing = {"binary_head.weight", "binary_head.bias"}
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(
            f"E1-A CLIP B/16 checkpoint mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return payload


def _load_shared_model(
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[SharedCLIPB16AttrQuery, dict[str, Any]]:
    payload = _load_payload(checkpoint, map_location="cpu")
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Not a shared CLIP B/16 E1 checkpoint: {checkpoint}")
    if payload.get("model_name") != MODEL_NAME:
        raise ValueError("Shared E1 checkpoint has an unexpected CLIP model name")
    stage = str(payload.get("model_stage", ""))
    if stage not in {"local", "joint"}:
        raise ValueError(f"Unexpected shared E1 model stage: {stage!r}")
    training_config = payload.get("training_config", {})
    model = SharedCLIPB16AttrQuery(
        AttriVision(pretrained=None, model_name=MODEL_NAME),
        payload["attribute_names"],
        stage=stage,
        num_heads=int(training_config.get("num_heads", 8)),
        dropout=float(training_config.get("dropout", 0.1)),
        use_ffn=bool(training_config.get("use_ffn", True)),
    )
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Shared E1 checkpoint has no model_state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.assert_trainability()
    return model, payload


def _new_model(
    *,
    stage: str,
    init_source: str,
    init_checkpoint: str | None,
    attributes: Sequence[str],
    config: dict[str, Any],
    device: torch.device,
    resume_payload: dict[str, Any] | None = None,
) -> SharedCLIPB16AttrQuery:
    if stage not in {"local", "joint"}:
        raise ValueError("E1 shared model stage must be local or joint")
    if resume_payload is not None:
        if resume_payload.get("architecture") != ARCHITECTURE:
            raise ValueError("Resume checkpoint is not a shared CLIP B/16 E1 checkpoint")
        if resume_payload.get("model_stage") != stage:
            raise ValueError("Resume checkpoint stage differs from the requested stage")
        if list(resume_payload.get("attribute_names", [])) != list(attributes):
            raise ValueError("Resume checkpoint attribute order differs from the dataset")
        base = AttriVision(pretrained=None, model_name=MODEL_NAME)
    elif init_source == "pretrained":
        if stage != "local":
            raise ValueError("Direct pretrained initialization is only valid for E1-0/local")
        base = AttriVision(
            pretrained=str(config.get("pretrained_tag", "openai")),
            model_name=MODEL_NAME,
        )
    else:
        base = AttriVision(pretrained=None, model_name=MODEL_NAME)

    model = SharedCLIPB16AttrQuery(
        base,
        attributes,
        stage=stage,
        num_heads=int(config["num_heads"]),
        dropout=float(config["dropout"]),
        use_ffn=bool(config["use_ffn"]),
    )
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
    elif init_source == "a7":
        if not init_checkpoint:
            raise ValueError("E1-B/local requires --init-checkpoint pointing to E1-A best")
        _load_a7_base(base, init_checkpoint, attributes)
    elif init_source == "shared":
        if not init_checkpoint:
            raise ValueError("E1-C/joint requires --init-checkpoint pointing to E1-B best")
        prior_payload = _load_payload(init_checkpoint, map_location="cpu")
        if prior_payload.get("architecture") != ARCHITECTURE or prior_payload.get("model_name") != MODEL_NAME:
            raise ValueError("E1-C initialization is not a shared CLIP ViT-B/16 checkpoint")
        if prior_payload.get("model_stage") != "local" or prior_payload.get("experiment") != "E1-B":
            raise ValueError("E1-C must initialize from an E1-B local-decoder checkpoint")
        if list(prior_payload.get("attribute_names", [])) != list(attributes):
            raise ValueError("E1-B checkpoint attribute order differs from the dataset")
        model.load_state_dict(prior_payload["model_state_dict"], strict=True)
    else:
        raise ValueError(f"Unsupported initialization source: {init_source!r}")
    model.to(device)
    model.assert_trainability()
    return model


def _build_loader(
    config: dict[str, Any],
    table: Any,
    model: SharedCLIPB16AttrQuery,
    data_root: Path,
    device: torch.device,
) -> DataLoader:
    annotation_path = Path(config["train_annotation_path"])
    dataset = AttriVisionDataset(
        table,
        [data_root, annotation_path.parent, REPOSITORY_ROOT],
        build_train_transform(
            int(config["image_size"]), float(config["rotation"]),
            str(config["augmentation"]),
        ),
        config.get("max_train_samples"),
    )
    collator = PromptCollator(
        table.attribute_names,
        model.base.tokenizer,
        text_sampling="single",
        multi_attributes=1,
        prompt_mode="category_complete",
        unique_prompts=False,
        stochastic=True,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        drop_last=len(dataset) >= int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["num_workers"]) > 0,
        collate_fn=collator,
    )


def _optimizer_groups(
    model: SharedCLIPB16AttrQuery,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    decay = float(config["weight_decay"])
    groups: list[dict[str, Any]] = []
    decoder = [p for p in model.local_head.parameters() if p.requires_grad]
    if decoder:
        groups.append({
            "params": decoder,
            "lr": float(config["decoder_learning_rate"]),
            "weight_decay": decay,
            "name": "local_projector_decoder",
        })
    vision = [p for p in model.base.clip.visual.parameters() if p.requires_grad]
    if vision:
        groups.append({
            "params": vision,
            "lr": float(config["vision_learning_rate"]),
            "weight_decay": decay,
            "name": "shared_vision_encoder",
        })
    if model.base.clip.logit_scale.requires_grad:
        groups.append({
            "params": [model.base.clip.logit_scale],
            "lr": float(config["vision_learning_rate"]),
            "weight_decay": 0.0,
            "name": "global_logit_scale",
        })
    if not groups:
        raise RuntimeError("No trainable parameters are configured for this E1 stage")
    return groups


def _gradient_norm(loss: torch.Tensor, parameters: Sequence[nn.Parameter]) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [gradient.detach().float().norm() for gradient in gradients if gradient is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def _train_one_epoch(
    model: SharedCLIPB16AttrQuery,
    loader: DataLoader,
    mapper: CategoryPromptMapper,
    state_features: torch.Tensor,
    global_criterion: FocalCLIPLoss,
    local_criterion: CategoryLocalFCELoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    model.assert_trainability()
    is_joint = model.stage == "joint"
    totals = {
        "total_loss": 0.0,
        "global_loss": 0.0,
        "local_loss": 0.0,
        "global_backbone_gradient": 0.0,
        "local_backbone_gradient": 0.0,
        "samples": 0.0,
        "batches": 0.0,
    }
    visual_parameters = [p for p in model.base.clip.visual.parameters() if p.requires_grad]
    for batch_index, batch in enumerate(loader, start=1):
        if not isinstance(batch, PromptBatch):
            raise TypeError("PromptCollator must return PromptBatch")
        images = batch.images.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        semantic_targets = mapper.encode(labels)
        tokens = batch.tokens.to(device, non_blocking=True)
        selected = batch.selected_semantics.to(device, non_blocking=True)
        owners = batch.text_owners.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, bool(config["amp"])):
            global_features, local_features = model.joint_features(images)
            local_output = local_criterion(
                local_features,
                state_features,
                semantic_targets,
                valid_category_mask(semantic_targets, mapper.category_indices()),
            )
            if is_joint:
                # E1-C reuses A7's sampled-prompt image/text focal contrastive
                # objective.  The E1-A text tower remains frozen.
                with torch.no_grad():
                    sampled_text_features = model.encode_text(tokens)
                logits = model.global_logits_from_features(
                    global_features, sampled_text_features,
                )
                global_output = global_criterion(
                    logits, batch.semantic_labels.to(device, non_blocking=True),
                    selected, owners,
                )
                global_loss = global_output.loss
                total = (
                    float(config["global_loss_weight"]) * global_loss
                    + float(config["local_loss_weight"]) * local_output.loss
                )
            else:
                global_loss = global_features.new_zeros(())
                total = local_output.loss

        if batch_index == 1 and is_joint:
            if not visual_parameters:
                raise RuntimeError("E1-C has no trainable shared visual parameters")
            global_gradient_norm = _gradient_norm(global_loss, visual_parameters)
            local_gradient_norm = _gradient_norm(local_output.loss, visual_parameters)
            if global_gradient_norm <= 0.0:
                raise RuntimeError("E1-C global loss does not reach the shared CLIP vision encoder")
            if local_gradient_norm <= 0.0:
                raise RuntimeError("E1-C local loss does not reach the shared CLIP vision encoder")
            totals["global_backbone_gradient"] += global_gradient_norm
            totals["local_backbone_gradient"] += local_gradient_norm

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        if float(config["grad_clip"]) > 0:
            nn.utils.clip_grad_norm_(model.trainable_parameters(), float(config["grad_clip"]))
        if batch_index == 1:
            if not any(parameter.grad is not None for parameter in model.local_head.parameters()):
                raise RuntimeError("The local objective did not produce a decoder gradient")
            if is_joint and not any(parameter.grad is not None for parameter in visual_parameters):
                raise RuntimeError("The first E1-C update has no shared vision gradient")
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        totals["total_loss"] += float(total.detach()) * count
        totals["global_loss"] += float(global_loss.detach()) * count
        totals["local_loss"] += float(local_output.loss.detach()) * count
        totals["samples"] += count
        totals["batches"] += 1.0

    samples = max(totals.pop("samples"), 1.0)
    batches = max(totals.pop("batches"), 1.0)
    global_gradient = totals["global_backbone_gradient"]
    local_gradient = totals["local_backbone_gradient"]
    result = {key: value / samples for key, value in totals.items()}
    # These two independent gradient norms are measured on the first batch.
    result["global_backbone_gradient"] = global_gradient
    result["local_backbone_gradient"] = local_gradient
    result["train_batches"] = batches
    result["average_batch_size"] = samples / batches
    result["epoch"] = epoch
    return result


def _training_state(
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    best_score: float,
    best_epoch: int,
    stale_evaluations: int,
) -> dict[str, Any]:
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "stale_evaluations": int(stale_evaluations),
        "rng": rng_state(),
    }


def _checkpoint_payload(
    model: SharedCLIPB16AttrQuery,
    attributes: Sequence[str],
    experiment: str,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": ARCHITECTURE,
        "model_name": MODEL_NAME,
        "model_id": "openai/clip-vit-base-patch16",
        "model_stage": model.stage,
        "experiment": experiment,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "attribute_names": list(attributes),
        "state_prompt_texts": list(model.mapper.prompts),
        "state_keys": list(model.mapper.keys),
        "category_groups": [list(group) for group in model.category_groups],
        "category_names": category_names(model.mapper),
        "epoch": int(epoch),
        "metrics": metrics,
        "training_config": config,
        "vision_encoder_instances": 1,
        "visual_hidden_dim": model.visual_dim,
        "joint_embedding_dim": model.text_dim,
        "spatial_patch_tokens_at_224": 196,
        "initialization": {
            "source": config["init_source"],
            "checkpoint": config.get("init_checkpoint"),
        },
    }
    if state is not None:
        payload["training_state"] = state
    return payload


def train_shared(
    *,
    stage: str,
    init_source: str,
    init_checkpoint: str | None,
    config: dict[str, Any],
    resume: str | None = None,
) -> Path:
    if stage == "joint":
        experiment = "E1-C"
        if init_source != "shared":
            raise ValueError("E1-C requires --init-source shared and an E1-B checkpoint")
    elif init_source == "pretrained":
        experiment = "E1-0"
        if stage != "local":
            raise ValueError("E1-0 uses the frozen local-decoder stage")
    elif init_source == "a7":
        experiment = "E1-B"
        if stage != "local":
            raise ValueError("E1-B uses the frozen local-decoder stage")
    else:
        raise ValueError(f"Unsupported E1 stage/init-source pair: {stage}/{init_source}")

    device = choose_device(config["device"])
    data_root = Path(config["data_root"]).resolve()
    train_annotation = find_annotation_file(data_root, "train")
    train_table = read_gt_csv(train_annotation)
    mapper = CategoryPromptMapper(train_table.attribute_names)
    if len(mapper.keys) != 52 or len(mapper.category_indices()) != 12:
        raise RuntimeError("Training annotations do not match the established 52-state ontology")

    config = {
        **config,
        "experiment": experiment,
        "model_name": MODEL_NAME,
        "model_id": "openai/clip-vit-base-patch16",
        "model_stage": stage,
        "init_source": init_source,
        "init_checkpoint": str(Path(init_checkpoint).resolve()) if init_checkpoint else None,
        "train_annotation_path": str(train_annotation.resolve()),
        "validation_protocol": "native52_category_nll",
        "selection_metric": "mADM",
        "selection_alpha": float(config["selection_alpha"]),
        "global_objective": "paper_fce" if stage == "joint" else "frozen_global_reference",
        "local_objective": "category_local_fce",
        "text_sampling": "single",
        "contrastive_target": "multi_positive",
        "category_temperature": float(config["category_temperature"]),
    }
    output_dir = Path(config["output_dir"]).resolve()
    config["output_dir"] = str(output_dir)
    if resume:
        resume_path = Path(resume).resolve()
        if resume_path.parent != output_dir:
            raise ValueError("Resume checkpoints must come from the same experiment output directory")
        if not (output_dir / "checkpoint_best.pth").is_file():
            raise FileNotFoundError("Resume requires the matching checkpoint_best.pth in the output directory")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing E1 output directory: {output_dir}. "
            "Choose a new --output-dir or resume from its checkpoint_last.pth."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "config.json", config)
    set_seed(int(config["seed"]), bool(config.get("deterministic", False)))

    resume_payload = _load_payload(resume) if resume else None
    if resume_payload is not None:
        if resume_payload.get("experiment") != experiment:
            raise ValueError("Resume checkpoint experiment differs from the requested E1 stage")
        if resume_payload.get("training_config", {}).get("init_source") != init_source:
            raise ValueError("Resume checkpoint initialization source differs from the requested run")
        if resume_payload.get("training_config", {}).get("init_checkpoint") != config["init_checkpoint"]:
            raise ValueError("Resume checkpoint initialization path differs from the requested run")

    model = _new_model(
        stage=stage,
        init_source=init_source,
        init_checkpoint=init_checkpoint,
        attributes=train_table.attribute_names,
        config=config,
        device=device,
        resume_payload=resume_payload,
    )
    atomic_json(output_dir / "parameter_counts.json", model.parameter_counts())
    loader = _build_loader(config, train_table, model, data_root, device)
    state_tokens = model.base.tokenize(mapper.prompts).to(device)
    model.eval()
    with torch.no_grad(), autocast(device, bool(config["amp"])):
        state_features = model.encode_text(state_tokens).float()
    global_criterion = FocalCLIPLoss(
        "focal_clip", "multi_positive",
        float(config["focal_alpha"]), float(config["focal_gamma"]),
    )
    local_criterion = CategoryLocalFCELoss(
        mapper.category_indices(),
        float(config["category_temperature"]),
        float(config["focal_gamma"]),
    ).to(device)
    optimizer = torch.optim.AdamW(_optimizer_groups(model, config))
    scheduler = build_scheduler(
        optimizer,
        len(loader),
        int(config["epochs"]),
        int(config["warmup_epochs"]),
        float(config["decoder_learning_rate"]),
        float(config["min_learning_rate"]),
    )
    scaler = _grad_scaler(bool(config["amp"]) and device.type == "cuda")
    best_path = output_dir / "checkpoint_best.pth"
    last_path = output_dir / "checkpoint_last.pth"
    best_score = -math.inf
    best_epoch = 0
    stale = 0
    start_epoch = 1
    if resume_payload is not None:
        training_state = resume_payload.get("training_state")
        if not isinstance(training_state, dict):
            raise ValueError("Resume checkpoint has no optimizer/scheduler training state")
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        scaler.load_state_dict(training_state["scaler"])
        if training_state.get("rng"):
            restore_rng(training_state["rng"])
        best_score = float(training_state["best_score"])
        best_epoch = int(training_state["best_epoch"])
        stale = int(training_state["stale_evaluations"])
        start_epoch = int(resume_payload["epoch"]) + 1

    history_path = output_dir / "metrics.csv"
    rows: list[dict[str, Any]] = []
    if history_path.is_file() and resume_payload is not None:
        with history_path.open(encoding="utf-8", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        started = time.perf_counter()
        train_metrics = _train_one_epoch(
            model, loader, mapper, state_features,
            global_criterion, local_criterion, optimizer, scheduler, scaler,
            config, device, epoch,
        )
        validation_rows, _details = evaluate_model(
            model,
            train_table.attribute_names,
            data_root,
            device,
            int(config["eval_batch_size"]),
            int(config["num_workers"]),
            bool(config["amp"]),
            float(config["category_temperature"]),
            config.get("max_val_samples"),
            alphas=(float(config["selection_alpha"]),),
            progress_every=int(config.get("progress_every", 0)),
        )
        fused = next(
            row for row in validation_rows
            if row.get("score_type") == "fusion"
            and abs(float(row.get("alpha", -1.0)) - float(config["selection_alpha"])) < 1e-9
        )
        current_score = float(fused["mADM"])
        elapsed = time.perf_counter() - started
        row = {
            "experiment": experiment,
            **train_metrics,
            "val_global_mADM": next(r["mADM"] for r in validation_rows if r["score_type"] == "global"),
            "val_local_mADM": next(r["mADM"] for r in validation_rows if r["score_type"] == "local"),
            "val_fused_mADM": current_score,
            "val_fused_rank1": fused.get("rank1"),
            "val_fused_mAP": fused.get("mAP"),
            "val_fused_mINP": fused.get("mINP"),
            "selection_alpha": float(config["selection_alpha"]),
            "selection_metric": "mADM",
            "epoch_seconds": elapsed,
        }
        rows.append(row)
        write_rows_csv(history_path, rows)
        atomic_json(output_dir / "metrics_latest.json", {
            "experiment": experiment,
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_rows,
            "selection_metric": "mADM",
            "best_mADM": max(best_score, current_score),
        })
        print(
            f"{experiment} epoch={epoch:03d}/{int(config['epochs'])}: "
            f"global={train_metrics['global_loss']:.5f} "
            f"local={train_metrics['local_loss']:.5f} "
            f"val(global/local/fused mADM)="
            f"{row['val_global_mADM']:.5f}/{row['val_local_mADM']:.5f}/{current_score:.5f}",
            flush=True,
        )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                _checkpoint_payload(
                    model, train_table.attribute_names, experiment, epoch,
                    {"scores": validation_rows, "selected": fused}, config,
                ),
                best_path,
            )
        else:
            stale += 1
        atomic_torch_save(
            _checkpoint_payload(
                model, train_table.attribute_names, experiment, epoch,
                {"scores": validation_rows, "selected": fused}, config,
                _training_state(optimizer, scheduler, scaler, best_score, best_epoch, stale),
            ),
            last_path,
        )
        if stale >= int(config["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}; best mADM={best_score:.6f} at {best_epoch}", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("Training completed without a validation-selected best checkpoint")
    atomic_json(output_dir / "training_summary.json", {
        "experiment": experiment,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "best_epoch": best_epoch,
        "best_mADM": best_score,
        "selection_metric": "mADM",
        "selection_alpha": float(config["selection_alpha"]),
    })
    return best_path


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def evaluate_checkpoint(
    *,
    checkpoint: str | Path,
    experiment: str,
    data_root: str | Path,
    output_path: str | Path | None = None,
    device_name: str = "auto",
    batch_size: int = 128,
    num_workers: int = 4,
    amp: bool = True,
    category_temperature: float = 0.01,
    max_val_samples: int | None = None,
) -> dict[str, Any]:
    device = choose_device(device_name)
    source = Path(checkpoint).resolve()
    if experiment == "E1-A":
        base, payload = load_attrivision_checkpoint(source, device)
        if payload.get("model_name") != MODEL_NAME:
            raise ValueError("E1-A evaluation requires a CLIP ViT-B/16 checkpoint")
        metrics = evaluate_native52_category_nll(
            base,
            payload["attribute_names"],
            Path(data_root).resolve(),
            build_eval_transform(224, "resize_pad_crop"),
            device,
            batch_size,
            num_workers,
            amp,
            max_val_samples,
            category_temperature,
        )
        result: dict[str, Any] = {
            "experiment": "E1-A",
            "checkpoint": str(source),
            "split": "val",
            "dataset_root": str(Path(data_root).resolve()),
            "validation_annotation": str(find_annotation_file(Path(data_root).resolve(), "val")),
            "descriptor": "global",
            "validation_protocol": "native52_category_nll",
            "category_temperature": category_temperature,
            "metrics": metrics,
        }
    else:
        model, payload = _load_shared_model(source, device)
        if payload.get("experiment") != experiment:
            raise ValueError(
                f"Requested {experiment}, checkpoint contains {payload.get('experiment')!r}"
            )
        rows, _details = evaluate_model(
            model,
            payload["attribute_names"],
            Path(data_root).resolve(),
            device,
            batch_size,
            num_workers,
            amp,
            category_temperature,
            max_val_samples,
            alphas=(0.5,),
        )
        result = {
            "experiment": experiment,
            "checkpoint": str(source),
            "split": "val",
            "dataset_root": str(Path(data_root).resolve()),
            "validation_annotation": str(find_annotation_file(Path(data_root).resolve(), "val")),
            "validation_protocol": "native52_category_nll",
            "category_temperature": category_temperature,
            "selection_alpha": 0.5,
            "scores": rows,
            "model_stage": payload["model_stage"],
            "vision_encoder_instances": 1,
        }
    if output_path:
        atomic_json(Path(output_path), result)
    return result


def train_a7_b16(args: Any) -> Path:
    """Use the established A7 loop with a fresh CLIP B/16 initialization."""
    from ..cli import run as run_attrivision
    from ..cli import validate_args
    from ..engine.trainer_attrivision import train

    args.clip_model = MODEL_NAME
    args.pretrained_tag = "openai"
    args.no_pretrained = False
    args.init_checkpoint = None
    args.prompt_mode = "category_complete"
    args.training_objective = "paper_fce"
    args.text_sampling = "single"
    args.multi_attributes = 3
    args.contrastive_target = "multi_positive"
    args.augmentation = "resize_pad_crop"
    args.use_fce = True
    args.lambda_attr = 0.0
    args.category_ce_mode = "off"
    args.category_temperature = 0.01
    args.validation_protocol = "native52_category_nll"
    args.selection_metric = "mADM"
    args.retrieval_scoring = "paired_l1"
    args.freeze_text_encoder = False
    args.freeze_logit_scale = False
    args.trainable_vision_blocks = 0
    if args.output_dir == "outputs/attrivision":
        args.output_dir = "outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-A"
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if resume_path.parent != output_dir:
            raise ValueError("E1-A resume must use the same output directory as the original run")
        if not resume_path.is_file():
            raise FileNotFoundError(f"E1-A resume checkpoint does not exist: {resume_path}")
    elif args.mode != "smoke" and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing E1-A output directory: {output_dir}. "
            "Choose a new --output-dir or resume from its checkpoint_last.pth."
        )
    if args.mode != "smoke":
        atomic_json(output_dir / "experiment_config.json", {
            **vars(args),
            "experiment": "E1-A",
            "model_id": "openai/clip-vit-base-patch16",
            "initialization": "openai pretrained CLIP ViT-B/16; no B/32 weights transferred",
            "vision_encoder_instances": 1,
            "training_objective": "A7 paper_fce with category_complete 52-state prompts",
            "validation_protocol": "native52_category_nll",
            "selection_metric": "mADM",
        })
    if args.mode == "smoke":
        run_attrivision(args)
        return output_dir / "checkpoint_best.pth"
    args.mode = "train"
    return train(args)
