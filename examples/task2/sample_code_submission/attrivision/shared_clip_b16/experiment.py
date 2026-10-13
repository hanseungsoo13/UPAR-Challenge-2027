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
import torch.nn.functional as F
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
    category_nll,
    evaluate_model,
    restore_rng,
    rng_state,
    valid_category_mask,
)
from upar.config import REPOSITORY_ROOT, choose_device, set_seed
from upar.data import ImagePathDataset, find_annotation_file, read_gt_csv
from upar.retrieval import autocast, official_retrieval_metrics
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
    if stage not in {"local", "joint", "full"}:
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
    if stage not in {"local", "joint", "full"}:
        raise ValueError("E1 shared model stage must be local, joint, or full")
    if resume_payload is not None:
        if resume_payload.get("architecture") != ARCHITECTURE:
            raise ValueError("Resume checkpoint is not a shared CLIP B/16 E1 checkpoint")
        if resume_payload.get("model_stage") != stage:
            raise ValueError("Resume checkpoint stage differs from the requested stage")
        if list(resume_payload.get("attribute_names", [])) != list(attributes):
            raise ValueError("Resume checkpoint attribute order differs from the dataset")
        base = AttriVision(pretrained=None, model_name=MODEL_NAME)
    elif init_source == "pretrained":
        if stage not in {"local", "full"}:
            raise ValueError("Direct pretrained initialization is valid for E1-0/local or full joint training")
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
            raise ValueError("E1-C/E1-D requires --init-checkpoint pointing to E1-B best")
        prior_payload = _load_payload(init_checkpoint, map_location="cpu")
        if prior_payload.get("architecture") != ARCHITECTURE or prior_payload.get("model_name") != MODEL_NAME:
            raise ValueError("E1-C/E1-D initialization is not a shared CLIP ViT-B/16 checkpoint")
        if prior_payload.get("model_stage") != "local" or prior_payload.get("experiment") != "E1-B":
            raise ValueError("E1-C/E1-D must initialize from an E1-B local-decoder checkpoint")
        if list(prior_payload.get("attribute_names", [])) != list(attributes):
            raise ValueError("E1-B checkpoint attribute order differs from the dataset")
        model.load_state_dict(prior_payload["model_state_dict"], strict=True)
    elif init_source == "pretrained":
        # The CLIP towers were loaded above; the local decoder keeps its fresh
        # random initialization for E1-0 and direct full-joint E1-E.
        pass
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


def _image_domain(path: str) -> str:
    return str(path).replace("\\", "/").split("/", 1)[0]


def _style_view(images: torch.Tensor) -> torch.Tensor:
    """Apply a fixed, geometry-preserving exposure/contrast style change."""
    mean = images.new_tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
    std = images.new_tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)
    rgb = (images * std + mean).clamp(0.0, 1.0)
    rgb = ((rgb - 0.5) * 1.08 + 0.5) * 0.92
    return (rgb.clamp(0.0, 1.0) - mean) / std


def _domain_retrieval_report(
    details: dict[str, Any], holdout_domain: str | None,
) -> dict[str, Any]:
    table = details["table"]
    queries = details["queries"]
    labels = details["labels"]
    ids = np.asarray(details["ids"], dtype=np.int64)
    domains = np.asarray([_image_domain(path) for path in table.image_paths], dtype=object)
    domain_names = sorted(set(domains.tolist()))
    if holdout_domain and holdout_domain not in domain_names:
        raise ValueError(
            f"holdout-domain={holdout_domain!r} was not found in validation annotations; "
            f"available domains: {domain_names}"
        )
    fused_log_probabilities = details["fusion_scores"]["alpha_0.500"]
    fused_probabilities = np.exp(fused_log_probabilities).astype(np.float32)
    mapper = CategoryPromptMapper(details["model_attribute_names"])
    groups = details["groups"]
    queries52 = mapper.encode(torch.from_numpy(queries).float()).numpy()
    distances = category_nll(queries52, fused_probabilities, groups)

    def summarize(domain: str) -> dict[str, float]:
        gallery_indices = np.flatnonzero(domains == domain).astype(np.int64)
        query_indices = np.unique(ids[gallery_indices]).astype(np.int64)
        local_id_by_global = {
            int(query_id): local_id for local_id, query_id in enumerate(query_indices)
        }
        local_gallery_ids = np.asarray(
            [local_id_by_global[int(query_id)] for query_id in ids[gallery_indices]],
            dtype=np.int64,
        )
        official = official_retrieval_metrics(
            distances[np.ix_(query_indices, gallery_indices)],
            queries[query_indices], labels[gallery_indices], local_gallery_ids,
        )
        return {
            "queries": float(len(query_indices)),
            "gallery_images": float(len(gallery_indices)),
            "rank1": float(official["Rank-1"]),
            "mAP": float(official["mAP"]),
            "mADM": float(official["mADM"]),
        }

    per_domain = {domain: summarize(domain) for domain in domain_names}
    result: dict[str, Any] = {
        "protocol": "domain-query retrieval against the same-domain gallery",
        "domains": per_domain,
        "holdout_domain": holdout_domain,
    }
    if holdout_domain:
        result["holdout"] = per_domain[holdout_domain]
        seen_domains = [domain for domain in domain_names if domain != holdout_domain]
        if seen_domains:
            result["seen_domain_macro"] = {
                key: float(np.mean([per_domain[domain][key] for domain in seen_domains]))
                for key in ("rank1", "mAP", "mADM")
            }
    return result


@torch.inference_mode()
def _attribute_query_stability(
    model: SharedCLIPB16AttrQuery,
    data_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    max_val_samples: int | None,
) -> dict[str, Any]:
    annotation_path = find_annotation_file(data_root, "val")
    table = read_gt_csv(annotation_path)
    if max_val_samples is not None:
        count = min(max_val_samples, len(table.image_paths))
        table = type(table)(table.image_paths[:count], table.labels[:count], table.attribute_names)
    loader = DataLoader(
        ImagePathDataset(
            table.image_paths,
            [data_root, annotation_path.parent, REPOSITORY_ROOT],
            build_eval_transform(224, "resize_pad_crop"),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    feature_sums: torch.Tensor | None = None
    attention_sums: torch.Tensor | None = None
    image_count = 0
    model.eval()
    for images in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device, amp):
            _, queries, _, attention = model.joint_features_with_tokens(images)
            _, styled_queries, _, styled_attention = model.joint_features_with_tokens(
                _style_view(images),
            )
        query_cosine = F.cosine_similarity(
            queries.float(), styled_queries.float(), dim=-1,
        )
        attention_cosine = F.cosine_similarity(
            attention.float(), styled_attention.float(), dim=-1,
        ).mean(dim=1)
        if feature_sums is None:
            feature_sums = torch.zeros(query_cosine.shape[-1], dtype=torch.float64)
            attention_sums = torch.zeros(attention_cosine.shape[-1], dtype=torch.float64)
        feature_sums += query_cosine.double().sum(dim=0).cpu()
        attention_sums += attention_cosine.double().sum(dim=0).cpu()
        image_count += len(images)
    if image_count == 0 or feature_sums is None or attention_sums is None:
        raise ValueError("Cannot compute query stability on an empty validation split")
    feature_by_category = (feature_sums / image_count).tolist()
    attention_by_category = (attention_sums / image_count).tolist()
    labels = category_names(model.mapper)
    return {
        "style_transform": {
            "type": "fixed exposure and contrast",
            "brightness_scale": 0.92,
            "contrast_scale": 1.08,
            "spatial_transform": "none",
        },
        "images": image_count,
        "attribute_query_cosine_mean": float(np.mean(feature_by_category)),
        "attribute_query_cosine_by_category": dict(zip(labels, feature_by_category)),
        "attention_cosine_mean": float(np.mean(attention_by_category)),
        "attention_cosine_by_category": dict(zip(labels, attention_by_category)),
        "attention_alignment": "identity; the style transform preserves the 14x14 patch coordinates",
    }


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
    text = [
        parameter
        for name, parameter in model.base.clip.named_parameters()
        if not name.startswith("visual.")
        and name != "logit_scale"
        and parameter.requires_grad
    ]
    if text:
        groups.append({
            "params": text,
            "lr": float(config["text_learning_rate"]),
            "weight_decay": decay,
            "name": "shared_text_encoder",
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
    anchor_model: SharedCLIPB16AttrQuery | None,
    loader: DataLoader,
    mapper: CategoryPromptMapper,
    state_tokens: torch.Tensor,
    state_features: torch.Tensor | None,
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
    is_full = model.stage == "full"
    trains_encoders = is_joint or is_full
    totals = {
        "total_loss": 0.0,
        "global_loss": 0.0,
        "local_loss": 0.0,
        "clip_feature_anchor_loss": 0.0,
        "feature_consistency_loss": 0.0,
        "attention_consistency_loss": 0.0,
        "global_backbone_gradient": 0.0,
        "local_backbone_gradient": 0.0,
        "clip_feature_anchor_backbone_gradient": 0.0,
        "global_text_gradient": 0.0,
        "local_text_gradient": 0.0,
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
            clip_feature_anchor_loss = global_features.new_zeros(())
            if anchor_model is not None:
                with torch.no_grad():
                    anchor_features = anchor_model.base.encode_image(images).float()
                anchor_similarity = F.cosine_similarity(
                    global_features.float(), anchor_features, dim=-1,
                )
                clip_feature_anchor_loss = (1.0 - anchor_similarity).mean()
            local_state_features = (
                model.encode_text(state_tokens).float() if is_full else state_features
            )
            if local_state_features is None:
                raise RuntimeError("Frozen local training requires cached CLIP text features")
            local_output = local_criterion(
                local_features,
                local_state_features,
                semantic_targets,
                valid_category_mask(semantic_targets, mapper.category_indices()),
            )
            if trains_encoders:
                # E1-C tunes vision while E1-D tunes both CLIP towers. Both use
                # A7's sampled-prompt image/text focal contrastive objective.
                if is_full:
                    sampled_text_features = model.encode_text(tokens)
                else:
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
                    + float(config.get("clip_feature_anchor_weight", 0.0))
                    * clip_feature_anchor_loss
                )
            else:
                global_loss = global_features.new_zeros(())
                total = local_output.loss

            feature_consistency = local_features.new_zeros(())
            attention_consistency = local_features.new_zeros(())
            if config.get("consistency_variant", "B0") in {"B1", "B2"}:
                # Disable decoder dropout for both paired views so the penalty
                # measures image-style sensitivity rather than dropout noise.
                model.eval()
                try:
                    with autocast(device, bool(config["amp"])):
                        if config.get("consistency_variant") == "B2":
                            _, original_queries, _, original_attention = model.joint_features_with_tokens(images)
                            _, styled_queries, _, styled_attention = model.joint_features_with_tokens(
                                _style_view(images),
                            )
                        else:
                            _, original_queries = model.joint_features(images)
                            _, styled_queries = model.joint_features(_style_view(images))
                        feature_consistency = 1.0 - F.cosine_similarity(
                            original_queries.float(), styled_queries.float(), dim=-1,
                        ).mean()
                        if config.get("consistency_variant") == "B2":
                            attention_consistency = 1.0 - F.cosine_similarity(
                                original_attention.float(), styled_attention.float(), dim=-1,
                            ).mean()
                finally:
                    model.train()
                total = (
                    total
                    + float(config["feature_consistency_weight"]) * feature_consistency
                    + float(config["attention_consistency_weight"]) * attention_consistency
                )

        if batch_index == 1 and trains_encoders:
            if not visual_parameters:
                raise RuntimeError("Joint E1 training has no trainable shared visual parameters")
            global_gradient_norm = _gradient_norm(global_loss, visual_parameters)
            local_gradient_norm = _gradient_norm(local_output.loss, visual_parameters)
            if global_gradient_norm <= 0.0:
                raise RuntimeError("Global loss does not reach the shared CLIP vision encoder")
            if local_gradient_norm <= 0.0:
                raise RuntimeError("Local loss does not reach the shared CLIP vision encoder")
            totals["global_backbone_gradient"] += global_gradient_norm
            totals["local_backbone_gradient"] += local_gradient_norm
            if anchor_model is not None:
                totals["clip_feature_anchor_backbone_gradient"] += _gradient_norm(
                    clip_feature_anchor_loss, visual_parameters,
                )
            if is_full:
                text_projection = getattr(model.base.clip, "text_projection", None)
                if not isinstance(text_projection, nn.Parameter):
                    raise RuntimeError("E1-D requires a trainable CLIP text projection parameter")
                global_text_gradient_norm = _gradient_norm(global_loss, [text_projection])
                local_text_gradient_norm = _gradient_norm(local_output.loss, [text_projection])
                if global_text_gradient_norm <= 0.0:
                    raise RuntimeError("E1-D global loss does not reach the CLIP text encoder")
                if local_text_gradient_norm <= 0.0:
                    raise RuntimeError("E1-D local loss does not reach the CLIP text encoder")
                totals["global_text_gradient"] += global_text_gradient_norm
                totals["local_text_gradient"] += local_text_gradient_norm

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        if float(config["grad_clip"]) > 0:
            nn.utils.clip_grad_norm_(model.trainable_parameters(), float(config["grad_clip"]))
        if batch_index == 1:
            if not any(parameter.grad is not None for parameter in model.local_head.parameters()):
                raise RuntimeError("The local objective did not produce a decoder gradient")
            if trains_encoders and not any(parameter.grad is not None for parameter in visual_parameters):
                raise RuntimeError("The first joint E1 update has no shared vision gradient")
            if is_full and not any(
                parameter.grad is not None
                for name, parameter in model.base.clip.named_parameters()
                if not name.startswith("visual.") and name != "logit_scale"
            ):
                raise RuntimeError("The first E1-D update has no CLIP text encoder gradient")
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        totals["total_loss"] += float(total.detach()) * count
        totals["global_loss"] += float(global_loss.detach()) * count
        totals["local_loss"] += float(local_output.loss.detach()) * count
        totals["clip_feature_anchor_loss"] += float(clip_feature_anchor_loss.detach()) * count
        totals["feature_consistency_loss"] += float(feature_consistency.detach()) * count
        totals["attention_consistency_loss"] += float(attention_consistency.detach()) * count
        totals["samples"] += count
        totals["batches"] += 1.0

    samples = max(totals.pop("samples"), 1.0)
    batches = max(totals.pop("batches"), 1.0)
    global_gradient = totals["global_backbone_gradient"]
    local_gradient = totals["local_backbone_gradient"]
    anchor_gradient = totals["clip_feature_anchor_backbone_gradient"]
    global_text_gradient = totals["global_text_gradient"]
    local_text_gradient = totals["local_text_gradient"]
    result = {key: value / samples for key, value in totals.items()}
    # These two independent gradient norms are measured on the first batch.
    result["global_backbone_gradient"] = global_gradient
    result["local_backbone_gradient"] = local_gradient
    result["clip_feature_anchor_backbone_gradient"] = anchor_gradient
    result["global_text_gradient"] = global_text_gradient
    result["local_text_gradient"] = local_text_gradient
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
    best_unseen_score: float = -math.inf,
    best_unseen_epoch: int = 0,
) -> dict[str, Any]:
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "stale_evaluations": int(stale_evaluations),
        "best_unseen_score": float(best_unseen_score),
        "best_unseen_epoch": int(best_unseen_epoch),
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
    consistency_variant = str(config.get("consistency_variant", "B0"))
    if consistency_variant not in {"B0", "B1", "B2"}:
        raise ValueError("consistency_variant must be B0, B1, or B2")
    feature_weight = float(config.get("feature_consistency_weight", 0.0))
    attention_weight = float(config.get("attention_consistency_weight", 0.0))
    clip_feature_anchor_weight = float(config.get("clip_feature_anchor_weight", 0.0))
    if feature_weight < 0.0 or attention_weight < 0.0:
        raise ValueError("Consistency loss weights must be non-negative")
    if not math.isfinite(clip_feature_anchor_weight) or clip_feature_anchor_weight < 0.0:
        raise ValueError("clip_feature_anchor_weight must be finite and non-negative")
    if clip_feature_anchor_weight > 0.0 and stage != "joint":
        raise ValueError("CLIP feature anchoring is only supported for E1-C joint training")
    if consistency_variant == "B0" and (feature_weight != 0.0 or attention_weight != 0.0):
        raise ValueError("B0 is the FCE-only baseline and requires zero consistency weights")
    if consistency_variant == "B1" and (feature_weight <= 0.0 or attention_weight != 0.0):
        raise ValueError("B1 requires a positive feature weight and zero attention weight")
    if consistency_variant == "B2" and (feature_weight <= 0.0 or attention_weight <= 0.0):
        raise ValueError("B2 requires positive feature and attention consistency weights")
    if stage == "joint":
        if consistency_variant != "B0":
            raise ValueError("Consistency variants are only supported for frozen local-query training")
        experiment = "E1-C"
        if init_source != "shared":
            raise ValueError("E1-C requires --init-source shared and an E1-B checkpoint")
    elif stage == "full":
        if consistency_variant != "B0":
            raise ValueError("Consistency variants are only supported for frozen local-query training")
        if init_source == "shared":
            experiment = "E1-D"
        elif init_source == "pretrained":
            experiment = "E1-E"
        else:
            raise ValueError("Full joint training requires shared E1-B or pretrained CLIP initialization")
    elif init_source == "pretrained":
        if consistency_variant != "B0":
            raise ValueError("Consistency variants require initialization from the E1-A checkpoint")
        experiment = "E1-0"
        if stage != "local":
            raise ValueError("E1-0 uses the frozen local-decoder stage")
    elif init_source == "a7":
        experiment = {"B0": "E1-B", "B1": "E1-B1", "B2": "E1-B2"}[
            consistency_variant
        ]
        if stage != "local":
            raise ValueError("E1-B consistency experiments use the frozen local-decoder stage")
    else:
        raise ValueError(f"Unsupported E1 stage/init-source pair: {stage}/{init_source}")

    device = choose_device(config["device"])
    data_root = Path(config["data_root"]).resolve()
    train_annotation = find_annotation_file(data_root, "train")
    train_table = read_gt_csv(train_annotation)
    holdout_domain = config.get("holdout_domain") if init_source in {"a7", "shared"} else None
    if holdout_domain:
        keep = np.asarray(
            [_image_domain(path) != str(holdout_domain) for path in train_table.image_paths],
            dtype=bool,
        )
        removed = int((~keep).sum())
        if removed == 0:
            available = sorted({_image_domain(path) for path in train_table.image_paths})
            raise ValueError(
                f"holdout-domain={holdout_domain!r} was not found in training annotations; "
                f"available domains: {available}"
            )
        train_table = type(train_table)(
            [path for path, selected in zip(train_table.image_paths, keep) if selected],
            train_table.labels[keep], train_table.attribute_names,
        )
        if not train_table.image_paths:
            raise ValueError("Holdout domain filtering left no training images")
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
        "global_objective": "paper_fce" if stage in {"joint", "full"} else "frozen_global_reference",
        "trainable_clip_towers": "vision_and_text" if stage == "full" else ("vision" if stage == "joint" else "frozen"),
        "local_objective": "category_local_fce",
        "text_sampling": "single",
        "contrastive_target": "multi_positive",
        "category_temperature": float(config["category_temperature"]),
        "consistency_variant": consistency_variant,
        "feature_consistency_weight": feature_weight,
        "attention_consistency_weight": attention_weight,
        "clip_feature_anchor_weight": clip_feature_anchor_weight,
        "holdout_domain": holdout_domain,
        "training_domains": sorted({_image_domain(path) for path in train_table.image_paths}),
        "training_images_after_holdout": len(train_table.image_paths),
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
        saved_config = resume_payload.get("training_config", {})
        if resume_payload.get("experiment") != experiment:
            raise ValueError("Resume checkpoint experiment differs from the requested E1 stage")
        if saved_config.get("init_source") != init_source:
            raise ValueError("Resume checkpoint initialization source differs from the requested run")
        if saved_config.get("init_checkpoint") != config["init_checkpoint"]:
            raise ValueError("Resume checkpoint initialization path differs from the requested run")
        for key, default in (
            ("consistency_variant", "B0"),
            ("feature_consistency_weight", 0.0),
            ("attention_consistency_weight", 0.0),
            ("clip_feature_anchor_weight", 0.0),
            ("holdout_domain", None),
        ):
            if saved_config.get(key, default) != config.get(key, default):
                raise ValueError(f"Resume checkpoint {key} differs from the requested run")

    model = _new_model(
        stage=stage,
        init_source=init_source,
        init_checkpoint=init_checkpoint,
        attributes=train_table.attribute_names,
        config=config,
        device=device,
        resume_payload=resume_payload,
    )
    anchor_model: SharedCLIPB16AttrQuery | None = None
    if clip_feature_anchor_weight > 0.0:
        if not init_checkpoint or init_source != "shared":
            raise ValueError("E1-C feature anchoring requires the E1-B --init-checkpoint")
        anchor_model, anchor_payload = _load_shared_model(init_checkpoint, device)
        if anchor_payload.get("experiment") != "E1-B" or anchor_payload.get("model_stage") != "local":
            raise ValueError("The feature-anchor teacher must be the E1-B frozen-encoder checkpoint")
        anchor_model.eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)
    atomic_json(output_dir / "parameter_counts.json", model.parameter_counts())
    loader = _build_loader(config, train_table, model, data_root, device)
    state_tokens = model.base.tokenize(mapper.prompts).to(device)
    model.eval()
    if stage == "full":
        state_features = None
    else:
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
    best_unseen_path = output_dir / "checkpoint_best_unseen_domain.pth"
    last_path = output_dir / "checkpoint_last.pth"
    best_score = -math.inf
    best_epoch = 0
    best_unseen_score = -math.inf
    best_unseen_epoch = 0
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
        best_unseen_score = float(training_state.get("best_unseen_score", -math.inf))
        best_unseen_epoch = int(training_state.get("best_unseen_epoch", 0))
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
            model, anchor_model, loader, mapper, state_tokens, state_features,
            global_criterion, local_criterion, optimizer, scheduler, scaler,
            config, device, epoch,
        )
        validation_rows, details = evaluate_model(
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
        domain_report = (
            _domain_retrieval_report(details, str(holdout_domain))
            if holdout_domain else None
        )
        current_unseen_score = (
            float(domain_report["holdout"]["mADM"])
            if domain_report and domain_report.get("holdout") else None
        )
        new_unseen_best = (
            current_unseen_score is not None
            and current_unseen_score > best_unseen_score
        )
        if new_unseen_best:
            best_unseen_score = float(current_unseen_score)
            best_unseen_epoch = epoch
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
            "unseen_mADM": current_unseen_score,
            "unseen_rank1": (
                float(domain_report["holdout"]["rank1"])
                if domain_report and domain_report.get("holdout") else None
            ),
            "unseen_mAP": (
                float(domain_report["holdout"]["mAP"])
                if domain_report and domain_report.get("holdout") else None
            ),
            "best_unseen_mADM": max(best_unseen_score, current_unseen_score)
            if current_unseen_score is not None else None,
            "best_unseen_epoch": best_unseen_epoch,
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
            "domain_retrieval": domain_report,
            "best_unseen_mADM": max(best_unseen_score, current_unseen_score)
            if current_unseen_score is not None else None,
            "best_unseen_epoch": best_unseen_epoch,
        })
        anchor_summary = (
            f"anchor={train_metrics['clip_feature_anchor_loss']:.5f} "
            if clip_feature_anchor_weight > 0.0 else ""
        )
        print(
            f"{experiment} epoch={epoch:03d}/{int(config['epochs'])}: "
            f"global={train_metrics['global_loss']:.5f} "
            f"local={train_metrics['local_loss']:.5f} "
            f"{anchor_summary}"
            f"val(global/local/fused mADM)="
            f"{row['val_global_mADM']:.5f}/{row['val_local_mADM']:.5f}/{current_score:.5f}"
            + (f", holdout({holdout_domain}) mADM={current_unseen_score:.5f}"
               if current_unseen_score is not None else ""),
            flush=True,
        )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            stale = 0
            atomic_torch_save(
                _checkpoint_payload(
                    model, train_table.attribute_names, experiment, epoch,
                    {"scores": validation_rows, "selected": fused,
                     "domain_retrieval": domain_report}, config,
                ),
                best_path,
            )
        else:
            stale += 1
        if new_unseen_best:
            atomic_torch_save(
                _checkpoint_payload(
                    model, train_table.attribute_names, experiment, epoch,
                    {"scores": validation_rows, "selected": fused,
                     "domain_retrieval": domain_report,
                     "checkpoint_selection": "holdout_domain_mADM"}, config,
                ),
                best_unseen_path,
            )
        atomic_torch_save(
            _checkpoint_payload(
                model, train_table.attribute_names, experiment, epoch,
                {"scores": validation_rows, "selected": fused}, config,
                _training_state(
                    optimizer, scheduler, scaler, best_score, best_epoch, stale,
                    best_unseen_score, best_unseen_epoch,
                ),
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
        "best_unseen_checkpoint": str(best_unseen_path) if best_unseen_path.is_file() else None,
        "best_unseen_epoch": best_unseen_epoch if best_unseen_epoch else None,
        "best_unseen_mADM": best_unseen_score if best_unseen_epoch else None,
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
        rows, details = evaluate_model(
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
        if experiment in {"E1-B", "E1-B1", "E1-B2", "E1-C"}:
            holdout_domain = payload.get("training_config", {}).get("holdout_domain")
            result["domain_retrieval"] = _domain_retrieval_report(
                details, str(holdout_domain) if holdout_domain else None,
            )
        if experiment in {"E1-B", "E1-B1", "E1-B2"}:
            holdout_domain = payload.get("training_config", {}).get("holdout_domain")
            fused = next(
                row for row in rows
                if row.get("score_type") == "fusion"
                and abs(float(row.get("alpha", -1.0)) - 0.5) < 1e-9
            )
            result["official_validation"] = {
                "mADM": float(fused["mADM"]),
                "Rank-1": float(fused["rank1"]),
                "mAP": float(fused["mAP"]),
                "fusion_alpha": 0.5,
            }
            result["attribute_query_stability"] = _attribute_query_stability(
                model,
                Path(data_root).resolve(),
                device,
                batch_size,
                num_workers,
                amp,
                max_val_samples,
            )
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
