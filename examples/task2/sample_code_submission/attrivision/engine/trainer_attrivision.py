"""Fine-tuning loop for the AttriVision CLIP baseline."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from upar.config import REPOSITORY_ROOT, choose_device, set_seed
from upar.data import AnnotationTable, find_annotation_file, read_gt_csv
from upar.retrieval import autocast

from ..checkpoint import resume_training, save_best, save_last
from ..datasets.attribute_prompts import (
    CategoryPromptMapper, MixedCategoryPromptMapper, PaperAttributePromptMapper,
    prompts_for_attributes,
)
from ..datasets.upar_abpr import (
    AttriVisionDataset,
    PromptBatch,
    PromptCollator,
    UniquePromptBatchSampler,
    semantic_label_matrix,
)
from ..losses.focal_clip_loss import FocalCLIPLoss
from ..losses.category_structured_ce import CategoryStructuredCELoss
from ..losses.task2_hybrid_loss import Task2HybridLoss
from ..losses.mixed_state_hybrid_loss import MixedStateHybridLoss
from ..models.attrivision import AttriVision
from ..tracking import RunLogger
from ..transforms import build_eval_transform, build_train_transform
from .evaluator_abpr import (
    evaluate_abpr, evaluate_binary_head, evaluate_native52,
    evaluate_native52_category_nll, evaluate_native52_category_nll_domainwise,
)
from .evaluator_mixed import evaluate_mixed_state_nll


def _grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_scheduler(optimizer: torch.optim.Optimizer, steps_per_epoch: int,
                    epochs: int, warmup_epochs: int,
                    learning_rate: float, min_learning_rate: float
                    ) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a step-wise linear-warmup and cosine-decay schedule."""
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps - 1)
    min_ratio = min_learning_rate / learning_rate

    def lr_scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step + 1, 1) / warmup_steps
        cosine_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / cosine_steps, 0.0), 1.0)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)


def _finish_epoch_totals(totals: dict[str, float], device: torch.device) -> dict[str, float]:
    samples = totals.pop("samples")
    batches = totals.pop("batches")
    denominator = max(samples, 1.0)
    result = {key: value / denominator for key, value in totals.items()}
    result["train_batches"] = batches
    result["average_batch_size"] = samples / max(batches, 1.0)
    if device.type == "cuda":
        gib = 1024 ** 3
        result["cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / gib
        result["cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / gib
    return result


def _optimizer_step(
    scaler: Any, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    """Advance the LR schedule only when GradScaler applied the update.

    With AMP, GradScaler can skip an optimizer update after an overflow. Calling
    ``scheduler.step`` for that skipped batch triggers PyTorch's misleading
    "scheduler.step before optimizer.step" warning and consumes the first LR.
    """
    previous_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    if float(scaler.get_scale()) >= previous_scale:
        scheduler.step()


def train_one_epoch_paper(
    model: AttriVision, loader: DataLoader, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: FocalCLIPLoss, scaler: Any, device: torch.device,
    amp: bool, grad_clip: float, binary_criterion: nn.Module | None = None,
    attr_weight: float = 1.0, diagnostics: bool = False,
    category_criterion: CategoryStructuredCELoss | None = None,
    prototype_tokens: torch.Tensor | None = None,
    category_mode: str = "off", category_weight: float = 1.0,
    teacher_model: AttriVision | None = None, distill_weight: float = 0.0,
    rsc_drop_ratio: float = 0.0,
) -> dict[str, float]:
    model.train()
    if category_mode not in {"off", "only", "combined"}:
        raise ValueError(f"Unknown Category CE mode: {category_mode}")
    if category_mode != "off" and (category_criterion is None or prototype_tokens is None):
        raise ValueError("Category CE requires its criterion and all 52 prototype tokens")
    if teacher_model is not None and distill_weight <= 0:
        raise ValueError("distill_weight must be positive when a teacher is enabled")
    if not 0 <= rsc_drop_ratio < 1:
        raise ValueError("rsc_drop_ratio must be in [0, 1)")
    if rsc_drop_ratio > 0 and category_mode == "only":
        raise ValueError("RSC requires the FCE logits and cannot use category_mode=only")
    if teacher_model is not None:
        teacher_model.eval()
    totals = {"loss": 0.0, "fce": 0.0, "category_ce": 0.0, "bce": 0.0,
              "distill": 0.0, "rsc_dropped": 0.0,
              "i2t": 0.0, "t2i": 0.0, "samples": 0.0, "batches": 0.0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, batch in enumerate(loader):
        if not isinstance(batch, PromptBatch):
            raise TypeError("PromptCollator must return PromptBatch")
        images = batch.images.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        semantic_labels = batch.semantic_labels.to(device, non_blocking=True)
        tokens = batch.tokens.to(device, non_blocking=True)
        selected = batch.selected_semantics.to(device, non_blocking=True)
        text_owners = batch.text_owners.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            teacher_features = None
            rsc_dropped = 0.0
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_features = teacher_model.encode_image(images)
            if category_mode == "only":
                image_features = model.encode_image(images)
                text_features = None
                logits = None
                fce = image_features.new_zeros(())
                i2t = image_features.new_zeros(())
                t2i = image_features.new_zeros(())
            else:
                image_features, text_features, logits = model(images, tokens)
                result = criterion(logits, semantic_labels, selected, text_owners)
                if rsc_drop_ratio > 0:
                    feature_grad = torch.autograd.grad(
                        result.loss, image_features, retain_graph=True,
                        create_graph=False,
                    )[0]
                    importance = (feature_grad.detach() * image_features.detach()).abs()
                    drop_count = max(1, int(round(importance.shape[1] * rsc_drop_ratio)))
                    drop_indices = importance.topk(drop_count, dim=1, largest=True).indices
                    keep = torch.ones_like(image_features)
                    keep.scatter_(1, drop_indices, 0.0)
                    challenged_features = F.normalize(image_features * keep, dim=-1)
                    scale = model.logit_scale.exp().clamp(max=100.0)
                    challenged_logits = scale * challenged_features @ text_features.T
                    result = criterion(
                        challenged_logits, semantic_labels, selected, text_owners,
                    )
                    image_features = challenged_features
                    rsc_dropped = float((1.0 - keep).mean().detach())
                else:
                    rsc_dropped = 0.0
                fce, i2t, t2i = result.loss, result.i2t, result.t2i
            if teacher_features is not None:
                distill = 1.0 - F.cosine_similarity(
                    image_features.float(), teacher_features.float(), dim=-1,
                ).mean()
            else:
                distill = image_features.new_zeros(())
            if category_mode != "off":
                state_features = model.encode_text(prototype_tokens)
                category_ce = category_criterion(
                    image_features, state_features, semantic_labels,
                )
            else:
                category_ce = image_features.new_zeros(())
            bce = attr_weight * binary_criterion(model.binary_logits_from_features(image_features), labels.float()) if binary_criterion is not None else image_features.new_zeros(())
            total = fce + category_weight * category_ce + bce + distill_weight * distill
        if diagnostics and batch_index == 0 and category_mode != "only":
            assert logits is not None
            positive_mask = criterion.positive_mask(
                logits, semantic_labels, selected, text_owners,
            )
            per_image = positive_mask.sum(dim=1).detach().cpu().tolist()
            per_text = positive_mask.sum(dim=0).detach().cpu().tolist()
            owner_counts = torch.bincount(text_owners, minlength=len(images)).cpu().tolist()
            print("AttriVision first-batch diagnostic", flush=True)
            print(f"batch image count = {len(images)}", flush=True)
            print(f"text count = {len(tokens)}", flush=True)
            print(f"texts per image = {owner_counts}", flush=True)
            print(f"image feature shape = {list(image_features.shape)}", flush=True)
            print(f"text feature shape = {list(text_features.shape)}", flush=True)
            print(f"similarity matrix shape = {list(logits.shape)}", flush=True)
            print(f"positive count per image = {per_image}", flush=True)
            print(f"positive count per text = {per_text}", flush=True)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        totals["loss"] += float(total.detach()) * count
        totals["fce"] += float(fce.detach()) * count
        totals["category_ce"] += float(category_ce.detach()) * count
        totals["bce"] += float(bce.detach()) * count
        totals["distill"] += float(distill.detach()) * count
        totals["rsc_dropped"] += rsc_dropped * count
        totals["i2t"] += float(i2t.detach()) * count
        totals["t2i"] += float(t2i.detach()) * count
        totals["samples"] += count
        totals["batches"] += 1
    return _finish_epoch_totals(totals, device)


@torch.inference_mode()
def evaluate_paper_fce(
    model: AttriVision, loader: DataLoader, criterion: FocalCLIPLoss,
    device: torch.device, amp: bool,
) -> dict[str, float]:
    """Measure FCE on a deterministic validation prompt selection."""
    model.eval()
    totals = {"fce_loss": 0.0, "i2t_loss": 0.0, "t2i_loss": 0.0}
    samples = 0
    for batch in loader:
        if not isinstance(batch, PromptBatch):
            raise TypeError("PromptCollator must return PromptBatch")
        images = batch.images.to(device, non_blocking=True)
        tokens = batch.tokens.to(device, non_blocking=True)
        semantic_labels = batch.semantic_labels.to(device, non_blocking=True)
        selected = batch.selected_semantics.to(device, non_blocking=True)
        text_owners = batch.text_owners.to(device, non_blocking=True)
        with autocast(device, amp):
            _, _, logits = model(images, tokens)
            result = criterion(logits, semantic_labels, selected, text_owners)
        count = len(images)
        totals["fce_loss"] += float(result.loss) * count
        totals["i2t_loss"] += float(result.i2t) * count
        totals["t2i_loss"] += float(result.t2i) * count
        samples += count
    return {key: value / max(samples, 1) for key, value in totals.items()}


def train_one_epoch_hybrid(
    model: AttriVision, loader: DataLoader, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: Task2HybridLoss, scaler: Any, device: torch.device,
    amp: bool, grad_clip: float, prototype_tokens: torch.Tensor,
    mapper: CategoryPromptMapper | None, binary_criterion: nn.Module | None = None, attr_weight: float = 1.0,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0, "fce": 0.0, "bce": 0.0, "prototype": 0.0, "set": 0.0,
        "i2t": 0.0, "t2i": 0.0, "samples": 0.0, "batches": 0.0,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        semantic_labels = mapper.encode(labels) if mapper is not None else labels.bool()
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            image_features = model.encode_image(images)
            text_features = model.encode_text(prototype_tokens)
            output = criterion(
                image_features, text_features, semantic_labels, model.logit_scale,
            )
            bce = attr_weight * binary_criterion(model.binary_logits_from_features(image_features), labels.float()) if binary_criterion is not None else image_features.new_zeros(())
            total = output.loss + bce
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        totals["loss"] += float(total.detach()) * count
        totals["fce"] += float(output.loss.detach()) * count
        totals["bce"] += float(bce.detach()) * count
        totals["prototype"] += float(output.prototype.detach()) * count
        totals["set"] += float(output.set_contrastive.detach()) * count
        totals["i2t"] += float(output.i2t.detach()) * count
        totals["t2i"] += float(output.t2i.detach()) * count
        totals["samples"] += count
        totals["batches"] += 1
    return _finish_epoch_totals(totals, device)


def train_one_epoch_mixed(
    model: AttriVision, loader: DataLoader, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: MixedStateHybridLoss, scaler: Any, device: torch.device,
    amp: bool, grad_clip: float, prototype_tokens: torch.Tensor,
    mapper: MixedCategoryPromptMapper,
) -> dict[str, float]:
    """Train A7-mixed with category-local CE and multi-label BCE."""
    model.train()
    totals = {
        "loss": 0.0, "fce": 0.0, "prototype": 0.0, "set": 0.0,
        "single_ce": 0.0, "multi_bce": 0.0, "consistency": 0.0,
        "i2t": 0.0, "t2i": 0.0, "samples": 0.0, "batches": 0.0,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        semantic_labels = mapper.encode(labels)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            image_features = model.encode_image(images)
            text_features = model.encode_text(prototype_tokens)
            output = criterion(
                image_features, text_features, semantic_labels, model.logit_scale,
                labels.float(),
            )
        scaler.scale(output.loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        totals["loss"] += float(output.loss.detach()) * count
        totals["fce"] += float(output.loss.detach()) * count
        totals["prototype"] += float(output.prototype.detach()) * count
        totals["set"] += float(output.set_contrastive.detach()) * count
        totals["single_ce"] += float(output.single_ce.detach()) * count
        totals["multi_bce"] += float(output.multi_bce.detach()) * count
        totals["consistency"] += float(output.consistency.detach()) * count
        totals["i2t"] += float(output.i2t.detach()) * count
        totals["t2i"] += float(output.t2i.detach()) * count
        totals["samples"] += count
        totals["batches"] += 1
    return _finish_epoch_totals(totals, device)


def train_one_epoch_mixed_fce_aux(
    model: AttriVision, loader: DataLoader, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    fce_criterion: FocalCLIPLoss,
    prototype_criterion: MixedStateHybridLoss,
    scaler: Any, device: torch.device, amp: bool, grad_clip: float,
    prototype_tokens: torch.Tensor, prototype_weight: float,
) -> dict[str, float]:
    """Train A7-FCE-50 with a small mixed prototype CE/BCE auxiliary term.

    The sampled multi-positive FCE remains the main A7 objective.  The mixed
    prototype criterion supplies category-local CE for single-state groups and
    focal BCE for multi-state groups; its set-level term is deliberately
    disabled for this ablation.
    """
    if prototype_weight < 0:
        raise ValueError("prototype_weight cannot be negative")
    model.train()
    totals = {
        "loss": 0.0, "fce": 0.0, "prototype": 0.0, "set": 0.0,
        "single_ce": 0.0, "multi_bce": 0.0, "consistency": 0.0,
        "i2t": 0.0, "t2i": 0.0, "samples": 0.0, "batches": 0.0,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        if not isinstance(batch, PromptBatch):
            raise TypeError("PromptCollator must return PromptBatch")
        images = batch.images.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        semantic_labels = batch.semantic_labels.to(device, non_blocking=True)
        tokens = batch.tokens.to(device, non_blocking=True)
        selected = batch.selected_semantics.to(device, non_blocking=True)
        text_owners = batch.text_owners.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            image_features, _, logits = model(images, tokens)
            fce_output = fce_criterion(logits, semantic_labels, selected, text_owners)
            prototype_text_features = model.encode_text(prototype_tokens)
            prototype_output = prototype_criterion(
                image_features, prototype_text_features, semantic_labels,
                model.logit_scale, labels.float(), include_set=False,
            )
            fce = fce_output.loss
            prototype = prototype_output.prototype
            total = fce + prototype_weight * prototype
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        totals["loss"] += float(total.detach()) * count
        totals["fce"] += float(fce.detach()) * count
        totals["prototype"] += float(prototype.detach()) * count
        totals["single_ce"] += float(prototype_output.single_ce.detach()) * count
        totals["multi_bce"] += float(prototype_output.multi_bce.detach()) * count
        totals["consistency"] += float(prototype_output.consistency.detach()) * count
        totals["i2t"] += float(fce_output.i2t.detach()) * count
        totals["t2i"] += float(fce_output.t2i.detach()) * count
        totals["samples"] += count
        totals["batches"] += 1
    return _finish_epoch_totals(totals, device)


def train_one_epoch_binary(
    model: AttriVision, loader: DataLoader, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler, criterion: nn.Module,
    scaler: Any, device: torch.device, amp: bool, grad_clip: float, attr_weight: float = 1.0,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "fce": 0.0, "bce": 0.0, "i2t": 0.0, "t2i": 0.0, "samples": 0.0, "batches": 0.0}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            loss = attr_weight * criterion(model.binary_logits(images), labels.float())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        _optimizer_step(scaler, optimizer, scheduler)
        count = len(images)
        totals["loss"] += float(loss.detach()) * count
        totals["bce"] += float(loss.detach()) * count
        totals["samples"] += count; totals["batches"] += 1
    return _finish_epoch_totals(totals, device)


def _optimizer_groups(
    model: nn.Module, weight_decay: float, learning_rate: float,
    visual_learning_rate: float | None = None,
    text_learning_rate: float | None = None,
) -> list[dict[str, Any]]:
    """Build AdamW groups with optional tower-specific learning rates."""
    grouped: dict[tuple[float, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("clip.visual.") and visual_learning_rate is not None:
            group_lr = visual_learning_rate
        elif name.startswith("clip.") and text_learning_rate is not None:
            group_lr = text_learning_rate
        else:
            group_lr = learning_rate
        use_decay = parameter.ndim >= 2 and not name.endswith("logit_scale")
        grouped.setdefault((group_lr, use_decay), []).append(parameter)
    groups: list[dict[str, Any]] = []
    for (group_lr, use_decay), parameters in sorted(grouped.items(), key=lambda item: item[0]):
        groups.append({
            "params": parameters,
            "lr": group_lr,
            "weight_decay": weight_decay if use_decay else 0.0,
        })
    return groups


def _configure_trainability(model: AttriVision, args: Any) -> None:
    """Apply feasibility-study freezing without changing the FCE objective."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    if getattr(args, "freeze_text_encoder", False):
        text_prefixes = (
            "clip.token_embedding", "clip.positional_embedding", "clip.transformer",
            "clip.ln_final", "clip.text_projection",
        )
        for name, parameter in model.named_parameters():
            if name.startswith(text_prefixes):
                parameter.requires_grad_(False)

    if getattr(args, "freeze_logit_scale", False):
        model.clip.logit_scale.requires_grad_(False)

    trainable_blocks = int(getattr(args, "trainable_vision_blocks", 0))
    if trainable_blocks:
        visual = model.clip.visual
        blocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
        if blocks is None:
            raise RuntimeError("The selected CLIP visual tower has no transformer resblocks")
        if trainable_blocks > len(blocks):
            raise ValueError(
                f"trainable-vision-blocks={trainable_blocks} exceeds visual depth {len(blocks)}"
            )
        for parameter in visual.parameters():
            parameter.requires_grad_(False)
        for block in list(blocks)[-trainable_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for module_name in ("ln_post", "proj"):
            module = getattr(visual, module_name, None)
            if module is not None:
                if isinstance(module, nn.Parameter):
                    module.requires_grad_(True)
                else:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)


class _UniformWeightAverage:
    """Small SWAD-style uniform average over post-burn-in model snapshots."""

    def __init__(self, start_epoch: int) -> None:
        self.start_epoch = int(start_epoch)
        self.count = 0
        self._state: dict[str, torch.Tensor] | None = None

    def update(self, model: nn.Module, epoch: int) -> bool:
        if epoch < self.start_epoch:
            return False
        current = model.state_dict()
        if self._state is None:
            self._state = {
                key: value.detach().cpu().clone() for key, value in current.items()
            }
            self.count = 1
            return True
        self.count += 1
        for key, value in current.items():
            if self._state[key].is_floating_point():
                self._state[key].add_(
                    (value.detach().cpu() - self._state[key]) / self.count
                )
            else:
                self._state[key] = value.detach().cpu().clone()
        return True

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise RuntimeError("SWAD average is not initialized")
        return self._state


def train(args: Any) -> Path:
    set_seed(args.seed, args.deterministic)
    device = choose_device(args.device)
    data_root = Path(args.data_root).resolve()
    train_gt = find_annotation_file(data_root, "train")
    table = read_gt_csv(train_gt)
    unseen_domain = getattr(args, "unseen_domain", None)
    if unseen_domain:
        keep = np.asarray(
            [str(path).split("/", 1)[0] != unseen_domain for path in table.image_paths],
            dtype=bool,
        )
        removed = int((~keep).sum())
        if removed == 0:
            raise ValueError(
                f"unseen-domain={unseen_domain!r} was not found in training annotations"
            )
        table = AnnotationTable(
            [path for path, selected in zip(table.image_paths, keep) if selected],
            table.labels[keep], table.attribute_names,
        )
    model = AttriVision(
        pretrained=None if args.no_pretrained or args.resume or args.init_checkpoint else args.pretrained_tag,
        model_name=args.clip_model,
    ).to(device)
    if args.init_checkpoint:
        init_payload = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        init_state = {
            key: value for key, value in init_payload["model_state_dict"].items()
            if not key.startswith("binary_head.")
        }
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        if set(missing) - {"binary_head.weight", "binary_head.bias"} or unexpected:
            raise ValueError(f"Invalid init checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    teacher_model: AttriVision | None = None
    if getattr(args, "dg_method", "none") == "miro":
        if args.no_pretrained:
            raise ValueError("MIRO-inspired distillation requires the pretrained CLIP teacher")
        teacher_model = AttriVision(
            pretrained=args.pretrained_tag, model_name=args.clip_model,
        ).to(device)
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        teacher_model.eval()
    _configure_trainability(model, args)
    if getattr(args, "freeze_binary_head", False):
        for parameter in model.binary_head.parameters():
            parameter.requires_grad_(False)
    dataset = AttriVisionDataset(
        table,
        [data_root, train_gt.parent, REPOSITORY_ROOT],
        build_train_transform(
            args.image_size, args.rotation, getattr(args, "augmentation", "current"),
        ),
        args.max_train_samples,
    )
    collator = None
    sampler = None
    loader_options: dict[str, Any] = {}
    semantic_labels = semantic_label_matrix(
        torch.from_numpy(dataset.labels.copy()), table.attribute_names, args.prompt_mode,
    )
    if args.prompt_mode == "category_complete":
        mapper = CategoryPromptMapper(table.attribute_names)
    elif args.prompt_mode == "mixed_category":
        mapper = MixedCategoryPromptMapper(table.attribute_names)
    else:
        mapper = None
    if args.prompt_mode == "paper_binary":
        prototype_prompts = PaperAttributePromptMapper(table.attribute_names).prompts
    else:
        prototype_prompts = mapper.prompts if mapper is not None else prompts_for_attributes(table.attribute_names)
    category_mode = getattr(args, "category_ce_mode", "off")
    prototype_tokens = (
        model.tokenize(prototype_prompts).to(device)
        if args.use_fce or category_mode != "off" else None
    )
    if args.use_fce and args.training_objective in {"paper_fce", "a7_fce_mixed_aux"}:
        # For paper_binary, "unique" means each image owns a distinct
        # sampled (image, attribute-state) pair.  The old global-state sampler
        # is intentionally not used because the same negative state is valid
        # for many images and globally forbidding it would shrink batches.
        collator_unique = args.unique_prompts and args.prompt_mode != "paper_binary"
        collator = PromptCollator(
            table.attribute_names, model.tokenizer, args.text_sampling, args.multi_attributes,
            args.prompt_mode, collator_unique,
        )
    if (
        args.use_fce and args.training_objective in {"paper_fce", "a7_fce_mixed_aux"}
        and args.unique_prompts and args.prompt_mode != "paper_binary"
    ):
        prompts_per_image = 1 if args.text_sampling == "single" else args.multi_attributes
        sampler = UniquePromptBatchSampler(
            semantic_labels, args.batch_size, prompts_per_image, args.seed,
        )
        collator.semantic_frequencies = sampler.frequencies
        loader_options["batch_sampler"] = sampler
    else:
        loader_options.update({
            "batch_size": args.batch_size,
            "shuffle": True,
            "generator": torch.Generator().manual_seed(args.seed),
            "drop_last": len(dataset) >= args.batch_size,
        })
    loader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
        **loader_options,
    )
    validation_fce_loader = None
    if args.use_fce and args.training_objective in {"paper_fce", "a7_fce_mixed_aux"}:
        val_gt = find_annotation_file(data_root, "val")
        val_table = read_gt_csv(val_gt)
        val_dataset = AttriVisionDataset(
            val_table, [data_root, val_gt.parent, REPOSITORY_ROOT],
            build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), args.max_val_samples,
        )
        validation_fce_loader = DataLoader(
            val_dataset, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=PromptCollator(
                val_table.attribute_names, model.tokenizer, args.text_sampling,
                args.multi_attributes, args.prompt_mode, False, stochastic=False,
            ),
        )
    criterion: FocalCLIPLoss | Task2HybridLoss | MixedStateHybridLoss | None
    aux_prototype_criterion: MixedStateHybridLoss | None = None
    if args.use_fce and args.training_objective == "a7_mixed":
        if not isinstance(mapper, MixedCategoryPromptMapper):
            raise ValueError("a7_mixed requires prompt_mode=mixed_category")
        criterion = MixedStateHybridLoss(
            semantic_labels.float().mean(dim=0),
            [(kind, indices) for _, kind, indices in mapper.category_specs()],
            args.focal_alpha, args.focal_gamma, args.balance_max_weight,
            args.prototype_loss_weight, args.set_loss_weight,
            getattr(args, "mixed_consistency_weight", 0.1),
            getattr(args, "mixed_set_positive", "exact"),
            getattr(args, "mixed_min_shared_categories", 8),
            getattr(args, "mixed_set_beta", 4.0),
        ).to(device)
    elif args.use_fce and args.training_objective == "a7_fce_mixed_aux":
        if not isinstance(mapper, MixedCategoryPromptMapper):
            raise ValueError("a7_fce_mixed_aux requires prompt_mode=mixed_category")
        criterion = FocalCLIPLoss(
            args.loss, args.contrastive_target, args.focal_alpha, args.focal_gamma,
        )
        aux_prototype_criterion = MixedStateHybridLoss(
            semantic_labels.float().mean(dim=0),
            [(kind, indices) for _, kind, indices in mapper.category_specs()],
            args.focal_alpha, args.focal_gamma, args.balance_max_weight,
            prototype_weight=1.0, set_weight=0.0,
            consistency_weight=getattr(args, "mixed_consistency_weight", 0.1),
            set_positive_mode=getattr(args, "mixed_set_positive", "exact"),
            min_shared_categories=getattr(args, "mixed_min_shared_categories", 8),
            beta=getattr(args, "mixed_set_beta", 4.0),
        ).to(device)
    elif args.use_fce and args.training_objective == "task2_hybrid":
        criterion = Task2HybridLoss(
            semantic_labels.float().mean(dim=0), args.focal_alpha, args.focal_gamma,
            args.balance_max_weight, args.prototype_loss_weight, args.set_loss_weight,
        ).to(device)
    elif args.use_fce:
        criterion = FocalCLIPLoss(
            args.loss, args.contrastive_target, args.focal_alpha, args.focal_gamma,
        )
    category_criterion = (
        CategoryStructuredCELoss(
            mapper.category_indices(), getattr(args, "category_temperature", 0.01),
        ).to(device)
        if category_mode != "off" and mapper is not None else None
    )
    if category_mode != "off" and not isinstance(mapper, CategoryPromptMapper):
        raise ValueError("Category CE requires prompt_mode=category_complete")
    positive_rates = torch.from_numpy(dataset.labels).float().mean(dim=0).clamp_min(1e-6)
    pos_weight = ((1.0 - positive_rates) / positive_rates).to(device)
    binary_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    training_config = {
        "model_name": args.clip_model,
        "training_objective": args.training_objective,
        "prompt_mode": args.prompt_mode,
        "text_sampling": args.text_sampling,
        "multi_attributes": args.multi_attributes,
        "unique_prompts": args.unique_prompts,
        "loss": args.loss,
        "contrastive_target": args.contrastive_target,
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "prototype_loss_weight": args.prototype_loss_weight,
        "set_loss_weight": args.set_loss_weight,
        "mixed_consistency_weight": getattr(args, "mixed_consistency_weight", 0.1),
        "mixed_set_positive": getattr(args, "mixed_set_positive", "exact"),
        "mixed_min_shared_categories": getattr(args, "mixed_min_shared_categories", 8),
        "mixed_set_beta": getattr(args, "mixed_set_beta", 4.0),
        "mixed_aux_prototype_weight": getattr(args, "mixed_aux_prototype_weight", 0.1),
        "balance_max_weight": args.balance_max_weight,
        "use_fce": args.use_fce,
        "lambda_attr": args.lambda_attr,
        "visual_learning_rate": getattr(args, "visual_learning_rate", None),
        "text_learning_rate": getattr(args, "text_learning_rate", None),
        "freeze_text_encoder": getattr(args, "freeze_text_encoder", False),
        "freeze_logit_scale": getattr(args, "freeze_logit_scale", False),
        "trainable_vision_blocks": getattr(args, "trainable_vision_blocks", 0),
        "category_ce_mode": category_mode,
        "category_ce_weight": getattr(args, "category_ce_weight", 1.0),
        "category_temperature": getattr(args, "category_temperature", 0.01),
        "freeze_binary_head": getattr(args, "freeze_binary_head", False),
        "augmentation": getattr(args, "augmentation", "current"),
        "selection_metric": getattr(args, "selection_metric", "map"),
        "paper_faithful": bool(getattr(args, "paper_faithful", False)),
        "dg_method": getattr(args, "dg_method", "none"),
        "distill_weight": getattr(args, "distill_weight", 0.0),
        "rsc_drop_ratio": getattr(args, "rsc_drop_ratio", 0.0),
        "swad_start_epoch": getattr(args, "swad_start_epoch", 3),
        "domain_wise_validation": getattr(args, "domain_wise_validation", False),
        "unseen_domain": unseen_domain,
        "positive_rates": positive_rates.tolist(),
    }
    optimizer = torch.optim.AdamW(
        _optimizer_groups(
            model, args.weight_decay, args.learning_rate,
            getattr(args, "visual_learning_rate", None),
            getattr(args, "text_learning_rate", None),
        ),
    )
    scheduler = build_scheduler(
        optimizer, len(loader), args.epochs, args.warmup_epochs,
        args.learning_rate, args.min_learning_rate,
    )
    scaler = _grad_scaler(args.amp and device.type == "cuda")
    output_dir = Path(args.output_dir)
    best_path = output_dir / "checkpoint_best.pth"
    last_path = output_dir / "checkpoint_last.pth"
    logger = RunLogger(
        output_dir, resume=bool(args.resume),
        csv_filename=getattr(args, "training_log_filename", "metrics.csv"),
    )
    best_map = -math.inf
    best_epoch = 0
    stale = 0
    selection_key = getattr(args, "selection_metric", "mADM")
    start_epoch = 1

    if args.resume:
        payload = resume_training(args.resume, model, optimizer, scheduler, scaler, device)
        if payload.get("model_name") != args.clip_model:
            raise ValueError(
                f"Resume model mismatch: checkpoint={payload.get('model_name')}, "
                f"requested={args.clip_model}"
            )
        if payload["attribute_names"] != table.attribute_names:
            raise ValueError("Resume checkpoint attribute order differs from training annotations")
        if payload.get("prompt_mode", "binary_positive") != args.prompt_mode:
            raise ValueError("Resume checkpoint prompt mode differs from --prompt-mode")
        if payload.get("training_config") != training_config:
            raise ValueError("Resume checkpoint training objective differs from current arguments")
        state = payload["training_state"]
        previous_selection_metric = state.get(
            "selection_metric",
            payload.get("training_config", {}).get("selection_metric"),
        )
        if previous_selection_metric and previous_selection_metric != selection_key:
            raise ValueError(
                f"Resume checkpoint selected {previous_selection_metric!r}, "
                f"but the current run selects {selection_key!r}; start a fresh output directory."
            )
        best_map = float(state.get("best_score", state["best_map"]))
        best_epoch = int(state["best_epoch"])
        stale = int(state["stale_evaluations"])
        start_epoch = int(payload["epoch"]) + 1
        if sampler is not None:
            sampler.epoch = start_epoch - 1
        if not best_path.is_file():
            raise FileNotFoundError(
                f"Best checkpoint is missing: {best_path}. Resume with its original output directory."
            )

    swad_average = (
        _UniformWeightAverage(getattr(args, "swad_start_epoch", 3))
        if getattr(args, "dg_method", "none") == "swad" else None
    )
    swad_eval_model = copy.deepcopy(model).to(device) if swad_average is not None else None
    if swad_eval_model is not None:
        for parameter in swad_eval_model.parameters():
            parameter.requires_grad_(False)

    logger.log(json.dumps({**vars(args), "resolved_device": str(device)}, indent=2, default=str))
    logger.log(
        f"Initialization checkpoint: {args.init_checkpoint if args.init_checkpoint else 'pretrained/default'}; "
        f"binary_head={'newly_initialized' if args.init_checkpoint else 'newly_initialized'}; "
        f"use_fce={args.use_fce}, lambda_attr={args.lambda_attr:g}"
    )
    logger.log(
        f"Feasibility freezing: freeze_text={getattr(args, 'freeze_text_encoder', False)}, "
        f"freeze_logit_scale={getattr(args, 'freeze_logit_scale', False)}, "
        f"trainable_vision_blocks={getattr(args, 'trainable_vision_blocks', 0)}, "
        f"visual_lr={getattr(args, 'visual_learning_rate', None)}, "
        f"text_lr={getattr(args, 'text_learning_rate', None)}"
    )
    logger.log(
        f"DG ablation: method={getattr(args, 'dg_method', 'none')}, "
        f"distill_weight={getattr(args, 'distill_weight', 0.0):g}, "
        f"rsc_drop_ratio={getattr(args, 'rsc_drop_ratio', 0.0):g}, "
        f"swad_start_epoch={getattr(args, 'swad_start_epoch', 3)}, "
        f"domain_wise_validation={getattr(args, 'domain_wise_validation', False)}"
    )
    logger.log(
        f"Training samples={len(dataset)}, attributes={len(table.attribute_names)}, "
        f"unseen_domain={unseen_domain or 'none'}"
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    clip_parameters = sum(parameter.numel() for parameter in model.clip.parameters())
    visual_parameters = sum(parameter.numel() for parameter in model.clip.visual.parameters())
    text_parameters = clip_parameters - visual_parameters
    binary_head_parameters = sum(parameter.numel() for parameter in model.binary_head.parameters())
    optimizer_parameter_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    binary_head_in_optimizer = all(
        id(parameter) in optimizer_parameter_ids for parameter in model.binary_head.parameters()
    )
    if getattr(args, "freeze_binary_head", False) and binary_head_in_optimizer:
        raise RuntimeError("Frozen binary_head parameters must not be in the optimizer")
    if not getattr(args, "freeze_binary_head", False) and not binary_head_in_optimizer:
        raise RuntimeError("binary_head parameters are missing from the optimizer")
    logger.log(
        f"Parameters: trainable={trainable_parameters:,}/{total_parameters:,}, "
        f"clip={clip_parameters:,}, visual={visual_parameters:,}, "
        f"text_and_scale={text_parameters:,}, binary_head={binary_head_parameters:,}, "
        f"binary_head_in_optimizer={binary_head_in_optimizer}; "
        f"optimizer_tensors={sum(len(group['params']) for group in optimizer.param_groups)}"
    )
    if sampler is not None:
        logger.log(
            f"Paper-style unique-prompt batches enabled: requested_batch={args.batch_size}, "
            f"first_epoch_batches={len(sampler)}"
        )
    if args.training_objective == "task2_hybrid":
        logger.log(
            f"Task2 hybrid objective: semantic_prototypes={semantic_labels.shape[1]}, "
            f"prototype_weight={args.prototype_loss_weight:g}, "
            f"set_weight={args.set_loss_weight:g}, "
            f"balance_cap={args.balance_max_weight:g}"
        )
    if args.training_objective == "a7_mixed":
        logger.log(
            f"A7-mixed objective: semantic_prototypes={semantic_labels.shape[1]}, "
            f"prototype_weight={args.prototype_loss_weight:g}, "
            f"set_weight={args.set_loss_weight:g}, "
            f"consistency_weight={getattr(args, 'mixed_consistency_weight', 0.1):g}, "
            f"set_positive={getattr(args, 'mixed_set_positive', 'exact')}, "
            f"min_shared_categories={getattr(args, 'mixed_min_shared_categories', 8)}, "
            f"set_beta={getattr(args, 'mixed_set_beta', 4.0):g}"
        )
    if args.training_objective == "a7_fce_mixed_aux":
        logger.log(
            f"A7-FCE-50 auxiliary objective: main=focal_clip/multi_positive, "
            f"mixed_prototype_weight={getattr(args, 'mixed_aux_prototype_weight', 0.1):g}, "
            "mixed_set_weight=0, "
            f"consistency_weight={getattr(args, 'mixed_consistency_weight', 0.1):g}"
        )
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            started = time.time()
            if not args.use_fce:
                losses = train_one_epoch_binary(
                    model, loader, optimizer, scheduler, binary_criterion, scaler,
                    device, args.amp, args.grad_clip, args.lambda_attr,
                )
            elif args.training_objective == "a7_fce_mixed_aux":
                assert isinstance(criterion, FocalCLIPLoss)
                assert isinstance(aux_prototype_criterion, MixedStateHybridLoss)
                assert isinstance(mapper, MixedCategoryPromptMapper)
                assert prototype_tokens is not None
                losses = train_one_epoch_mixed_fce_aux(
                    model, loader, optimizer, scheduler, criterion,
                    aux_prototype_criterion, scaler, device, args.amp,
                    args.grad_clip, prototype_tokens,
                    getattr(args, "mixed_aux_prototype_weight", 0.1),
                )
            elif args.training_objective == "a7_mixed":
                assert isinstance(criterion, MixedStateHybridLoss)
                assert isinstance(mapper, MixedCategoryPromptMapper)
                assert prototype_tokens is not None
                losses = train_one_epoch_mixed(
                    model, loader, optimizer, scheduler, criterion, scaler, device,
                    args.amp, args.grad_clip, prototype_tokens, mapper,
                )
            elif args.training_objective == "task2_hybrid":
                assert isinstance(criterion, Task2HybridLoss)
                losses = train_one_epoch_hybrid(
                    model, loader, optimizer, scheduler, criterion, scaler, device,
                    args.amp, args.grad_clip, prototype_tokens, mapper,
                    binary_criterion if args.lambda_attr > 0 else None, args.lambda_attr,
                )
            else:
                assert isinstance(criterion, FocalCLIPLoss)
                losses = train_one_epoch_paper(
                    model, loader, optimizer, scheduler, criterion, scaler, device,
                    args.amp, args.grad_clip,
                    binary_criterion if args.lambda_attr > 0 else None, args.lambda_attr,
                    diagnostics=(
                        bool(getattr(args, "batch_diagnostics", False))
                        and epoch == start_epoch
                    ),
                    category_criterion=category_criterion,
                    prototype_tokens=prototype_tokens,
                    category_mode=category_mode,
                    category_weight=getattr(args, "category_ce_weight", 1.0),
                    teacher_model=teacher_model,
                    distill_weight=getattr(args, "distill_weight", 0.0),
                    rsc_drop_ratio=(
                        getattr(args, "rsc_drop_ratio", 0.0)
                        if getattr(args, "dg_method", "none") == "rsc" else 0.0
                    ),
                )
            metrics: dict[str, float] = {}
            should_stop = False
            if epoch % args.retrieval_interval == 0 or epoch == args.epochs:
                evaluation_model = model
                if swad_average is not None and swad_average.update(model, epoch):
                    assert swad_eval_model is not None
                    swad_eval_model.load_state_dict(swad_average.state_dict())
                    evaluation_model = swad_eval_model
                if getattr(args, "validation_protocol", "binary_head") == "native52_category_nll":
                    if getattr(args, "domain_wise_validation", False):
                        metrics = evaluate_native52_category_nll_domainwise(
                            evaluation_model, table.attribute_names, data_root,
                            build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), device,
                            args.eval_batch_size, args.num_workers, args.amp,
                            args.max_val_samples, getattr(args, "category_temperature", 0.01),
                            unseen_domain=getattr(args, "unseen_domain", None),
                        )
                    else:
                        metrics = evaluate_native52_category_nll(
                            evaluation_model, table.attribute_names, data_root,
                            build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), device,
                            args.eval_batch_size, args.num_workers, args.amp,
                            args.max_val_samples, getattr(args, "category_temperature", 0.01),
                        )
                elif getattr(args, "validation_protocol", "binary_head") == "mixed_state_nll":
                    metrics = evaluate_mixed_state_nll(
                        evaluation_model, table.attribute_names, data_root,
                        build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), device,
                        args.eval_batch_size, args.num_workers, args.amp,
                        args.max_val_samples, getattr(args, "category_temperature", 0.01),
                    )
                elif getattr(args, "validation_protocol", "binary_head") == "native52":
                    metrics = evaluate_native52(
                        evaluation_model, table.attribute_names, data_root,
                        build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), device,
                        args.eval_batch_size, args.num_workers, args.amp,
                        args.max_val_samples, args.attribute_temperature,
                    )
                elif getattr(args, "validation_protocol", "binary_head") == "paired_l1":
                    metrics = evaluate_abpr(
                        evaluation_model, table.attribute_names, data_root,
                        build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), device,
                        args.eval_batch_size, args.num_workers, args.amp,
                        args.max_val_samples, args.prompt_mode,
                        "paired_l1", args.attribute_temperature,
                    )
                else:
                    metrics = evaluate_binary_head(
                        evaluation_model, table.attribute_names, data_root,
                        build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), device,
                        args.eval_batch_size, args.num_workers, args.amp,
                        args.max_val_samples,
                    )
                if validation_fce_loader is not None:
                    assert isinstance(criterion, FocalCLIPLoss)
                    metrics.update(evaluate_paper_fce(
                        evaluation_model, validation_fce_loader, criterion, device, args.amp,
                    ))
                if selection_key not in metrics:
                    raise RuntimeError(
                        f"Validation protocol {getattr(args, 'validation_protocol', 'unknown')!r} "
                        f"did not produce the requested selection metric {selection_key!r}. "
                        f"Available metrics: {sorted(metrics)}"
                    )
                current_score = float(metrics[selection_key])
                if current_score > best_map:
                    best_map = current_score
                    best_epoch = epoch
                    stale = 0
                    save_best(
                        best_path, evaluation_model, table.attribute_names, epoch,
                        {**losses, **metrics}, args.prompt_mode, training_config,
                    )
                    logger.log(
                        f"Saved new best checkpoint: {best_path} "
                        f"({selection_key}={current_score:.6f})"
                    )
                else:
                    stale += 1
                    should_stop = (
                        epoch >= args.minimum_training_epochs
                        and stale >= args.early_stopping_patience
                    )

            elapsed = time.time() - started
            lr = float(optimizer.param_groups[0]["lr"])
            domain_log = ""
            if "domain_macro_mADM" in metrics:
                domain_parts = []
                for domain in ("PA100k", "PETA", "Market1501"):
                    prefix = f"domain_{domain}_"
                    if f"{prefix}mADM" not in metrics:
                        continue
                    domain_parts.append(
                        f"{domain} R1={100 * metrics[f'{prefix}rank1']:.2f}%, "
                        f"mADM={100 * metrics[f'{prefix}mADM']:.2f}%"
                    )
                domain_log = ", " + ", ".join(domain_parts)
                if "unseen_mADM" in metrics:
                    domain_log += (
                        f", Unseen({metrics['unseen_domain']}) "
                        f"R1={100 * metrics['unseen_rank1']:.2f}%, "
                        f"mADM={100 * metrics['unseen_mADM']:.2f}%"
                    )
            logger.log(
                f"Epoch {epoch:03d}/{args.epochs}: total={losses['loss']:.6f}, "
                f"fce={losses.get('fce', 0.0):.6f}, bce={losses.get('bce', 0.0):.6f}, "
                + f"catce={losses.get('category_ce', 0.0):.6f}, "
                + (f"prototype={losses['prototype']:.6f}, set={losses['set']:.6f}, "
                   if args.training_objective in {"task2_hybrid", "a7_mixed", "a7_fce_mixed_aux"} else "")
                + (f"single_ce={losses.get('single_ce', 0.0):.6f}, "
                   f"multi_bce={losses.get('multi_bce', 0.0):.6f}, "
                   f"consistency={losses.get('consistency', 0.0):.6f}, "
                   if args.training_objective in {"a7_mixed", "a7_fce_mixed_aux"} else "")
                + f"distill={losses.get('distill', 0.0):.6f}, "
                + f"rsc_drop={losses.get('rsc_dropped', 0.0):.4f}, "
                + f"i2t={losses['i2t']:.6f}, t2i={losses['t2i']:.6f}, lr={lr:.3e}"
                + (f", AUROC={metrics.get('macro_auroc', float('nan')):.4f}, "
                   f"AP={metrics.get('macro_ap', float('nan')):.4f}, "
                   f"F1={metrics.get('instance_f1', metrics.get('macro_f1', float('nan'))):.4f}, "
                   f"BitErr={metrics.get('mean_hamming_error', float('nan')):.3f}, "
                   f"Exact={100 * metrics.get('exact_match', float('nan')):.2f}%, "
                   f"Rank-1={100 * metrics['rank1']:.2f}%, "
                   f"Rank-5={100 * metrics['rank5']:.2f}%, Rank-10={100 * metrics['rank10']:.2f}%, "
                   f"mAP={100 * metrics['map']:.2f}%, "
                   f"mADM={100 * metrics.get('mADM', float('nan')):.2f}%, "
                   f"SemTop1={100 * metrics['semantic_top1']:.2f}%"
                   + (f", DomainMacro R1={100 * metrics['domain_macro_rank1']:.2f}%, "
                      f"mADM={100 * metrics['domain_macro_mADM']:.2f}%, "
                      f"Worst={100 * metrics['worst_domain_mADM']:.2f}%"
                      if "domain_macro_mADM" in metrics else "")
                   + domain_log
                   if metrics else "")
                + (f", batch_avg={losses['average_batch_size']:.1f}, "
                   f"VRAM={losses['cuda_peak_allocated_gib']:.2f}/"
                   f"{losses['cuda_peak_reserved_gib']:.2f} GiB" if device.type == "cuda" else "")
                + f" ({elapsed:.1f}s)"
            )
            row = {
                "epoch": epoch, "total_loss": losses["loss"],
                "i2t_loss": metrics.get("i2t_loss", losses["i2t"]),
                "t2i_loss": metrics.get("t2i_loss", losses["t2i"]),
                "fce_loss": metrics.get("fce_loss", losses.get("fce", 0.0)),
                "category_ce_loss": losses.get("category_ce", 0.0),
                "bce_loss": losses.get("bce", 0.0), "learning_rate": lr,
                "distill_loss": losses.get("distill", 0.0),
                "rsc_dropped_fraction": losses.get("rsc_dropped", 0.0),
                "prototype_loss": losses.get("prototype", ""),
                "set_loss": losses.get("set", ""),
                "single_ce_loss": losses.get("single_ce", ""),
                "multi_bce_loss": losses.get("multi_bce", ""),
                "consistency_loss": losses.get("consistency", ""),
                "best_map": best_map, "best_epoch": best_epoch,
                "best_score": best_map, "selection_metric": selection_key,
                "stale_evaluations": stale, "elapsed_seconds": elapsed,
                "train_batches": losses["train_batches"],
                "average_batch_size": losses["average_batch_size"],
                "cuda_peak_allocated_gib": losses.get("cuda_peak_allocated_gib", ""),
                "cuda_peak_reserved_gib": losses.get("cuda_peak_reserved_gib", ""),
                **metrics,
            }
            logger.metrics(row)
            save_last(
                last_path, model, table.attribute_names, epoch, {**losses, **metrics},
                optimizer, scheduler, scaler, best_map, best_epoch, stale,
                args.prompt_mode, training_config,
            )
            if should_stop:
                logger.log(
                    f"Early stopping at epoch {epoch}; best "
                    f"{getattr(args, 'selection_metric', 'map')}={100 * best_map:.2f}% "
                    f"at epoch {best_epoch}."
                )
                break
    finally:
        logger.close()
    if not best_path.is_file():
        raise RuntimeError("Training completed without a validation-selected checkpoint")
    return best_path
