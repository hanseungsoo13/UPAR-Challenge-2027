"""Fine-tuning loop for the AttriVision CLIP baseline."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from upar.config import REPOSITORY_ROOT, choose_device, set_seed
from upar.data import find_annotation_file, read_gt_csv
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
    evaluate_native52_category_nll,
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
) -> dict[str, float]:
    model.train()
    if category_mode not in {"off", "only", "combined"}:
        raise ValueError(f"Unknown Category CE mode: {category_mode}")
    if category_mode != "off" and (category_criterion is None or prototype_tokens is None):
        raise ValueError("Category CE requires its criterion and all 52 prototype tokens")
    totals = {"loss": 0.0, "fce": 0.0, "category_ce": 0.0, "bce": 0.0,
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
                fce, i2t, t2i = result.loss, result.i2t, result.t2i
            if category_mode != "off":
                state_features = model.encode_text(prototype_tokens)
                category_ce = category_criterion(
                    image_features, state_features, semantic_labels,
                )
            else:
                category_ce = image_features.new_zeros(())
            bce = attr_weight * binary_criterion(model.binary_logits_from_features(image_features), labels.float()) if binary_criterion is not None else image_features.new_zeros(())
            total = fce + category_weight * category_ce + bce
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


def _optimizer_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Apply AdamW decay only to matrix/kernel weights, as in standard CLIP tuning."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("logit_scale"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train(args: Any) -> Path:
    set_seed(args.seed, args.deterministic)
    device = choose_device(args.device)
    data_root = Path(args.data_root).resolve()
    train_gt = find_annotation_file(data_root, "train")
    table = read_gt_csv(train_gt)
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
    if args.use_fce and args.training_objective == "paper_fce":
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
        args.use_fce and args.training_objective == "paper_fce"
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
    if args.use_fce and args.training_objective == "paper_fce":
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
    if args.use_fce and args.training_objective == "a7_mixed":
        if not isinstance(mapper, MixedCategoryPromptMapper):
            raise ValueError("a7_mixed requires prompt_mode=mixed_category")
        criterion = MixedStateHybridLoss(
            semantic_labels.float().mean(dim=0),
            [(kind, indices) for _, kind, indices in mapper.category_specs()],
            args.focal_alpha, args.focal_gamma, args.balance_max_weight,
            args.prototype_loss_weight, args.set_loss_weight,
            getattr(args, "mixed_consistency_weight", 0.1),
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
        "balance_max_weight": args.balance_max_weight,
        "use_fce": args.use_fce,
        "lambda_attr": args.lambda_attr,
        "category_ce_mode": category_mode,
        "category_ce_weight": getattr(args, "category_ce_weight", 1.0),
        "category_temperature": getattr(args, "category_temperature", 0.01),
        "freeze_binary_head": getattr(args, "freeze_binary_head", False),
        "augmentation": getattr(args, "augmentation", "current"),
        "selection_metric": getattr(args, "selection_metric", "map"),
        "paper_faithful": bool(getattr(args, "paper_faithful", False)),
        "positive_rates": positive_rates.tolist(),
    }
    optimizer = torch.optim.AdamW(
        _optimizer_groups(model, args.weight_decay), lr=args.learning_rate,
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

    logger.log(json.dumps({**vars(args), "resolved_device": str(device)}, indent=2, default=str))
    logger.log(
        f"Initialization checkpoint: {args.init_checkpoint if args.init_checkpoint else 'pretrained/default'}; "
        f"binary_head={'newly_initialized' if args.init_checkpoint else 'newly_initialized'}; "
        f"use_fce={args.use_fce}, lambda_attr={args.lambda_attr:g}"
    )
    logger.log(f"Training samples={len(dataset)}, attributes={len(table.attribute_names)}")
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
                )
            metrics: dict[str, float] = {}
            should_stop = False
            if epoch % args.retrieval_interval == 0 or epoch == args.epochs:
                if getattr(args, "validation_protocol", "binary_head") == "native52_category_nll":
                    metrics = evaluate_native52_category_nll(
                        model, table.attribute_names, data_root,
                        build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), device,
                        args.eval_batch_size, args.num_workers, args.amp,
                        args.max_val_samples, getattr(args, "category_temperature", 0.01),
                    )
                elif getattr(args, "validation_protocol", "binary_head") == "mixed_state_nll":
                    metrics = evaluate_mixed_state_nll(
                        model, table.attribute_names, data_root,
                        build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), device,
                        args.eval_batch_size, args.num_workers, args.amp,
                        args.max_val_samples, getattr(args, "category_temperature", 0.01),
                    )
                elif getattr(args, "validation_protocol", "binary_head") == "native52":
                    metrics = evaluate_native52(
                        model, table.attribute_names, data_root,
                        build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), device,
                        args.eval_batch_size, args.num_workers, args.amp,
                        args.max_val_samples, args.attribute_temperature,
                    )
                elif getattr(args, "validation_protocol", "binary_head") == "paired_l1":
                    metrics = evaluate_abpr(
                        model, table.attribute_names, data_root,
                        build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), device,
                        args.eval_batch_size, args.num_workers, args.amp,
                        args.max_val_samples, args.prompt_mode,
                        "paired_l1", args.attribute_temperature,
                    )
                else:
                    metrics = evaluate_binary_head(
                        model, table.attribute_names, data_root,
                        build_eval_transform(args.image_size, getattr(args, "augmentation", "current")), device,
                        args.eval_batch_size, args.num_workers, args.amp,
                        args.max_val_samples,
                    )
                if validation_fce_loader is not None:
                    assert isinstance(criterion, FocalCLIPLoss)
                    metrics.update(evaluate_paper_fce(
                        model, validation_fce_loader, criterion, device, args.amp,
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
                        best_path, model, table.attribute_names, epoch,
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
            logger.log(
                f"Epoch {epoch:03d}/{args.epochs}: total={losses['loss']:.6f}, "
                f"fce={losses.get('fce', 0.0):.6f}, bce={losses.get('bce', 0.0):.6f}, "
                + f"catce={losses.get('category_ce', 0.0):.6f}, "
                + (f"prototype={losses['prototype']:.6f}, set={losses['set']:.6f}, "
                   if args.training_objective in {"task2_hybrid", "a7_mixed"} else "")
                + (f"single_ce={losses.get('single_ce', 0.0):.6f}, "
                   f"multi_bce={losses.get('multi_bce', 0.0):.6f}, "
                   f"consistency={losses.get('consistency', 0.0):.6f}, "
                   if args.training_objective == "a7_mixed" else "")
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
