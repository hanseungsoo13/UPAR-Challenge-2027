"""Command-line interface and model/loss sanity checks for AttriVision."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from upar.config import REPOSITORY_ROOT, choose_device, set_seed

from .checkpoint import load_model
from .datasets.attribute_prompts import prompts_for_attributes
from .engine.evaluator_abpr import (
    evaluate_abpr, evaluate_native52, evaluate_native52_category_nll,
    print_evaluation,
)
from .engine.trainer_attrivision import train
from .losses.focal_clip_loss import FocalCLIPLoss
from .losses.task2_hybrid_loss import Task2HybridLoss
from .models.attrivision import AttriVision
from .transforms import build_eval_transform, build_train_transform


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AttriVision CLIP ViT-B/32 baseline for UPAR Task 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=("task2",), default="task2")
    parser.add_argument("--model", choices=("attrivision",), default="attrivision")
    parser.add_argument("--mode", choices=("train", "eval", "train_eval", "smoke"), default="train_eval")
    parser.add_argument(
        "--paper-faithful", action="store_true",
        help="use the paper-style 40-attribute presence/absence FCE recipe",
    )
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", default="outputs/attrivision")
    parser.add_argument("--checkpoint")
    parser.add_argument("--init-checkpoint", help="initialize CLIP weights from an existing checkpoint; binary head is reset")
    parser.add_argument("--resume", nargs="?", const="__LAST__", metavar="CHECKPOINT")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--clip-model", default="ViT-B-32-quickgelu",
        help="OpenCLIP architecture; OpenAI ViT-B/32 weights require QuickGELU",
    )
    parser.add_argument("--pretrained-tag", default="openai")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--rotation", type=float, default=10.0)
    parser.add_argument("--augmentation", choices=("current", "paper_like"), default="current")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--text-sampling", "--text_sampling", choices=("single", "multi"), default="single")
    parser.add_argument("--multi-attributes", type=int, default=3)
    parser.add_argument(
        "--paper-multi-attributes", type=int,
        help="override the paper preset's sampled-state count (the paper does not publish K)",
    )
    parser.add_argument(
        "--paper-contrastive-target", choices=("diagonal", "multi_positive"),
        help="override the paper preset target for K-sampling ablations",
    )
    parser.add_argument(
        "--paper-batch-size", type=int,
        help="override the paper preset batch size (default: 32)",
    )
    parser.add_argument(
        "--prompt-mode", choices=("category_complete", "binary_positive", "paper_binary"),
        default="category_complete",
        help="12-category states, positive-only attributes, or paper-style positive/negative binary states",
    )
    parser.add_argument("--loss", choices=("clip", "focal_clip"), default="focal_clip")
    parser.add_argument(
        "--training-objective", choices=("task2_hybrid", "paper_fce"),
        default="task2_hybrid",
        help="Task-2-aligned prototype/set loss or the paper-equation FCE ablation",
    )
    parser.add_argument("--prototype-loss-weight", type=float, default=0.25)
    parser.add_argument("--set-loss-weight", type=float, default=1.0)
    parser.add_argument("--lambda-attr", type=float, default=1.0)
    parser.add_argument(
        "--category-ce-mode", choices=("off", "only", "combined"), default="off",
        help="optional 12-category Native52 CE objective",
    )
    parser.add_argument("--category-ce-weight", type=float, default=1.0)
    parser.add_argument("--category-temperature", type=float, default=0.01)
    parser.add_argument(
        "--freeze-binary-head", action=argparse.BooleanOptionalAction, default=False,
        help="exclude the unused 40-D head from gradients and optimizer groups",
    )
    parser.add_argument("--use-fce", type=lambda value: value.lower() in {"1", "true", "yes", "y"}, default=True,
                        help="include the existing 52-state FCE objective (true/false)")
    parser.add_argument("--balance-max-weight", type=float, default=10.0)
    parser.add_argument("--focal-alpha", "--focal_alpha", type=float, default=1.0)
    parser.add_argument("--focal-gamma", "--focal_gamma", type=float, default=2.0)
    parser.add_argument(
        "--contrastive-target", "--contrastive_target",
        choices=("diagonal", "multi_positive"), default="diagonal",
    )
    parser.add_argument(
        "--unique-prompts", action=argparse.BooleanOptionalAction, default=False,
        help="construct batches whose sampled attribute phrases do not repeat (paper section 4.2)",
    )
    parser.add_argument("--query-aggregation", choices=("mean",), default="mean")
    parser.add_argument(
        "--retrieval-scoring", choices=("cosine_set", "paired_l1"), default="cosine_set",
        help="query-text cosine or 40 paired-prompt probabilities with official L1 distance",
    )
    parser.add_argument(
        "--attribute-temperature", type=float,
        help="paired_l1 softmax temperature T (default: checkpoint's learned CLIP temperature)",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--minimum-training-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--retrieval-interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--smoke-batch-size", type=int, default=2)
    parser.add_argument("--batch-diagnostics", action="store_true")
    parser.add_argument(
        "--validation-protocol", choices=(
            "binary_head", "native52", "native52_category_nll", "paired_l1",
        ),
        default="binary_head",
    )
    parser.add_argument(
        "--selection-metric", choices=("map", "mADM"), default="map",
        help="validation metric used for checkpoint selection",
    )
    return parser


def apply_paper_preset(args: argparse.Namespace) -> None:
    """Resolve one reproducible paper-style recipe before validation/training."""
    if not args.paper_faithful:
        return
    args.clip_model = "ViT-B-32-quickgelu"
    args.prompt_mode = "paper_binary"
    args.training_objective = "paper_fce"
    args.text_sampling = "multi"
    args.multi_attributes = (
        args.paper_multi_attributes
        if args.paper_multi_attributes is not None else 3
    )
    args.batch_size = args.paper_batch_size if args.paper_batch_size is not None else 32
    args.contrastive_target = args.paper_contrastive_target or "diagonal"
    args.loss = "focal_clip"
    args.use_fce = True
    args.unique_prompts = True
    args.lambda_attr = 0.0
    args.freeze_binary_head = True
    args.validation_protocol = "paired_l1"
    args.retrieval_scoring = "paired_l1"
    args.selection_metric = "mADM"
    # The paper specifies rotation/flip augmentation but not random resized
    # crop; keep the CLIP evaluation geometry and avoid the extra crop policy.
    args.augmentation = "paper_like"


def validate_args(args: argparse.Namespace) -> None:
    if args.debug:
        args.epochs = min(args.epochs, 1)
        args.max_train_samples = args.max_train_samples or 64
        args.max_val_samples = args.max_val_samples or 64
    for field in (
        "epochs", "batch_size", "eval_batch_size", "num_workers", "image_size",
        "multi_attributes", "minimum_training_epochs", "early_stopping_patience",
        "retrieval_interval", "smoke_batch_size",
    ):
        if getattr(args, field) < (0 if field == "num_workers" else 1):
            raise ValueError(f"{field} has an invalid value: {getattr(args, field)}")
    if args.warmup_epochs < 0:
        raise ValueError("warmup-epochs cannot be negative")
    if args.learning_rate <= 0 or args.min_learning_rate < 0 or args.weight_decay < 0 or args.rotation < 0:
        raise ValueError("learning-rate must be positive; weight-decay and rotation cannot be negative")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("min-learning-rate cannot exceed learning-rate")
    if args.warmup_epochs >= args.epochs and not args.debug:
        raise ValueError("warmup-epochs must be smaller than epochs")
    if args.image_size != 224:
        raise ValueError("Vanilla CLIP ViT-B/32 requires --image-size 224")
    if args.focal_alpha < 0 or args.focal_gamma < 0:
        raise ValueError("focal alpha and gamma cannot be negative")
    if args.prototype_loss_weight < 0 or args.set_loss_weight < 0:
        raise ValueError("hybrid loss weights cannot be negative")
    if args.prototype_loss_weight + args.set_loss_weight <= 0:
        raise ValueError("at least one hybrid loss weight must be positive")
    if args.lambda_attr < 0:
        raise ValueError("lambda-attr cannot be negative")
    if args.category_ce_weight < 0 or args.category_temperature <= 0:
        raise ValueError("category-ce-weight must be non-negative and category-temperature positive")
    if args.category_ce_mode == "combined" and args.category_ce_weight <= 0:
        raise ValueError("combined Category CE requires a positive category-ce-weight")
    if args.balance_max_weight < 1:
        raise ValueError("balance-max-weight must be at least 1")
    if args.attribute_temperature is not None and args.attribute_temperature <= 0:
        raise ValueError("attribute-temperature must be positive")
    if args.resume == "__LAST__":
        args.resume = str(Path(args.output_dir) / "checkpoint_last.pth")
    if args.resume and args.mode not in {"train", "train_eval"}:
        raise ValueError("--resume is valid only in train or train_eval mode")


def run_smoke(args: argparse.Namespace) -> None:
    set_seed(args.seed, deterministic=True)
    device = choose_device(args.device)
    model = AttriVision(pretrained=None, model_name=args.clip_model).to(device)
    batch_size = args.smoke_batch_size
    images = torch.randn(batch_size, 3, args.image_size, args.image_size, device=device)
    transform = build_train_transform(args.image_size, args.rotation, args.augmentation)
    transform_names = [type(item).__name__ for item in transform.transforms]
    if args.augmentation == "paper_like" and "RandomResizedCrop" in transform_names:
        raise AssertionError("paper_like augmentation must not use RandomResizedCrop")
    if args.training_objective == "task2_hybrid":
        semantic_count = batch_size + 2
        texts = [f"a photo of semantic state {index}" for index in range(semantic_count)]
        labels = torch.zeros(batch_size, semantic_count, dtype=torch.bool, device=device)
        labels[torch.arange(batch_size), torch.arange(batch_size)] = True
        labels[:, -1] = True
        tokens = model.tokenize(texts).to(device)
        criterion = Task2HybridLoss(
            labels.float().mean(dim=0).clamp_min(1 / max(batch_size, 1)),
            args.focal_alpha, args.focal_gamma, args.balance_max_weight,
            args.prototype_loss_weight, args.set_loss_weight,
        ).to(device)
        model.train()
        image_features = model.encode_image(images)
        text_features = model.encode_text(tokens)
        output = criterion(image_features, text_features, labels, model.logit_scale)
        output.loss.backward()
        assert image_features.shape == (batch_size, 512)
        assert text_features.shape == (semantic_count, 512)
        assert torch.isfinite(output.loss)
        assert any(parameter.grad is not None for parameter in model.parameters())
        print(
            "Task2 hybrid sanity checks passed: "
            f"image={tuple(image_features.shape)}, text={tuple(text_features.shape)}, "
            f"loss={float(output.loss.detach()):.6f}, backward=finite"
        )
        return
    if args.prompt_mode == "paper_binary":
        from .datasets.attribute_prompts import ATTRIBUTE_PROMPTS, PaperAttributePromptMapper

        mapper = PaperAttributePromptMapper(list(ATTRIBUTE_PROMPTS))
        assert len(mapper.prompts) == 80
        attribute_labels = torch.zeros(batch_size, 40, dtype=torch.float32, device=device)
        rows = torch.arange(batch_size, device=device)
        attribute_labels[rows, rows % 40] = 1
        labels = mapper.encode(attribute_labels)
        count = 1 if args.text_sampling == "single" else min(args.multi_attributes, 40)
        selected_indices = [torch.nonzero(row, as_tuple=False).flatten()[:count] for row in labels]
        texts = [mapper.prompts[int(index)] for row in selected_indices for index in row]
        owners = [image for image, row in enumerate(selected_indices) for _ in row]
        attributes = [int(index) for row in selected_indices for index in row]
        selected = torch.zeros(len(texts), 80, dtype=torch.bool, device=device)
        selected[torch.arange(len(texts), device=device), torch.tensor(attributes, device=device)] = True
        text_owners = torch.tensor(owners, dtype=torch.long, device=device)
    elif args.text_sampling == "single":
        texts = [f"a photo of person number {index}" for index in range(batch_size)]
        labels = torch.eye(batch_size, device=device)
        selected = torch.eye(batch_size, device=device, dtype=torch.bool)
        text_owners = torch.arange(batch_size, device=device)
    else:
        # Each image owns K phrases. One is unique and the remainder are
        # shared semantic states, exercising a rectangular [B,K*B] matrix.
        texts = []
        owners = []
        attributes = []
        semantic_count = batch_size + max(args.multi_attributes - 1, 0)
        for index in range(batch_size):
            own_attributes = [index] + list(range(batch_size, semantic_count))
            texts.extend([f"a photo of semantic state {value}" for value in own_attributes])
            owners.extend([index] * len(own_attributes))
            attributes.extend(own_attributes)
        labels = torch.zeros(batch_size, semantic_count, device=device)
        labels[torch.arange(batch_size), torch.arange(batch_size)] = 1
        if semantic_count > batch_size:
            labels[:, batch_size:] = 1
        selected = torch.zeros(len(texts), semantic_count, dtype=torch.bool, device=device)
        selected[torch.arange(len(texts)), torch.tensor(attributes, device=device)] = True
        text_owners = torch.tensor(owners, dtype=torch.long, device=device)
    tokens = model.tokenize(texts).to(device)
    criterion = FocalCLIPLoss(args.loss, args.contrastive_target, args.focal_alpha, args.focal_gamma)
    model.train()
    image_features, text_features, logits = model(images, tokens)
    output = criterion(logits, labels, selected, text_owners)
    positive_mask = criterion.positive_mask(logits, labels, selected, text_owners)
    output.loss.backward()
    assert image_features.shape == (batch_size, 512)
    assert text_features.shape == (len(texts), 512)
    assert logits.shape == (batch_size, len(texts))
    assert torch.isfinite(output.loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    print(
        "Sanity checks passed: "
        f"image={tuple(image_features.shape)}, text={tuple(text_features.shape)}, "
        f"similarity={tuple(logits.shape)}, loss={float(output.loss.detach()):.6f}, backward=finite"
    )
    print(
        f"Smoke configuration: sampling={args.text_sampling}, "
        f"target={args.contrastive_target}, augmentation={args.augmentation}, "
        f"texts/image={torch.bincount(text_owners, minlength=batch_size).tolist()}, "
        f"positive/image={positive_mask.sum(1).tolist()}, "
        f"positive/text={positive_mask.sum(0).tolist()}"
    )


def run(args: argparse.Namespace) -> None:
    apply_paper_preset(args)
    validate_args(args)
    if args.mode == "smoke":
        run_smoke(args)
        return

    checkpoint = train(args) if args.mode in {"train", "train_eval"} else None
    if args.mode == "train":
        return
    checkpoint = Path(args.checkpoint) if args.checkpoint else checkpoint
    if checkpoint is None:
        checkpoint = Path(args.output_dir) / "checkpoint_best.pth"
    device = choose_device(args.device)
    model, payload = load_model(checkpoint, device)
    prompts_for_attributes(payload["attribute_names"])
    if args.validation_protocol == "native52_category_nll":
        metrics = evaluate_native52_category_nll(
            model, payload["attribute_names"], Path(args.data_root).resolve(),
            build_eval_transform(args.image_size), device, args.eval_batch_size,
            args.num_workers, args.amp, args.max_val_samples,
            args.category_temperature,
        )
    elif args.validation_protocol == "native52":
        metrics = evaluate_native52(
            model, payload["attribute_names"], Path(args.data_root).resolve(),
            build_eval_transform(args.image_size), device, args.eval_batch_size,
            args.num_workers, args.amp, args.max_val_samples,
            args.attribute_temperature,
        )
    else:
        metrics = evaluate_abpr(
            model, payload["attribute_names"], Path(args.data_root).resolve(),
            build_eval_transform(args.image_size), device, args.eval_batch_size,
            args.num_workers, args.amp, args.max_val_samples,
            payload.get("prompt_mode", "binary_positive"),
            args.retrieval_scoring, args.attribute_temperature,
        )
    print_evaluation(metrics, str(checkpoint))


def main() -> None:
    run(build_parser().parse_args())
