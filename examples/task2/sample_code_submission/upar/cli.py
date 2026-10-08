"""Command-line interface and data-independent smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .checkpoint import cached_probabilities, load_checkpoint
from .config import NUM_ATTRIBUTES, REPOSITORY_ROOT, PreprocessingConfig, choose_device, set_seed
from .data import build_eval_transform, find_annotation_file, read_gt_csv
from .engine import evaluate_model, print_evaluation, train
from .modeling import UPARModel, WeightedBCELoss
from .retrieval import infer_probabilities, l1_attribute_distances, load_retrieval_annotations, reorder_columns


def apply_ablation(args: argparse.Namespace) -> None:
    """Apply the reproducible E1-E3 ladder without changing other defaults."""
    if args.ablation == "NONE":
        return
    # E1 is the common evaluation/checkpoint-selection anchor for the ladder.
    args.selection_metric = "madm"
    if args.ablation in {"E2", "E3"}:
        args.image_size = 256
        args.image_width = 128
        args.resize_size = 256
        args.resize_width = 128
    if args.ablation == "E3":
        # Keep AugMix unchanged so E3 isolates the spatial crop policy.
        args.crop_policy = "reference"


def run_smoke_tests(args: argparse.Namespace) -> None:
    set_seed(args.seed, deterministic=True)
    data_root = Path(args.data_root).resolve()
    table = read_gt_csv(find_annotation_file(data_root, "train"))
    val_path = find_annotation_file(data_root, "val")
    val_table = read_gt_csv(val_path)
    val_queries, val_ids, val_names = load_retrieval_annotations(val_path.parent, val_table)

    criterion = WeightedBCELoss(table.labels.mean(0), args.label_smoothing)
    logits = torch.randn(3, NUM_ATTRIBUTES, requires_grad=True)
    targets = torch.from_numpy(table.labels[:3].copy())
    criterion(logits, targets).backward()
    probabilities = torch.sigmoid(logits.detach()).numpy().astype(np.float32)
    queries = table.labels[:2]
    distances = l1_attribute_distances(queries, probabilities)
    reverse_names = list(reversed(table.attribute_names))
    reversed_probabilities = reorder_columns(probabilities, table.attribute_names, reverse_names)
    restored = reorder_columns(reversed_probabilities, reverse_names, table.attribute_names)

    assert table.labels.shape[1] == NUM_ATTRIBUTES
    assert logits.shape == (3, NUM_ATTRIBUTES)
    assert 0 <= probabilities.min() <= probabilities.max() <= 1
    assert distances.shape == (len(queries), 3) and np.isfinite(distances).all()
    assert np.allclose(restored, probabilities)
    assert val_queries.ndim == 2 and val_queries.shape[1] == NUM_ATTRIBUTES
    assert val_ids.shape == (len(val_table.image_paths),)
    assert val_names == table.attribute_names

    if args.smoke_model:
        device = choose_device(args.device)
        model = UPARModel(dropout=args.dropout, pretrained=False).to(device).eval()
        height = max(32, args.image_size)
        width = max(32, args.image_width or args.image_size)
        with torch.inference_mode():
            output = model(torch.randn(1, 3, height, width, device=device))
        assert output.shape == (1, NUM_ATTRIBUTES)
    print(
        "Smoke tests passed: train/val CSV, [B,40] loss, [Q,40] queries, "
        "sigmoid, [Q,G] distance, finite values, attribute ordering"
        + (", ConvNeXt forward" if args.smoke_model else "")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UPAR ConvNeXt-B Task 2 baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("train", "eval", "train_eval", "smoke"), default="train_eval")
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__OUTPUT_LAST__",
        metavar="CHECKPOINT",
        help="resume training from CHECKPOINT, or from OUTPUT_DIR/last.pth when omitted",
    )
    parser.add_argument(
        "--ablation",
        type=str.upper,
        choices=("NONE", "E1", "E2", "E3"),
        default="NONE",
        help="reproducible ablation preset: E1=mADM, E2=256x128, E3=reference crop",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--resize-size", type=int)
    parser.add_argument("--resize-width", type=int)
    parser.add_argument("--crop-policy", choices=("current", "reference"), default="current")
    parser.add_argument("--augmix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--unweighted-bce", action="store_true")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9998)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-factor", type=float, default=0.1)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
        help="stop after this many retrieval evaluations without an mAP improvement",
    )
    parser.add_argument("--retrieval-interval", type=int, default=1)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--selection-metric", choices=("map", "madm"), default="map")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--cache-val-probs", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--smoke-model", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.debug:
        args.max_train_samples = args.max_train_samples or 256
        args.max_val_samples = args.max_val_samples or 256
        args.epochs = min(args.epochs, 1)
    for field in ("epochs", "batch_size", "eval_batch_size", "image_size",
                  "retrieval_interval", "query_chunk_size", "early_stopping_patience"):
        if getattr(args, field) <= 0:
            raise ValueError(f"{field} must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if args.resize_size is not None and args.resize_size < args.image_size:
        raise ValueError("resize_size must be at least image_size")
    if args.image_width is not None and args.image_width <= 0:
        raise ValueError("image_width must be positive")
    if args.resize_width is not None and args.resize_width <= 0:
        raise ValueError("resize_width must be positive")
    if args.resize_width is not None and args.image_width is None:
        raise ValueError("resize_width requires image_width")
    if (
        args.image_width is not None
        and args.resize_width is not None
        and args.resize_width < args.image_width
    ):
        raise ValueError("resize_width must be at least image_width")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise ValueError("max_train_samples must be positive")
    if args.max_val_samples is not None and args.max_val_samples <= 0:
        raise ValueError("max_val_samples must be positive")
    if args.resume and args.mode not in {"train", "train_eval"}:
        raise ValueError("--resume can only be used with train or train_eval mode")
    if args.resume == "__OUTPUT_LAST__":
        args.resume = str(Path(args.output_dir) / "last.pth")


def main() -> None:
    args = build_parser().parse_args()
    apply_ablation(args)
    validate_args(args)
    if args.mode == "smoke":
        run_smoke_tests(args)
        return

    checkpoint_path = train(args) if args.mode in {"train", "train_eval"} else None
    if args.mode not in {"eval", "train_eval"}:
        return
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif checkpoint_path is None:
        checkpoint_path = Path(args.output_dir) / "best.pth"
        legacy_path = Path(args.output_dir) / "model_best.pth"
        if not checkpoint_path.is_file() and legacy_path.is_file():
            checkpoint_path = legacy_path
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    preprocessing = PreprocessingConfig.from_dict(checkpoint["preprocessing"])

    def compute() -> np.ndarray:
        gt_path = find_annotation_file(Path(args.data_root).resolve(), "val")
        table = read_gt_csv(gt_path)
        paths = table.image_paths[:args.max_val_samples] if args.max_val_samples else table.image_paths
        return infer_probabilities(
            model, paths,
            [Path(args.data_root).resolve(), gt_path.parent, REPOSITORY_ROOT],
            build_eval_transform(preprocessing), device,
            args.eval_batch_size, args.num_workers, args.amp,
        )

    probabilities = None
    if args.cache_val_probs:
        probabilities = cached_probabilities(
            Path(args.output_dir) / "val_probs.npy",
            checkpoint_path.resolve(),
            compute,
        )
    metrics = evaluate_model(
        model, checkpoint["attribute_names"], Path(args.data_root).resolve(),
        preprocessing, device, args.eval_batch_size, args.num_workers, args.amp,
        args.max_val_samples, args.query_chunk_size, probabilities,
    )
    print_evaluation(metrics, str(checkpoint_path))
