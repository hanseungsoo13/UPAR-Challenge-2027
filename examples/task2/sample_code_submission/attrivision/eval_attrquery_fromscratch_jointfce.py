"""Evaluate AttrQuery-FromScratchJointFCE and run fusion/attention diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.attrquery_fromscratch_jointfce import (  # noqa: E402
    ALPHAS,
    atomic_json,
    evaluate_model,
    load_gallery_scores,
    plot_alpha_curve,
    plot_attention_similarity_summary,
    protocol_validation,
    retrieval_views,
    save_attention_visualizations,
    save_gallery_scores,
    write_rows_csv,
)
from attrivision.models.attrquery_fromscratch_jointfce import load_experiment_model  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import REPOSITORY_ROOT, choose_device  # noqa: E402


def _parse_float_list(value: list[float] | None) -> tuple[float, ...]:
    values = tuple(ALPHAS if value is None else value)
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("--alphas must contain values in [0,1]")
    return values


def _parse_indices(value: str | None, count: int, total: int) -> list[int]:
    if value:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        indices = list(range(min(count, total)))
    if any(index < 0 or index >= total for index in indices):
        raise ValueError(f"Attention indices must be in [0,{total})")
    return indices


def _parse_categories(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_summary(
    path: Path,
    payload: dict,
    rows: list[dict],
) -> None:
    global_row = next(row for row in rows if row["score_type"] == "global")
    local_row = next(row for row in rows if row["score_type"] == "local")
    fusion_rows = [row for row in rows if row["score_type"] == "fusion"]
    best = max(fusion_rows, key=lambda row: float(row["mADM"]))

    def metric_block(row: dict) -> str:
        return (
            f"Rank-1={100*float(row['rank1']):.3f}%, "
            f"mAP={100*float(row['mAP']):.3f}%, "
            f"mADM={100*float(row['mADM']):.3f}%, "
            f"mINP={100*float(row['mINP']):.3f}%"
        )

    lines = [
        "# AttrQuery-Fusion-Diagnostics / FromScratchJointFCE",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Epoch: `{payload.get('epoch', 'unknown')}`",
        "- Initialization: original pretrained CLIP; A7 and AttrQuery checkpoints were not used",
        "- Validation: same validation gallery/query IDs, exact 52-state mapping, category-NLL",
        "",
        "## Validation views",
        "",
        f"- Global-only: {metric_block(global_row)}",
        f"- Local-only: {metric_block(local_row)}",
        f"- Best fusion alpha: `{float(best['alpha']):.2f}` ({metric_block(best)})",
        f"- Best-alpha mADM improvement vs Global: {100*(float(best['mADM'])-float(global_row['mADM'])):.3f} percentage points",
        f"- Best-alpha mADM improvement vs Local: {100*(float(best['mADM'])-float(local_row['mADM'])):.3f} percentage points",
        "",
        "## Interpretation guardrail",
        "",
        "Attention maps show where category queries draw information from. "
        "They do not by themselves establish correct attribute localization; "
        "the maps should be compared with retrieval/category errors and image evidence.",
        "",
        "## Files",
        "",
        "- `fusion_alpha_sweep.csv` and `fusion_alpha_curve.png`: alpha sweep",
        "- `global_local_metrics.csv`: fixed Global-only and Local-only metrics",
        "- `gallery_scores.npz` and `fusion_scores.npz`: reusable global/local/fusion scores",
        "- `protocol_validation.json`: explicit A7 protocol checks",
        "- `attention_heatmaps/`: optional category/head heatmaps and similarity matrices",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate FromScratchJointFCE with fixed Category-NLL and fusion alpha sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=str(REPOSITORY_ROOT / "data"))
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--category-temperature", type=float)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--alphas", type=float, nargs="*", default=None)
    parser.add_argument("--gallery-scores")
    parser.add_argument(
        "--attention-count", type=int, default=0,
        help="number of validation images for attention visualization; 0 disables it",
    )
    parser.add_argument("--attention-indices", help="comma-separated validation indices")
    parser.add_argument("--attention-categories", help="comma-separated category names")
    parser.add_argument("--save-all-attention-heads", action=argparse.BooleanOptionalAction, default=True)
    return parser


def eval_main() -> None:
    args = build_parser().parse_args()
    if args.eval_batch_size <= 0 or args.num_workers < 0 or args.progress_every < 0:
        raise ValueError("Invalid evaluation loader/progress setting")
    if args.attention_count < 0:
        raise ValueError("--attention-count cannot be negative")
    device = choose_device(args.device)
    data_root = Path(args.data_root).resolve()
    model, checkpoint_payload = load_experiment_model(args.checkpoint, device)
    training_config = dict(checkpoint_payload["training_config"])
    temperature = float(args.category_temperature or training_config.get("category_temperature", 0.01))
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        condition = training_config.get("condition_name", "FromScratchJointFCE")
        output_dir = (Path("outputs/AttrQuery-FromScratchJointFCE") / condition / "evaluation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mapper = model.mapper
    protocol = protocol_validation(training_config, mapper)
    atomic_json(output_dir / "protocol_validation.json", protocol)
    alphas = _parse_float_list(args.alphas)
    cached_scores = load_gallery_scores(Path(args.gallery_scores)) if args.gallery_scores else None
    rows, details = evaluate_model(
        model, model.attribute_names, data_root, device, args.eval_batch_size,
        args.num_workers, args.amp, temperature, args.max_val_samples,
        alphas=alphas, progress_every=args.progress_every,
        gallery_scores=cached_scores,
    )
    save_gallery_scores(output_dir / "gallery_scores.npz", (
        details["global_scores"], details["local_scores"],
    ))
    fusion_score_path = output_dir / "fusion_scores.npz"
    import numpy as np  # noqa: PLC0415
    np.savez_compressed(fusion_score_path, **details["fusion_scores"])
    fusion_rows = [row for row in rows if row["score_type"] == "fusion"]
    fixed_rows = [row for row in rows if row["score_type"] in {"global", "local"}]
    write_rows_csv(output_dir / "fusion_alpha_sweep.csv", fusion_rows)
    write_rows_csv(output_dir / "global_local_metrics.csv", fixed_rows)
    write_rows_csv(output_dir / "metrics_summary.csv", rows)
    plot_alpha_curve(rows, output_dir / "fusion_alpha_curve.png")
    atomic_json(output_dir / "evaluation_metadata.json", {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": checkpoint_payload.get("epoch"),
        "device": str(device),
        "category_temperature": temperature,
        "alphas": list(alphas),
        "initialization": checkpoint_payload.get("initialization"),
        "gallery_scores": str(output_dir / "gallery_scores.npz"),
    })
    summary_payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": checkpoint_payload.get("epoch"),
    }
    _write_summary(output_dir / "analysis_summary.md", summary_payload, rows)

    if args.attention_count or args.attention_indices:
        gt_path, table, _, _, _ = retrieval_views(data_root, "val", args.max_val_samples)
        indices = _parse_indices(args.attention_indices, args.attention_count, len(table.image_paths))
        selected_categories = _parse_categories(args.attention_categories)
        attention_dir = output_dir / "attention_heatmaps"
        attention_rows = save_attention_visualizations(
            model, table, [data_root, gt_path.parent, REPOSITORY_ROOT],
            build_eval_transform(224, "resize_pad_crop"), indices, attention_dir,
            device, args.amp, selected_categories, args.save_all_attention_heads,
        )
        write_rows_csv(attention_dir / "attention_images.csv", attention_rows)
        if selected_categories is None:
            labels = model.category_names
        else:
            labels = selected_categories
        plot_attention_similarity_summary(attention_dir, attention_rows, labels, output_dir)
        with (output_dir / "analysis_summary.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\n- Attention samples: `{len(indices)}` in `{attention_dir}`\n")
    print(
        f"FromScratchJointFCE evaluation complete: output={output_dir}; "
        f"best_alpha={max(fusion_rows, key=lambda row: float(row['mADM']))['alpha']}; "
        f"device={device}", flush=True,
    )


if __name__ == "__main__":
    eval_main()
