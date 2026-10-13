"""Run the two E1-A CLIP-preservation experiments requested for UPAR."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

SUBMISSION_DIR = Path(__file__).resolve().parents[1]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.checkpoint import load_model  # noqa: E402
from attrivision.cli import build_parser, validate_args  # noqa: E402
from attrivision.engine.evaluator_abpr import (  # noqa: E402
    evaluate_native52_category_nll_domainwise,
)
from attrivision.engine.trainer_attrivision import train  # noqa: E402
from attrivision.models.attrivision import AttriVision  # noqa: E402
from attrivision.transforms import build_eval_transform  # noqa: E402
from upar.config import choose_device  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("outputs/E1A_CLIP_Preservation")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
    temporary.replace(path)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _configure_training(args: argparse.Namespace, condition: str) -> None:
    """Keep the E1-A fine-tuning recipe fixed across the two conditions."""
    args.mode = "train_eval"
    args.clip_model = args.clip_model or "ViT-B-32-quickgelu"
    args.pretrained_tag = args.pretrained_tag or "openai"
    args.no_pretrained = False
    args.init_checkpoint = None
    args.image_size = 224
    args.prompt_mode = "category_complete"
    args.training_objective = "paper_fce"
    args.text_sampling = "single"
    args.multi_attributes = 3
    args.contrastive_target = "multi_positive"
    args.augmentation = "resize_pad_crop"
    args.loss = "focal_clip"
    args.use_fce = True
    args.unique_prompts = False
    args.lambda_attr = 0.0
    args.category_ce_mode = "off"
    args.validation_protocol = "native52_category_nll"
    args.retrieval_scoring = "paired_l1"
    args.selection_metric = args.selection_metric or "mADM"
    args.domain_wise_validation = True
    args.training_log_filename = "training_log.csv"
    args.paper_faithful = False
    args.distill_weight = 0.0
    args.rsc_drop_ratio = 0.0
    args.unseen_domain = args.unseen_domain or None

    if condition == "domain_balance":
        args.domain_sampling = "sqrt"
        args.dg_method = "none"
        args.output_dir = str(Path(args.output_root) / "E1A_DomainBalanced")
    elif condition == "wise_source":
        # WiSE-FT needs a plain fine-tuned endpoint and its matching pretrained
        # initialization. The interpolation itself is produced after this run.
        args.domain_sampling = "natural"
        args.dg_method = "none"
        args.output_dir = str(Path(args.output_root) / "E1A_WiSEFT" / "source")
    else:
        raise ValueError(f"Unknown E1-A training condition: {condition}")
    validate_args(args)


def _evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path, label: str) -> dict[str, Any]:
    device = choose_device(args.device)
    model, payload = load_model(checkpoint, device)
    metrics = evaluate_native52_category_nll_domainwise(
        model, payload["attribute_names"], Path(args.data_root).resolve(),
        build_eval_transform(args.image_size, args.augmentation), device,
        args.eval_batch_size, args.num_workers, args.amp, args.max_val_samples,
        args.category_temperature,
        unseen_domain=args.unseen_domain,
    )
    metrics.update({
        "experiment": label,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "selection_metric": args.selection_metric,
        "domain_sampling": payload.get("training_config", {}).get("domain_sampling", "natural"),
        "dg_method": payload.get("training_config", {}).get("dg_method", "none"),
    })
    _atomic_json(Path(args.output_dir) / "validation_metrics.json", metrics)
    return metrics


def _wise_checkpoint(
    source_payload: dict[str, Any], pretrained_state: dict[str, torch.Tensor],
    alpha: float, pretrained_tag: str, output_path: Path,
) -> None:
    source_state = source_payload["model_state_dict"]
    blended_state = {key: value.detach().cpu().clone() for key, value in source_state.items()}
    blended_keys = 0
    for key, fine_tuned in source_state.items():
        if not key.startswith("clip."):
            # Keep the task-specific binary head and any non-CLIP state at the
            # fine-tuned endpoint. The query head is retrained downstream.
            continue
        if key not in pretrained_state or pretrained_state[key].shape != fine_tuned.shape:
            raise ValueError(f"Cannot interpolate pretrained/fine-tuned parameter {key!r}")
        initial = pretrained_state[key].detach().cpu()
        fine_tuned_cpu = fine_tuned.detach().cpu()
        blended_state[key] = (
            alpha * fine_tuned_cpu + (1.0 - alpha) * initial
            if fine_tuned_cpu.is_floating_point() else fine_tuned_cpu.clone()
        )
        blended_keys += 1
    if blended_keys == 0:
        raise ValueError("The fine-tuned checkpoint has no CLIP parameters to interpolate")

    payload = dict(source_payload)
    payload["model_state_dict"] = blended_state
    payload["metrics"] = {}
    payload["wise_ft"] = {
        "alpha_finetuned": float(alpha),
        "alpha_pretrained": float(1.0 - alpha),
        "pretrained_tag": pretrained_tag,
        "interpolated_clip_tensor_count": blended_keys,
        "query_retraining_required": True,
    }
    payload["training_config"] = dict(source_payload.get("training_config", {}))
    payload["training_config"]["wise_ft"] = payload["wise_ft"]
    _atomic_torch_save(payload, output_path)


def run_domain_balance(args: argparse.Namespace) -> None:
    _configure_training(args, "domain_balance")
    output_dir = Path(args.output_dir)
    _atomic_json(output_dir / "config.json", vars(args))
    checkpoint = train(args)
    metrics = _evaluate_checkpoint(args, checkpoint, "E1A_DomainBalanced")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def run_wise_ft(args: argparse.Namespace) -> None:
    _configure_training(args, "wise_source")
    source_dir = Path(args.output_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(source_dir / "config.json", vars(args))
    source_checkpoint = (
        Path(args.finetuned_checkpoint).resolve()
        if args.finetuned_checkpoint else train(args)
    )
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Fine-tuned source checkpoint not found: {source_checkpoint}")

    try:
        source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        source_payload = torch.load(source_checkpoint, map_location="cpu")
    if source_payload.get("model_name") != args.clip_model:
        raise ValueError(
            f"WiSE-FT expects {args.clip_model}, got {source_payload.get('model_name')!r}"
        )
    source_config = source_payload.get("training_config", {})
    if source_config.get("dg_method", "none") != "none":
        raise ValueError("WiSE-FT source must be a plain E1A fine-tuning checkpoint")
    if source_config.get("domain_sampling", "natural") != "natural":
        raise ValueError("WiSE-FT source must use natural domain sampling")
    _evaluate_checkpoint(args, source_checkpoint, "E1A_WiSEFT_Source")

    pretrained_model = AttriVision(
        pretrained=args.pretrained_tag, model_name=args.clip_model,
    )
    pretrained_state = pretrained_model.state_dict()
    wise_root = Path(args.output_root) / "E1A_WiSEFT"
    for alpha in args.wise_alphas:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"WiSE-FT alpha must be in [0, 1], got {alpha}")
        alpha_name = f"alpha_{alpha:.2f}"
        output_dir = wise_root / alpha_name
        args.output_dir = str(output_dir)
        checkpoint = output_dir / "checkpoint_wise.pth"
        _wise_checkpoint(
            source_payload, pretrained_state, alpha, args.pretrained_tag, checkpoint,
        )
        _atomic_json(output_dir / "config.json", {
            **vars(args),
            "experiment": "E1A_WiSEFT",
            "source_checkpoint": str(source_checkpoint.resolve()),
            "alpha_finetuned": float(alpha),
            "alpha_pretrained": float(1.0 - alpha),
        })
        metrics = _evaluate_checkpoint(args, checkpoint, f"E1A_WiSEFT_{alpha_name}")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main() -> None:
    parser = build_parser()
    parser.description = "E1-A CLIP preservation: domain-balanced sampling and WiSE-FT"
    parser.set_defaults(epochs=30)
    parser.add_argument(
        "--experiment", choices=("domain_balance", "wise_ft"), required=True,
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--finetuned-checkpoint",
        help="reuse an E1A plain fine-tuned checkpoint for WiSE-FT; otherwise train one",
    )
    parser.add_argument(
        "--wise-alphas", nargs="+", type=float, default=(0.2, 0.4, 0.6, 0.8, 1.0),
        help="fine-tuned weight in alpha*finetuned + (1-alpha)*pretrained",
    )
    args = parser.parse_args()
    if args.experiment == "domain_balance":
        run_domain_balance(args)
    else:
        run_wise_ft(args)


if __name__ == "__main__":
    main()
