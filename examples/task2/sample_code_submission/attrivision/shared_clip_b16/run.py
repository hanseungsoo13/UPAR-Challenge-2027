"""Command-line entry point for the isolated shared CLIP B/16 E1 experiment."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

SUBMISSION_DIR = Path(__file__).resolve().parents[2]
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from attrivision.cli import build_parser as build_a7_parser  # noqa: E402
from attrivision.datasets.attribute_prompts import ATTRIBUTE_PROMPTS, CategoryPromptMapper  # noqa: E402
from attrivision.models.attrivision import AttriVision  # noqa: E402
from attrivision.shared_clip_b16.experiment import (  # noqa: E402
    _checkpoint_payload,
    atomic_json,
    evaluate_checkpoint,
    train_a7_b16,
    train_shared,
)
from attrivision.shared_clip_b16.model import (  # noqa: E402
    ARCHITECTURE,
    MODEL_NAME,
    SharedCLIPB16AttrQuery,
)
from upar.config import choose_device, set_seed  # noqa: E402


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "e1_shared_clip_b16.json"
)


def _read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model_name") != MODEL_NAME:
        raise ValueError(f"E1 config must select {MODEL_NAME}")
    return config


def _add_training_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--data-root")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--decoder-learning-rate", type=float)
    parser.add_argument("--vision-learning-rate", type=float)
    parser.add_argument("--text-learning-rate", type=float)
    parser.add_argument("--global-loss-weight", type=float)
    parser.add_argument("--local-loss-weight", type=float)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)


def _training_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _read_config(args.config)
    for key in (
        "output_root", "data_root", "device", "epochs", "batch_size",
        "eval_batch_size", "num_workers", "decoder_learning_rate",
        "vision_learning_rate", "text_learning_rate", "global_loss_weight", "local_loss_weight",
        "max_train_samples", "max_val_samples", "amp",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.output_dir:
        config["output_dir"] = str(Path(args.output_dir).resolve())
    else:
        config["output_dir"] = str(
            Path(config["output_root"]).resolve() / args.experiment_name
        )
    if int(config["image_size"]) != 224:
        raise ValueError("CLIP ViT-B/16 E1 requires 224x224 input")
    if float(config["selection_alpha"]) != 0.5:
        raise ValueError("The default E1 evaluation protocol fixes fusion alpha=0.5")
    return config


def _run_train(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Train an E1 shared CLIP B/16 AttriQuery stage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage", choices=("control", "local", "joint", "full"), required=True)
    _add_training_overrides(parser)
    args = parser.parse_args(argv)
    if args.stage == "control":
        stage, init_source, experiment = "local", "pretrained", "E1-0"
    elif args.stage == "local":
        if not args.init_checkpoint:
            parser.error("--stage local requires --init-checkpoint with the E1-A best checkpoint")
        stage, init_source, experiment = "local", "a7", "E1-B"
    elif args.stage == "joint":
        if not args.init_checkpoint:
            parser.error("--stage joint requires --init-checkpoint with the E1-B best checkpoint")
        stage, init_source, experiment = "joint", "shared", "E1-C"
    else:
        if not args.init_checkpoint:
            parser.error("--stage full requires --init-checkpoint with the E1-B best checkpoint")
        stage, init_source, experiment = "full", "shared", "E1-D"
    if args.resume and args.init_checkpoint is None and stage != "local":
        parser.error("--resume must be used with the same --init-checkpoint as the original run")
    args.experiment_name = experiment
    config = _training_config(args)
    best = train_shared(
        stage=stage,
        init_source=init_source,
        init_checkpoint=args.init_checkpoint,
        config=config,
        resume=args.resume,
    )
    result = evaluate_checkpoint(
        checkpoint=best,
        experiment=experiment,
        data_root=config["data_root"],
        output_path=Path(config["output_dir"]) / "evaluation_val.json",
        device_name=config["device"],
        batch_size=int(config["eval_batch_size"]),
        num_workers=int(config["num_workers"]),
        amp=bool(config["amp"]),
        category_temperature=float(config["category_temperature"]),
        max_val_samples=config.get("max_val_samples"),
    )
    print(f"Best checkpoint: {best}", flush=True)
    print(f"Validation result: {Path(config['output_dir']) / 'evaluation_val.json'}", flush=True)
    if experiment == "E1-B":
        print("E1-B evaluation includes global-only, local-only, and alpha=0.5 fusion.", flush=True)
    if experiment == "E1-C":
        print("E1-C evaluation includes global-only, local-only, and alpha=0.5 fusion.", flush=True)
    if experiment == "E1-D":
        print("E1-D evaluation includes global-only, local-only, and alpha=0.5 fusion.", flush=True)
    del result


def _run_a7(argv: list[str]) -> None:
    args = build_a7_parser().parse_args(argv)
    if args.mode not in {"train", "train_eval", "smoke"}:
        raise ValueError("Use the top-level eval command to evaluate an E1-A checkpoint")
    checkpoint = train_a7_b16(args)
    if args.mode == "smoke":
        return
    output_dir = Path(args.output_dir).resolve()
    result = evaluate_checkpoint(
        checkpoint=checkpoint,
        experiment="E1-A",
        data_root=args.data_root,
        output_path=output_dir / "evaluation_val.json",
        device_name=args.device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
        category_temperature=args.category_temperature,
        max_val_samples=args.max_val_samples,
    )
    print(
        f"E1-A best checkpoint: {checkpoint}; "
        f"validation mADM={result['metrics'].get('mADM', float('nan')):.6f}",
        flush=True,
    )


def _run_eval(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an E1 checkpoint on the UPAR validation split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", choices=("E1-A", "E1-0", "E1-B", "E1-C", "E1-D"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--category-temperature", type=float, default=0.01)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    output = Path(args.output) if args.output else Path(args.checkpoint).resolve().parent / "evaluation_val.json"
    evaluate_checkpoint(
        checkpoint=args.checkpoint,
        experiment=args.experiment,
        data_root=args.data_root,
        output_path=output,
        device_name=args.device,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
        category_temperature=args.category_temperature,
        max_val_samples=args.max_val_samples,
    )
    print(f"Validation result: {output.resolve()}", flush=True)


def _smoke(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Check E1 shared-encoder shapes, trainability, gradients, and checkpoint I/O",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    set_seed(args.seed, deterministic=True)
    device = choose_device(args.device)
    attributes = list(ATTRIBUTE_PROMPTS)
    mapper = CategoryPromptMapper(attributes)
    model = SharedCLIPB16AttrQuery(
        AttriVision(pretrained=None, model_name=MODEL_NAME),
        attributes,
        stage="local",
    ).to(device)
    model.assert_trainability()
    image = torch.randn(args.batch_size, 3, 224, 224, device=device)
    model.eval()
    with torch.no_grad():
        global_features, local_features, patches = model.joint_features_with_tokens(image)
        state_features = model.encode_text(model.base.tokenize(mapper.prompts).to(device))
    assert global_features.shape == (args.batch_size, 512), tuple(global_features.shape)
    assert patches.shape == (args.batch_size, 196, 768), tuple(patches.shape)
    assert local_features.shape == (args.batch_size, 12, 512), tuple(local_features.shape)
    assert state_features.shape == (52, 512), tuple(state_features.shape)
    visual_names = [name for name, _module in model.named_modules() if name == "base.clip.visual"]
    assert len(visual_names) == 1, f"expected one visual encoder, got {visual_names}"
    assert model.parameter_counts()["vision_encoder_instances"] == 1

    model.stage = "joint"
    model._set_trainability()
    model.assert_trainability()
    model.train()
    global_features, local_features = model.joint_features(image)
    global_gradient = torch.autograd.grad(
        global_features[:, 0].sum(), list(model.base.clip.visual.parameters()),
        retain_graph=True, allow_unused=True,
    )
    local_gradient = torch.autograd.grad(
        local_features[..., 0].sum(), list(model.base.clip.visual.parameters()),
        allow_unused=True,
    )
    assert any(value is not None and value.abs().sum() > 0 for value in global_gradient)
    assert any(value is not None and value.abs().sum() > 0 for value in local_gradient)

    model.stage = "full"
    model._set_trainability()
    model.assert_trainability()
    model.train()
    global_features, local_features = model.joint_features(image)
    text_projection = model.base.clip.text_projection
    state_features = model.encode_text(model.base.tokenize(mapper.prompts).to(device))
    global_loss = global_features[:, 0].sum() + state_features[:, 0].sum()
    local_loss = local_features[..., 0].sum() + state_features[:, 1].sum()
    for loss in (global_loss, local_loss):
        vision_gradient, text_gradient = torch.autograd.grad(
            loss,
            [next(model.base.clip.visual.parameters()), text_projection],
            retain_graph=loss is global_loss,
            allow_unused=True,
        )
        assert vision_gradient is not None and vision_gradient.abs().sum() > 0
        assert text_gradient is not None and text_gradient.abs().sum() > 0

    model.stage = "local"
    model._set_trainability()
    model.assert_trainability()
    payload = _checkpoint_payload(
        model, attributes, "E1-B", 1, {},
        {"num_heads": 8, "dropout": 0.1, "use_ffn": True, "init_source": "a7"},
    )
    with tempfile.TemporaryDirectory(prefix="e1_shared_clip_b16_") as temporary:
        path = Path(temporary) / "checkpoint.pth"
        torch.save(payload, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(loaded["model_state_dict"], strict=True)
    assert payload["architecture"] == ARCHITECTURE
    print(
        "Shared CLIP B/16 smoke checks passed: one vision encoder; "
        "global=[B,512], patches=[B,196,768], local=[B,12,512], text=[52,512]; "
        "E1-C vision gradients and E1-D global/local vision/text gradients passed; checkpoint I/O passed.",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: run.py {a7|train|eval|smoke} ...", file=sys.stderr)
        raise SystemExit(2)
    command, argv = sys.argv[1], sys.argv[2:]
    if command == "a7":
        _run_a7(argv)
    elif command == "train":
        _run_train(argv)
    elif command == "eval":
        _run_eval(argv)
    elif command == "smoke":
        _smoke(argv)
    else:
        raise SystemExit(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
