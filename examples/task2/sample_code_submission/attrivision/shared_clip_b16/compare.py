"""Build E1 comparison CSV and Markdown from recorded validation results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sidecar_metadata(path: Path) -> dict[str, Any]:
    for name in ("experiment_config.json", "config.json"):
        candidate = path.parent / name
        if candidate.is_file():
            return _read_json(candidate)
    return {}


def _metric(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None or row.get(key) in (None, "", "nan", "NaN"):
        return None
    return float(row[key])


def _load_result(path: Path, expected_experiment: str, primary: str) -> dict[str, Any]:
    value = _read_json(path)
    metadata = _sidecar_metadata(path)
    if value.get("split", "val") != "val":
        raise ValueError(f"{expected_experiment} result is not from the validation split: {path}")
    protocol = value.get("validation_protocol") or metadata.get("validation_protocol")
    dataset_root = value.get("dataset_root") or metadata.get("data_root")
    if dataset_root:
        dataset_root = str(Path(dataset_root).resolve())
    temperature = value.get("category_temperature", metadata.get("category_temperature"))

    views: dict[str, dict[str, Any]] = {}
    if isinstance(value.get("scores"), list):
        for row in value["scores"]:
            kind = str(row.get("score_type", ""))
            if kind == "fusion" and abs(float(row.get("alpha", -1.0)) - 0.5) > 1e-9:
                continue
            views[kind] = row
    else:
        metrics = value.get("metrics", value)
        views["global"] = metrics

    selected = views.get(primary)
    if selected is None:
        raise ValueError(
            f"{expected_experiment} is missing the requested {primary} metric view in {path}"
        )
    return {
        "experiment": expected_experiment,
        "source": str(path.resolve()),
        "protocol": protocol,
        "dataset_root": dataset_root,
        "category_temperature": float(temperature) if temperature is not None else None,
        "primary": selected,
        "global": views.get("global"),
        "local": views.get("local"),
        "fused": views.get("fusion"),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{100 * float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def build_comparison(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = (
        ("기존 A7-B/32", Path(args.a7_b32), "global"),
        ("E1-0", Path(args.e1_0), "fusion"),
        ("E1-A", Path(args.e1_a), "global"),
        ("E1-B", Path(args.e1_b), "fusion"),
        ("E1-C", Path(args.e1_c), "fusion"),
        ("E1-D", Path(args.e1_d), "fusion"),
        ("E1-E", Path(args.e1_e), "fusion"),
    )
    results = {
        name: _load_result(path, name, primary)
        for name, path, primary in sources
    }
    protocols = {row["protocol"] for row in results.values()}
    datasets = {row["dataset_root"] for row in results.values()}
    temperatures = {row["category_temperature"] for row in results.values()}
    comparable = (
        protocols == {"native52_category_nll"}
        and len(datasets) == 1
        and None not in datasets
        and temperatures == {0.01}
    )

    settings = {
        "기존 A7-B/32": ("CLIP ViT-B/32", "A7 full fine-tuning", "global"),
        "E1-0": ("pretrained CLIP ViT-B/16", "frozen encoder + local decoder", "global + local fusion"),
        "E1-A": ("pretrained CLIP ViT-B/16", "A7 full fine-tuning", "global"),
        "E1-B": ("E1-A best checkpoint", "frozen encoder + local decoder", "global + local fusion"),
        "E1-C": ("E1-B best checkpoint", "shared vision + local decoder joint fine-tuning", "global + local fusion"),
        "E1-D": ("E1-B best checkpoint", "full CLIP B/16 + full local decoder joint fine-tuning", "global + local fusion"),
        "E1-E": ("OpenAI pretrained CLIP ViT-B/16", "full CLIP B/16 + full local decoder joint fine-tuning", "global + local fusion"),
    }
    rows: list[dict[str, Any]] = []
    for name, result in results.items():
        primary = result["primary"]
        row: dict[str, Any] = {
            "Experiment": name,
            "Initialization": settings[name][0],
            "Training": settings[name][1],
            "Primary descriptor": settings[name][2],
            "Rank-1": _metric(primary, "rank1"),
            "mAP": _metric(primary, "mAP"),
            "mADM": _metric(primary, "mADM"),
            "mINP": _metric(primary, "mINP"),
            "Global mADM": _metric(result["global"], "mADM"),
            "Local mADM": _metric(result["local"], "mADM"),
            "Fused mADM": _metric(result["fused"], "mADM"),
            "Validation split": "val",
            "Validation protocol": result["protocol"] or "unknown",
            "Category temperature": result["category_temperature"],
            "Comparable": comparable,
            "Result file": result["source"],
        }
        for descriptor in ("global", "local", "fused"):
            descriptor_row = result[descriptor]
            for metric_name, metric_key in (
                ("Rank-1", "rank1"), ("mAP", "mAP"),
                ("mADM", "mADM"), ("mINP", "mINP"),
            ):
                row[f"{descriptor.title()} {metric_name}"] = _metric(
                    descriptor_row, metric_key,
                )
        rows.append(row)

    def m_adm(name: str) -> float | None:
        return _metric(results[name]["primary"], "mADM")

    pairs = (
        ("E1-A vs existing A7-B/32", "E1-A", "기존 A7-B/32"),
        ("E1-B vs E1-A", "E1-B", "E1-A"),
        ("E1-C vs E1-B", "E1-C", "E1-B"),
        ("E1-D vs E1-C", "E1-D", "E1-C"),
        ("E1-D vs E1-B", "E1-D", "E1-B"),
        ("E1-E vs E1-D", "E1-E", "E1-D"),
        ("E1-E vs E1-0", "E1-E", "E1-0"),
        ("E1-B vs E1-0", "E1-B", "E1-0"),
    )
    deltas = [
        {
            "Comparison": label,
            "mADM delta": (
                _difference(m_adm(first), m_adm(second)) if comparable else None
            ),
        }
        for label, first, second in pairs
    ]
    return rows, deltas


def _difference(first: float | None, second: float | None) -> float | None:
    return None if first is None or second is None else first - second


def write_comparison(
    rows: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    delta_path = output_dir / "deltas.csv"
    with delta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)

    markdown_path = output_dir / "comparison.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# Shared CLIP ViT-B/16 E1 validation comparison\n\n")
        handle.write("All metrics below are from the recorded validation split. A fused descriptor uses the fixed local weight alpha=0.5.\n\n")
        handle.write("| Experiment | Initialization | Training | Descriptor | Rank-1 | mAP | mADM | mINP | Global mADM | Local mADM | Fused mADM | Comparable |\n")
        handle.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
        for row in rows:
            metrics = [
                _fmt(row[key]) for key in (
                    "Rank-1", "mAP", "mADM", "mINP", "Global mADM", "Local mADM", "Fused mADM",
                )
            ]
            handle.write(
                f"| {row['Experiment']} | {row['Initialization']} | {row['Training']} | "
                f"{row['Primary descriptor']} | " + " | ".join(metrics)
                + f" | {row['Comparable']} |\n"
            )
        handle.write("\n## mADM changes\n\n| Comparison | Delta (percentage points) |\n|---|---:|\n")
        for row in deltas:
            value = row["mADM delta"]
            formatted = "-" if value is None else f"{100 * value:+.2f} pp"
            handle.write(f"| {row['Comparison']} | {formatted} |\n")
        if not rows[0]["Comparable"]:
            handle.write(
                "\nDirect comparisons are marked unavailable because the recorded dataset, "
                "validation protocol, or category temperature differ or are missing.\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the E1 result comparison CSV and Markdown table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--a7-b32", default="outputs/attrivision_ablation_mADM/A7/validation_metrics.json")
    parser.add_argument("--e1-0", default="outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-0/evaluation_val.json")
    parser.add_argument("--e1-a", default="outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-A/evaluation_val.json")
    parser.add_argument("--e1-b", default="outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-B/evaluation_val.json")
    parser.add_argument("--e1-c", default="outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-C/evaluation_val.json")
    parser.add_argument("--e1-d", default="outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-D/evaluation_val.json")
    parser.add_argument("--e1-e", default="outputs/BackboneUpgrade/E1_SharedCLIP_B16/E1-E/evaluation_val.json")
    parser.add_argument("--output-dir", default="outputs/BackboneUpgrade/E1_SharedCLIP_B16/Comparison")
    args = parser.parse_args()
    rows, deltas = build_comparison(args)
    write_comparison(rows, deltas, Path(args.output_dir))
    print(f"Wrote {Path(args.output_dir) / 'comparison.csv'} and comparison.md", flush=True)


if __name__ == "__main__":
    main()
