from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_dataset import build_dataset
from .build_manifest import build_manifest
from .common import load_config
from .export_submission import export
from .parse_replays import build_alignment_report
from .reporting import generate_reports
from .train_ranker import train_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    manifest, _, manifest_stats = build_manifest(config, base / "data_processed")
    alignment = build_alignment_report(manifest, base / "reports" / "alignment_report.json")
    _, _, _, dataset_stats = build_dataset(config, base / "data_processed", manifest)
    if not args.skip_training:
        train_all(
            base / "data_processed", base / "models", base / "reports",
            seed=int(config["seed"]), confidence_threshold=float(config["confidence_threshold"]),
        )
    generate_reports(base)
    package = export(base)
    print(json.dumps({
        "manifest": manifest_stats,
        "alignment": alignment.get("selected_method"),
        "dataset": dataset_stats,
        "package": package,
    }, indent=2))


if __name__ == "__main__":
    main()
