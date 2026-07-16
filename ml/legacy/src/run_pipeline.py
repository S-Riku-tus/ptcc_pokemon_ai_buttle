from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .build_dataset import build_dataset
from .build_manifest import build_manifest
from .common import load_config
from .export_submission import export
from .reporting import generate_reports
from ml.core.ranker_training import train_all


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument(
        "--reuse-processed", action="store_true",
        help="Reuse bundled manifest/dataset artifacts; useful when raw replay ZIPs are not mounted.",
    )
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    workspace = Path(__file__).resolve().parents[3]
    processed = workspace / "data" / "ml" / "alakazam" / "processed"

    if args.reuse_processed:
        manifest = pd.read_csv(processed / "manifest.csv")
        manifest_stats = _load_json(processed / "manifest_stats.json")
        dataset_stats = _load_json(processed / "dataset_stats.json")
    else:
        manifest, _, manifest_stats = build_manifest(config, processed)
        if args.skip_dataset:
            dataset_stats = _load_json(processed / "dataset_stats.json")
        else:
            _, _, _, dataset_stats = build_dataset(config, processed, manifest)

    training = None
    if not args.skip_training:
        training = train_all(
            processed,
            workspace / "data" / "ml" / "alakazam" / "models",
            workspace / "data" / "ml" / "alakazam" / "reports",
        )
        shutil.copy2(
            workspace / "data" / "ml" / "alakazam" / "models" / "ranker_model.json",
            workspace / "agents" / "alakazam_ml_v2_expanded" / "ranker_model.json",
        )

    generated = generate_reports(base)
    package = export(base)
    print(json.dumps({
        "manifest": manifest_stats,
        "dataset": dataset_stats,
        "training_splits": (training or {}).get("splits"),
        "reports": {name: str(path) for name, path in generated.items()},
        "package": package,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
