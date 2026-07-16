from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .dataset import build_dataset
from .manifest import build_deck_clusters, build_manifest
from .replay_io import discover_zip_paths
from .train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", help="ZIP files or directories containing replay ZIPs")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    workdir = Path(args.workdir)
    processed = workdir / "data" / "processed"
    artifacts = workdir / "artifacts"
    reports = workdir / "reports"
    zips = discover_zip_paths(args.roots)
    manifest, _, _ = build_manifest(zips, processed)
    build_deck_clusters(manifest, processed / "deck_clusters.csv")
    rows, decisions, _ = build_dataset(manifest, processed)
    if not args.skip_train:
        run_training(rows, decisions, artifacts, reports)


if __name__ == "__main__":
    main()
