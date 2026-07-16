from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import load_config
from ml.core.dataset import build_dataset as build_expanded_dataset
from ml.core.splits import make_splits


def _apply_splits(decisions: pd.DataFrame) -> pd.DataFrame:
    decisions = decisions.copy()
    for split in make_splits(decisions):
        column = "split_" + split.name.replace("_holdout", "")
        decisions[column] = "train"
        decisions.loc[decisions["decision_id"].isin(split.test_decisions), column] = "test"
    # Keep a strict calibration/validation subset inside the chronological
    # training episodes. It is deterministic and never overlaps the test set.
    if "split_time" in decisions:
        validation = (
            decisions["split_time"].eq("train")
            & decisions["decision_id"].map(
                lambda value: int(hashlib.sha1(str(value).encode()).hexdigest(), 16) % 10 == 0
            )
        )
        decisions.loc[validation, "split_time"] = "validation"
    return decisions


def _write_candidate_parquet(rows_path: Path, output_path: Path, split_maps: dict[str, dict[str, str]]) -> int:
    """Stream the 1M-row candidate table to Parquet without a full-memory copy."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        # The canonical compressed CSV remains complete and reproducible.
        # Parquet is an optional acceleration artifact.
        return -1

    writer = None
    total = 0
    try:
        for chunk in pd.read_csv(rows_path, chunksize=50_000):
            chunk = chunk.rename(columns={"label": "selected", "sample_weight": "teacher_weight"})
            chunk["selected"] = chunk["selected"].astype(bool)
            for column, mapping in split_maps.items():
                chunk[column] = chunk["decision_id"].map(mapping)
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            total += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    return total


def build_dataset(
    config: dict[str, Any],
    processed_dir: Path,
    manifest: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if manifest is None:
        path = processed_dir / "manifest.csv"
        if not path.exists():
            path = processed_dir / "episode_manifest.csv"
        manifest = pd.read_csv(path)
    if "usable_manifest" not in manifest and "usable" in manifest:
        manifest["usable_manifest"] = manifest["usable"].fillna(False).astype(bool)

    _rows, decisions, stats = build_expanded_dataset(
        manifest,
        processed_dir,
        load_rows=False,
        workers=int(config.get("dataset_workers") or 0) or None,
    )
    decisions = _apply_splits(decisions)
    decisions["teacher_weight"] = decisions["sample_weight"]
    decisions["outcome"] = decisions.apply(
        lambda row: "win" if bool(row["target_win"]) else "loss" if bool(row["target_loss"]) else "draw",
        axis=1,
    )
    decisions["value_target"] = decisions["outcome"].map({"win": 1.0, "loss": 0.0, "draw": 0.5})
    decisions["unique_legal"] = decisions["candidate_count"].eq(1)
    decisions["high_importance"] = decisions["selected_action_type"].isin(
        {"attack", "boss", "energy", "retreat", "xerosic", "hammer", "evolve"}
    )
    try:
        decisions.to_parquet(processed_dir / "decision_dataset.parquet", index=False)
    except (ImportError, ModuleNotFoundError, ValueError):
        decisions.to_csv(processed_dir / "decision_dataset.csv.gz", index=False, compression="gzip")

    weight_columns = [
        "decision_id", "rank_weight", "deck_weight", "outcome_weight",
        "action_balance_weight", "seat_confidence", "alignment_confidence",
        "sample_weight",
    ]
    weights = decisions[weight_columns].rename(columns={"sample_weight": "teacher_weight"}).copy()
    weights["data_quality_weight"] = weights["seat_confidence"] * weights["alignment_confidence"]
    weights["importance_weight"] = weights["action_balance_weight"]
    weights["post_action_quality_weight"] = 1.0
    weights["agreement_weight"] = 1.0
    weights["rank_outcome_weight"] = weights["rank_weight"] * weights["outcome_weight"]
    try:
        weights.to_parquet(processed_dir / "expert_weights.parquet", index=False)
    except (ImportError, ModuleNotFoundError, ValueError):
        weights.to_csv(processed_dir / "expert_weights.csv.gz", index=False, compression="gzip")

    split_maps = {
        column: decisions.set_index("decision_id")[column].to_dict()
        for column in ("split_time", "split_team", "split_submission", "split_deck")
        if column in decisions
    }
    candidate_count = _write_candidate_parquet(
        processed_dir / "dataset_rows.csv.gz",
        processed_dir / "legal_candidate_dataset.parquet",
        split_maps,
    )

    stats.update({
        "decision_count": int(len(decisions)),
        "candidate_count": int(stats.get("candidate_row_count", 0) if candidate_count < 0 else candidate_count),
        "parquet_materialized": bool(candidate_count >= 0),
        "episode_count": int(decisions["episode_id"].nunique()),
        "team_count": int(decisions["target_team"].nunique()),
        "submission_count": int(decisions["submission_id"].nunique()),
        "deck_count": int(decisions["deck_hash"].nunique()),
        "split_counts": {
            column: decisions[column].value_counts().to_dict()
            for column in ("split_time", "split_team", "split_submission", "split_deck")
            if column in decisions
        },
        "outcome_counts": decisions.groupby("outcome")["episode_id"].nunique().to_dict(),
        "leakage_guards": {
            "opponent_hand_cards_read": False,
            "visualize_frames_used_for_features": False,
            "future_logs_used_as_features": False,
            "outcome_used_as_policy_feature": False,
            "initial_deck_used_as_policy_feature": False,
            "split_unit": "decision grouped by episode/submission/team/deck",
        },
    })
    (processed_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Feed the recovered count back into manifest statistics.
    stats_path = processed_dir / "manifest_stats.json"
    if stats_path.exists():
        manifest_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        manifest_stats["usable_decision_count"] = int(len(decisions))
        stats_path.write_text(json.dumps(manifest_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return decisions, pd.DataFrame(), weights, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument(
        "--processed", default=str(Path(__file__).resolve().parents[1] / "data_processed")
    )
    args = parser.parse_args()
    _, _, _, stats = build_dataset(load_config(args.config), Path(args.processed))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
