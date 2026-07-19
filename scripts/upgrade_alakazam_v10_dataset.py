"""Derive the v10 training table from the validated v8 replay-aligned table.

The replay labels and split identities are unchanged.  Only observation-time
features that can be reconstructed exactly from existing columns are added,
and teacher/source weights are recomputed with the v10 balancing policy.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.weights import add_decision_weights


PIVOT_IDS = {140, 305, 343, 741, 742}


def _add_v10_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    active_present = (frame["self_active_id"] >= 0).astype("int8")
    frame["self_board_count"] = frame["self_bench_count"] + active_present
    frame["self_last_body_risk"] = (frame["self_board_count"] <= 1).astype("int8")
    frame["has_backup_route_body"] = (frame["field_route_body_count"] >= 2).astype("int8")
    frame["active_is_one_energy_pivot"] = frame["self_active_id"].isin(PIVOT_IDS).astype("int8")
    frame["support_pivot_ready"] = (
        frame["self_active_id"].isin(PIVOT_IDS)
        & (frame["self_active_energy"] == 0)
        & (frame["ready_bench_alakazam_count"] > 0)
    ).astype("int8")
    frame["deck_runway_margin"] = (
        frame["self_deck_count"] - frame["self_prize_count"] - 3
    )
    frame["deck_pressure_risk"] = (frame["deck_runway_margin"] <= 4).astype("int8")

    # Card 112 was not materialized in the v8 key-card table.  It cannot be
    # reconstructed from aggregate fields, so it is explicitly unknown/zero
    # rather than inferred from hidden information.
    for prefix in ("hand", "field", "discard", "opp_field"):
        frame[f"{prefix}_112"] = 0
    frame["opp_munkidori_count"] = 0
    frame["opp_has_munkidori"] = 0
    frame["opp_spread_package_count"] = (
        frame["opp_froslass_count"] + frame["opp_field_648"]
    )
    frame["self_has_shaymin"] = (frame["field_343"] > 0).astype("int8")

    action = frame["action_type"].astype(str)
    target_area = frame["candidate_inplay_area"]
    frame["energy_support_pivot_value"] = (
        action.eq("energy")
        & target_area.eq(4)
        & frame["candidate_target_id"].isin(PIVOT_IDS)
        & frame["support_pivot_ready"].eq(1)
    ).astype("int8")
    frame["bench_shaymin_into_spread_package"] = (
        action.eq("bench")
        & frame["candidate_card_id"].eq(343)
        & frame["opp_spread_package_count"].gt(0)
    ).astype("int8")
    frame["bench_backup_route_under_boardout_risk"] = (
        action.eq("bench")
        & frame["candidate_card_id"].eq(741)
        & frame["self_board_count"].le(2)
    ).astype("int8")
    frame["optional_draw_under_deck_pressure"] = (
        action.isin({"ability", "trainer"}) & frame["deck_pressure_risk"].eq(1)
    ).astype("int8")
    frame["retreat_support_pivot_value"] = (
        action.eq("retreat") & frame["support_pivot_ready"].eq(1)
    ).astype("int8")
    return frame


def upgrade(source: Path, target: Path, chunk_size: int = 50_000) -> dict[str, object]:
    target.mkdir(parents=True, exist_ok=True)
    decisions = pd.read_csv(source / "decisions.csv")
    decisions = add_decision_weights(decisions)
    decisions_tmp = target / "decisions.csv.tmp"
    decisions.to_csv(decisions_tmp, index=False)
    decisions_tmp.replace(target / "decisions.csv")
    weight_map = decisions.set_index("decision_id")["sample_weight"]

    rows_source = source / "dataset_rows.csv.gz"
    rows_tmp = target / "dataset_rows.csv.gz.tmp"
    first = True
    row_count = 0
    feature_columns: list[str] = []
    with gzip.open(rows_tmp, "wt", newline="", encoding="utf-8", compresslevel=3) as handle:
        for chunk in pd.read_csv(rows_source, chunksize=chunk_size):
            chunk = _add_v10_features(chunk)
            chunk["sample_weight"] = chunk["decision_id"].map(weight_map).astype("float32")
            chunk.to_csv(handle, index=False, header=first)
            if first:
                feature_columns = [
                    column for column in chunk.columns
                    if column not in {"decision_id", "label", "candidate_index", "sample_weight"}
                ]
                first = False
            row_count += len(chunk)
            print(json.dumps({"event": "v10_dataset_chunk", "rows": row_count}), flush=True)
    rows_tmp.replace(target / "dataset_rows.csv.gz")

    for name in ("manifest.csv", "manifest_stats.json", "deck_clusters.csv"):
        path = source / name
        if path.exists():
            shutil.copy2(path, target / name)
    stats_path = source / "dataset_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    stats.update({
        "candidate_row_count": row_count,
        "usable_decision_count": len(decisions),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "v10_upgrade_source": str(source),
        "v10_source_balancing": True,
        "v10_munkidori_columns": "zero_unknown_from_v8_aggregate_table",
    })
    (target / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"decisions": len(decisions), "rows": row_count, "features": len(feature_columns)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    print(json.dumps(upgrade(args.source, args.target, args.chunk_size), indent=2))


if __name__ == "__main__":
    main()
