from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from zipfile import ZipFile

try:
    import orjson  # type: ignore
except ImportError:  # portable fallback
    class _OrjsonCompat:
        JSONDecodeError = json.JSONDecodeError
        @staticmethod
        def loads(data):
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            return json.loads(data)
    orjson = _OrjsonCompat()
import pandas as pd

from .align import infer_main_selected_option
from .features import action_type, assert_no_leakage, option_features, state_features
from .splits import make_splits
from .weights import add_decision_weights

# Candidate rows intentionally contain only features, labels and the decision key.
# Rich replay/team/deck metadata lives once per decision in decisions.csv.
META_COLUMNS = {"decision_id", "label", "candidate_index", "sample_weight"}


def _trajectory_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    # pandas may emit NaN for optional values.
    if isinstance(value, float) and pd.isna(value):
        return default
    return value


def _process_zip_job(job: dict[str, Any]) -> dict[str, Any]:
    """Worker: extract one ZIP to a compressed candidate-row part and JSONL decisions."""
    zip_path = str(job["zip_path"])
    part_index = int(job["part_index"])
    part_dir = Path(job["part_dir"])
    trajectories = job["trajectories"]
    rows_path = part_dir / f"rows_{part_index:03d}.csv.gz"
    decisions_path = part_dir / f"decisions_{part_index:03d}.jsonl"

    by_replay: dict[str, list[dict[str, Any]]] = {}
    for row in trajectories:
        by_replay.setdefault(str(row["replay_path"]), []).append(row)

    considered = unresolved = candidate_rows = 0
    skipped_inactive_main = skipped_empty_or_single = 0
    writer: csv.DictWriter | None = None
    feature_fields: list[str] | None = None
    decision_count = 0

    with ZipFile(zip_path) as zf, gzip.open(rows_path, "wt", newline="", encoding="utf-8", compresslevel=3) as rows_handle, decisions_path.open("w", encoding="utf-8") as decision_handle:
        for replay_path, replay_trajectories in by_replay.items():
            replay = orjson.loads(zf.read(replay_path))
            steps = replay.get("steps") or []
            for trajectory in replay_trajectories:
                seat = int(trajectory["target_seat"])
                for step_index in range(max(0, len(steps) - 1)):
                    if seat >= len(steps[step_index]) or seat >= len(steps[step_index + 1]):
                        continue
                    agent_state = steps[step_index][seat]
                    observation = agent_state.get("observation") or {}
                    current = observation.get("current")
                    select = observation.get("select") or {}
                    options = select.get("option") or []
                    if int(select.get("type", -1)) != 0:
                        continue
                    if agent_state.get("status") != "ACTIVE":
                        skipped_inactive_main += 1
                        continue
                    if not current or len(options) < 2:
                        skipped_empty_or_single += 1
                        continue
                    considered += 1
                    selected, align_method, align_confidence = infer_main_selected_option(
                        current, select, options, steps[step_index + 1][seat]
                    )
                    if selected is None or not (0 <= selected < len(options)):
                        unresolved += 1
                        continue

                    selected_type = action_type(current, options[selected])
                    decision_id = f"{trajectory['trajectory_id']}:{step_index}"
                    record = {
                        "decision_id": decision_id,
                        "trajectory_id": trajectory["trajectory_id"],
                        "zip_name": trajectory["zip_name"],
                        "submission_id": int(trajectory["submission_id"]),
                        "episode_id": int(trajectory["episode_id"]),
                        "target_seat": seat,
                        "target_team": trajectory["target_team"],
                        "rank": int(trajectory["rank"]),
                        "deck_hash": trajectory["deck_hash"],
                        "majkel_distance": _trajectory_value(trajectory, "majkel_distance", None),
                        "target_win": bool(trajectory["target_win"]),
                        "target_loss": bool(trajectory["target_loss"]),
                        "seat_confidence": float(trajectory["seat_confidence"]),
                        "alignment_method": align_method,
                        "alignment_confidence": float(align_confidence),
                        "selected_action_type": selected_type,
                        "candidate_count": len(options),
                    }
                    decision_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    decision_count += 1

                    base_state = state_features(current)
                    for candidate_index, option in enumerate(options):
                        feature = option_features(current, select, option, base_state=base_state)
                        feature.update({
                            "decision_id": decision_id,
                            "candidate_index": candidate_index,
                            "label": int(candidate_index == selected),
                        })
                        if writer is None:
                            feature_fields = list(feature.keys())
                            writer = csv.DictWriter(rows_handle, fieldnames=feature_fields)
                            writer.writeheader()
                        writer.writerow(feature)
                        candidate_rows += 1

    return {
        "zip_path": zip_path,
        "rows_path": str(rows_path),
        "decisions_path": str(decisions_path),
        "feature_fields": feature_fields or [],
        "considered": considered,
        "unresolved": unresolved,
        "candidate_rows": candidate_rows,
        "decision_count": decision_count,
        "skipped_inactive_main": skipped_inactive_main,
        "skipped_empty_or_single": skipped_empty_or_single,
    }


def _jobs_from_manifest(usable: pd.DataFrame, part_dir: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for index, (zip_path, frame) in enumerate(usable.groupby("zip_path", sort=False)):
        records = frame.where(pd.notna(frame), None).to_dict(orient="records")
        jobs.append({
            "zip_path": str(zip_path),
            "part_index": index,
            "part_dir": str(part_dir),
            "trajectories": records,
        })
    return jobs


def build_dataset(
    manifest: pd.DataFrame,
    output_dir: str | Path,
    *,
    load_rows: bool = True,
    workers: int | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame, dict[str, Any]]:
    """Build a leakage-safe legal-option ranking dataset.

    ZIPs are independently processed in worker processes. Policy features use
    only observation t and the legal candidate option. The label is the exact
    CABT legal-option index serialized on the same seat at replay step t+1.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    usable = manifest[manifest["usable_manifest"] == True].copy()
    final_path = output / "dataset_rows.csv.gz"
    part_dir = output / ".dataset_parts"
    shutil.rmtree(part_dir, ignore_errors=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    jobs = _jobs_from_manifest(usable, part_dir)
    requested_workers = workers or min(4, max(1, os.cpu_count() or 1))
    worker_count = max(1, min(requested_workers, len(jobs)))
    results: list[dict[str, Any]] = []
    if worker_count == 1:
        results = [_process_zip_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_process_zip_job, job): job for job in jobs}
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda x: x["rows_path"])

    decision_records: list[dict[str, Any]] = []
    for result in results:
        with open(result["decisions_path"], encoding="utf-8") as handle:
            decision_records.extend(json.loads(line) for line in handle if line.strip())
    decisions = pd.DataFrame(decision_records)
    if decisions.empty:
        raise RuntimeError("No aligned decisions were produced")
    decisions = add_decision_weights(decisions)
    for split in make_splits(decisions):
        column = f"split_{split.name.removesuffix('_holdout')}"
        decisions[column] = "train"
        decisions.loc[decisions["decision_id"].isin(split.test_decisions), column] = "test"
    decisions.to_csv(output / "decisions.csv", index=False)
    weight_map = decisions.set_index("decision_id")["sample_weight"].to_dict()

    first = True
    with gzip.open(final_path, "wt", newline="", encoding="utf-8", compresslevel=3) as output_handle:
        for result in results:
            rows_path = result["rows_path"]
            if not result["feature_fields"]:
                continue
            for chunk in pd.read_csv(rows_path, chunksize=50_000):
                chunk["sample_weight"] = chunk["decision_id"].map(weight_map).astype(float)
                chunk.to_csv(output_handle, index=False, header=first)
                first = False
    if first:
        raise RuntimeError("No candidate rows were produced")

    sample = pd.read_csv(final_path, nrows=5)
    feature_cols = [c for c in sample.columns if c not in META_COLUMNS]
    assert_no_leakage(feature_cols)
    considered = sum(int(x["considered"]) for x in results)
    unresolved = sum(int(x["unresolved"]) for x in results)
    candidate_rows = sum(int(x["candidate_rows"]) for x in results)
    stats = {
        "usable_trajectory_count": int(usable["trajectory_id"].nunique()),
        "considered_active_main_decisions": considered,
        "usable_decision_count": int(len(decisions)),
        "candidate_row_count": candidate_rows,
        "alignment_rate": float(len(decisions) / considered) if considered else 0.0,
        "unresolved_decision_count": unresolved,
        "skipped_inactive_main_observation_count": sum(int(x["skipped_inactive_main"]) for x in results),
        "skipped_empty_or_single_option_count": sum(int(x["skipped_empty_or_single"]) for x in results),
        "mean_candidates_per_decision": float(decisions["candidate_count"].mean()),
        "action_type_counts": {str(k): int(v) for k, v in decisions["selected_action_type"].value_counts().items()},
        "alignment_methods": {str(k): int(v) for k, v in decisions["alignment_method"].value_counts().items()},
        "team_count": int(decisions["target_team"].nunique()),
        "submission_count": int(decisions["submission_id"].nunique()),
        "deck_count": int(decisions["deck_hash"].nunique()),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "policy_feature_provenance": "observation_t_and_legal_option_only",
        "label_provenance": "same_seat_action_from_replay_step_t_plus_1",
        "dataset_workers": worker_count,
        "zip_part_stats": [
            {k: v for k, v in result.items() if k not in {"feature_fields", "rows_path", "decisions_path"}}
            for result in results
        ],
    }
    (output / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(part_dir, ignore_errors=True)
    rows = pd.read_csv(final_path) if load_rows else None
    return rows, decisions, stats


def feature_columns(rows: pd.DataFrame) -> list[str]:
    cols = [c for c in rows.columns if c not in META_COLUMNS]
    assert_no_leakage(cols)
    return cols


def load_dataset_rows(path: str | Path) -> pd.DataFrame:
    """Load the candidate table with compact dtypes suitable for 4 GB hosts."""
    path = Path(path)
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    dtype: dict[str, str] = {}
    for column in columns:
        if column == "decision_id":
            dtype[column] = "category"
        elif column == "action_type":
            dtype[column] = "category"
        elif column == "label":
            dtype[column] = "int8"
        elif column == "candidate_index":
            dtype[column] = "int16"
        else:
            dtype[column] = "float32"
    return pd.read_csv(path, dtype=dtype)
