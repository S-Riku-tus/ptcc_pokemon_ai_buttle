from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import assert_no_leakage


def build_matrix_store(processed_dir: str | Path, chunk_size: int = 20_000, progress: bool = False) -> dict[str, Any]:
    processed = Path(processed_dir)
    matrix_dir = processed / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads((processed / "dataset_stats.json").read_text(encoding="utf-8"))
    feature_columns = list(stats["feature_columns"])
    assert_no_leakage(feature_columns)
    row_count = int(stats["candidate_row_count"])
    decisions = pd.read_csv(processed / "decisions.csv")
    decisions["decision_numeric_id"] = np.arange(len(decisions), dtype=np.int32)
    decisions.to_csv(matrix_dir / "decisions_indexed.csv", index=False)
    decision_map = dict(zip(decisions["decision_id"].astype(str), decisions["decision_numeric_id"].astype(int)))

    action_values = sorted(set(stats.get("action_type_counts", {})) | {
        "ability", "attack", "bench", "boss", "end", "energy", "evolve",
        "hammer", "other", "retreat", "trainer", "xerosic",
    })
    action_map = {value: index for index, value in enumerate(action_values)}

    x = np.lib.format.open_memmap(matrix_dir / "features.npy", mode="w+", dtype=np.float32, shape=(row_count, len(feature_columns)))
    y = np.lib.format.open_memmap(matrix_dir / "labels.npy", mode="w+", dtype=np.int8, shape=(row_count,))
    decision_index = np.lib.format.open_memmap(matrix_dir / "decision_index.npy", mode="w+", dtype=np.int32, shape=(row_count,))
    candidate_index = np.lib.format.open_memmap(matrix_dir / "candidate_index.npy", mode="w+", dtype=np.int16, shape=(row_count,))
    sample_weight = np.lib.format.open_memmap(matrix_dir / "sample_weight.npy", mode="w+", dtype=np.float32, shape=(row_count,))
    candidate_action_type = np.lib.format.open_memmap(matrix_dir / "candidate_action_type.npy", mode="w+", dtype=np.int8, shape=(row_count,))

    usecols = ["decision_id", "candidate_index", "label", "sample_weight"] + feature_columns
    dtypes: dict[str, str] = {
        "decision_id": "string", "candidate_index": "int16", "label": "int8", "sample_weight": "float32", "action_type": "string",
    }
    for column in feature_columns:
        if column != "action_type":
            dtypes[column] = "float32"

    offset = 0
    for chunk_no, chunk in enumerate(pd.read_csv(processed / "dataset_rows.csv.gz", usecols=usecols, dtype=dtypes, chunksize=chunk_size)):
        n = len(chunk)
        if offset + n > row_count:
            raise ValueError("Dataset has more rows than dataset_stats.json")
        action_codes = chunk["action_type"].astype(str).map(action_map)
        if action_codes.isna().any():
            missing = sorted(chunk.loc[action_codes.isna(), "action_type"].astype(str).unique())
            raise ValueError(f"Unknown action types: {missing}")
        numeric = chunk[feature_columns].copy()
        numeric["action_type"] = action_codes.astype(np.float32)
        x[offset:offset+n] = numeric.to_numpy(dtype=np.float32, copy=False)
        mapped = chunk["decision_id"].astype(str).map(decision_map)
        if mapped.isna().any():
            raise ValueError("Candidate row references an unknown decision_id")
        decision_index[offset:offset+n] = mapped.to_numpy(dtype=np.int32, copy=False)
        candidate_index[offset:offset+n] = chunk["candidate_index"].to_numpy(dtype=np.int16, copy=False)
        y[offset:offset+n] = chunk["label"].to_numpy(dtype=np.int8, copy=False)
        sample_weight[offset:offset+n] = chunk["sample_weight"].to_numpy(dtype=np.float32, copy=False)
        candidate_action_type[offset:offset+n] = action_codes.to_numpy(dtype=np.int8, copy=False)
        offset += n
        if progress and chunk_no % 10 == 0:
            print(json.dumps({"event": "matrix_progress", "rows": offset, "total": row_count}), flush=True)
    if offset != row_count:
        raise ValueError(f"Expected {row_count} rows, wrote {offset}")

    for array in (x, y, decision_index, candidate_index, sample_weight, candidate_action_type):
        array.flush()

    counts = np.bincount(np.asarray(decision_index), minlength=len(decisions))
    positives = np.bincount(np.asarray(decision_index), weights=np.asarray(y, dtype=np.int64), minlength=len(decisions))
    expected_counts = decisions["candidate_count"].to_numpy(dtype=np.int64)
    integrity = {
        "all_decisions_present": bool(np.all(counts > 0)),
        "one_positive_per_decision": bool(np.all(positives == 1)),
        "candidate_counts_match": bool(np.array_equal(counts.astype(np.int64), expected_counts)),
        "unknown_action_type_count": 0,
    }
    if not (integrity["all_decisions_present"] and integrity["one_positive_per_decision"] and integrity["candidate_counts_match"] and integrity["unknown_action_type_count"] == 0):
        raise ValueError(f"Matrix integrity failure: {integrity}")

    schema = {
        "row_count": row_count,
        "decision_count": int(len(decisions)),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "action_type_map": action_map,
        "integrity": integrity,
        "files": {
            "features": "features.npy", "labels": "labels.npy", "decision_index": "decision_index.npy",
            "candidate_index": "candidate_index.npy", "sample_weight": "sample_weight.npy",
            "candidate_action_type": "candidate_action_type.npy", "decisions": "decisions_indexed.csv",
        },
    }
    (matrix_dir / "matrix_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    return schema


def load_matrix_store(processed_dir: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame]:
    matrix_dir = Path(processed_dir) / "matrix"
    schema = json.loads((matrix_dir / "matrix_schema.json").read_text(encoding="utf-8"))
    arrays = {
        "features": np.load(matrix_dir / "features.npy", mmap_mode="r"),
        "labels": np.load(matrix_dir / "labels.npy", mmap_mode="r"),
        "decision_index": np.load(matrix_dir / "decision_index.npy", mmap_mode="r"),
        "candidate_index": np.load(matrix_dir / "candidate_index.npy", mmap_mode="r"),
        "sample_weight": np.load(matrix_dir / "sample_weight.npy", mmap_mode="r"),
        "candidate_action_type": np.load(matrix_dir / "candidate_action_type.npy", mmap_mode="r"),
    }
    decisions = pd.read_csv(matrix_dir / "decisions_indexed.csv")
    return schema, arrays, decisions
