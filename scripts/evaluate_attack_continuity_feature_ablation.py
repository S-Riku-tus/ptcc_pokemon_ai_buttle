"""Train an exact-split ablation with newly added continuity features disabled."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.core.matrix import load_matrix_store
from ml.core.matrix_train import (
    CATEGORICAL_NAMES,
    _calibration_ids,
    _decision_numeric_set,
    _fit_temperature_arrays,
    _group_sizes,
    _make_model,
    _row_indices,
    evaluate_arrays,
)
from ml.core.splits import make_splits


def _extract_with_zeroed_features(arrays, rows, zero_indices):
    x = np.asarray(arrays["features"][rows], dtype=np.float32)
    x[:, zero_indices] = 0.0
    y = np.asarray(arrays["labels"][rows], dtype=np.int8)
    weights = np.asarray(arrays["sample_weight"][rows], dtype=np.float32)
    groups = _group_sizes(np.asarray(arrays["decision_index"][rows], dtype=np.int32))
    return x, y, weights, groups


def _score_with_zeroed_features(model, arrays, rows, zero_indices, batch_size=100_000):
    output = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        end = min(len(rows), start + batch_size)
        x = np.asarray(arrays["features"][rows[start:end]], dtype=np.float32)
        x[:, zero_indices] = 0.0
        output[start:end] = model.predict(x).astype(np.float32)
    return output


def _metric_delta(baseline, candidate):
    keys = ("top1", "top3", "mrr", "fallback_rate", "accepted_top1")
    result = {
        key: float(candidate[key] - baseline[key])
        for key in keys
        if baseline.get(key) is not None and candidate.get(key) is not None
    }
    actions = {}
    for action in sorted(
        set(baseline.get("action_type_metrics", {}))
        & set(candidate.get("action_type_metrics", {}))
    ):
        old = baseline["action_type_metrics"][action]
        new = candidate["action_type_metrics"][action]
        actions[action] = {
            key: float(new[key] - old[key])
            for key in ("top1", "top3", "mrr", "fallback_rate")
            if old.get(key) is not None and new.get(key) is not None
        }
    result["action_type_metrics"] = actions
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--old-schema", required=True, type=Path)
    parser.add_argument("--candidate-metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    schema, arrays, decisions = load_matrix_store(args.processed_dir)
    features = list(schema["feature_columns"])
    old_schema = json.loads(args.old_schema.read_text(encoding="utf-8"))
    old_features = set(old_schema["feature_columns"])
    added_features = [name for name in features if name not in old_features]
    zero_indices = [features.index(name) for name in added_features]
    if not zero_indices:
        raise RuntimeError("No newly added features were found")

    split = next(item for item in make_splits(decisions) if item.name == "time_holdout")
    train_ids = _decision_numeric_set(decisions, split.train_decisions)
    test_ids = _decision_numeric_set(decisions, split.test_decisions)
    fit_ids, calibration_ids = _calibration_ids(train_ids)
    decision_count = int(schema["decision_count"])
    fit_rows = _row_indices(arrays["decision_index"], fit_ids, decision_count)
    calibration_rows = _row_indices(arrays["decision_index"], calibration_ids, decision_count)
    test_rows = _row_indices(arrays["decision_index"], test_ids, decision_count)

    print(json.dumps({
        "event": "ablation_fit",
        "disabled_features": added_features,
        "fit_decisions": len(fit_ids),
        "test_decisions": len(test_ids),
        "fit_rows": len(fit_rows),
    }), flush=True)
    x, y, weights, groups = _extract_with_zeroed_features(arrays, fit_rows, zero_indices)
    categorical = [features.index(name) for name in CATEGORICAL_NAMES if name in features]
    model = _make_model(50)
    model.fit(
        X=x,
        y=y,
        group=groups,
        sample_weight=weights,
        categorical_feature=categorical,
        feature_name=features,
    )
    del x, y, weights, groups, fit_rows
    gc.collect()

    calibration_scores = _score_with_zeroed_features(
        model, arrays, calibration_rows, zero_indices
    )
    temperature = _fit_temperature_arrays(
        arrays, calibration_rows, calibration_scores
    )
    test_scores = _score_with_zeroed_features(model, arrays, test_rows, zero_indices)
    ablation_metrics, _ = evaluate_arrays(
        arrays,
        decisions,
        test_rows,
        test_scores,
        schema["action_type_map"],
        temperature=temperature,
    )
    candidate_metrics = json.loads(args.candidate_metrics.read_text(encoding="utf-8"))
    report = {
        "definition": "same time split and hyperparameters; added continuity features fixed to zero",
        "disabled_features": added_features,
        "ablation": ablation_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_ablation": _metric_delta(ablation_metrics, candidate_metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "candidate_minus_ablation": report["candidate_minus_ablation"],
    }, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
