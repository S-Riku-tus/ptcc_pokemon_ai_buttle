"""Check v33 compact-runtime selector scores against saved offline scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

import train_alakazam_v33_oof_selector as training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--decisions", type=int, default=25)
    parser.add_argument("--tolerance", type=float, default=2e-4)
    args = parser.parse_args()

    sys.path.insert(0, str(args.agent_dir.resolve()))
    import ml_runtime as runtime  # noqa: PLC0415

    selector = json.loads(
        (args.agent_dir / "selector_model.json").read_text(
            encoding="utf-8"
        )
    )
    bases = [
        json.loads(
            (args.agent_dir / artifact).read_text(encoding="utf-8")
        )
        for artifact in selector["base_artifacts"]
    ]
    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        feature_names = cached["feature_names"].astype(str).tolist()
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
    with np.load(args.scores, allow_pickle=False) as saved:
        validation_scores = saved["validation_scores"]
        validation_groups = saved["validation_groups"]

    validation_decisions = np.flatnonzero(splits == "validation")
    global_starts, global_ends = training._ranges(groups)
    local_starts, local_ends = training._ranges(validation_groups)
    raw_names = list(selector["raw_feature_names"])
    raw_columns = np.asarray(
        [feature_names.index(name) for name in raw_names],
        dtype=np.int64,
    )
    ranker = runtime.HybridRanker.__new__(runtime.HybridRanker)
    ranker.selector_model = selector
    ranker.selector_bases = bases

    maximum_error = 0.0
    checked = min(args.decisions, len(validation_decisions))
    sample = np.linspace(
        0,
        len(validation_decisions) - 1,
        checked,
        dtype=np.int64,
    )
    for local_decision in sample:
        decision = int(validation_decisions[local_decision])
        global_start = int(global_starts[decision])
        global_end = int(global_ends[decision])
        local_start = int(local_starts[local_decision])
        local_end = int(local_ends[local_decision])
        raw = features[global_start:global_end]
        score_sets = validation_scores[:, local_start:local_end]
        expected_x = training._meta_features(
            raw,
            raw_columns,
            score_sets,
            [global_end - global_start],
        )
        expected = np.asarray([
            runtime._selector_tree_score(row.tolist(), selector)
            for row in expected_x
        ])
        feature_dicts = [
            {
                name: float(value)
                for name, value in zip(feature_names, row)
            }
            for row in raw
        ]
        actual = np.asarray(
            ranker._selector_scores(
                feature_dicts,
                list(range(len(feature_dicts))),
            )
        )
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(expected - actual))),
        )

    report = {
        "decisions_checked": int(checked),
        "maximum_absolute_score_error": maximum_error,
        "tolerance": float(args.tolerance),
        "passed": bool(maximum_error <= args.tolerance),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
