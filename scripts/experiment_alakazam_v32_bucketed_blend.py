"""Tune conservative per-context weights for an existing v32 ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402


def _correct(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    starts, ends = ensemble._ranges(groups)
    return np.asarray([
        labels[start + int(np.argmax(scores[start:end]))] == 1
        for start, end in zip(starts, ends)
    ])


def _blend(
    score_sets: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    return sum(
        float(weight) * scores
        for weight, scores in zip(weights, score_sets)
    )


def _accuracy_subset(
    score_sets: list[np.ndarray],
    weights: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    decisions: np.ndarray,
) -> float:
    return float(
        _correct(
            _blend(score_sets, weights),
            labels,
            groups,
        )[decisions].mean()
    )


def _context_values(
    cache: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache, allow_pickle=False) as saved:
        names = saved["feature_names"].astype(str).tolist()
        groups = saved["groups"]
        splits = saved["splits"].astype(str)
        starts, _ = ensemble._ranges(groups)
        decisions = np.flatnonzero(splits == split)
        rows = starts[decisions]
        fallback = saved["features"][
            rows,
            names.index("fallback_action_type"),
        ].astype(np.int16)
        turn = saved["features"][
            rows,
            names.index("turn"),
        ].astype(np.int16)
    turn_bin = np.where(turn <= 4, 0, np.where(turn <= 10, 1, 2))
    return fallback, turn_bin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    previous = json.loads(
        args.blend_report.read_text(encoding="utf-8")
    )
    model_names = list(previous["model_order"])
    global_weights = np.asarray(
        previous["selected_weights"],
        dtype=np.float64,
    )
    with np.load(args.scores, allow_pickle=False) as saved:
        validation_scores = [
            saved[f"validation_{name}"] for name in model_names
        ]
        test_scores = [
            saved[f"test_{name}"] for name in model_names
        ]
        validation_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]
        validation_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]
    validation_fallback, validation_turn = _context_values(
        args.cache,
        "validation",
    )
    test_fallback, test_turn = _context_values(args.cache, "test")

    # Tune broad fallback-action buckets.  Retain the global weights unless a
    # bucket has at least 100 examples and gains at least two validation hits.
    bucket_weights: dict[int, np.ndarray] = {}
    bucket_rows = []
    for bucket in sorted(set(validation_fallback.tolist())):
        decisions = np.flatnonzero(validation_fallback == bucket)
        base_correct = int(round(
            _accuracy_subset(
                validation_scores,
                global_weights,
                validation_labels,
                validation_groups,
                decisions,
            ) * len(decisions)
        ))
        weights = global_weights.copy()
        best_correct = base_correct
        if len(decisions) >= 100:
            for _ in range(3):
                changed = False
                for coordinate in range(1, len(weights)):
                    selected = float(weights[coordinate])
                    low = max(0.0, global_weights[coordinate] - 0.75)
                    high = global_weights[coordinate] + 0.75
                    for value in np.arange(low, high + 0.001, 0.05):
                        candidate = weights.copy()
                        candidate[coordinate] = value
                        correct = int(round(
                            _accuracy_subset(
                                validation_scores,
                                candidate,
                                validation_labels,
                                validation_groups,
                                decisions,
                            ) * len(decisions)
                        ))
                        if correct > best_correct:
                            best_correct = correct
                            selected = float(value)
                    if selected != weights[coordinate]:
                        changed = True
                    weights[coordinate] = selected
                if not changed:
                    break
        accepted = best_correct >= base_correct + 2
        bucket_weights[bucket] = (
            weights if accepted else global_weights.copy()
        )
        bucket_rows.append({
            "fallback_action_type": bucket,
            "validation_decisions": len(decisions),
            "base_correct": base_correct,
            "tuned_correct": best_correct,
            "accepted": accepted,
            "weights": bucket_weights[bucket].tolist(),
        })

    validation_correct = np.zeros(len(validation_groups), dtype=bool)
    test_correct = np.zeros(len(test_groups), dtype=bool)
    for bucket, weights in bucket_weights.items():
        validation_mask = validation_fallback == bucket
        test_mask = test_fallback == bucket
        validation_correct[validation_mask] = _correct(
            _blend(validation_scores, weights),
            validation_labels,
            validation_groups,
        )[validation_mask]
        test_correct[test_mask] = _correct(
            _blend(test_scores, weights),
            test_labels,
            test_groups,
        )[test_mask]

    # Any unseen bucket keeps the global ensemble.
    global_validation_correct = _correct(
        _blend(validation_scores, global_weights),
        validation_labels,
        validation_groups,
    )
    global_test_correct = _correct(
        _blend(test_scores, global_weights),
        test_labels,
        test_groups,
    )
    known_validation = np.isin(
        validation_fallback,
        list(bucket_weights),
    )
    known_test = np.isin(test_fallback, list(bucket_weights))
    validation_correct[~known_validation] = global_validation_correct[
        ~known_validation
    ]
    test_correct[~known_test] = global_test_correct[~known_test]

    report = {
        "model_order": model_names,
        "global_weights": global_weights.tolist(),
        "bucket_definition": "fallback_action_type",
        "validation_bucket_counts": dict(Counter(
            validation_fallback.astype(int).tolist()
        )),
        "test_bucket_counts": dict(Counter(
            test_fallback.astype(int).tolist()
        )),
        "bucket_models": bucket_rows,
        "global_validation_top1": float(
            global_validation_correct.mean()
        ),
        "bucketed_validation_top1": float(validation_correct.mean()),
        "global_test_top1": float(global_test_correct.mean()),
        "bucketed_test_top1": float(test_correct.mean()),
        "turn_bins_audited_not_used": {
            "validation": dict(Counter(validation_turn.tolist())),
            "test": dict(Counter(test_turn.tolist())),
        },
        "target_top1": 0.9,
        "target_met": float(test_correct.mean()) >= 0.9,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
