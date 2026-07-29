"""Evaluate parameter-free consensus gates over a frozen v32 ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _evaluate(
    score_sets: list[np.ndarray],
    weights: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    threshold: int,
) -> tuple[float, float]:
    starts, ends = _ranges(groups)
    correct = consensus_used = 0
    for start, end in zip(starts, ends):
        selections = [
            int(np.argmax(scores[start:end]))
            for scores in score_sets
        ]
        counts = np.bincount(selections, minlength=end - start)
        blended = sum(
            float(weight) * scores[start:end]
            for weight, scores in zip(weights, score_sets)
        )
        best_vote = int(np.max(counts))
        if best_vote >= threshold:
            candidates = np.flatnonzero(counts == best_vote)
            selected = int(
                candidates[np.argmax(blended[candidates])]
            )
            consensus_used += 1
        else:
            selected = int(np.argmax(blended))
        correct += int(labels[start + selected] == 1)
    return correct / len(groups), consensus_used / len(groups)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(
        args.blend_report.read_text(encoding="utf-8")
    )
    names = report["model_order"]
    weights = np.asarray(report["selected_weights"])
    with np.load(args.scores, allow_pickle=False) as saved:
        validation_scores = [
            saved[f"validation_{name}"] for name in names
        ]
        test_scores = [
            saved[f"test_{name}"] for name in names
        ]
        validation_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]
        validation_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]
    rows = []
    for threshold in range(2, len(names) + 2):
        validation_top1, validation_usage = _evaluate(
            validation_scores,
            weights,
            validation_labels,
            validation_groups,
            threshold,
        )
        test_top1, test_usage = _evaluate(
            test_scores,
            weights,
            test_labels,
            test_groups,
            threshold,
        )
        rows.append({
            "threshold": threshold,
            "validation_top1": validation_top1,
            "validation_consensus_usage": validation_usage,
            "test_top1": test_top1,
            "test_consensus_usage": test_usage,
        })
    selected = max(
        rows,
        key=lambda row: (
            row["validation_top1"],
            row["threshold"],
        ),
    )
    output = {
        "model_order": names,
        "experiments": rows,
        "selected_on_validation": selected,
        "reference_test_top1": report["test_top1"],
        "target_top1": 0.9,
        "target_met": selected["test_top1"] >= 0.9,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
