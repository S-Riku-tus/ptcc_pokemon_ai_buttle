"""Select recency tree or fixed six-model blend by observable context."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _correct(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    starts, ends = _ranges(groups)
    return np.asarray([
        labels[start + int(np.argmax(scores[start:end]))] == 1
        for start, end in zip(starts, ends)
    ])


def _contexts(cache: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(cache, allow_pickle=False) as saved:
        names = saved["feature_names"].astype(str).tolist()
        splits = saved["splits"].astype(str)
        groups = saved["groups"]
        starts, _ = _ranges(groups)
        decisions = np.flatnonzero(splits == split)
        rows = starts[decisions]
        features = saved["features"]
    fallback = features[
        rows,
        names.index("fallback_action_type"),
    ].astype(np.int16)
    turns = features[rows, names.index("turn")].astype(np.int16)
    option_count = groups[decisions].astype(np.int16)
    return {
        "fallback_action": fallback,
        "fallback_turn": fallback * 4 + np.minimum(turns // 5, 3),
        "fallback_option": fallback * 4 + np.minimum(option_count // 4, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("six_scores", type=Path)
    parser.add_argument("six_report", type=Path)
    parser.add_argument("recency_scores", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--recency-index", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    blend = json.loads(args.six_report.read_text(encoding="utf-8"))
    names = list(blend["model_order"])
    weights = np.asarray(blend["selected_weights"])
    with np.load(args.six_scores, allow_pickle=False) as saved:
        validation_scores = [
            saved[f"validation_{name}"] for name in names
        ]
        test_scores = [saved[f"test_{name}"] for name in names]
        validation_labels = saved["validation_labels"]
        validation_groups = saved["validation_groups"]
        test_labels = saved["test_labels"]
        test_groups = saved["test_groups"]
    with np.load(args.recency_scores, allow_pickle=False) as saved:
        validation_recency = saved["validation_scores"][
            args.recency_index
        ]
        test_recency = saved["test_scores"][args.recency_index]
    validation_blend = sum(
        float(weight) * scores
        for weight, scores in zip(weights, validation_scores)
    )
    test_blend = sum(
        float(weight) * scores
        for weight, scores in zip(weights, test_scores)
    )
    validation_base_correct = _correct(
        validation_blend, validation_labels, validation_groups
    )
    validation_recency_correct = _correct(
        validation_recency, validation_labels, validation_groups
    )
    test_base_correct = _correct(test_blend, test_labels, test_groups)
    test_recency_correct = _correct(
        test_recency, test_labels, test_groups
    )
    validation_contexts = _contexts(args.cache, "validation")
    test_contexts = _contexts(args.cache, "test")

    experiments = []
    for definition, validation_values in validation_contexts.items():
        test_values = test_contexts[definition]
        selected_buckets = []
        rows = []
        for bucket in sorted(set(validation_values.tolist())):
            mask = validation_values == bucket
            base_hits = int(validation_base_correct[mask].sum())
            recency_hits = int(validation_recency_correct[mask].sum())
            selected = bool(
                mask.sum() >= 50 and recency_hits >= base_hits + 2
            )
            if selected:
                selected_buckets.append(bucket)
            rows.append({
                "bucket": int(bucket),
                "count": int(mask.sum()),
                "base_hits": base_hits,
                "recency_hits": recency_hits,
                "selected_recency": selected,
            })
        validation_selected = np.isin(
            validation_values, selected_buckets
        )
        test_selected = np.isin(test_values, selected_buckets)
        validation_correct = np.where(
            validation_selected,
            validation_recency_correct,
            validation_base_correct,
        )
        test_correct = np.where(
            test_selected,
            test_recency_correct,
            test_base_correct,
        )
        experiments.append({
            "definition": definition,
            "selected_buckets": list(map(int, selected_buckets)),
            "validation_recency_rate": float(
                validation_selected.mean()
            ),
            "test_recency_rate": float(test_selected.mean()),
            "validation_top1": float(validation_correct.mean()),
            "test_top1": float(test_correct.mean()),
            "buckets": rows,
        })
    selected = max(
        experiments,
        key=lambda row: row["validation_top1"],
    )
    report = {
        "base_validation_top1": float(validation_base_correct.mean()),
        "base_test_top1": float(test_base_correct.mean()),
        "recency_validation_top1": float(
            validation_recency_correct.mean()
        ),
        "recency_test_top1": float(test_recency_correct.mean()),
        "context_counts": {
            name: dict(Counter(values.tolist()))
            for name, values in validation_contexts.items()
        },
        "experiments": experiments,
        "selected": selected,
        "target_top1": 0.9,
        "target_met": selected["test_top1"] >= 0.9,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "base_test_top1": report["base_test_top1"],
        "recency_test_top1": report["recency_test_top1"],
        "experiments": [{
            key: row[key]
            for key in (
                "definition",
                "selected_buckets",
                "validation_top1",
                "test_top1",
            )
        } for row in experiments],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
