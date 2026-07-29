"""Break down v32 recency-tree and six-model errors by decision context."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ACTION_TYPES = (
    "ability",
    "attack",
    "bench",
    "boss",
    "end",
    "energy",
    "evolve",
    "hammer",
    "other",
    "retreat",
    "trainer",
    "xerosic",
)


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("six_scores", type=Path)
    parser.add_argument("six_report", type=Path)
    parser.add_argument("recency_scores", type=Path)
    parser.add_argument("--recency-index", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    blend = json.loads(args.six_report.read_text(encoding="utf-8"))
    names = list(blend["model_order"])
    weights = np.asarray(blend["selected_weights"])
    with np.load(args.six_scores, allow_pickle=False) as saved:
        six = [saved[f"test_{name}"] for name in names]
        labels = saved["test_labels"]
        groups = saved["test_groups"]
    with np.load(args.recency_scores, allow_pickle=False) as saved:
        recency = saved["test_scores"][args.recency_index]
    with np.load(args.cache, allow_pickle=False) as cached:
        splits = cached["splits"].astype(str)
        teacher_actions = cached["teacher_action_types"][splits == "test"]
        feature_names = cached["feature_names"].astype(str).tolist()
        all_groups = cached["groups"]
        all_starts, _ = _ranges(all_groups)
        test_decisions = np.flatnonzero(splits == "test")
        first_rows = all_starts[test_decisions]
        turns = cached["features"][
            first_rows,
            feature_names.index("turn"),
        ]

    starts, ends = _ranges(groups)
    blend_scores = sum(
        float(weight) * score
        for weight, score in zip(weights, six)
    )
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "recency_correct": 0,
            "blend_correct": 0,
            "oracle_correct": 0,
            "recency_only": 0,
            "blend_only": 0,
        }
    )
    for decision, (start, end) in enumerate(zip(starts, ends)):
        recency_correct = int(
            labels[start + int(np.argmax(recency[start:end]))] == 1
        )
        blend_correct = int(
            labels[start + int(np.argmax(blend_scores[start:end]))] == 1
        )
        oracle_correct = int(any(
            labels[start + int(np.argmax(score[start:end]))] == 1
            for score in six
        ))
        action = int(teacher_actions[decision])
        action_name = (
            ACTION_TYPES[action]
            if 0 <= action < len(ACTION_TYPES)
            else str(action)
        )
        option_count = int(groups[decision])
        turn = int(turns[decision])
        for key in (
            "all",
            f"action:{action_name}",
            f"options:{min(option_count, 12)}+"
            if option_count >= 12
            else f"options:{option_count}",
            f"turn:{min(turn // 5 * 5, 30)}+"
            if turn >= 30
            else f"turn:{turn // 5 * 5}-{turn // 5 * 5 + 4}",
        ):
            row = buckets[key]
            row["count"] += 1
            row["recency_correct"] += recency_correct
            row["blend_correct"] += blend_correct
            row["oracle_correct"] += oracle_correct
            row["recency_only"] += int(
                recency_correct and not blend_correct
            )
            row["blend_only"] += int(
                blend_correct and not recency_correct
            )
    output = {}
    for key, row in buckets.items():
        count = max(1, row["count"])
        output[key] = {
            **row,
            "recency_top1": row["recency_correct"] / count,
            "blend_top1": row["blend_correct"] / count,
            "oracle_top1": row["oracle_correct"] / count,
        }
    report = {
        "models": names,
        "buckets": dict(sorted(output.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: value
        for key, value in output.items()
        if key == "all" or key.startswith("action:")
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
