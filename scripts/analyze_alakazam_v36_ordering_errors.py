"""Describe which action families are reversed by a held-out ranker."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_alakazam_v35_residual import load_cache  # noqa: E402
from scripts.train_alakazam_v33_teacher import (  # noqa: E402
    ACTION_TYPES,
    ranges,
    rows_for,
    turn_blocks,
    turn_pick_sets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache = load_cache(args.cache)
    features = cache["features"]
    labels = cache["labels"]
    groups = cache["groups"]
    names = cache["names"]
    blocks = turn_blocks(
        features, groups, cache["episode_ids"], names
    )
    pick_sets, semantic_columns = turn_pick_sets(
        features, labels, groups, blocks, names
    )
    action_column = names.index("action_type")
    turn_position_column = names.index("turn_decision_index")

    with np.load(args.scores, allow_pickle=False) as stored:
        held_out_scores = {
            split: stored[split] for split in ("validation", "test")
        }

    report = {}
    for split in ("validation", "test"):
        decisions = np.flatnonzero(cache["splits"] == split)
        absolute_rows = rows_for(groups, decisions)
        group_sizes = groups[decisions].astype(np.int64)
        starts, ends = ranges(group_sizes)
        pair_counts: Counter[tuple[str, str]] = Counter()
        position_counts: Counter[int] = Counter()
        category_counts: Counter[str] = Counter()
        for local, (start, end) in enumerate(zip(starts, ends)):
            rows = absolute_rows[start:end]
            block_labels = labels[rows]
            teacher_local = int(np.flatnonzero(block_labels == 1)[0])
            predicted_local = int(
                np.argmax(held_out_scores[split][start:end])
            )
            if teacher_local == predicted_local:
                category_counts["correct"] += 1
                continue
            predicted_row = int(rows[predicted_local])
            predicted_semantic = tuple(
                features[predicted_row, semantic_columns].tolist()
            )
            decision = int(decisions[local])
            if predicted_semantic in pick_sets[decision]:
                category_counts["same_turn_ordering"] += 1
                teacher_action = ACTION_TYPES[
                    int(cache["action_types"][decision])
                ]
                predicted_action = ACTION_TYPES[
                    int(features[predicted_row, action_column])
                ]
                pair_counts[(teacher_action, predicted_action)] += 1
                position_counts[int(
                    features[rows[teacher_local], turn_position_column]
                )] += 1
            else:
                category_counts["divergence"] += 1

        report[split] = {
            "decisions": int(len(decisions)),
            "categories": dict(category_counts),
            "top_teacher_to_predicted_pairs": [
                {
                    "teacher": teacher,
                    "predicted_later": predicted,
                    "count": count,
                }
                for (teacher, predicted), count in pair_counts.most_common()
            ],
            "ordering_errors_by_turn_position": {
                str(position): count
                for position, count in sorted(position_counts.items())
            },
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
