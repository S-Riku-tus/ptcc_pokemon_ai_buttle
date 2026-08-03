"""Re-rank stage-one near ties with leakage-free empirical action order.

Most v34 Top-1 misses are harmless permutations of actions the teacher plays
in the same turn.  A row-wise ranker cannot directly represent a preference
such as "play card A before evolving" because its score for A does not see
the competing candidate.  This experiment learns pairwise precedence tables
from out-of-fold training shortlists and backs off from card-level signatures
to action families when an exact pair is sparse.

All choices are made on validation.  Test is evaluated once after the
signature level, shortlist size, smoothing and blend are fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_alakazam_v35_residual import load_cache  # noqa: E402
from scripts.train_alakazam_v33_teacher import (  # noqa: E402
    graded_labels,
    ranges,
    rows_for,
    turn_blocks,
)
from scripts.train_alakazam_v34_teacher import recency_multiplier  # noqa: E402


def signature(features: np.ndarray, row: int, columns: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(features[row, column]) for column in columns)


def canonical_pair(left: tuple[int, ...], right: tuple[int, ...]):
    if left <= right:
        return (left, right), False
    return (right, left), True


def topk_rows(scores: np.ndarray, groups: np.ndarray, k: int) -> list[np.ndarray]:
    starts, ends = ranges(groups)
    return [
        a + np.argsort(-scores[a:b], kind="stable")[: min(k, b - a)]
        for a, b in zip(starts, ends)
    ]


def learn_tables(
    features: np.ndarray,
    absolute_rows: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    graded: np.ndarray,
    episode_weights: np.ndarray,
    signature_columns: list[tuple[int, ...]],
    k: int,
) -> list[dict[tuple[tuple[int, ...], tuple[int, ...]], list[float]]]:
    tables = [defaultdict(lambda: [0.0, 0.0]) for _ in signature_columns]
    picks = topk_rows(scores, groups, k)
    for decision, local_rows in enumerate(picks):
        weight = float(episode_weights[decision])
        for offset, left_local in enumerate(local_rows):
            left_row = int(absolute_rows[left_local])
            left_grade = int(graded[left_row])
            for right_local in local_rows[offset + 1:]:
                right_row = int(absolute_rows[right_local])
                right_grade = int(graded[right_row])
                if left_grade == right_grade:
                    continue
                left_wins = left_grade > right_grade
                for table, columns in zip(tables, signature_columns):
                    left = signature(features, left_row, columns)
                    right = signature(features, right_row, columns)
                    if left == right:
                        continue
                    key, reversed_pair = canonical_pair(left, right)
                    canonical_left_wins = left_wins != reversed_pair
                    table[key][0 if canonical_left_wins else 1] += weight
    return [dict(table) for table in tables]


def pair_logit(
    left: tuple[int, ...],
    right: tuple[int, ...],
    table: dict,
    smoothing: float,
    min_count: float,
) -> float | None:
    if left == right:
        return None
    key, reversed_pair = canonical_pair(left, right)
    wins = table.get(key)
    if wins is None or sum(wins) < min_count:
        return None
    left_wins, right_wins = wins
    if reversed_pair:
        left_wins, right_wins = right_wins, left_wins
    probability = (left_wins + smoothing) / (
        left_wins + right_wins + 2.0 * smoothing
    )
    return math.log(probability / max(1.0 - probability, 1e-12))


def rerank(
    features: np.ndarray,
    absolute_rows: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    tables: list[dict],
    signature_columns: list[tuple[int, ...]],
    *,
    k: int,
    smoothing: float,
    min_count: float,
    alpha: float,
    max_level: int,
) -> np.ndarray:
    out = scores.astype(np.float64).copy()
    starts, ends = ranges(groups)
    for a, b in zip(starts, ends):
        shortlist = a + np.argsort(-scores[a:b], kind="stable")[: min(k, b - a)]
        block = scores[shortlist].astype(np.float64)
        scale = max(float(block.std()), 1e-5)
        z = (block - float(block.mean())) / scale
        priority = np.zeros(len(shortlist), dtype=np.float64)
        for left_index, left_local in enumerate(shortlist):
            left_row = int(absolute_rows[left_local])
            for right_index, right_local in enumerate(shortlist):
                if left_index == right_index:
                    continue
                right_row = int(absolute_rows[right_local])
                value = None
                # Most specific table first, then deterministic backoff.
                for level in range(max_level - 1, -1, -1):
                    columns = signature_columns[level]
                    value = pair_logit(
                        signature(features, left_row, columns),
                        signature(features, right_row, columns),
                        tables[level], smoothing, min_count,
                    )
                    if value is not None:
                        break
                if value is not None:
                    priority[left_index] += value
        if len(shortlist) > 1:
            priority /= len(shortlist) - 1
        out[shortlist] = z + alpha * priority
        tail = np.ones(b - a, dtype=bool)
        tail[shortlist - a] = False
        out[a:b][tail] -= 1e6
    return out.astype(np.float32)


def metrics(scores: np.ndarray, labels: np.ndarray, rows: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    starts, ends = ranges(groups)
    correct = top2 = top3 = 0
    for a, b in zip(starts, ends):
        order = np.argsort(-scores[a:b], kind="stable")
        block_labels = labels[rows[a:b]]
        correct += int(block_labels[order[0]] == 1)
        top2 += int(np.any(block_labels[order[:2]] == 1))
        top3 += int(np.any(block_labels[order[:3]] == 1))
    n = max(len(groups), 1)
    return {
        "decisions": int(len(groups)),
        "top1": correct / n,
        "top2": top2 / n,
        "top3": top3 / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("stage1_scores", type=Path)
    parser.add_argument("oof_scores", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--episode-fraction", type=float, default=0.875)
    args = parser.parse_args()

    cache = load_cache(args.cache)
    features, labels, groups = cache["features"], cache["labels"], cache["groups"]
    names, episodes = cache["names"], cache["episode_ids"]
    decisions = {
        split: np.flatnonzero(cache["splits"] == split)
        for split in ("train", "validation", "test")
    }
    ordered = np.unique(episodes[decisions["train"]])
    kept = ordered[-max(1, int(round(len(ordered) * args.episode_fraction))):]
    fit_decisions = decisions["train"][np.isin(episodes[decisions["train"]], kept)]
    block_decisions = {
        "train": fit_decisions,
        "validation": decisions["validation"],
        "test": decisions["test"],
    }
    block_rows = {key: rows_for(groups, value) for key, value in block_decisions.items()}
    block_groups = {key: groups[value].astype(np.int64) for key, value in block_decisions.items()}
    with np.load(args.oof_scores, allow_pickle=False) as stored:
        train_scores = stored["scores"]
    with np.load(args.stage1_scores, allow_pickle=False) as stored:
        block_scores = {
            "train": train_scores,
            "validation": stored["validation"],
            "test": stored["test"],
        }

    turn_blocks_value = turn_blocks(features, groups, episodes, names)
    graded, _ = graded_labels(features, labels, groups, turn_blocks_value, names)
    signature_columns = [
        (names.index("action_type"),),
        (names.index("action_type"), names.index("candidate_card_id")),
        (
            names.index("action_type"), names.index("candidate_card_id"),
            names.index("candidate_attack_id"), names.index("candidate_inplay_area"),
        ),
    ]
    episode_weight = recency_multiplier(
        episodes[fit_decisions], floor=0.25, power=2.0
    )

    validation_runs = []
    tables_by_k = {}
    for k in (3, 5):
        tables = learn_tables(
            features, block_rows["train"], block_groups["train"],
            block_scores["train"], graded, episode_weight,
            signature_columns, k,
        )
        tables_by_k[k] = tables
        for max_level in (1, 2, 3):
            for min_count in (2.0, 5.0, 10.0, 20.0):
                for smoothing in (1.0, 3.0, 10.0):
                    for alpha in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
                        candidate_scores = rerank(
                            features, block_rows["validation"],
                            block_groups["validation"], block_scores["validation"],
                            tables, signature_columns, k=k,
                            smoothing=smoothing, min_count=min_count,
                            alpha=alpha, max_level=max_level,
                        )
                        row = {
                            "k": k, "max_level": max_level,
                            "min_count": min_count, "smoothing": smoothing,
                            "alpha": alpha,
                            **metrics(
                                candidate_scores, labels,
                                block_rows["validation"],
                                block_groups["validation"],
                            ),
                        }
                        validation_runs.append(row)
        best_k = max(
            (row for row in validation_runs if row["k"] == k),
            key=lambda row: (row["top1"], -row["alpha"]),
        )
        print(json.dumps({f"k{k}_best": best_k}), flush=True)

    selected = max(
        validation_runs,
        key=lambda row: (row["top1"], -row["alpha"], -row["k"]),
    )
    test_scores = rerank(
        features, block_rows["test"], block_groups["test"], block_scores["test"],
        tables_by_k[selected["k"]], signature_columns,
        k=selected["k"], smoothing=selected["smoothing"],
        min_count=selected["min_count"], alpha=selected["alpha"],
        max_level=selected["max_level"],
    )
    test = metrics(test_scores, labels, block_rows["test"], block_groups["test"])
    report = {
        "method": "OOF empirical pair precedence with hierarchical backoff",
        "selection_rule": "maximum validation strict Top-1; test scored once",
        "fit_episodes": int(len(kept)),
        "fit_decisions": int(len(fit_decisions)),
        "table_pair_counts": {
            str(k): [len(table) for table in tables]
            for k, tables in tables_by_k.items()
        },
        "selected": selected,
        "test": test,
        "target_top1": 0.90,
        "target_met": bool(test["top1"] > 0.90),
        "validation_runs": validation_runs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected": selected, "test": test}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
