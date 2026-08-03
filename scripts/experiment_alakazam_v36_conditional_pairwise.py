"""State-conditioned pairwise re-ranker for the v34/v35 Top-3 shortlist.

Training uses out-of-fold stage-one scores.  Each example compares two
candidates on the same board using (a) important state features, (b) every
candidate-feature difference, (c) both candidates' semantic identities and
(d) their stage-one score gap.  Symmetric examples prevent positional bias.

The tree count and blend are selected on validation only.  The chronological
test block is scored once after selection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import lightgbm as lgb
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


def split_counts(node: dict[str, Any], counts: Counter[int]) -> None:
    if "v" in node:
        return
    counts[int(node["f"])] += 1
    split_counts(node["l"], counts)
    split_counts(node["r"], counts)


def important_state_columns(
    model_path: Path,
    cache_names: list[str],
    state_end: int,
    limit: int,
) -> list[int]:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model_names = model["feature_names"]
    counts: Counter[int] = Counter()
    for tree in model["trees"]:
        split_counts(tree, counts)
    selected_names = [
        model_names[index]
        for index, _ in counts.most_common()
        if index < len(model_names) and model_names[index] in cache_names[:state_end]
    ][:limit]
    return [cache_names.index(name) for name in selected_names]


def shortlist(scores: np.ndarray, groups: np.ndarray, k: int) -> list[np.ndarray]:
    starts, ends = ranges(groups)
    return [
        a + np.argsort(-scores[a:b], kind="stable")[: min(k, b - a)]
        for a, b in zip(starts, ends)
    ]


def pair_matrix(
    features: np.ndarray,
    absolute_rows: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    graded: np.ndarray,
    state_columns: list[int],
    candidate_columns: list[int],
    semantic_columns: list[int],
    *,
    k: int,
    decision_weights: np.ndarray | None = None,
    include_equal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """Build symmetric pair examples and retain decision/pair provenance."""
    starts, _ = ranges(groups)
    decision_ids: list[int] = []
    left_rows: list[int] = []
    right_rows: list[int] = []
    score_gaps: list[float] = []
    targets: list[int] = []
    weights: list[float] = []
    provenance: list[tuple[int, int, int]] = []

    for decision, picked in enumerate(shortlist(scores, groups, k)):
        for left_index in range(len(picked)):
            for right_index in range(left_index + 1, len(picked)):
                left_local = int(picked[left_index])
                right_local = int(picked[right_index])
                left_absolute = int(absolute_rows[left_local])
                right_absolute = int(absolute_rows[right_local])
                left_grade = int(graded[left_absolute])
                right_grade = int(graded[right_absolute])
                if left_grade == right_grade and not include_equal:
                    continue
                base_weight = (
                    float(decision_weights[decision])
                    if decision_weights is not None else 1.0
                )
                # Chosen-vs-other comparisons are the strict Top-1 signal.
                grade_weight = 2.0 if max(left_grade, right_grade) == 3 else 0.5
                for reverse in (False, True):
                    lhs = right_absolute if reverse else left_absolute
                    rhs = left_absolute if reverse else right_absolute
                    lhs_local = right_local if reverse else left_local
                    rhs_local = left_local if reverse else right_local
                    decision_ids.append(decision)
                    left_rows.append(lhs)
                    right_rows.append(rhs)
                    score_gaps.append(float(scores[lhs_local] - scores[rhs_local]))
                    target = int((right_grade > left_grade) if reverse else (left_grade > right_grade))
                    targets.append(target)
                    weights.append(base_weight * grade_weight)
                    provenance.append((decision, lhs_local, rhs_local))

    decision_array = np.asarray(decision_ids, dtype=np.int64)
    left_array = np.asarray(left_rows, dtype=np.int64)
    right_array = np.asarray(right_rows, dtype=np.int64)
    context_rows = absolute_rows[starts][decision_array]
    matrix = np.hstack([
        features[context_rows][:, state_columns],
        features[left_array][:, candidate_columns]
        - features[right_array][:, candidate_columns],
        features[left_array][:, semantic_columns],
        features[right_array][:, semantic_columns],
        np.asarray(score_gaps, dtype=np.float32)[:, None],
    ]).astype(np.float32, copy=False)
    return (
        np.ascontiguousarray(matrix),
        np.asarray(targets, dtype=np.int8),
        np.asarray(weights, dtype=np.float32),
        provenance,
    )


def rerank(
    model: lgb.LGBMClassifier,
    trees: int,
    features: np.ndarray,
    labels: np.ndarray,
    absolute_rows: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    graded: np.ndarray,
    state_columns: list[int],
    candidate_columns: list[int],
    semantic_columns: list[int],
    *,
    k: int,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix, _, _, provenance = pair_matrix(
        features, absolute_rows, groups, scores, graded, state_columns,
        candidate_columns, semantic_columns, k=k, include_equal=True,
    )
    probabilities = model.predict_proba(
        matrix, num_iteration=trees
    )[:, 1]
    pair_values: dict[tuple[int, int, int], float] = {
        key: float(value) for key, value in zip(provenance, probabilities)
    }
    out = scores.astype(np.float64).copy()
    starts, ends = ranges(groups)
    correct = top2 = top3 = 0
    for decision, (a, b) in enumerate(zip(starts, ends)):
        picked = a + np.argsort(-scores[a:b], kind="stable")[: min(k, b - a)]
        block = scores[picked].astype(np.float64)
        z = (block - block.mean()) / max(float(block.std()), 1e-5)
        preference = np.zeros(len(picked), dtype=np.float64)
        for left_index, left_local in enumerate(picked):
            for right_index, right_local in enumerate(picked):
                if left_index == right_index:
                    continue
                probability = pair_values.get(
                    (decision, int(left_local), int(right_local)), 0.5
                )
                preference[left_index] += probability - 0.5
        if len(picked) > 1:
            preference /= len(picked) - 1
        out[picked] = z + alpha * preference
        tail = np.ones(b - a, dtype=bool)
        tail[picked - a] = False
        out[a:b][tail] -= 1e6
        order = np.argsort(-out[a:b], kind="stable")
        block_labels = labels[absolute_rows[a:b]]
        correct += int(block_labels[order[0]] == 1)
        top2 += int(np.any(block_labels[order[:2]] == 1))
        top3 += int(np.any(block_labels[order[:3]] == 1))
    n = max(len(groups), 1)
    return out.astype(np.float32), {
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
    parser.add_argument("ranker_model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--state-features", type=int, default=120)
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
    graded, _ = graded_labels(
        features, labels, groups, turn_blocks(features, groups, episodes, names), names
    )

    state_end = names.index("option_type")
    turn_names = {
        "turn_decision_index", "turn_candidate_offer_count",
        "turn_candidate_passed_over", "turn_candidate_offered_previous",
        "turn_candidate_first_offer_index", "turn_class_passed_over",
        "turn_class_offer_count", "turn_new_candidate",
    }
    state_columns = important_state_columns(
        args.ranker_model, names, state_end, args.state_features
    )
    candidate_columns = [
        index for index in range(state_end, len(names))
        if names[index] not in turn_names
    ]
    semantic_columns = [
        names.index(name) for name in (
            "option_type", "action_type", "candidate_card_id",
            "candidate_attack_id", "candidate_target_id",
            "candidate_inplay_area",
        )
    ]
    decision_weights = recency_multiplier(
        episodes[fit_decisions], floor=0.25, power=2.0
    )
    train_x, train_y, train_w, _ = pair_matrix(
        features, block_rows["train"], block_groups["train"],
        block_scores["train"], graded, state_columns, candidate_columns,
        semantic_columns, k=args.k, decision_weights=decision_weights,
    )
    validation_x, validation_y, validation_w, _ = pair_matrix(
        features, block_rows["validation"], block_groups["validation"],
        block_scores["validation"], graded, state_columns, candidate_columns,
        semantic_columns, k=args.k,
    )
    print(
        f"pair matrices train={train_x.shape} validation={validation_x.shape}",
        flush=True,
    )

    started = time.perf_counter()
    model = lgb.LGBMClassifier(
        objective="binary", metric="binary_logloss", n_estimators=500,
        learning_rate=0.035, num_leaves=63, min_child_samples=60,
        max_depth=12, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.0,
        random_state=3636, n_jobs=20, verbosity=-1,
    )
    model.fit(
        train_x, train_y, sample_weight=train_w,
        eval_set=[(validation_x, validation_y)],
        eval_sample_weight=[validation_w],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_iteration = int(model.best_iteration_ or 500)
    print(
        f"fit {time.perf_counter() - started:.1f}s best={best_iteration}",
        flush=True,
    )

    validation_runs = []
    for trees in sorted(set([
        max(10, best_iteration // 2), max(10, best_iteration * 3 // 4),
        best_iteration,
    ])):
        for alpha in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            _, result = rerank(
                model, trees, features, labels, block_rows["validation"],
                block_groups["validation"], block_scores["validation"],
                graded, state_columns, candidate_columns, semantic_columns,
                k=args.k, alpha=alpha,
            )
            validation_runs.append({"trees": trees, "alpha": alpha, **result})
    selected = max(
        validation_runs,
        key=lambda row: (row["top1"], -row["alpha"], -row["trees"]),
    )
    _, test = rerank(
        model, selected["trees"], features, labels, block_rows["test"],
        block_groups["test"], block_scores["test"], graded, state_columns,
        candidate_columns, semantic_columns, k=args.k,
        alpha=selected["alpha"],
    )
    report = {
        "method": "OOF state-conditioned symmetric pairwise Top-3 reranker",
        "selection_rule": "maximum validation strict Top-1; test scored once",
        "fit_episodes": int(len(kept)),
        "fit_decisions": int(len(fit_decisions)),
        "train_pairs": int(len(train_y)),
        "matrix_features": int(train_x.shape[1]),
        "state_features": [names[index] for index in state_columns],
        "candidate_difference_features": len(candidate_columns),
        "best_iteration_binary_logloss": best_iteration,
        "fit_seconds": time.perf_counter() - started,
        "validation_runs": validation_runs,
        "selected": selected,
        "test": test,
        "target_top1": 0.90,
        "target_met": bool(test["top1"] > 0.90),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"selected": selected, "test": test}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
