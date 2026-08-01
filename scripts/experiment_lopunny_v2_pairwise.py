"""Validation-only explicit pairwise reranker for Lopunny v2.

The classifier receives (state, left-right candidate delta, selected critical
fields for both sides) and learns whether left should precede right.  Base
scores are deliberately excluded from its training features, so hard-negative
mining cannot leak an in-sample confidence signal.  At validation it conducts
a round robin inside the base ranker's Top-K and blends the resulting logits
with decision-local base z-scores.  Test is never read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_lopunny_top1_teacher as v1  # noqa: E402


CRITICAL = (
    "option_type", "candidate_card_id", "candidate_attack_id",
    "candidate_area", "candidate_inplay_area", "candidate_target_id",
    "candidate_target_hp", "candidate_target_energy",
    "candidate_target_special_energy", "candidate_target_appear_this_turn",
    "candidate_hand_cost", "candidate_total_draw_count",
    "candidate_net_hand_delta", "post_action_hand_count",
    "candidate_option_position", "candidate_raw_inplay_index",
    "same_action_option_count", "same_card_option_count", "action_type",
    "is_attack", "is_retreat", "is_end", "is_evolve", "is_ability",
    "is_energy", "is_boss", "is_xerosic", "is_bench", "is_trainer",
    "candidate_is_lopunny", "candidate_is_wally", "candidate_is_lillie",
    "candidate_is_air_balloon", "candidate_is_ultra_ball",
    "candidate_is_pokegear", "candidate_is_lopunny_energy",
    "target_is_buneary", "target_is_lopunny",
    "lopunny_attack_damage_estimate", "lopunny_attack_lethal_estimate",
)


def _schema(names: list[str]) -> tuple[int, np.ndarray, list[str]]:
    candidate_start = names.index("option_type")
    critical = np.asarray(
        [names.index(name) for name in CRITICAL if name in names],
        dtype=np.int64,
    )
    output = (
        [f"state__{name}" for name in names[:candidate_start]]
        + [f"delta__{name}" for name in names[candidate_start:]]
        + [f"left__{names[index]}" for index in critical]
        + [f"right__{names[index]}" for index in critical]
    )
    return candidate_start, critical, output


def _row(
    left: np.ndarray,
    right: np.ndarray,
    candidate_start: int,
    critical: np.ndarray,
) -> np.ndarray:
    return np.concatenate((
        left[:candidate_start],
        left[candidate_start:] - right[candidate_start:],
        left[critical],
        right[critical],
    )).astype(np.float32)


def _fit_base(
    arrays: dict[str, np.ndarray],
    names: list[str],
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    rankable = train[
        (arrays["chosen_counts"][train] > 0)
        & (arrays["chosen_counts"][train] < arrays["groups"][train])
        & (arrays["forced"][train] == 0)
    ]
    rows = v1._rows_for(arrays["groups"], rankable)
    varying = v1._varying_columns(arrays["features"], rows)
    selected_names = [names[index] for index in varying]
    model = lgb.LGBMRanker(**v1._ranker_params(55137818, 900, False))
    group_sizes = arrays["groups"][rankable].astype(int)
    model.fit(
        arrays["features"][rows][:, varying], arrays["labels"][rows],
        group=group_sizes,
        sample_weight=np.repeat(
            v1._episode_recency(arrays["episode_ids"][rankable], 0.40, 2.0),
            group_sizes,
        ),
        feature_name=selected_names,
        categorical_feature=v1._categorical_columns(selected_names),
    )
    train_rows = v1._rows_for(arrays["groups"], train)
    validation_rows = v1._rows_for(arrays["groups"], validation)
    train_scores = model.predict(
        arrays["features"][train_rows][:, varying], num_iteration=900
    ).astype(np.float32)
    validation_scores = model.predict(
        arrays["features"][validation_rows][:, varying], num_iteration=900
    ).astype(np.float32)

    count_names = arrays["count_feature_names"].astype(str).tolist()
    variable = train[arrays["minimums"][train] < arrays["maximums"][train]]
    count = lgb.LGBMRegressor(**v1._count_params(55137818, 200))
    count.fit(
        arrays["count_features"][variable], arrays["chosen_counts"][variable],
        sample_weight=v1._episode_recency(
            arrays["episode_ids"][variable], 0.40, 2.0
        ),
        feature_name=count_names,
        categorical_feature=v1._categorical_columns(count_names),
    )
    counts = v1._predict_counts(
        count, arrays["count_features"], validation,
        arrays["minimums"], arrays["maximums"], num_iteration=200,
    )
    return train_scores, validation_scores, counts


def _training_pairs(
    arrays: dict[str, np.ndarray],
    train: np.ndarray,
    train_scores: np.ndarray,
    candidate_start: int,
    critical: np.ndarray,
    hard_negatives: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_starts, global_ends = v1._group_ranges(arrays["groups"])
    local_starts, local_ends = v1._group_ranges(arrays["groups"][train])
    rows: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    recency = v1._episode_recency(arrays["episode_ids"][train], 0.40, 2.0)
    for local, decision in enumerate(train):
        decision = int(decision)
        if (
            int(arrays["select_contexts"][decision]) != 0
            or bool(arrays["forced"][decision])
            or int(arrays["chosen_counts"][decision]) != 1
        ):
            continue
        start, end = int(global_starts[decision]), int(global_ends[decision])
        score_start, score_end = int(local_starts[local]), int(local_ends[local])
        chosen = int(np.flatnonzero(arrays["labels"][start:end] == 1)[0])
        chosen_semantic = tuple(arrays["semantics"][start + chosen].tolist())
        order = np.argsort(-train_scores[score_start:score_end], kind="stable")
        negatives = [
            int(index) for index in order
            if tuple(arrays["semantics"][start + index].tolist()) != chosen_semantic
        ][:hard_negatives]
        positive_row = arrays["features"][start + chosen]
        pair_weight = float(recency[local]) / max(1, len(negatives))
        for negative in negatives:
            negative_row = arrays["features"][start + negative]
            rows.append(_row(positive_row, negative_row, candidate_start, critical))
            labels.append(1)
            weights.append(pair_weight)
            rows.append(_row(negative_row, positive_row, candidate_start, critical))
            labels.append(0)
            weights.append(pair_weight)
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(weights, dtype=np.float32),
    )


def _pair_plan(
    arrays: dict[str, np.ndarray],
    validation: np.ndarray,
    base_scores: np.ndarray,
    candidate_start: int,
    critical: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, list[tuple[Any, ...]]]:
    global_starts, global_ends = v1._group_ranges(arrays["groups"])
    local_starts, local_ends = v1._group_ranges(arrays["groups"][validation])
    all_pair_rows: list[np.ndarray] = []
    plans: list[tuple[Any, ...]] = []
    for local, decision in enumerate(validation):
        decision = int(decision)
        if int(arrays["select_contexts"][decision]) != 0:
            continue
        start, end = int(global_starts[decision]), int(global_ends[decision])
        score_start, score_end = int(local_starts[local]), int(local_ends[local])
        block = base_scores[score_start:score_end]
        order = np.argsort(-block, kind="stable")[:min(top_k, len(block))]
        pair_begin = len(all_pair_rows)
        coordinates = []
        for left_position, left in enumerate(order):
            for right_position in range(left_position + 1, len(order)):
                right = order[right_position]
                all_pair_rows.append(_row(
                    arrays["features"][start + left],
                    arrays["features"][start + right],
                    candidate_start, critical,
                ))
                coordinates.append((left_position, right_position))
        plans.append((
            score_start, score_end, order,
            pair_begin, len(all_pair_rows), coordinates,
        ))
    return np.asarray(all_pair_rows, dtype=np.float32), plans


def _pair_scores(
    model: lgb.LGBMClassifier,
    pair_matrix: np.ndarray,
    plans: list[tuple[Any, ...]],
    base_scores: np.ndarray,
    trees: int,
) -> np.ndarray:
    probabilities = model.predict_proba(
        pair_matrix, num_iteration=trees
    )[:, 1]
    output = base_scores.copy()
    for score_start, score_end, order, pair_begin, pair_end, coordinates in plans:
        totals = np.zeros(len(order), dtype=np.float32)
        for probability, (left, right) in zip(
            probabilities[pair_begin:pair_end], coordinates
        ):
            probability = float(np.clip(probability, 1e-5, 1 - 1e-5))
            logit = math.log(probability / (1 - probability))
            totals[left] += logit
            totals[right] -= logit
        output[score_start:score_end] -= 1e6
        output[score_start + order] = totals
    return output


def _blend_pair_base(
    pair: np.ndarray,
    base: np.ndarray,
    groups: np.ndarray,
    decisions: np.ndarray,
    alpha: float,
) -> np.ndarray:
    starts, ends = v1._group_ranges(groups[decisions])
    result = pair.copy()
    for start, end in zip(starts, ends):
        selected = pair[start:end] > -1e5
        if not np.any(selected):
            continue
        block = base[start:end][selected].astype(np.float64)
        block = (block - block.mean()) / max(block.std(), 1e-5)
        indices = start + np.flatnonzero(selected)
        result[indices] += alpha * block
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=1000)
    parser.add_argument("--hard-negatives", type=int, default=6)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files}
    names = arrays["feature_names"].astype(str).tolist()
    splits = arrays["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    starts, _ = v1._group_ranges(arrays["groups"])
    arrays["decision_turns"] = np.rint(
        arrays["features"][starts, names.index("turn")]
    ).astype(np.int16)
    arrays["turn_pick_sets"] = v1._turn_pick_sets(arrays)

    started = time.perf_counter()
    train_scores, validation_scores, counts = _fit_base(
        arrays, names, train, validation
    )
    candidate_start, critical, pair_names = _schema(names)
    pair_x, pair_y, pair_weight = _training_pairs(
        arrays, train, train_scores, candidate_start, critical,
        args.hard_negatives,
    )
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=args.trees,
        learning_rate=0.025, num_leaves=127, min_child_samples=35,
        max_depth=-1, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.86, reg_alpha=0.25, reg_lambda=1.5,
        random_state=55137818, n_jobs=20, verbosity=-1,
    )
    model.fit(
        pair_x, pair_y, sample_weight=pair_weight, feature_name=pair_names
    )
    base_metrics = v1.evaluate(
        validation_scores, validation, arrays, counts
    )
    experiments = []
    best = None
    plans = {
        top_k: _pair_plan(
            arrays, validation, validation_scores,
            candidate_start, critical, top_k,
        )
        for top_k in (3, 5, 8)
    }
    for trees in range(100, args.trees + 1, 100):
        for top_k in (3, 5, 8):
            pair_matrix, pair_plan = plans[top_k]
            pair_scores = _pair_scores(
                model, pair_matrix, pair_plan, validation_scores, trees,
            )
            for alpha in (0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0):
                scores = _blend_pair_base(
                    pair_scores, validation_scores, arrays["groups"],
                    validation, alpha,
                )
                metrics = v1.evaluate(scores, validation, arrays, counts)
                row = {
                    "trees": trees, "top_k": top_k, "alpha": alpha,
                    "nonforced_semantic_exact": metrics[
                        "nonforced_semantic_exact"
                    ],
                    "single_top1": metrics["single_choice_semantic_top1"],
                    "main_top1": metrics[
                        "main_single_choice_semantic_top1"
                    ],
                }
                experiments.append(row)
                if best is None or (
                    row["nonforced_semantic_exact"], row["main_top1"],
                    -row["trees"], -row["top_k"],
                ) > (
                    best["nonforced_semantic_exact"], best["main_top1"],
                    -best["trees"], -best["top_k"],
                ):
                    best = row
    report: dict[str, Any] = {
        "cache": str(args.cache.resolve()),
        "test_read": False,
        "pair_rows": int(len(pair_y)),
        "pair_features": int(pair_x.shape[1]),
        "fit_seconds": time.perf_counter() - started,
        "base_validation": base_metrics,
        "selected": best,
        "experiments": experiments,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "base_exact": base_metrics["nonforced_semantic_exact"],
        "selected": best,
        "pair_rows": len(pair_y),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
