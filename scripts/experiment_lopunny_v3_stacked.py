"""Validation-only chronological stacker for Lopunny v3.

The base ranker usually contains the teacher action in its Top-3, while the
hierarchical action head is better at identifying the next semantic action
family.  This experiment learns how to arbitrate those two signals on a
chronologically later calibration slice of the training games.  The official
validation games are used only for the final model comparison and test is
never read.
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

from scripts import train_lopunny_top1_teacher as v1  # noqa: E402
from scripts.experiment_lopunny_v3_action_head import (  # noqa: E402
    _categorical,
    _decision_matrix,
    _signature,
)
from scripts.experiment_lopunny_v3_contextual_topk import (  # noqa: E402
    SEED,
    _fit_base,
    _predict_global,
    _rankable,
)


DERIVED_NAMES = [
    "base_z", "base_rank", "base_margin", "head_log_probability",
    "head_probability", "head_rank", "head_max_probability",
    "head_entropy", "signature_train_log_count", "option_position",
]


def _fit_head(
    arrays: dict[str, np.ndarray], decisions: np.ndarray, names: list[str],
    slots: int, trees: int, seed: int,
) -> tuple[lgb.LGBMClassifier, list[tuple[int, int, int]], dict[tuple[int, int, int], int], Counter]:
    matrix, kept, signatures, head_names = _decision_matrix(
        arrays, decisions, names, slots
    )
    if len(kept) != len(matrix):
        raise AssertionError("decision matrix alignment failure")
    counts = Counter(signatures)
    classes = sorted(counts)
    class_by_signature = {signature: index for index, signature in enumerate(classes)}
    labels = np.asarray([class_by_signature[value] for value in signatures], dtype=np.int32)
    model = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=trees, learning_rate=0.025,
        num_leaves=63, min_child_samples=28, max_depth=-1,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.82,
        reg_alpha=0.25, reg_lambda=1.8, random_state=seed,
        n_jobs=20, verbosity=-1,
    )
    model.fit(
        matrix, labels,
        sample_weight=v1._episode_recency(
            arrays["episode_ids"][np.asarray(kept, dtype=np.int64)], 0.35, 2.0
        ),
        feature_name=head_names, categorical_feature=_categorical(head_names),
    )
    return model, classes, class_by_signature, counts


def _stack_rows(
    arrays: dict[str, np.ndarray], decisions: np.ndarray, names: list[str],
    base_global: np.ndarray, head: lgb.LGBMClassifier,
    classes: list[tuple[int, int, int]],
    class_by_signature: dict[tuple[int, int, int], int],
    signature_counts: Counter, slots: int, trees: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    matrix, kept, _, _ = _decision_matrix(arrays, decisions, names, slots)
    probabilities = head.predict_proba(matrix, num_iteration=trees)
    starts, ends = v1._group_ranges(arrays["groups"])
    columns = {name: names.index(name) for name in (
        "option_type", "action_type", "candidate_card_id",
        "candidate_attack_id", "candidate_area", "candidate_inplay_area",
        "candidate_target_id",
    )}
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    used: list[int] = []
    for probability_row, decision in zip(probabilities, kept):
        start, end = int(starts[decision]), int(ends[decision])
        base = base_global[start:end].astype(np.float64)
        base_z = (base - base.mean()) / max(float(base.std()), 1e-5)
        base_order = np.argsort(np.argsort(-base, kind="stable"), kind="stable")
        base_margin = base - np.partition(base, -2)[-2] if len(base) > 1 else np.ones(1)
        candidate_probabilities = []
        candidate_support = []
        for candidate in arrays["features"][start:end]:
            signature = _signature(candidate, columns)
            class_index = class_by_signature.get(signature)
            candidate_probabilities.append(
                float(probability_row[class_index]) if class_index is not None else 1e-9
            )
            candidate_support.append(int(signature_counts.get(signature, 0)))
        candidate_probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
        head_order = np.argsort(
            np.argsort(-candidate_probabilities, kind="stable"), kind="stable"
        )
        entropy = -float(np.sum(probability_row * np.log(np.maximum(probability_row, 1e-9))))
        for local in range(end - start):
            rows.append([
                float(base_z[local]), float(base_order[local]),
                float(base_margin[local]),
                float(np.log(max(candidate_probabilities[local], 1e-9))),
                float(candidate_probabilities[local]), float(head_order[local]),
                float(probability_row.max()), entropy,
                float(np.log1p(candidate_support[local])),
                float(local / max(1, end - start - 1)),
            ])
            labels.append(int(arrays["labels"][start + local]))
        groups.append(end - start)
        used.append(decision)
    return (
        np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int8),
        np.asarray(groups, dtype=np.int32), used,
    )


def _replace_main_scores(
    arrays: dict[str, np.ndarray], decisions: np.ndarray,
    base_global: np.ndarray, stack_scores: np.ndarray, stack_decisions: list[int],
) -> np.ndarray:
    starts, ends = v1._group_ranges(arrays["groups"])
    rows = v1._rows_for(arrays["groups"], decisions)
    output = base_global[rows].copy()
    offsets = {
        int(decision): int(offset) for decision, offset in zip(
            decisions, np.r_[0, np.cumsum(arrays["groups"][decisions])[:-1]]
        )
    }
    source = 0
    for decision in stack_decisions:
        size = int(ends[decision] - starts[decision])
        target = offsets[decision]
        output[target:target + size] = stack_scores[source:source + size]
        source += size
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-trees", type=int, default=600)
    parser.add_argument("--head-trees", type=int, default=300)
    parser.add_argument("--stack-trees", type=int, default=600)
    parser.add_argument("--slots", type=int, default=24)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files}
    names = arrays["feature_names"].astype(str).tolist()
    count_names = arrays["count_feature_names"].astype(str).tolist()
    splits = arrays["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    starts, _ = v1._group_ranges(arrays["groups"])
    arrays["decision_turns"] = np.rint(
        arrays["features"][starts, names.index("turn")]
    ).astype(np.int16)
    arrays["turn_pick_sets"] = v1._turn_pick_sets(arrays)
    episodes = np.unique(arrays["episode_ids"][train])
    cut = max(1, min(len(episodes) - 1, int(round(
        len(episodes) * (1.0 - args.calibration_fraction)
    ))))
    fit_episode_set = set(int(value) for value in episodes[:cut])
    fit_train = train[np.asarray([
        int(arrays["episode_ids"][decision]) in fit_episode_set for decision in train
    ])]
    calibration = train[np.asarray([
        int(arrays["episode_ids"][decision]) not in fit_episode_set for decision in train
    ])]
    varying = v1._varying_columns(
        arrays["features"], v1._rows_for(arrays["groups"], _rankable(arrays, train))
    )
    started = time.perf_counter()

    calibration_base = _fit_base(
        arrays, names, fit_train, varying, args.base_trees, SEED + 3100
    )
    calibration_global = np.full(len(arrays["features"]), np.nan, dtype=np.float32)
    _predict_global(
        calibration_base, arrays, calibration, varying,
        args.base_trees, calibration_global,
    )
    calibration_head, calibration_classes, calibration_class_map, calibration_counts = _fit_head(
        arrays, fit_train, names, args.slots, args.head_trees, SEED + 3200
    )
    stack_x, stack_y, stack_groups, stack_decisions = _stack_rows(
        arrays, calibration, names, calibration_global, calibration_head,
        calibration_classes, calibration_class_map, calibration_counts,
        args.slots, args.head_trees,
    )
    stacker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=args.stack_trees,
        learning_rate=0.025, num_leaves=31, min_child_samples=35,
        max_depth=-1, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, reg_alpha=0.2, reg_lambda=1.5,
        random_state=SEED + 3300, n_jobs=20, verbosity=-1,
    )
    stacker.fit(
        stack_x, stack_y, group=stack_groups,
        sample_weight=np.repeat(
            v1._episode_recency(
                arrays["episode_ids"][np.asarray(stack_decisions)], 0.25, 2.0
            ), stack_groups,
        ),
        feature_name=DERIVED_NAMES,
    )

    final_base = _fit_base(arrays, names, train, varying, args.base_trees, SEED)
    validation_global = np.full(len(arrays["features"]), np.nan, dtype=np.float32)
    _predict_global(
        final_base, arrays, validation, varying, args.base_trees, validation_global
    )
    final_head, classes, class_map, signature_counts = _fit_head(
        arrays, train, names, args.slots, args.head_trees, SEED + 900
    )
    validation_x, _, _, validation_stack_decisions = _stack_rows(
        arrays, validation, names, validation_global, final_head, classes,
        class_map, signature_counts, args.slots, args.head_trees,
    )

    variable = train[arrays["minimums"][train] < arrays["maximums"][train]]
    count = lgb.LGBMRegressor(**v1._count_params(SEED, 250))
    count.fit(
        arrays["count_features"][variable], arrays["chosen_counts"][variable],
        feature_name=count_names,
        categorical_feature=v1._categorical_columns(count_names),
    )
    counts = v1._predict_counts(
        count, arrays["count_features"], validation,
        arrays["minimums"], arrays["maximums"], num_iteration=250,
    )
    base_rows = v1._rows_for(arrays["groups"], validation)
    base_metrics = v1.evaluate(
        validation_global[base_rows], validation, arrays, counts
    )
    experiments: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for trees in range(100, args.stack_trees + 1, 100):
        stacked = stacker.predict(validation_x, num_iteration=trees).astype(np.float32)
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            blended = stacked + alpha * validation_x[:, 0]
            scores = _replace_main_scores(
                arrays, validation, validation_global, blended,
                validation_stack_decisions,
            )
            metrics = v1.evaluate(scores, validation, arrays, counts)
            row = {
                "trees": trees, "base_z_alpha": alpha,
                "nonforced_semantic_exact": metrics["nonforced_semantic_exact"],
                "single_top1": metrics["single_choice_semantic_top1"],
                "main_top1": metrics["main_single_choice_semantic_top1"],
            }
            experiments.append(row)
            if best is None or (
                row["nonforced_semantic_exact"], row["main_top1"], -row["trees"]
            ) > (
                best["nonforced_semantic_exact"], best["main_top1"], -best["trees"]
            ):
                best = row
    report: dict[str, Any] = {
        "cache": str(args.cache.resolve()), "test_read": False,
        "fit_train_episodes": int(cut),
        "calibration_episodes": int(len(episodes) - cut),
        "calibration_main_decisions": int(len(stack_decisions)),
        "base_validation": base_metrics, "selected": best,
        "fit_seconds": time.perf_counter() - started,
        "target": {
            "metric": "validation_nonforced_semantic_exact", "value": 0.85,
            "met": bool(best and best["nonforced_semantic_exact"] >= 0.85),
        },
        "experiments": experiments,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "base": base_metrics["nonforced_semantic_exact"],
        "selected": best, "target": report["target"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
