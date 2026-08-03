"""Validation-only blend of next-action and within-turn plan rankers.

One ranker learns the exact next action.  The second assigns a positive label
to every legal semantic action Majkel will use at any later point in the same
turn.  Their decision-local z-score blend tests whether filtering to the
teacher's turn plan can remove off-plan Top-1 errors without losing ordering.
The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_lopunny_top1_teacher as v1  # noqa: E402
from scripts.experiment_lopunny_v3_contextual_topk import (  # noqa: E402
    SEED,
    _fit_base,
    _predict_global,
    _rankable,
)


def _future_labels(arrays: dict[str, np.ndarray]) -> np.ndarray:
    labels = np.zeros(len(arrays["features"]), dtype=np.int8)
    starts, ends = v1._group_ranges(arrays["groups"])
    for decision, future in enumerate(arrays["turn_pick_sets"]):
        start, end = int(starts[decision]), int(ends[decision])
        for row in range(start, end):
            key = tuple(int(value) for value in arrays["semantics"][row])
            labels[row] = int(key in future)
    return labels


def _blend(
    arrays: dict[str, np.ndarray],
    decisions: np.ndarray,
    base_global: np.ndarray,
    plan_global: np.ndarray,
    alpha: float,
) -> np.ndarray:
    starts, ends = v1._group_ranges(arrays["groups"])
    blocks: list[np.ndarray] = []
    for decision_value in decisions:
        decision = int(decision_value)
        start, end = int(starts[decision]), int(ends[decision])
        base = base_global[start:end]
        plan = plan_global[start:end]
        base_z = (base - base.mean()) / max(float(base.std()), 1e-5)
        plan_z = (plan - plan.mean()) / max(float(plan.std()), 1e-5)
        blocks.append(plan_z + alpha * base_z)
    return np.concatenate(blocks).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=600)
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
    future_labels = _future_labels(arrays)
    rankable = _rankable(arrays, train)
    varying = v1._varying_columns(
        arrays["features"], v1._rows_for(arrays["groups"], rankable)
    )
    selected_names = [names[index] for index in varying]
    train_rows = v1._rows_for(arrays["groups"], rankable)
    train_groups = arrays["groups"][rankable].astype(int)
    started = time.perf_counter()

    base = _fit_base(arrays, names, train, varying, args.trees, SEED)
    plan = lgb.LGBMRanker(**v1._ranker_params(SEED + 2100, args.trees, False))
    plan.fit(
        arrays["features"][train_rows][:, varying], future_labels[train_rows],
        group=train_groups,
        sample_weight=np.repeat(
            v1._episode_recency(arrays["episode_ids"][rankable], 0.35, 2.0),
            train_groups,
        ),
        feature_name=selected_names,
        categorical_feature=v1._categorical_columns(selected_names),
    )
    base_global = np.full(len(arrays["features"]), np.nan, dtype=np.float32)
    plan_global = np.full(len(arrays["features"]), np.nan, dtype=np.float32)
    _predict_global(base, arrays, validation, varying, args.trees, base_global)
    _predict_global(plan, arrays, validation, varying, args.trees, plan_global)

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
    base_metrics = v1.evaluate(base_global[base_rows], validation, arrays, counts)
    experiments: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        scores = _blend(arrays, validation, base_global, plan_global, alpha)
        metrics = v1.evaluate(scores, validation, arrays, counts)
        row = {
            "alpha": alpha,
            "nonforced_semantic_exact": metrics["nonforced_semantic_exact"],
            "single_top1": metrics["single_choice_semantic_top1"],
            "single_turn_set": metrics["single_choice_turn_set"],
            "main_top1": metrics["main_single_choice_semantic_top1"],
        }
        experiments.append(row)
        if best is None or (
            row["nonforced_semantic_exact"], row["main_top1"]
        ) > (
            best["nonforced_semantic_exact"], best["main_top1"]
        ):
            best = row
    report: dict[str, Any] = {
        "cache": str(args.cache.resolve()), "test_read": False,
        "future_positive_rate": float(future_labels[train_rows].mean()),
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
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"base": base_metrics["nonforced_semantic_exact"],
                      "selected": best, "target": report["target"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
