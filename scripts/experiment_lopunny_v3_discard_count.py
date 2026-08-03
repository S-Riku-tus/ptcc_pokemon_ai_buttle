"""Validation-only specialist for multi-discard choice and pick count.

The global ranker shares trees across thirteen unrelated selection contexts;
context 8 is the only common multi-pick discard menu and is its largest error
bucket.  This experiment fits a context-8-only ranker plus an integer count
classifier, then blends the specialist and global scores locally.  Test is
never read.
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


def _zblend_context(
    arrays: dict[str, np.ndarray],
    decisions: np.ndarray,
    base_global: np.ndarray,
    specialist_global: np.ndarray,
    alpha: float,
) -> np.ndarray:
    starts, ends = v1._group_ranges(arrays["groups"])
    rows = v1._rows_for(arrays["groups"], decisions)
    output = base_global[rows].copy()
    offset = 0
    for decision_value in decisions:
        decision = int(decision_value)
        start, end = int(starts[decision]), int(ends[decision])
        size = end - start
        if int(arrays["select_contexts"][decision]) == 8:
            base = base_global[start:end]
            specialist = specialist_global[start:end]
            base_z = (base - base.mean()) / max(float(base.std()), 1e-5)
            specialist_z = (
                specialist - specialist.mean()
            ) / max(float(specialist.std()), 1e-5)
            output[offset:offset + size] = specialist_z + alpha * base_z
        offset += size
    return output


def _count_predictions(
    model: lgb.LGBMClassifier,
    arrays: dict[str, np.ndarray],
    decisions: np.ndarray,
    trees: int,
) -> dict[int, int]:
    variable = decisions[
        arrays["minimums"][decisions] < arrays["maximums"][decisions]
    ]
    predicted = model.predict(
        arrays["count_features"][variable], num_iteration=trees
    ).astype(int)
    return {
        int(decision): int(np.clip(
            value, arrays["minimums"][decision], arrays["maximums"][decision]
        ))
        for decision, value in zip(variable, predicted)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-trees", type=int, default=400)
    parser.add_argument("--specialist-trees", type=int, default=1000)
    parser.add_argument("--count-trees", type=int, default=600)
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
    varying = v1._varying_columns(
        arrays["features"],
        v1._rows_for(arrays["groups"], _rankable(arrays, train)),
    )
    started = time.perf_counter()

    base = _fit_base(arrays, names, train, varying, args.base_trees, SEED)
    validation_global = np.full(len(arrays["features"]), np.nan, dtype=np.float32)
    _predict_global(
        base, arrays, validation, varying, args.base_trees, validation_global
    )

    discard_train = train[
        (arrays["select_contexts"][train] == 8)
        & (arrays["chosen_counts"][train] > 0)
        & (arrays["chosen_counts"][train] < arrays["groups"][train])
        & (arrays["forced"][train] == 0)
    ]
    discard_rows = v1._rows_for(arrays["groups"], discard_train)
    discard_varying = v1._varying_columns(arrays["features"], discard_rows)
    discard_names = [names[index] for index in discard_varying]
    discard_groups = arrays["groups"][discard_train].astype(int)
    specialist = lgb.LGBMRanker(
        **v1._ranker_params(SEED + 1200, args.specialist_trees, False)
    )
    specialist.fit(
        arrays["features"][discard_rows][:, discard_varying],
        arrays["labels"][discard_rows], group=discard_groups,
        sample_weight=np.repeat(
            v1._episode_recency(arrays["episode_ids"][discard_train], 0.25, 2.0),
            discard_groups,
        ),
        feature_name=discard_names,
        categorical_feature=v1._categorical_columns(discard_names),
    )
    discard_validation = validation[arrays["select_contexts"][validation] == 8]
    specialist_global = np.full(len(arrays["features"]), np.nan, dtype=np.float32)
    specialist_rows = v1._rows_for(arrays["groups"], discard_validation)
    specialist_global[specialist_rows] = specialist.predict(
        arrays["features"][specialist_rows][:, discard_varying],
        num_iteration=args.specialist_trees,
    ).astype(np.float32)

    variable_train = train[
        arrays["minimums"][train] < arrays["maximums"][train]
    ]
    count = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=args.count_trees,
        learning_rate=0.03, num_leaves=63, min_child_samples=20,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.86,
        reg_alpha=0.15, reg_lambda=1.2, random_state=SEED + 1400,
        n_jobs=20, verbosity=-1,
    )
    count.fit(
        arrays["count_features"][variable_train],
        arrays["chosen_counts"][variable_train].astype(int),
        sample_weight=v1._episode_recency(
            arrays["episode_ids"][variable_train], 0.25, 2.0
        ),
        feature_name=count_names,
        categorical_feature=v1._categorical_columns(count_names),
    )

    experiments: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    base_counts = _count_predictions(count, arrays, validation, args.count_trees)
    base_rows = v1._rows_for(arrays["groups"], validation)
    base_metrics = v1.evaluate(
        validation_global[base_rows], validation, arrays, base_counts
    )
    for specialist_trees in range(100, args.specialist_trees + 1, 100):
        specialist_global[specialist_rows] = specialist.predict(
            arrays["features"][specialist_rows][:, discard_varying],
            num_iteration=specialist_trees,
        ).astype(np.float32)
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            scores = _zblend_context(
                arrays, validation, validation_global, specialist_global, alpha
            )
            metrics = v1.evaluate(scores, validation, arrays, base_counts)
            discard_metrics = metrics["by_context"].get("type_1_context_8", {})
            row = {
                "trees": specialist_trees, "alpha": alpha,
                "nonforced_semantic_exact": metrics["nonforced_semantic_exact"],
                "discard_semantic_exact": discard_metrics.get(
                    "nonforced_semantic_exact", 0.0
                ),
                "variable_count_accuracy": metrics["variable_count_accuracy"],
            }
            experiments.append(row)
            if best is None or (
                row["nonforced_semantic_exact"], row["discard_semantic_exact"],
                -row["trees"]
            ) > (
                best["nonforced_semantic_exact"], best["discard_semantic_exact"],
                -best["trees"]
            ):
                best = row

    report: dict[str, Any] = {
        "cache": str(args.cache.resolve()), "test_read": False,
        "discard_train_decisions": int(len(discard_train)),
        "discard_validation_decisions": int(len(discard_validation)),
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
    print(json.dumps({
        "base": base_metrics["nonforced_semantic_exact"],
        "selected": best, "target": report["target"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
