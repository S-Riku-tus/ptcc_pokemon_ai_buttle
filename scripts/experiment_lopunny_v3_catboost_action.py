"""Validation-only CatBoost challenger for the v3 hierarchical action head.

CatBoost's ordered categorical statistics are a useful counterpoint to
LightGBM for sparse card/action IDs.  All categorical columns are cast to
strings and one-hot encoded (high threshold), keeping the learned structure
portable and avoiding target-statistic leakage across rows.  Test is not read.
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
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_lopunny_top1_teacher as v1  # noqa: E402
from scripts.experiment_lopunny_v3_action_head import (  # noqa: E402
    _categorical,
    _decision_matrix,
    _scores,
)
from scripts.experiment_lopunny_v3_contextual_topk import (  # noqa: E402
    SEED,
    _fit_base,
    _predict_global,
    _rankable,
)


def _frame(matrix: np.ndarray, names: list[str], categorical: list[int]) -> pd.DataFrame:
    frame = pd.DataFrame(matrix, columns=names)
    for index in categorical:
        name = names[index]
        frame[name] = np.rint(frame[name]).astype(np.int64).astype(str)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-trees", type=int, default=400)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--slots", type=int, default=16)
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

    train_x, train_kept, train_signatures, head_names = _decision_matrix(
        arrays, train, names, args.slots
    )
    validation_x, validation_kept, validation_signatures, _ = _decision_matrix(
        arrays, validation, names, args.slots
    )
    signature_counts = Counter(train_signatures)
    ordered_signatures = sorted(signature_counts)
    class_by_signature = {
        signature: index for index, signature in enumerate(ordered_signatures)
    }
    train_y = np.asarray([
        class_by_signature[signature] for signature in train_signatures
    ], dtype=np.int32)
    categorical = _categorical(head_names)
    categorical_names = [head_names[index] for index in categorical]
    train_frame = _frame(train_x, head_names, categorical)
    validation_frame = _frame(validation_x, head_names, categorical)
    train_pool = Pool(
        train_frame, train_y, cat_features=categorical_names,
        weight=v1._episode_recency(
            arrays["episode_ids"][np.asarray(train_kept)], 0.35, 2.0
        ),
    )
    validation_pool = Pool(validation_frame, cat_features=categorical_names)
    head = CatBoostClassifier(
        iterations=args.iterations, depth=8, learning_rate=0.05,
        loss_function="MultiClass", random_seed=SEED + 1700,
        l2_leaf_reg=4.0, random_strength=0.35, rsm=0.85,
        one_hot_max_size=2048, thread_count=20, verbose=100,
        allow_writing_files=False,
    )
    head.fit(train_pool)

    count_names = arrays["count_feature_names"].astype(str).tolist()
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
    known = np.asarray([
        int(signature in class_by_signature) for signature in validation_signatures
    ], dtype=np.int8)
    for trees in range(100, args.iterations + 1, 100):
        probabilities = np.asarray(
            head.predict_proba(validation_pool, ntree_end=trees), dtype=np.float32
        )
        predicted = probabilities.argmax(axis=1)
        signature_accuracy = float(np.mean([
            bool(known[index])
            and ordered_signatures[int(predicted[index])] == signature
            for index, signature in enumerate(validation_signatures)
        ]))
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            scores = _scores(
                arrays, validation, validation_global, probabilities,
                validation_kept, class_by_signature, names, alpha,
            )
            metrics = v1.evaluate(scores, validation, arrays, counts)
            row = {
                "trees": trees, "alpha": alpha,
                "signature_accuracy": signature_accuracy,
                "nonforced_semantic_exact": metrics["nonforced_semantic_exact"],
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
        "classes": len(ordered_signatures),
        "categorical_features": len(categorical),
        "validation_known_signature_rate": float(known.mean()),
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
