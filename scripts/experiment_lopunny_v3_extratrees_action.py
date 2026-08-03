"""Validation-only ExtraTrees challenger for the Lopunny action head.

LightGBM treats many public-state identifiers as ordered/categorical splits.
ExtraTrees provides a materially different high-variance ensemble over the
same hierarchical action-signature task.  It is accepted only if it improves
the untouched chronological validation split; test is never loaded here.
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
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_lopunny_top1_teacher as v1  # noqa: E402
from scripts.experiment_lopunny_v3_action_head import (  # noqa: E402
    _decision_matrix,
    _scores,
)
from scripts.experiment_lopunny_v3_contextual_topk import (  # noqa: E402
    SEED,
    _fit_base,
    _predict_global,
    _rankable,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-trees", type=int, default=600)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--slots", type=int, default=24)
    parser.add_argument("--max-features", type=float, default=0.45)
    parser.add_argument("--min-leaf", type=int, default=2)
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

    train_x, train_kept, train_signatures, _ = _decision_matrix(
        arrays, train, names, args.slots
    )
    validation_x, validation_kept, validation_signatures, _ = _decision_matrix(
        arrays, validation, names, args.slots
    )
    minimum = np.minimum(train_x.min(axis=0), validation_x.min(axis=0))
    maximum = np.maximum(train_x.max(axis=0), validation_x.max(axis=0))
    keep = np.flatnonzero(minimum != maximum)
    train_x = train_x[:, keep]
    validation_x = validation_x[:, keep]
    signature_counts = Counter(train_signatures)
    ordered_signatures = sorted(signature_counts)
    class_by_signature = {
        signature: index for index, signature in enumerate(ordered_signatures)
    }
    train_y = np.asarray(
        [class_by_signature[value] for value in train_signatures], dtype=np.int32
    )
    head = ExtraTreesClassifier(
        n_estimators=args.trees, criterion="entropy",
        max_features=args.max_features, min_samples_leaf=args.min_leaf,
        bootstrap=False, class_weight=None, random_state=SEED + 4100,
        n_jobs=20, verbose=0,
    )
    head.fit(
        train_x, train_y,
        sample_weight=v1._episode_recency(
            arrays["episode_ids"][np.asarray(train_kept, dtype=np.int64)],
            0.35, 2.0,
        ),
    )
    probabilities = head.predict_proba(validation_x)

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
    predictions = probabilities.argmax(axis=1)
    signature_accuracy = float(np.mean([
        ordered_signatures[int(prediction)] == truth
        for prediction, truth in zip(predictions, validation_signatures)
    ]))
    experiments: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        scores = _scores(
            arrays, validation, validation_global, probabilities,
            validation_kept, class_by_signature, names, alpha,
        )
        metrics = v1.evaluate(scores, validation, arrays, counts)
        row = {
            "alpha": alpha, "signature_accuracy": signature_accuracy,
            "nonforced_semantic_exact": metrics["nonforced_semantic_exact"],
            "single_top1": metrics["single_choice_semantic_top1"],
            "main_top1": metrics["main_single_choice_semantic_top1"],
        }
        experiments.append(row)
        if best is None or (
            row["nonforced_semantic_exact"], row["main_top1"]
        ) > (best["nonforced_semantic_exact"], best["main_top1"]):
            best = row
    report: dict[str, Any] = {
        "cache": str(args.cache.resolve()), "test_read": False,
        "parameters": {
            "trees": args.trees, "max_features": args.max_features,
            "min_leaf": args.min_leaf, "input_features": int(len(keep)),
        },
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
