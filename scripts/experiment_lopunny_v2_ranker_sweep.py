"""Validation-only capacity, recency, and seed sweep for Lopunny v2.

Each model fits a fixed tree budget on train and selects its tree prefix on
validation semantic exact.  No test arrays are scored.  Decision-local z-score
ensembles and a semantic oracle expose whether variance reduction or a learned
selector could plausibly reach the v2 target before either is implemented.
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


CONFIGS = [
    {
        "name": "v1_seed",
        "leaves": 63, "child": 24, "lr": 0.035, "colsample": 0.78,
        "seed": 55137818, "fraction": 1.0, "floor": 0.40,
    },
    {
        "name": "v1_seed7",
        "leaves": 63, "child": 24, "lr": 0.035, "colsample": 0.78,
        "seed": 7, "fraction": 1.0, "floor": 0.40,
    },
    {
        "name": "v1_seed42",
        "leaves": 63, "child": 24, "lr": 0.035, "colsample": 0.78,
        "seed": 42, "fraction": 1.0, "floor": 0.40,
    },
    {
        "name": "large",
        "leaves": 127, "child": 20, "lr": 0.025, "colsample": 0.85,
        "seed": 55137818, "fraction": 1.0, "floor": 0.35,
    },
    {
        "name": "xlarge",
        "leaves": 255, "child": 16, "lr": 0.020, "colsample": 0.90,
        "seed": 55137818, "fraction": 1.0, "floor": 0.30,
    },
    {
        "name": "recent_0875",
        "leaves": 127, "child": 20, "lr": 0.025, "colsample": 0.85,
        "seed": 55137818, "fraction": 0.875, "floor": 0.25,
    },
    {
        "name": "recent_075",
        "leaves": 127, "child": 20, "lr": 0.025, "colsample": 0.85,
        "seed": 55137818, "fraction": 0.75, "floor": 0.25,
    },
    {
        "name": "uniform",
        "leaves": 127, "child": 20, "lr": 0.025, "colsample": 0.85,
        "seed": 55137818, "fraction": 1.0, "floor": 1.0,
    },
]


def _normalise(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    starts, ends = v1._group_ranges(groups)
    out = np.empty_like(scores, dtype=np.float32)
    for start, end in zip(starts, ends):
        block = scores[start:end]
        out[start:end] = (
            (block - float(block.mean())) / max(float(block.std()), 1e-5)
        )
    return out


def _semantic_hits(
    scores: np.ndarray,
    decisions: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> np.ndarray:
    starts, ends = v1._group_ranges(arrays["groups"][decisions])
    absolute_rows = v1._rows_for(arrays["groups"], decisions)
    group_starts, group_ends = v1._group_ranges(arrays["groups"])
    result = np.zeros(len(decisions), dtype=bool)
    for local, (start, end) in enumerate(zip(starts, ends)):
        decision = int(decisions[local])
        picked = start + int(np.argmax(scores[start:end]))
        predicted = tuple(int(value) for value in arrays["semantics"][absolute_rows[picked]])
        teacher_rows = np.flatnonzero(
            arrays["labels"][group_starts[decision]:group_ends[decision]] == 1
        )
        teacher = {
            tuple(int(value) for value in arrays["semantics"][
                group_starts[decision] + row
            ])
            for row in teacher_rows
        }
        result[local] = predicted in teacher
    return result


def _fit_count(
    arrays: dict[str, np.ndarray],
    names: list[str],
    train: np.ndarray,
) -> tuple[lgb.LGBMRegressor, dict[int, int]]:
    variable = train[arrays["minimums"][train] < arrays["maximums"][train]]
    model = lgb.LGBMRegressor(**v1._count_params(55137818, 200))
    model.fit(
        arrays["count_features"][variable], arrays["chosen_counts"][variable],
        sample_weight=v1._episode_recency(
            arrays["episode_ids"][variable], 0.40, 2.0
        ),
        feature_name=names,
        categorical_feature=v1._categorical_columns(names),
    )
    return model, {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=1400)
    parser.add_argument("--step", type=int, default=100)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files}
    feature_names = arrays["feature_names"].astype(str).tolist()
    count_names = arrays["count_feature_names"].astype(str).tolist()
    splits = arrays["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    starts, _ = v1._group_ranges(arrays["groups"])
    arrays["decision_turns"] = np.rint(
        arrays["features"][starts, feature_names.index("turn")]
    ).astype(np.int16)
    arrays["turn_pick_sets"] = v1._turn_pick_sets(arrays)

    all_rankable = train[
        (arrays["chosen_counts"][train] > 0)
        & (arrays["chosen_counts"][train] < arrays["groups"][train])
        & (arrays["forced"][train] == 0)
    ]
    varying = v1._varying_columns(
        arrays["features"], v1._rows_for(arrays["groups"], all_rankable)
    )
    matrix = np.ascontiguousarray(arrays["features"][:, varying])
    names = [feature_names[index] for index in varying]
    categorical = v1._categorical_columns(names)
    validation_rows = v1._rows_for(arrays["groups"], validation)
    count_model, _ = _fit_count(arrays, count_names, train)
    validation_counts = v1._predict_counts(
        count_model, arrays["count_features"], validation,
        arrays["minimums"], arrays["maximums"], num_iteration=200,
    )

    results: list[dict[str, Any]] = []
    score_bank: dict[str, np.ndarray] = {}
    hit_bank: dict[str, np.ndarray] = {}
    train_episodes = np.unique(arrays["episode_ids"][train])
    for config in CONFIGS:
        keep_count = max(1, int(round(len(train_episodes) * config["fraction"])))
        kept = train_episodes[-keep_count:]
        fit_decisions = all_rankable[
            np.isin(arrays["episode_ids"][all_rankable], kept)
        ]
        fit_rows = v1._rows_for(arrays["groups"], fit_decisions)
        fit_groups = arrays["groups"][fit_decisions].astype(int)
        params = {
            "objective": "lambdarank", "metric": "None",
            "n_estimators": args.trees, "learning_rate": config["lr"],
            "num_leaves": config["leaves"],
            "min_child_samples": config["child"], "max_depth": -1,
            "subsample": 0.9, "subsample_freq": 1,
            "colsample_bytree": config["colsample"],
            "reg_alpha": 0.2, "reg_lambda": 1.5,
            "random_state": config["seed"], "n_jobs": 20,
            "verbosity": -1, "label_gain": [0, 1],
        }
        model = lgb.LGBMRanker(**params)
        fit_started = time.perf_counter()
        model.fit(
            matrix[fit_rows], arrays["labels"][fit_rows], group=fit_groups,
            sample_weight=np.repeat(
                v1._episode_recency(
                    arrays["episode_ids"][fit_decisions],
                    config["floor"], 2.0,
                ),
                fit_groups,
            ),
            feature_name=names, categorical_feature=categorical,
        )
        curve = []
        best = None
        best_scores = None
        for trees in range(args.step, args.trees + 1, args.step):
            scores = model.predict(
                matrix[validation_rows], num_iteration=trees
            ).astype(np.float32)
            metrics = v1.evaluate(
                scores, validation, arrays, validation_counts
            )
            point = {
                "trees": trees,
                "nonforced_semantic_exact": metrics["nonforced_semantic_exact"],
                "single_top1": metrics["single_choice_semantic_top1"],
                "main_top1": metrics["main_single_choice_semantic_top1"],
            }
            curve.append(point)
            if best is None or (
                point["nonforced_semantic_exact"], point["main_top1"], -trees
            ) > (
                best["nonforced_semantic_exact"], best["main_top1"],
                -best["trees"],
            ):
                best = point
                best_scores = scores
        assert best is not None and best_scores is not None
        tag = config["name"]
        score_bank[tag] = _normalise(best_scores, arrays["groups"][validation])
        hit_bank[tag] = _semantic_hits(best_scores, validation, arrays)
        row = {
            **config,
            "fit_episodes": keep_count,
            "fit_decisions": int(len(fit_decisions)),
            "fit_seconds": time.perf_counter() - fit_started,
            "best": best,
            "curve": curve,
        }
        results.append(row)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"partial": results}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({tag: best}), flush=True)

    ordered = sorted(
        results,
        key=lambda row: row["best"]["nonforced_semantic_exact"],
        reverse=True,
    )
    ensemble = []
    used: list[str] = []
    running = None
    for row in ordered:
        tag = row["name"]
        used.append(tag)
        running = score_bank[tag].copy() if running is None else running + score_bank[tag]
        metrics = v1.evaluate(
            running / len(used), validation, arrays, validation_counts
        )
        ensemble.append({
            "members": list(used),
            "nonforced_semantic_exact": metrics["nonforced_semantic_exact"],
            "single_top1": metrics["single_choice_semantic_top1"],
            "main_top1": metrics["main_single_choice_semantic_top1"],
        })
    oracle = np.logical_or.reduce(list(hit_bank.values()))
    nonforced_single = (
        (arrays["forced"][validation] == 0)
        & (arrays["chosen_counts"][validation] == 1)
    )
    main_single = nonforced_single & (arrays["select_contexts"][validation] == 0)
    report = {
        "cache": str(args.cache.resolve()),
        "test_read": False,
        "features": len(names),
        "models": results,
        "ensemble_ordered_by_individual_validation": ensemble,
        "semantic_oracle": {
            "all_decisions": float(oracle.mean()),
            "nonforced_single": float(oracle[nonforced_single].mean()),
            "main_single": float(oracle[main_single].mean()),
        },
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "best": ordered[0]["best"],
        "best_name": ordered[0]["name"],
        "best_ensemble": max(
            ensemble, key=lambda row: row["nonforced_semantic_exact"]
        ),
        "oracle": report["semantic_oracle"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
