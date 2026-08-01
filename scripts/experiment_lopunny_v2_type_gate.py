"""Validation-only action-type gate for Lopunny v2.

The v1 candidate ranker must learn two different questions in one score:
which action family the teacher takes next, and which card/target within that
family is preferred.  This experiment gives the first question a dedicated
listwise multiclass model built from one candidate-independent row per
decision.  Its masked action-type probabilities are blended with v1 scores;
the v1 ranker still resolves cards and targets inside a type.

Hyperparameters, tree prefixes, blend strengths, and confidence gates are
selected on validation.  Test is not read unless ``--evaluate-test`` is
explicitly supplied after the design has been frozen.
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


CONFIGS = {
    "small": {"num_leaves": 31, "min_child_samples": 50, "reg_lambda": 2.0},
    "medium": {"num_leaves": 63, "min_child_samples": 30, "reg_lambda": 1.5},
    "large": {"num_leaves": 127, "min_child_samples": 20, "reg_lambda": 2.0},
}


def _teacher_action_types(arrays: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    starts, ends = v1._group_ranges(arrays["groups"])
    action_column = names.index("action_type")
    result = np.full(len(starts), -1, dtype=np.int8)
    for decision, (start, end) in enumerate(zip(starts, ends)):
        chosen = np.flatnonzero(arrays["labels"][start:end] == 1)
        if len(chosen) == 1:
            result[decision] = int(
                round(arrays["features"][start + chosen[0], action_column])
            )
    return result


def _fit_base(
    arrays: dict[str, np.ndarray],
    names: list[str],
    train: np.ndarray,
    trees: int,
) -> tuple[lgb.LGBMRanker, np.ndarray, list[str]]:
    groups = arrays["groups"]
    rankable = train[
        (arrays["chosen_counts"][train] > 0)
        & (arrays["chosen_counts"][train] < groups[train])
        & (arrays["forced"][train] == 0)
    ]
    rows = v1._rows_for(groups, rankable)
    varying = v1._varying_columns(arrays["features"], rows)
    selected_names = [names[index] for index in varying]
    model = lgb.LGBMRanker(**v1._ranker_params(55137818, trees, False))
    group_sizes = groups[rankable].astype(int)
    model.fit(
        arrays["features"][rows][:, varying],
        arrays["labels"][rows],
        group=group_sizes,
        sample_weight=np.repeat(
            v1._episode_recency(arrays["episode_ids"][rankable], 0.40, 2.0),
            group_sizes,
        ),
        feature_name=selected_names,
        categorical_feature=v1._categorical_columns(selected_names),
    )
    return model, varying, selected_names


def _fit_count(
    arrays: dict[str, np.ndarray],
    count_names: list[str],
    train: np.ndarray,
) -> lgb.LGBMRegressor:
    variable = train[arrays["minimums"][train] < arrays["maximums"][train]]
    model = lgb.LGBMRegressor(**v1._count_params(55137818, 200))
    model.fit(
        arrays["count_features"][variable],
        arrays["chosen_counts"][variable],
        sample_weight=v1._episode_recency(
            arrays["episode_ids"][variable], 0.40, 2.0
        ),
        feature_name=count_names,
        categorical_feature=v1._categorical_columns(count_names),
    )
    return model


def _score_rows(
    model: lgb.LGBMRanker,
    arrays: dict[str, np.ndarray],
    varying: np.ndarray,
    decisions: np.ndarray,
    trees: int,
) -> np.ndarray:
    rows = v1._rows_for(arrays["groups"], decisions)
    return model.predict(
        arrays["features"][rows][:, varying], num_iteration=trees
    ).astype(np.float32)


def _probability_table(
    model: lgb.LGBMClassifier,
    matrix: np.ndarray,
    decisions: np.ndarray,
    trees: int,
) -> np.ndarray:
    compact = model.predict_proba(matrix[decisions], num_iteration=trees)
    table = np.full((len(decisions), 12), 1e-8, dtype=np.float32)
    for column, action_type in enumerate(model.classes_):
        table[:, int(action_type)] = compact[:, column]
    return table


def _blend(
    base_scores: np.ndarray,
    arrays: dict[str, np.ndarray],
    names: list[str],
    decisions: np.ndarray,
    probabilities: np.ndarray,
    alpha: float,
    confidence: float,
    teacher_types: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    groups = arrays["groups"]
    starts, ends = v1._group_ranges(groups[decisions])
    action_column = names.index("action_type")
    absolute_rows = v1._rows_for(groups, decisions)
    action_types = np.rint(
        arrays["features"][absolute_rows, action_column]
    ).astype(np.int16)
    out = base_scores.astype(np.float64).copy()
    applied = 0
    type_correct = 0
    type_total = 0
    for local, (start, end) in enumerate(zip(starts, ends)):
        decision = int(decisions[local])
        if int(arrays["select_contexts"][decision]) != 0:
            continue
        block_types = action_types[start:end]
        available = np.unique(block_types)
        masked = probabilities[local, available]
        order = np.argsort(-masked, kind="stable")
        predicted_type = int(available[order[0]])
        probability = float(masked[order[0]])
        type_total += int(teacher_types[decision] >= 0)
        type_correct += int(predicted_type == teacher_types[decision])
        if probability < confidence:
            continue
        block = out[start:end]
        scale = max(float(block.std()), 1e-5)
        block = (block - float(block.mean())) / scale
        prior = np.log(np.maximum(probabilities[local, block_types], 1e-8))
        out[start:end] = block + alpha * prior
        applied += 1
    return out.astype(np.float32), {
        "applied_main_decisions": applied,
        "masked_type_accuracy": type_correct / max(1, type_total),
        "main_decisions": type_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--base-trees", type=int, default=900)
    parser.add_argument("--type-trees", type=int, default=1200)
    parser.add_argument("--tree-step", type=int, default=100)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files}
    names = arrays["feature_names"].astype(str).tolist()
    count_names = arrays["count_feature_names"].astype(str).tolist()
    splits = arrays["splits"].astype(str)
    decisions = {
        split: np.flatnonzero(splits == split)
        for split in ("train", "validation", "test")
    }
    teacher_types = _teacher_action_types(arrays, names)
    group_starts, _ = v1._group_ranges(arrays["groups"])
    turn_column = names.index("turn")
    arrays["decision_turns"] = np.rint(
        arrays["features"][group_starts, turn_column]
    ).astype(np.int16)
    arrays["turn_pick_sets"] = v1._turn_pick_sets(arrays)
    type_train = decisions["train"][
        (arrays["select_contexts"][decisions["train"]] == 0)
        & (arrays["forced"][decisions["train"]] == 0)
        & (arrays["chosen_counts"][decisions["train"]] == 1)
        & (teacher_types[decisions["train"]] > 0)
    ]

    started = time.perf_counter()
    base, varying, selected_names = _fit_base(
        arrays, names, decisions["train"], args.base_trees
    )
    count_model = _fit_count(arrays, count_names, decisions["train"])
    base_scores = {
        split: _score_rows(
            base, arrays, varying, values, args.base_trees
        )
        for split, values in decisions.items()
        if split != "test" or args.evaluate_test
    }
    count_predictions = {
        split: v1._predict_counts(
            count_model,
            arrays["count_features"],
            values,
            arrays["minimums"],
            arrays["maximums"],
            num_iteration=200,
        )
        for split, values in decisions.items()
        if split != "test" or args.evaluate_test
    }
    base_validation = v1.evaluate(
        base_scores["validation"],
        decisions["validation"],
        arrays,
        count_predictions["validation"],
    )
    print(
        json.dumps({
            "base_validation_exact": base_validation["nonforced_semantic_exact"],
            "base_main_top1": base_validation["main_single_choice_semantic_top1"],
            "fit_seconds": time.perf_counter() - started,
        }),
        flush=True,
    )

    type_varying = v1._varying_columns(arrays["count_features"], type_train)
    type_names = [count_names[index] for index in type_varying]
    type_matrix = np.ascontiguousarray(
        arrays["count_features"][:, type_varying]
    )
    experiments: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_model: lgb.LGBMClassifier | None = None
    for config_name, config in CONFIGS.items():
        fit_started = time.perf_counter()
        model = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=args.type_trees,
            learning_rate=0.03,
            num_leaves=config["num_leaves"],
            min_child_samples=config["min_child_samples"],
            max_depth=-1,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.82,
            reg_alpha=0.25,
            reg_lambda=config["reg_lambda"],
            random_state=55137818,
            n_jobs=20,
            verbosity=-1,
        )
        model.fit(
            type_matrix[type_train],
            teacher_types[type_train],
            sample_weight=v1._episode_recency(
                arrays["episode_ids"][type_train], 0.40, 2.0
            ),
            feature_name=type_names,
            categorical_feature=v1._categorical_columns(type_names),
        )
        fit_seconds = time.perf_counter() - fit_started
        for trees in range(args.tree_step, args.type_trees + 1, args.tree_step):
            probabilities = _probability_table(
                model, type_matrix, decisions["validation"], trees
            )
            for confidence in (0.0, 0.35, 0.45, 0.55, 0.65, 0.75):
                for alpha in (0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0):
                    scores, diagnostics = _blend(
                        base_scores["validation"], arrays, names,
                        decisions["validation"], probabilities,
                        alpha, confidence,
                        teacher_types,
                    )
                    metrics = v1.evaluate(
                        scores, decisions["validation"], arrays,
                        count_predictions["validation"],
                    )
                    row = {
                        "config": config_name,
                        "trees": trees,
                        "alpha": alpha,
                        "confidence": confidence,
                        "validation_nonforced_semantic_exact": metrics[
                            "nonforced_semantic_exact"
                        ],
                        "validation_main_top1": metrics[
                            "main_single_choice_semantic_top1"
                        ],
                        **diagnostics,
                    }
                    if best is None or (
                        row["validation_nonforced_semantic_exact"],
                        row["validation_main_top1"],
                        -row["trees"],
                    ) > (
                        best["validation_nonforced_semantic_exact"],
                        best["validation_main_top1"],
                        -best["trees"],
                    ):
                        best = row
                        best_model = model
        config_best = max(
            (
                row for row in ([best] if best and best["config"] == config_name else [])
            ),
            key=lambda row: row["validation_nonforced_semantic_exact"],
            default=None,
        )
        experiments.append({
            "config": config_name,
            "fit_seconds": fit_seconds,
            "best_if_global_at_completion": config_best,
        })
        print(json.dumps({"config": config_name, "global_best": best}), flush=True)

    assert best is not None and best_model is not None
    report: dict[str, Any] = {
        "cache": str(args.cache.resolve()),
        "test_read": bool(args.evaluate_test),
        "train_type_decisions": int(len(type_train)),
        "base_feature_count": int(len(selected_names)),
        "type_feature_count": int(len(type_names)),
        "base_validation": base_validation,
        "experiments": experiments,
        "selected_on_validation": best,
    }
    if args.evaluate_test:
        test_probabilities = _probability_table(
            best_model,
            type_matrix,
            decisions["test"],
            int(best["trees"]),
        )
        test_scores, diagnostics = _blend(
            base_scores["test"], arrays, names, decisions["test"],
            test_probabilities, float(best["alpha"]),
            float(best["confidence"]),
            teacher_types,
        )
        report["test"] = v1.evaluate(
            test_scores, decisions["test"], arrays,
            count_predictions["test"],
        )
        report["test_type_diagnostics"] = diagnostics
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": best, "test_read": args.evaluate_test}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
