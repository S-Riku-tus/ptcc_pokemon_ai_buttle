"""Fit a state-conditioned stack over the five v32 policy challengers.

Base models are frozen after training on the chronological training split.
The stack is selected on an early/late division of the next validation
episodes, refitted on all validation episodes, and evaluated once on the
strictly later test episodes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import experiment_alakazam_v32_deepset as deepset  # noqa: E402
import experiment_alakazam_v32_score_stack as score_stack  # noqa: E402
import train_alakazam_v31_teacher as teacher  # noqa: E402


def _fit(
    arrays: dict[str, Any],
    names: list[str],
    train: np.ndarray,
    validation: np.ndarray | None,
    config: dict[str, Any],
    *,
    iterations: int,
) -> lgb.LGBMRanker:
    x, y, weights, groups = teacher._select_decisions(arrays, train)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=iterations,
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["leaves"]),
        min_child_samples=int(config["minimum"]),
        max_depth=int(config.get("max_depth", -1)),
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=float(config["column_fraction"]),
        reg_alpha=float(config["reg_alpha"]),
        reg_lambda=float(config["reg_lambda"]),
        random_state=743,
        n_jobs=6,
        verbosity=-1,
    )
    fit_kwargs: dict[str, Any] = {}
    if validation is not None:
        vx, vy, _, validation_groups = teacher._select_decisions(
            arrays,
            validation,
        )
        fit_kwargs.update({
            "eval_set": [(vx, vy)],
            "eval_group": [validation_groups],
            "callbacks": [lgb.early_stopping(60, verbose=False)],
        })
    model.fit(
        x,
        y,
        group=groups,
        sample_weight=weights,
        feature_name=names,
        **fit_kwargs,
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("schema_cache", type=Path)
    parser.add_argument(
        "--blend-report",
        type=Path,
        help="Use the dynamic model order from an existing blend report.",
    )
    parser.add_argument(
        "--model",
        action="append",
        type=Path,
        required=True,
        help="Compact tree model used to choose stable raw features.",
    )
    parser.add_argument("--feature-limit", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_names = (
        json.loads(args.blend_report.read_text(encoding="utf-8"))[
            "model_order"
        ]
        if args.blend_report is not None
        else ["large", "numeric", "deep", "history", "attention"]
    )
    with np.load(args.scores, allow_pickle=False) as saved:
        validation_scores = [
            saved[f"validation_{name}"] for name in model_names
        ]
        test_scores = [
            saved[f"test_{name}"] for name in model_names
        ]
        validation_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]
        validation_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]

    with np.load(args.schema_cache, allow_pickle=False) as schema:
        schema_names = schema["feature_names"].astype(str).tolist()
    with np.load(args.cache, allow_pickle=False) as cached:
        cache_names = cached["feature_names"].astype(str).tolist()
        columns = [cache_names.index(name) for name in schema_names]
        base_features = cached["features"][:, columns]
        labels = cached["labels"]
        weights = cached["weights"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]

    selected_columns = deepset._select_features(
        schema_names,
        args.model,
        args.feature_limit,
    )
    selected_names = [schema_names[index] for index in selected_columns]
    validation_decisions = np.flatnonzero(splits == "validation")
    test_decisions = np.flatnonzero(splits == "test")
    validation_raw, _, _, raw_validation_groups = teacher._select_decisions(
        {
            "features": base_features[:, selected_columns],
            "labels": labels,
            "weights": weights,
            "groups": groups,
        },
        validation_decisions,
    )
    test_raw, _, _, raw_test_groups = teacher._select_decisions(
        {
            "features": base_features[:, selected_columns],
            "labels": labels,
            "weights": weights,
            "groups": groups,
        },
        test_decisions,
    )
    if (
        raw_validation_groups != validation_groups.tolist()
        or raw_test_groups != test_groups.tolist()
    ):
        raise RuntimeError("Score and raw-feature group alignment changed")

    validation_score_features, score_names = score_stack._features(
        validation_scores,
        validation_groups,
    )
    test_score_features, _ = score_stack._features(
        test_scores,
        test_groups,
    )
    validation_x = np.column_stack((
        validation_raw,
        validation_score_features,
    )).astype(np.float32)
    test_x = np.column_stack((
        test_raw,
        test_score_features,
    )).astype(np.float32)
    names = [f"raw__{name}" for name in selected_names] + score_names

    validation_episodes = episode_ids[validation_decisions]
    ordered_episodes = np.unique(validation_episodes)
    ordered_episodes.sort()
    cut = max(1, int(len(ordered_episodes) * 0.60))
    early_episodes = set(ordered_episodes[:cut].tolist())
    meta_train = np.flatnonzero(np.asarray([
        episode in early_episodes for episode in validation_episodes
    ]))
    meta_validation = np.flatnonzero(np.asarray([
        episode not in early_episodes for episode in validation_episodes
    ]))
    stack_arrays: dict[str, Any] = {
        "features": validation_x,
        "labels": validation_labels,
        "weights": np.ones(len(validation_labels), dtype=np.float32),
        "groups": validation_groups,
    }
    configs = [
        {
            "leaves": 7,
            "minimum": 80,
            "max_depth": 4,
            "learning_rate": 0.025,
            "column_fraction": 0.70,
            "reg_alpha": 0.5,
            "reg_lambda": 4.0,
        },
        {
            "leaves": 15,
            "minimum": 80,
            "max_depth": 5,
            "learning_rate": 0.025,
            "column_fraction": 0.75,
            "reg_alpha": 0.5,
            "reg_lambda": 4.0,
        },
        {
            "leaves": 31,
            "minimum": 60,
            "max_depth": 6,
            "learning_rate": 0.02,
            "column_fraction": 0.75,
            "reg_alpha": 0.8,
            "reg_lambda": 5.0,
        },
        {
            "leaves": 31,
            "minimum": 100,
            "max_depth": 6,
            "learning_rate": 0.015,
            "column_fraction": 0.60,
            "reg_alpha": 1.0,
            "reg_lambda": 8.0,
        },
    ]
    experiments = []
    for config in configs:
        model = _fit(
            stack_arrays,
            names,
            meta_train,
            meta_validation,
            config,
            iterations=1200,
        )
        validation_x_late, validation_y_late, _, groups_late = (
            teacher._select_decisions(
                stack_arrays,
                meta_validation,
            )
        )
        row = {
            **config,
            "best_iteration": int(model.best_iteration_ or 1200),
            "meta_validation_top1": ensemble._accuracy(
                model.predict(validation_x_late),
                validation_y_late,
                groups_late,
            ),
        }
        experiments.append(row)
        print(json.dumps(row), flush=True)
    selected = max(
        experiments,
        key=lambda row: (
            row["meta_validation_top1"],
            -row["leaves"],
        ),
    )
    final = _fit(
        stack_arrays,
        names,
        np.arange(len(validation_groups), dtype=np.int64),
        None,
        selected,
        iterations=selected["best_iteration"],
    )
    test_top1 = ensemble._accuracy(
        final.predict(test_x),
        test_labels,
        test_groups.tolist(),
    )
    reference_weights = (
        np.asarray(json.loads(
            args.blend_report.read_text(encoding="utf-8")
        )["selected_weights"])
        if args.blend_report is not None
        else np.asarray([1.0, 0.55, 0.60, 1.10, 0.0])
    )
    weighted = sum(
        float(weight) * scores
        for weight, scores in zip(reference_weights, test_scores)
    )
    report = {
        "base_models": list(model_names),
        "raw_features": len(selected_names),
        "stack_features": len(names),
        "meta_train_decisions": len(meta_train),
        "meta_validation_decisions": len(meta_validation),
        "experiments": experiments,
        "selected": selected,
        "test_top1": test_top1,
        "weighted_blend_reference": ensemble._accuracy(
            weighted,
            test_labels,
            test_groups.tolist(),
        ),
        "target_top1": 0.9,
        "target_met": test_top1 >= 0.9,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
