"""Distill the six-model v32 ensemble into a standard-library tree student."""

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
import train_alakazam_v31_teacher as teacher  # noqa: E402


def _teacher_scores(
    saved: Any,
    split: str,
    names: list[str],
    weights: np.ndarray,
) -> np.ndarray:
    return sum(
        float(weight) * saved[f"{split}_{name}"]
        for name, weight in zip(names, weights)
    ).astype(np.float32)


def _fit(
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    names: list[str],
    config: dict[str, Any],
    *,
    validation: tuple[np.ndarray, np.ndarray] | None,
    iterations: int,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression_l2",
        metric="l2",
        n_estimators=iterations,
        learning_rate=config["learning_rate"],
        num_leaves=config["leaves"],
        max_depth=config["max_depth"],
        min_child_samples=config["minimum"],
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=config["column_fraction"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        random_state=743,
        n_jobs=6,
        verbosity=-1,
    )
    kwargs: dict[str, Any] = {}
    if validation is not None:
        kwargs.update({
            "eval_set": [validation],
            "callbacks": [lgb.early_stopping(60, verbose=False)],
        })
    model.fit(
        x,
        target,
        sample_weight=weights,
        feature_name=names,
        **kwargs,
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    blend = json.loads(
        args.blend_report.read_text(encoding="utf-8")
    )
    model_names = list(blend["model_order"])
    blend_weights = np.asarray(blend["selected_weights"])
    with np.load(args.scores, allow_pickle=False) as saved:
        validation_target = _teacher_scores(
            saved,
            "validation",
            model_names,
            blend_weights,
        )
        test_target = _teacher_scores(
            saved,
            "test",
            model_names,
            blend_weights,
        )
        validation_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]
        validation_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]
    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        labels = cached["labels"]
        row_weights = cached["weights"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
        names = cached["feature_names"].astype(str).tolist()
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    validation_x, _, validation_w, check_validation_groups = (
        teacher._select_decisions(
            {
                "features": features,
                "labels": labels,
                "weights": row_weights,
                "groups": groups,
            },
            validation,
        )
    )
    test_x, _, _, check_test_groups = teacher._select_decisions(
        {
            "features": features,
            "labels": labels,
            "weights": row_weights,
            "groups": groups,
        },
        test,
    )
    if (
        check_validation_groups != validation_groups.tolist()
        or check_test_groups != test_groups.tolist()
    ):
        raise RuntimeError("Score/cache group alignment changed")

    validation_episodes = episode_ids[validation]
    ordered = np.unique(validation_episodes)
    ordered.sort()
    cut = max(1, int(len(ordered) * 0.60))
    early = set(ordered[:cut].tolist())
    early_decisions = np.flatnonzero(np.asarray([
        episode in early for episode in validation_episodes
    ]))
    late_decisions = np.flatnonzero(np.asarray([
        episode not in early for episode in validation_episodes
    ]))
    local_arrays = {
        "features": validation_x,
        "labels": validation_labels,
        "weights": validation_w,
        "groups": validation_groups,
    }
    early_x, _, early_w, _ = teacher._select_decisions(
        local_arrays,
        early_decisions,
    )
    late_x, late_labels, _, late_groups = teacher._select_decisions(
        local_arrays,
        late_decisions,
    )
    target_arrays = {
        "features": validation_target[:, None],
        "labels": validation_labels,
        "weights": validation_w,
        "groups": validation_groups,
    }
    early_target, _, _, _ = teacher._select_decisions(
        target_arrays,
        early_decisions,
    )
    late_target, _, _, _ = teacher._select_decisions(
        target_arrays,
        late_decisions,
    )
    early_target = early_target[:, 0]
    late_target = late_target[:, 0]

    configs = [
        {
            "leaves": 31,
            "max_depth": 6,
            "minimum": 80,
            "learning_rate": 0.03,
            "column_fraction": 0.65,
            "reg_alpha": 0.5,
            "reg_lambda": 4.0,
        },
        {
            "leaves": 63,
            "max_depth": 8,
            "minimum": 60,
            "learning_rate": 0.025,
            "column_fraction": 0.72,
            "reg_alpha": 0.5,
            "reg_lambda": 4.0,
        },
        {
            "leaves": 127,
            "max_depth": 10,
            "minimum": 50,
            "learning_rate": 0.02,
            "column_fraction": 0.75,
            "reg_alpha": 0.8,
            "reg_lambda": 5.0,
        },
    ]
    experiments = []
    for config in configs:
        model = _fit(
            early_x,
            early_target,
            early_w,
            names,
            config,
            validation=(late_x, late_target),
            iterations=1200,
        )
        scores = model.predict(late_x)
        experiments.append({
            **config,
            "best_iteration": int(model.best_iteration_ or 1200),
            "late_teacher_mse": float(
                np.mean(np.square(scores - late_target))
            ),
            "late_teacher_top1_agreement": ensemble._accuracy(
                scores,
                (
                    np.concatenate([
                        np.eye(1, end - start, int(np.argmax(
                            late_target[start:end]
                        )), dtype=np.int8).ravel()
                        for start, end in zip(
                            *ensemble._ranges(late_groups)
                        )
                    ])
                ),
                late_groups,
            ),
            "late_actual_top1": ensemble._accuracy(
                scores,
                late_labels,
                late_groups,
            ),
        })
        print(json.dumps(experiments[-1]), flush=True)
    selected = max(
        experiments,
        key=lambda row: (
            row["late_teacher_top1_agreement"],
            row["late_actual_top1"],
        ),
    )
    final = _fit(
        validation_x,
        validation_target,
        validation_w,
        names,
        selected,
        validation=None,
        iterations=selected["best_iteration"],
    )
    test_scores = final.predict(test_x)
    teacher_labels = np.zeros_like(test_labels)
    for start, end in zip(*ensemble._ranges(test_groups)):
        teacher_labels[start + int(np.argmax(test_target[start:end]))] = 1
    report = {
        "teacher_models": model_names,
        "teacher_weights": blend_weights.tolist(),
        "meta_train_decisions": len(early_decisions),
        "meta_validation_decisions": len(late_decisions),
        "experiments": experiments,
        "selected": selected,
        "test_teacher_top1_agreement": ensemble._accuracy(
            test_scores,
            teacher_labels,
            test_groups.tolist(),
        ),
        "test_actual_top1": ensemble._accuracy(
            test_scores,
            test_labels,
            test_groups.tolist(),
        ),
        "teacher_test_top1": blend["test_top1"],
        "target_top1": 0.9,
    }
    report["target_met"] = report["test_actual_top1"] >= 0.9
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
