"""Train a six-policy selector on a separate higher-ranked teacher corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import experiment_alakazam_v32_score_stack as stack  # noqa: E402


def _fit(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray,
    train_groups: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_groups: np.ndarray,
    names: list[str],
    config: dict[str, Any],
) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=900,
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["leaves"]),
        max_depth=int(config["max_depth"]),
        min_child_samples=int(config["minimum"]),
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=float(config["column_fraction"]),
        reg_alpha=float(config["reg_alpha"]),
        reg_lambda=float(config["reg_lambda"]),
        random_state=3203,
        n_jobs=6,
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        group=train_groups.tolist(),
        sample_weight=train_w,
        feature_name=names,
        eval_set=[(validation_x, validation_y)],
        eval_group=[validation_groups.tolist()],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    return model


def _base_scores(
    saved: Any,
    split: str,
    names: list[str],
    weights: np.ndarray,
) -> np.ndarray:
    return sum(
        float(weight) * saved[f"{split}_{name}"]
        for name, weight in zip(names, weights)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("external_scores", type=Path)
    parser.add_argument("target_scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.blend_report.read_text(encoding="utf-8"))
    model_names = list(report["model_order"])
    blend_weights = np.asarray(
        report["selected_weights"],
        dtype=np.float32,
    )
    with np.load(args.external_scores, allow_pickle=False) as saved:
        external_score_sets = [
            saved[f"external_{name}"] for name in model_names
        ]
        external_labels = saved["external_labels"]
        external_weights = saved["external_weights"]
        external_groups = saved["external_groups"]
    with np.load(args.target_scores, allow_pickle=False) as saved:
        validation_score_sets = [
            saved[f"validation_{name}"] for name in model_names
        ]
        test_score_sets = [
            saved[f"test_{name}"] for name in model_names
        ]
        validation_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]
        validation_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]
        base_validation = _base_scores(
            saved,
            "validation",
            model_names,
            blend_weights,
        )
        base_test = _base_scores(
            saved,
            "test",
            model_names,
            blend_weights,
        )

    external_x, feature_names = stack._features(
        external_score_sets,
        external_groups,
    )
    validation_x, _ = stack._features(
        validation_score_sets,
        validation_groups,
    )
    test_x, _ = stack._features(test_score_sets, test_groups)
    configs = [
        {
            "leaves": 3,
            "max_depth": 2,
            "minimum": 150,
            "learning_rate": 0.02,
            "column_fraction": 0.75,
            "reg_alpha": 1.0,
            "reg_lambda": 8.0,
        },
        {
            "leaves": 7,
            "max_depth": 3,
            "minimum": 120,
            "learning_rate": 0.02,
            "column_fraction": 0.80,
            "reg_alpha": 1.0,
            "reg_lambda": 8.0,
        },
        {
            "leaves": 15,
            "max_depth": 4,
            "minimum": 100,
            "learning_rate": 0.015,
            "column_fraction": 0.80,
            "reg_alpha": 1.5,
            "reg_lambda": 10.0,
        },
        {
            "leaves": 31,
            "max_depth": 5,
            "minimum": 100,
            "learning_rate": 0.012,
            "column_fraction": 0.75,
            "reg_alpha": 2.0,
            "reg_lambda": 12.0,
        },
    ]
    experiments = []
    models = []
    validation_predictions = []
    for config in configs:
        model = _fit(
            external_x,
            external_labels,
            external_weights,
            external_groups,
            validation_x,
            validation_labels,
            validation_groups,
            feature_names,
            config,
        )
        scores = ensemble._normalize(
            model.predict(validation_x),
            validation_groups.tolist(),
        )
        row = {
            **config,
            "best_iteration": int(model.best_iteration_ or 900),
            "validation_top1": ensemble._accuracy(
                scores,
                validation_labels,
                validation_groups.tolist(),
            ),
        }
        experiments.append(row)
        models.append(model)
        validation_predictions.append(scores)
        print(json.dumps(row), flush=True)

    selected_index = max(
        range(len(experiments)),
        key=lambda index: (
            experiments[index]["validation_top1"],
            -experiments[index]["leaves"],
        ),
    )
    selected = experiments[selected_index]
    validation_stack = validation_predictions[selected_index]
    test_stack = ensemble._normalize(
        models[selected_index].predict(test_x),
        test_groups.tolist(),
    )
    best_blend = (
        ensemble._accuracy(
            base_validation,
            validation_labels,
            validation_groups.tolist(),
        ),
        0.0,
    )
    for alpha in np.arange(0.05, 2.01, 0.05):
        accuracy = ensemble._accuracy(
            base_validation + float(alpha) * validation_stack,
            validation_labels,
            validation_groups.tolist(),
        )
        best_blend = max(best_blend, (accuracy, -float(alpha)))
    alpha = -best_blend[1]
    strict_test = ensemble._accuracy(
        base_test + alpha * test_stack,
        test_labels,
        test_groups.tolist(),
    )
    output = {
        "teacher_for_selector": "Majkel1337 rank 2",
        "target_teacher": "Yushin Ito rank 3",
        "selector_train_decisions": len(external_groups),
        "selector_features": len(feature_names),
        "experiments": experiments,
        "selected_on_validation": selected,
        "selected_blend_alpha": alpha,
        "blend_validation_top1": best_blend[0],
        "strict_test_top1": strict_test,
        "base_validation_top1": report["validation_top1"],
        "base_test_top1": report["test_top1"],
        "target_top1": 0.9,
        "target_met": strict_test >= 0.9,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
