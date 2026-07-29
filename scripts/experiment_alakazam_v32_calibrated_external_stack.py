"""Calibrate a higher-ranked external selector on early target validation."""

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
import experiment_alakazam_v32_external_stack as external  # noqa: E402
import experiment_alakazam_v32_score_stack as stack  # noqa: E402
import train_alakazam_v31_teacher as teacher  # noqa: E402


def _select(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    decisions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    return teacher._select_decisions(
        {
            "features": features,
            "labels": labels,
            "weights": weights,
            "groups": groups,
        },
        decisions,
    )


def _fit_final(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    names: list[str],
    config: dict[str, Any],
) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=int(config["best_iteration"]),
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
        x,
        y,
        group=groups.tolist(),
        sample_weight=weights,
        feature_name=names,
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("external_scores", type=Path)
    parser.add_argument("target_scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    blend = json.loads(args.blend_report.read_text(encoding="utf-8"))
    model_names = list(blend["model_order"])
    blend_weights = np.asarray(blend["selected_weights"], dtype=np.float32)
    with np.load(args.external_scores, allow_pickle=False) as saved:
        external_sets = [
            saved[f"external_{name}"] for name in model_names
        ]
        external_labels = saved["external_labels"]
        external_weights = saved["external_weights"]
        external_groups = saved["external_groups"]
    with np.load(args.target_scores, allow_pickle=False) as saved:
        validation_sets = [
            saved[f"validation_{name}"] for name in model_names
        ]
        test_sets = [saved[f"test_{name}"] for name in model_names]
        validation_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]
        validation_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]
        base_validation = external._base_scores(
            saved,
            "validation",
            model_names,
            blend_weights,
        )
        base_test = external._base_scores(
            saved,
            "test",
            model_names,
            blend_weights,
        )

    external_x, feature_names = stack._features(
        external_sets,
        external_groups,
    )
    validation_x, _ = stack._features(
        validation_sets,
        validation_groups,
    )
    test_x, _ = stack._features(test_sets, test_groups)
    validation_count = len(validation_groups)
    cut = max(1, int(validation_count * 0.60))
    early = np.arange(cut, dtype=np.int64)
    late = np.arange(cut, validation_count, dtype=np.int64)
    validation_weights = np.ones(
        len(validation_labels),
        dtype=np.float32,
    )
    early_x, early_y, early_w, early_groups = _select(
        validation_x,
        validation_labels,
        validation_weights,
        validation_groups,
        early,
    )
    late_x, late_y, _, late_groups = _select(
        validation_x,
        validation_labels,
        validation_weights,
        validation_groups,
        late,
    )
    base_arrays = {
        "features": base_validation[:, None],
        "labels": validation_labels,
        "weights": validation_weights,
        "groups": validation_groups,
    }
    late_base_x, _, _, _ = teacher._select_decisions(
        base_arrays,
        late,
    )
    late_base = late_base_x[:, 0]
    train_x = np.concatenate((external_x, early_x))
    train_y = np.concatenate((external_labels, early_y))
    train_w = np.concatenate((external_weights, early_w))
    train_groups = np.concatenate((
        external_groups,
        np.asarray(early_groups),
    ))
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
    for config in configs:
        model = external._fit(
            train_x,
            train_y,
            train_w,
            train_groups,
            late_x,
            late_y,
            np.asarray(late_groups),
            feature_names,
            config,
        )
        scores = ensemble._normalize(
            model.predict(late_x),
            late_groups,
        )
        best = (
            ensemble._accuracy(
                late_base,
                late_y,
                late_groups,
            ),
            0.0,
        )
        for alpha in np.arange(0.05, 2.01, 0.05):
            accuracy = ensemble._accuracy(
                late_base + float(alpha) * scores,
                late_y,
                late_groups,
            )
            best = max(best, (accuracy, -float(alpha)))
        row = {
            **config,
            "best_iteration": int(model.best_iteration_ or 900),
            "late_selector_top1": ensemble._accuracy(
                scores,
                late_y,
                late_groups,
            ),
            "selected_alpha": -best[1],
            "late_blend_top1": best[0],
        }
        experiments.append(row)
        print(json.dumps(row), flush=True)

    selected = max(
        experiments,
        key=lambda row: (
            row["late_blend_top1"],
            -row["selected_alpha"],
            -row["leaves"],
        ),
    )
    all_train_x = np.concatenate((external_x, validation_x))
    all_train_y = np.concatenate((external_labels, validation_labels))
    all_train_w = np.concatenate((
        external_weights,
        validation_weights,
    ))
    all_train_groups = np.concatenate((
        external_groups,
        validation_groups,
    ))
    final = _fit_final(
        all_train_x,
        all_train_y,
        all_train_w,
        all_train_groups,
        feature_names,
        selected,
    )
    test_stack = ensemble._normalize(
        final.predict(test_x),
        test_groups.tolist(),
    )
    strict_test = ensemble._accuracy(
        base_test + float(selected["selected_alpha"]) * test_stack,
        test_labels,
        test_groups.tolist(),
    )
    output = {
        "external_teacher": "Majkel1337 rank 2",
        "target_teacher": "Yushin Ito rank 3",
        "external_decisions": len(external_groups),
        "early_target_calibration_decisions": len(early),
        "late_target_selection_decisions": len(late),
        "experiments": experiments,
        "selected_on_late_validation": selected,
        "strict_test_top1": strict_test,
        "base_test_top1": blend["test_top1"],
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
