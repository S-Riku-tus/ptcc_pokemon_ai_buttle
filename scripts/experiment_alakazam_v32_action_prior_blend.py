"""Blend a chronological action-family classifier into frozen policy scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_hierarchy as hierarchy  # noqa: E402
import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402


def _expanded_log_prior(
    probabilities: np.ndarray,
    action_types: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    starts, ends = ensemble._ranges(groups)
    output = np.empty(int(groups.sum()), dtype=np.float32)
    for decision, (start, end) in enumerate(zip(starts, ends)):
        actions = action_types[start:end].astype(np.int64)
        output[start:end] = np.log(np.clip(
            probabilities[decision, actions],
            1e-6,
            1.0,
        ))
    return ensemble._normalize(output, groups.tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    previous = json.loads(args.blend_report.read_text(encoding="utf-8"))
    model_names = list(previous["model_order"])
    weights = np.asarray(previous["selected_weights"])
    with np.load(args.scores, allow_pickle=False) as saved:
        validation_scores = [
            saved[f"validation_{name}"] for name in model_names
        ]
        test_scores = [saved[f"test_{name}"] for name in model_names]
        validation_labels = saved["validation_labels"]
        validation_groups = saved["validation_groups"]
        test_labels = saved["test_labels"]
        test_groups = saved["test_groups"]
    with np.load(args.cache, allow_pickle=False) as cached:
        arrays: dict[str, Any] = {
            key: cached[key]
            for key in (
                "features",
                "labels",
                "weights",
                "groups",
                "teacher_action_types",
            )
        }
        splits = cached["splits"].astype(str)
        feature_names = cached["feature_names"].astype(str).tolist()
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    starts, _ = ensemble._ranges(arrays["groups"])
    decision_weights = arrays["weights"][starts]
    decision_x, decision_names = hierarchy._decision_features(
        arrays["features"],
        arrays["groups"],
        feature_names,
    )
    action_column = feature_names.index("action_type")
    validation_action_rows, _, _, validation_raw_groups = (
        __import__("train_alakazam_v31_teacher")._select_decisions(
            {
                "features": arrays["features"][:, [action_column]],
                "labels": arrays["labels"],
                "weights": arrays["weights"],
                "groups": arrays["groups"],
            },
            validation,
        )
    )
    test_action_rows, _, _, test_raw_groups = (
        __import__("train_alakazam_v31_teacher")._select_decisions(
            {
                "features": arrays["features"][:, [action_column]],
                "labels": arrays["labels"],
                "weights": arrays["weights"],
                "groups": arrays["groups"],
            },
            test,
        )
    )
    if (
        validation_raw_groups != validation_groups.tolist()
        or test_raw_groups != test_groups.tolist()
    ):
        raise RuntimeError("score/cache group alignment changed")

    base_validation = sum(
        float(weight) * score
        for weight, score in zip(weights, validation_scores)
    )
    base_test = sum(
        float(weight) * score
        for weight, score in zip(weights, test_scores)
    )
    experiments = []
    priors = []
    for leaves, minimum in ((63, 30), (127, 25), (255, 20)):
        classifier = hierarchy._fit_action_classifier(
            decision_x,
            arrays["teacher_action_types"],
            decision_weights,
            decision_names,
            train,
            validation,
            num_leaves=leaves,
            min_child_samples=minimum,
        )
        validation_probabilities = hierarchy._full_probabilities(
            classifier, decision_x[validation]
        )
        test_probabilities = hierarchy._full_probabilities(
            classifier, decision_x[test]
        )
        validation_prior = _expanded_log_prior(
            validation_probabilities,
            validation_action_rows[:, 0],
            validation_groups,
        )
        test_prior = _expanded_log_prior(
            test_probabilities,
            test_action_rows[:, 0],
            test_groups,
        )
        grid = []
        for alpha in np.arange(0.0, 3.001, 0.025):
            top1 = ensemble._accuracy(
                base_validation + float(alpha) * validation_prior,
                validation_labels,
                validation_groups.tolist(),
            )
            grid.append((top1, float(alpha)))
        validation_top1, alpha = max(grid)
        test_top1 = ensemble._accuracy(
            base_test + alpha * test_prior,
            test_labels,
            test_groups.tolist(),
        )
        row = {
            "leaves": leaves,
            "minimum": minimum,
            "best_iteration": int(classifier.best_iteration_ or 1600),
            "alpha": alpha,
            "validation_top1": validation_top1,
            "test_top1": test_top1,
        }
        experiments.append(row)
        priors.append((validation_prior, test_prior))
        print(json.dumps(row), flush=True)
    selected_index = max(
        range(len(experiments)),
        key=lambda index: experiments[index]["validation_top1"],
    )
    selected = experiments[selected_index]
    report = {
        "model_order": model_names,
        "base_validation_top1": ensemble._accuracy(
            base_validation,
            validation_labels,
            validation_groups.tolist(),
        ),
        "base_test_top1": ensemble._accuracy(
            base_test,
            test_labels,
            test_groups.tolist(),
        ),
        "experiments": experiments,
        "selected": selected,
        "target_top1": 0.9,
        "target_met": selected["test_top1"] >= 0.9,
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
