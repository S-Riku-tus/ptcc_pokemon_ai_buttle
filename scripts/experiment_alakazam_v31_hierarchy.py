"""Evaluate a leakage-free hierarchical v31 policy on a cached teacher corpus.

The v30 ranker asks one model to solve two different problems:

1. which action family should happen next (attack, trainer, energy, ...);
2. which card or target is best inside that family.

This experiment trains a decision-level action-family classifier and uses a
candidate ranker only after the family has been selected.  All fitting uses
the frozen chronological train split; validation and test remain untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_alakazam_v31_teacher as teacher  # noqa: E402


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    return starts, ends


def _decision_features(
    features: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build one row per decision using only candidate-invariant values."""
    starts, _ = _ranges(groups)
    first_candidate = feature_names.index("option_type")
    columns = list(range(first_candidate))
    extra_names = {
        "fallback_action_type",
        "fallback_card_id",
        "fallback_legacy_agree",
        "v29_deterministic_agree",
    }
    columns.extend(
        index
        for index, name in enumerate(feature_names)
        if (
            name.startswith("offered_")
            or name in extra_names
        )
        and index not in columns
    )
    return features[starts][:, columns], [feature_names[index] for index in columns]


def _classifier(
    n_estimators: int,
    *,
    num_leaves: int,
    min_child_samples: int,
) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(teacher.ACTION_TYPES),
        n_estimators=n_estimators,
        learning_rate=0.025,
        num_leaves=num_leaves,
        max_depth=-1,
        min_child_samples=min_child_samples,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.15,
        reg_lambda=1.0,
        random_state=741,
        n_jobs=4,
        verbosity=-1,
    )


def _fit_action_classifier(
    x: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    feature_names: list[str],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    num_leaves: int,
    min_child_samples: int,
) -> lgb.LGBMClassifier:
    categorical = [
        index
        for index, name in enumerate(feature_names)
        if name in teacher.BASE_CATEGORICAL or name.endswith("_id")
    ]
    model = _classifier(
        1600,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
    )
    model.fit(
        x[train_indices],
        labels[train_indices],
        sample_weight=weights[train_indices],
        feature_name=feature_names,
        categorical_feature=categorical,
        eval_set=[(x[validation_indices], labels[validation_indices])],
        callbacks=[lgb.early_stopping(70, verbose=False)],
    )
    return model


def _full_probabilities(
    model: lgb.LGBMClassifier,
    x: np.ndarray,
) -> np.ndarray:
    """Map sklearn probability columns back to the fixed action enum.

    No teacher selects the generic ``other`` family in this corpus, so
    ``classes_`` is not contiguous. Treating probability-column positions as
    enum values silently shifted retreat/trainer/xerosic in the first probe.
    """
    compact = model.predict_proba(x)
    full = np.zeros((len(x), len(teacher.ACTION_TYPES)), dtype=np.float32)
    for column, action in enumerate(model.classes_):
        full[:, int(action)] = compact[:, column]
    return full


def _evaluate(
    decision_indices: np.ndarray,
    action_probabilities: np.ndarray,
    ranker_scores: np.ndarray,
    arrays: dict[str, Any],
) -> dict[str, Any]:
    starts, ends = _ranges(arrays["groups"])
    correct_action = 0
    correct_final = 0
    oracle_action = 0
    final_by_action: dict[int, Counter[str]] = {}
    confusion = np.zeros(
        (len(teacher.ACTION_TYPES), len(teacher.ACTION_TYPES)),
        dtype=np.int64,
    )
    for local_index, decision in enumerate(decision_indices):
        start = starts[decision]
        end = ends[decision]
        teacher_action = int(arrays["teacher_action_types"][decision])
        predicted_action = int(np.argmax(action_probabilities[local_index]))
        confusion[teacher_action, predicted_action] += 1
        correct_action += int(predicted_action == teacher_action)

        candidate_actions = arrays["features"][start:end, arrays["action_column"]]
        available_actions = {int(value) for value in candidate_actions}
        ordered_actions = np.argsort(
            -action_probabilities[local_index],
            kind="stable",
        )
        chosen_action = next(
            (
                int(action)
                for action in ordered_actions
                if int(action) in available_actions
            ),
            int(candidate_actions[0]),
        )
        local_candidates = np.flatnonzero(candidate_actions == chosen_action)
        chosen_local = int(
            local_candidates[
                np.argmax(ranker_scores[start:end][local_candidates])
            ]
        )
        label = int(arrays["labels"][start + chosen_local])
        correct_final += label
        oracle_action += int(
            np.any(
                arrays["labels"][start:end][
                    np.flatnonzero(candidate_actions == teacher_action)
                ]
                == 1
            )
        )
        stats = final_by_action.setdefault(teacher_action, Counter())
        stats["count"] += 1
        stats["correct"] += label

    count = len(decision_indices)
    return {
        "decisions": count,
        "action_top1": correct_action / count if count else 0.0,
        "hierarchical_semantic_top1": (
            correct_final / count if count else 0.0
        ),
        "teacher_action_representable": (
            oracle_action / count if count else 0.0
        ),
        "by_teacher_action": {
            teacher.ACTION_TYPES[action]: {
                "count": int(stats["count"]),
                "semantic_top1": (
                    stats["correct"] / stats["count"]
                    if stats["count"]
                    else 0.0
                ),
            }
            for action, stats in sorted(final_by_action.items())
        },
        "action_confusion": {
            teacher.ACTION_TYPES[row]: {
                teacher.ACTION_TYPES[column]: int(confusion[row, column])
                for column in range(len(teacher.ACTION_TYPES))
                if confusion[row, column]
            }
            for row in range(len(teacher.ACTION_TYPES))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        arrays: dict[str, Any] = {
            key: cached[key]
            for key in (
                "features",
                "labels",
                "weights",
                "groups",
                "fallback_correct",
                "teacher_action_types",
                "episode_ids",
                "ranks",
            )
        }
        arrays["splits"] = cached["splits"].astype(str)
        feature_names = cached["feature_names"].astype(str).tolist()

    split_values = np.asarray(arrays["splits"])
    train_indices = np.flatnonzero(split_values == "train")
    validation_indices = np.flatnonzero(split_values == "validation")
    test_indices = np.flatnonzero(split_values == "test")
    starts, _ = _ranges(arrays["groups"])
    decision_weights = arrays["weights"][starts]
    decision_x, decision_feature_names = _decision_features(
        arrays["features"],
        arrays["groups"],
        feature_names,
    )
    arrays["action_column"] = feature_names.index("action_type")

    ranker = teacher._fit(
        arrays,
        feature_names,
        train_indices,
        n_estimators=900,
        validation_indices=validation_indices,
    )
    all_ranker_scores = ranker.predict(arrays["features"]).astype(np.float32)

    experiments = []
    for num_leaves, min_child_samples in (
        (63, 30),
        (127, 25),
        (255, 20),
    ):
        classifier = _fit_action_classifier(
            decision_x,
            arrays["teacher_action_types"],
            decision_weights,
            decision_feature_names,
            train_indices,
            validation_indices,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
        )
        validation_probabilities = _full_probabilities(
            classifier,
            decision_x[validation_indices]
        )
        test_probabilities = _full_probabilities(
            classifier,
            decision_x[test_indices]
        )
        experiments.append({
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "best_iteration": int(classifier.best_iteration_ or 1600),
            "validation": _evaluate(
                validation_indices,
                validation_probabilities,
                all_ranker_scores,
                arrays,
            ),
            "test": _evaluate(
                test_indices,
                test_probabilities,
                all_ranker_scores,
                arrays,
            ),
        })

    report = {
        "cache": str(args.cache.resolve()),
        "features": len(feature_names),
        "decision_features": len(decision_feature_names),
        "split_decisions": {
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        },
        "ranker_best_iteration": int(ranker.best_iteration_ or 900),
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "experiments": [
            {
                "leaves": row["num_leaves"],
                "validation_action": row["validation"]["action_top1"],
                "validation_top1": row["validation"][
                    "hierarchical_semantic_top1"
                ],
                "test_action": row["test"]["action_top1"],
                "test_top1": row["test"]["hierarchical_semantic_top1"],
            }
            for row in experiments
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
