"""Predict the chosen position from the complete ordered legal-option list.

The candidate ranker scores options independently.  This experiment flattens
the bounded (at most 30 semantic candidates) option list so a model can learn
comparisons, stable tie-breaking, and action-order patterns directly.
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

import train_alakazam_v31_teacher as teacher  # noqa: E402


SLOT_FEATURES = (
    "option_type",
    "candidate_option_position",
    "candidate_option_reverse_position",
    "candidate_raw_index",
    "candidate_raw_inplay_index",
    "candidate_raw_player_relative",
    "candidate_serial",
    "candidate_target_serial",
    "candidate_same_action_preceding",
    "candidate_same_card_preceding",
    "candidate_card_id",
    "candidate_attack_id",
    "candidate_area",
    "candidate_inplay_area",
    "candidate_target_id",
    "candidate_target_hp",
    "candidate_target_max_hp",
    "candidate_target_energy",
    "candidate_target_special_energy",
    "candidate_target_appear_this_turn",
    "candidate_hand_cost",
    "candidate_total_draw_count",
    "candidate_net_hand_delta",
    "post_action_hand_count",
    "post_action_powerful_hand_damage",
    "breaks_current_ko_estimate",
    "preserves_current_ko_estimate",
    "attack_lethal_estimate",
    "same_action_option_count",
    "same_card_option_count",
    "action_type",
    "fallback_selected",
    "fallback_policy_score",
    "fallback_policy_rank",
    "legacy_ranker_score",
    "legacy_ranker_rank",
    "legacy_ranker_selected",
    "v29_selected",
    "v29_ranker_score",
    "v29_ranker_rank",
    "v29_ranker_raw_selected",
)


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    return starts, ends


def _flatten(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    starts, ends = _ranges(groups)
    candidate_start = feature_names.index("option_type")
    slot_indices = [
        feature_names.index(name)
        for name in SLOT_FEATURES
        if name in feature_names
    ]
    slot_names = [feature_names[index] for index in slot_indices]
    max_options = int(groups.max())
    width = candidate_start + max_options * len(slot_indices)
    flat = np.full((len(groups), width), -1.0, dtype=np.float32)
    flat[:, :candidate_start] = features[starts, :candidate_start]
    targets = np.empty(len(groups), dtype=np.int16)
    for decision, (start, end) in enumerate(zip(starts, ends)):
        group = features[start:end][:, slot_indices]
        flat[
            decision,
            candidate_start:candidate_start + group.size,
        ] = group.reshape(-1)
        targets[decision] = int(
            np.flatnonzero(labels[start:end] == 1)[0]
        )
    names = feature_names[:candidate_start] + [
        f"slot_{slot}_{name}"
        for slot in range(max_options)
        for name in slot_names
    ]
    return flat, targets, names, max_options


def _fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    names: list[str],
    train: np.ndarray,
    validation: np.ndarray,
    max_options: int,
) -> lgb.LGBMClassifier:
    categorical = [
        index
        for index, name in enumerate(names)
        if (
            name.endswith("_id")
            or name.endswith("::action_type")
            or name.endswith("::option_type")
            or name == "action_type"
            or name == "option_type"
            or name.endswith("_action_type")
            or name.endswith("_option_type")
        )
    ]
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=max_options,
        n_estimators=650,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=35,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.2,
        random_state=741,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(
        x[train],
        y[train],
        sample_weight=weights[train],
        feature_name=names,
        categorical_feature=categorical,
        eval_set=[(x[validation], y[validation])],
        callbacks=[lgb.early_stopping(45, verbose=False)],
    )
    return model


def _full_probabilities(
    model: lgb.LGBMClassifier,
    x: np.ndarray,
    max_options: int,
) -> np.ndarray:
    compact = model.predict_proba(x)
    full = np.zeros((len(x), max_options), dtype=np.float32)
    for column, value in enumerate(model.classes_):
        full[:, int(value)] = compact[:, column]
    return full


def _ranker_local_scores(
    base_scores: np.ndarray,
    groups: np.ndarray,
    decision_indices: np.ndarray,
    max_options: int,
) -> np.ndarray:
    starts, ends = _ranges(groups)
    out = np.full((len(decision_indices), max_options), -20.0, dtype=np.float32)
    for local, decision in enumerate(decision_indices):
        scores = base_scores[starts[decision]:ends[decision]]
        scores = (
            (scores - float(scores.mean()))
            / max(float(scores.std()), 1e-5)
        )
        out[local, :len(scores)] = scores
    return out


def _accuracy(
    probabilities: np.ndarray,
    ranker_scores: np.ndarray,
    targets: np.ndarray,
    counts: np.ndarray,
    alpha: float,
) -> float:
    correct = 0
    for index, count in enumerate(counts):
        scores = (
            np.log(np.maximum(probabilities[index, :count], 1e-12))
            + alpha * ranker_scores[index, :count]
        )
        correct += int(int(np.argmax(scores)) == int(targets[index]))
    return correct / len(targets)


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
            )
        }
        splits = cached["splits"].astype(str)
        feature_names = cached["feature_names"].astype(str).tolist()
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    starts, _ = _ranges(arrays["groups"])
    decision_weights = arrays["weights"][starts]
    x, targets, names, max_options = _flatten(
        arrays["features"],
        arrays["labels"],
        arrays["groups"],
        feature_names,
    )
    model = _fit(
        x,
        targets,
        decision_weights,
        names,
        train,
        validation,
        max_options,
    )
    ranker = teacher._fit(
        arrays,
        feature_names,
        train,
        n_estimators=900,
        validation_indices=validation,
    )
    base_scores = ranker.predict(arrays["features"]).astype(np.float32)
    validation_probabilities = _full_probabilities(
        model, x[validation], max_options
    )
    test_probabilities = _full_probabilities(
        model, x[test], max_options
    )
    validation_ranker = _ranker_local_scores(
        base_scores,
        arrays["groups"],
        validation,
        max_options,
    )
    test_ranker = _ranker_local_scores(
        base_scores,
        arrays["groups"],
        test,
        max_options,
    )
    grid = [
        {
            "alpha": float(alpha),
            "top1": _accuracy(
                validation_probabilities,
                validation_ranker,
                targets[validation],
                arrays["groups"][validation],
                float(alpha),
            ),
        }
        for alpha in np.arange(0.0, 3.01, 0.10)
    ]
    best = max(grid, key=lambda row: row["top1"])
    report = {
        "cache": str(args.cache.resolve()),
        "flat_features": len(names),
        "slot_features": len(SLOT_FEATURES),
        "max_options": max_options,
        "best_iteration": int(model.best_iteration_ or 650),
        "ranker_best_iteration": int(ranker.best_iteration_ or 900),
        "selected_alpha": best["alpha"],
        "validation_top1": best["top1"],
        "test_top1": _accuracy(
            test_probabilities,
            test_ranker,
            targets[test],
            arrays["groups"][test],
            float(best["alpha"]),
        ),
        "validation_grid": grid,
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
