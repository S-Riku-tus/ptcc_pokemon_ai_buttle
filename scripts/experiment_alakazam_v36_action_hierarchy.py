"""Learn a decision-level action prior on top of the frozen v34 ranker.

The v34/v35 ranker scores candidates independently.  Its strict holdout
errors are concentrated in choosing *which family* should happen next
(trainer, energy, attack, ...), even though the exact teacher candidate is in
the ranker's Top-3 98.5% of the time.  This experiment gives the policy one
view of the complete candidate set before it chooses a family.

Stage-one training scores are out-of-fold.  Hyperparameters and the blend
strength are selected on validation only; the chronological test block is
evaluated once after selection.
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

from scripts.experiment_alakazam_v35_residual import load_cache  # noqa: E402
from scripts.train_alakazam_v33_teacher import (  # noqa: E402
    ACTION_TYPES,
    ranges,
    rows_for,
)
from scripts.train_alakazam_v34_teacher import recency_multiplier  # noqa: E402


def stage1_summary(
    features: np.ndarray,
    rows: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
    action_column: int,
) -> np.ndarray:
    """Candidate-set summary reproducible from scores available at runtime."""
    starts, ends = ranges(groups)
    n_actions = len(ACTION_TYPES)
    # Per action: offered count, maximum score, score probability mass and
    # gap from the decision-wide maximum.  The tail contains global margin,
    # entropy and candidate count.
    out = np.zeros((len(groups), n_actions * 4 + 4), dtype=np.float32)
    for decision, (a, b) in enumerate(zip(starts, ends)):
        absolute = rows[a:b]
        block = scores[a:b].astype(np.float64)
        actions = features[absolute, action_column].astype(np.int64)
        top = float(block.max())
        exp = np.exp(np.clip(block - top, -50.0, 0.0))
        prob = exp / max(float(exp.sum()), 1e-12)
        for action in np.unique(actions):
            if not 0 <= int(action) < n_actions:
                continue
            mask = actions == action
            maximum = float(block[mask].max())
            offset = int(action) * 4
            out[decision, offset:offset + 4] = (
                int(mask.sum()), maximum, float(prob[mask].sum()), maximum - top
            )
        order = np.sort(block)[::-1]
        margin = float(order[0] - order[1]) if len(order) > 1 else 50.0
        entropy = float(-(prob * np.log(np.maximum(prob, 1e-12))).sum())
        out[decision, -4:] = (top, margin, entropy, len(block))
    return out


def decision_matrix(
    features: np.ndarray,
    names: list[str],
    rows: np.ndarray,
    groups: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """State features plus a compact view of every offered action family."""
    starts, _ = ranges(groups)
    first_candidate = names.index("option_type")
    # Everything before option_type is observation/state context and is
    # candidate invariant by construction.  Include the reproducible turn
    # position, which is also invariant within a decision.
    columns = list(range(first_candidate))
    if "turn_decision_index" in names:
        columns.append(names.index("turn_decision_index"))
    state = features[rows[starts]][:, columns]
    action_column = names.index("action_type")
    summary = stage1_summary(
        features, rows, groups, scores, action_column
    )
    summary_names = [
        f"set_{ACTION_TYPES[action]}_{suffix}"
        for action in range(len(ACTION_TYPES))
        for suffix in ("count", "max_score", "prob_mass", "gap_to_top")
    ] + ["set_top_score", "set_margin", "set_entropy", "set_candidates"]
    return (
        np.ascontiguousarray(np.hstack([state, summary]), dtype=np.float32),
        [names[column] for column in columns] + summary_names,
    )


def score_policy(
    features: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    groups: np.ndarray,
    stage1: np.ndarray,
    probabilities: np.ndarray,
    action_column: int,
    alpha: float,
) -> dict[str, Any]:
    starts, ends = ranges(groups)
    correct = 0
    action_correct = 0
    base_correct = 0
    teacher_actions: list[int] = []
    predicted_actions: list[int] = []
    for decision, (a, b) in enumerate(zip(starts, ends)):
        absolute = rows[a:b]
        block = stage1[a:b].astype(np.float64)
        actions = features[absolute, action_column].astype(np.int64)
        target_local = int(np.flatnonzero(labels[absolute] == 1)[0])
        teacher_action = int(actions[target_local])
        base_local = int(np.argmax(block))
        base_correct += int(base_local == target_local)

        # Decision-local normalisation keeps alpha stable across boards.
        scale = max(float(block.std()), 1e-5)
        z = (block - float(block.mean())) / scale
        prior = np.log(np.maximum(probabilities[decision, actions], 1e-8))
        picked = int(np.argmax(z + alpha * prior))
        predicted_action = int(actions[picked])
        correct += int(picked == target_local)
        action_correct += int(predicted_action == teacher_action)
        teacher_actions.append(teacher_action)
        predicted_actions.append(predicted_action)

    n = max(len(groups), 1)
    by_action = {}
    teacher_array = np.asarray(teacher_actions)
    predicted_array = np.asarray(predicted_actions)
    for action, name in enumerate(ACTION_TYPES):
        mask = teacher_array == action
        if np.any(mask):
            by_action[name] = {
                "count": int(mask.sum()),
                "action_accuracy": float(
                    np.mean(predicted_array[mask] == teacher_array[mask])
                ),
            }
    return {
        "decisions": int(len(groups)),
        "top1": correct / n,
        "base_top1": base_correct / n,
        "action_accuracy": action_correct / n,
        "by_teacher_action": by_action,
    }


def full_probabilities(model: lgb.LGBMClassifier, matrix: np.ndarray) -> np.ndarray:
    compact = model.predict_proba(matrix)
    out = np.full(
        (len(matrix), len(ACTION_TYPES)), 1e-8, dtype=np.float32
    )
    for column, action in enumerate(model.classes_):
        out[:, int(action)] = compact[:, column]
    out /= out.sum(axis=1, keepdims=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("stage1_scores", type=Path)
    parser.add_argument("oof_scores", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--episode-fraction", type=float, default=0.875)
    args = parser.parse_args()

    cache = load_cache(args.cache)
    features, labels, groups = (
        cache["features"], cache["labels"], cache["groups"]
    )
    names, episodes = cache["names"], cache["episode_ids"]
    decisions = {
        split: np.flatnonzero(cache["splits"] == split)
        for split in ("train", "validation", "test")
    }
    train_episodes = np.unique(episodes[decisions["train"]])
    kept_episodes = train_episodes[-max(
        1, int(round(len(train_episodes) * args.episode_fraction))
    ):]
    fit_decisions = decisions["train"][
        np.isin(episodes[decisions["train"]], kept_episodes)
    ]
    block_decisions = {
        "train": fit_decisions,
        "validation": decisions["validation"],
        "test": decisions["test"],
    }
    block_rows = {
        split: rows_for(groups, value)
        for split, value in block_decisions.items()
    }
    block_groups = {
        split: groups[value].astype(np.int64)
        for split, value in block_decisions.items()
    }
    with np.load(args.oof_scores, allow_pickle=False) as stored:
        train_scores = stored["scores"]
    with np.load(args.stage1_scores, allow_pickle=False) as stored:
        scores = {
            "train": train_scores,
            "validation": stored["validation"],
            "test": stored["test"],
        }
    for split in scores:
        if len(scores[split]) != len(block_rows[split]):
            raise RuntimeError(f"{split} stage-one score length mismatch")

    matrices = {}
    feature_names = None
    for split in ("train", "validation", "test"):
        matrices[split], current_names = decision_matrix(
            features, names, block_rows[split], block_groups[split],
            scores[split],
        )
        if feature_names is None:
            feature_names = current_names
        elif current_names != feature_names:
            raise RuntimeError("decision feature schema drift")

    targets = cache["action_types"][fit_decisions].astype(np.int64)
    decision_weights = recency_multiplier(
        episodes[fit_decisions], floor=0.25, power=2.0
    )
    validation_targets = cache["action_types"][
        decisions["validation"]
    ].astype(np.int64)

    configurations = (
        {"num_leaves": 31, "min_child_samples": 50, "max_depth": 10},
        {"num_leaves": 63, "min_child_samples": 40, "max_depth": 12},
    )
    alpha_grid = [
        0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0,
        1.5, 2.0,
    ]
    validation_runs = []
    fitted = []
    for config_index, config in enumerate(configurations):
        started = time.perf_counter()
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(ACTION_TYPES),
            n_estimators=250,
            learning_rate=0.05,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=3600 + config_index,
            n_jobs=20,
            verbosity=-1,
            **config,
        )
        model.fit(
            matrices["train"], targets,
            sample_weight=decision_weights,
            feature_name=feature_names,
            eval_set=[(matrices["validation"], validation_targets)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        probabilities = full_probabilities(model, matrices["validation"])
        for alpha in alpha_grid:
            metrics = score_policy(
                features, labels, block_rows["validation"],
                block_groups["validation"], scores["validation"],
                probabilities, names.index("action_type"), alpha,
            )
            validation_runs.append({
                "config": config_index,
                "alpha": alpha,
                "best_iteration": int(model.best_iteration_ or 250),
                "fit_seconds": time.perf_counter() - started,
                **metrics,
            })
        fitted.append(model)
        best_here = max(
            (row for row in validation_runs if row["config"] == config_index),
            key=lambda row: (row["top1"], -row["alpha"]),
        )
        print(json.dumps({"config": config_index, "best": best_here}), flush=True)

    selected = max(
        validation_runs,
        key=lambda row: (row["top1"], -row["alpha"], -row["config"]),
    )
    selected_model = fitted[selected["config"]]
    test_probability = full_probabilities(
        selected_model, matrices["test"]
    )
    test = score_policy(
        features, labels, block_rows["test"], block_groups["test"],
        scores["test"], test_probability, names.index("action_type"),
        selected["alpha"],
    )
    report = {
        "method": "OOF stage-one decision-level action hierarchy",
        "cache": str(args.cache.resolve()),
        "fit_episodes": int(len(kept_episodes)),
        "fit_decisions": int(len(fit_decisions)),
        "decision_features": int(matrices["train"].shape[1]),
        "selection_rule": "maximum validation strict Top-1; test scored once",
        "validation_runs": validation_runs,
        "selected": selected,
        "test": test,
        "target_top1": 0.90,
        "target_met": bool(test["top1"] > 0.90),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected": selected, "test": test}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
