"""Learn which frozen v32 base policy to trust for each decision.

The six base policies are trained only on the chronological training split.
This script derives one row per (decision, policy), trains a LambdaRank gate
on early validation episodes, selects its capacity on later validation
episodes, refits on all validation episodes, and evaluates the untouched test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    starts, ends = _ranges(groups)
    return float(np.mean([
        labels[start + int(np.argmax(scores[start:end]))] == 1
        for start, end in zip(starts, ends)
    ]))


def _tree_importance(model: dict[str, Any]) -> np.ndarray:
    counts = np.zeros(len(model["feature_names"]), dtype=np.float64)
    stack = list(model["trees"])
    while stack:
        node = stack.pop()
        if "v" in node:
            continue
        counts[int(node["f"])] += 1.0
        stack.extend((node["l"], node["r"]))
    return counts


def _selected_raw_columns(
    cache_names: list[str],
    importance_model: dict[str, Any],
    limit: int,
) -> tuple[np.ndarray, list[str]]:
    model_names = importance_model["feature_names"]
    counts = _tree_importance(importance_model)
    by_name = {
        name: float(counts[index])
        for index, name in enumerate(model_names)
    }
    mandatory = {
        "turn",
        "turn_action_count",
        "legal_option_count",
        "option_type",
        "action_type",
        "candidate_card_id",
        "candidate_attack_id",
        "candidate_target_id",
        "candidate_area",
        "candidate_inplay_area",
        "candidate_target_hp",
        "candidate_target_max_hp",
        "candidate_target_energy",
        "candidate_target_special_energy",
        "self_hand_count",
        "self_deck_count",
        "self_prize_count",
        "self_board_count",
        "opp_hand_count",
        "opp_deck_count",
        "opp_prize_count",
        "self_active_id",
        "opp_active_id",
        "has_ready_active_alakazam",
        "current_powerful_hand_damage",
        "fallback_policy_score",
        "legacy_ranker_score",
        "v29_ranker_score",
    }
    ordered = sorted(
        range(len(cache_names)),
        key=lambda index: (
            cache_names[index] not in mandatory,
            -by_name.get(cache_names[index], 0.0),
            index,
        ),
    )
    selected = np.asarray(ordered[:limit], dtype=np.int64)
    return selected, [cache_names[index] for index in selected]


def _semantic_columns(cache_names: list[str]) -> np.ndarray:
    wanted = (
        "option_type",
        "candidate_card_id",
        "candidate_attack_id",
        "candidate_target_id",
        "candidate_target_hp",
        "candidate_target_max_hp",
        "candidate_target_energy",
        "candidate_target_special_energy",
        "candidate_inplay_area",
        "candidate_area",
    )
    return np.asarray([
        cache_names.index(name) for name in wanted if name in cache_names
    ], dtype=np.int64)


def _softmax_peak(values: np.ndarray) -> float:
    shifted = np.clip(values - float(values.max()), -40.0, 0.0)
    probabilities = np.exp(shifted)
    return float(probabilities.max() / max(float(probabilities.sum()), 1e-9))


def _decision_rows(
    score_sets: list[np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    raw: np.ndarray,
    raw_columns: np.ndarray,
    semantic_columns: np.ndarray,
    blend_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    model_count = len(score_sets)
    starts, ends = _ranges(groups)
    rows: list[np.ndarray] = []
    targets: list[int] = []
    selected_actions: list[int] = []
    score_names = [
        "model_id",
        "own_top_score",
        "own_margin",
        "own_softmax_peak",
        "own_range",
        "agreement_count",
        "weighted_vote",
        "unique_action_count",
        "agrees_fixed_blend",
        "chosen_cross_model_mean",
        "chosen_cross_model_min",
        "chosen_cross_model_max",
        "chosen_cross_model_std",
        "chosen_average_rank",
    ]
    for start, end in zip(starts, ends):
        local_scores = np.stack([
            scores[start:end] for scores in score_sets
        ])
        tops = np.argmax(local_scores, axis=1)
        fixed = np.sum(
            blend_weights[:, None] * local_scores,
            axis=0,
        )
        fixed_top = int(np.argmax(fixed))
        keys = [
            tuple(raw[start + int(top), semantic_columns].tolist())
            for top in tops
        ]
        fixed_key = tuple(raw[start + fixed_top, semantic_columns].tolist())
        unique_count = len(set(keys))
        ranks = np.argsort(
            np.argsort(-local_scores, axis=1, kind="stable"),
            axis=1,
            kind="stable",
        )
        for model_index, top_value in enumerate(tops):
            top = int(top_value)
            own = local_scores[model_index]
            ordered = np.sort(own)
            margin = (
                float(ordered[-1] - ordered[-2])
                if len(ordered) > 1
                else 0.0
            )
            key = keys[model_index]
            agreement = np.asarray([
                other == key for other in keys
            ], dtype=np.float32)
            cross = local_scores[:, top]
            derived = np.asarray([
                model_index,
                float(own[top]),
                margin,
                _softmax_peak(own),
                float(own.max() - own.min()),
                float(agreement.sum()),
                float(np.sum(blend_weights * agreement)),
                float(unique_count),
                float(key == fixed_key),
                float(cross.mean()),
                float(cross.min()),
                float(cross.max()),
                float(cross.std()),
                float(ranks[:, top].mean()),
            ], dtype=np.float32)
            rows.append(np.concatenate((
                raw[start + top, raw_columns].astype(np.float32),
                derived,
            )))
            targets.append(int(labels[start + top] == 1))
            selected_actions.append(top)
    feature_names = [
        f"chosen__{index}" for index in range(len(raw_columns))
    ] + score_names
    return (
        np.stack(rows),
        np.asarray(targets, dtype=np.int8),
        feature_names,
        np.asarray(selected_actions, dtype=np.int64),
    )


def _gate_accuracy(
    gate_scores: np.ndarray,
    gate_targets: np.ndarray,
    model_count: int,
) -> float:
    scores = gate_scores.reshape(-1, model_count)
    targets = gate_targets.reshape(-1, model_count)
    choices = np.argmax(scores, axis=1)
    return float(np.mean(
        targets[np.arange(len(choices)), choices] == 1
    ))


def _fit(
    x: np.ndarray,
    y: np.ndarray,
    train_decisions: np.ndarray,
    model_count: int,
    config: dict[str, Any],
    *,
    iterations: int,
    validation_decisions: np.ndarray | None = None,
) -> lgb.LGBMRanker:
    row_indices = (
        train_decisions[:, None] * model_count
        + np.arange(model_count)[None, :]
    ).reshape(-1)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=iterations,
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["leaves"]),
        max_depth=int(config["max_depth"]),
        min_child_samples=int(config["minimum"]),
        colsample_bytree=float(config["column_fraction"]),
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=float(config["reg_alpha"]),
        reg_lambda=float(config["reg_lambda"]),
        random_state=3200,
        n_jobs=6,
        verbosity=-1,
    )
    fit_kwargs: dict[str, Any] = {}
    if validation_decisions is not None:
        validation_rows = (
            validation_decisions[:, None] * model_count
            + np.arange(model_count)[None, :]
        ).reshape(-1)
        fit_kwargs = {
            "eval_set": [(x[validation_rows], y[validation_rows])],
            "eval_group": [[model_count] * len(validation_decisions)],
            "callbacks": [lgb.early_stopping(80, verbose=False)],
        }
    model.fit(
        x[row_indices],
        y[row_indices],
        group=[model_count] * len(train_decisions),
        categorical_feature=["model_id"],
        feature_name=[*[
            f"f{index}" for index in range(x.shape[1] - 14)
        ], *[
            "model_id",
            "own_top_score",
            "own_margin",
            "own_softmax_peak",
            "own_range",
            "agreement_count",
            "weighted_vote",
            "unique_action_count",
            "agrees_fixed_blend",
            "chosen_cross_model_mean",
            "chosen_cross_model_min",
            "chosen_cross_model_max",
            "chosen_cross_model_std",
            "chosen_average_rank",
        ]],
        **fit_kwargs,
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--importance-model", type=Path, required=True)
    parser.add_argument("--feature-limit", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    blend = json.loads(args.blend_report.read_text(encoding="utf-8"))
    model_names = list(blend["model_order"])
    blend_weights = np.asarray(blend["selected_weights"], dtype=np.float32)
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
    with np.load(args.cache, allow_pickle=False) as cached:
        cache_names = cached["feature_names"].astype(str).tolist()
        features = cached["features"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
    importance_model = json.loads(
        args.importance_model.read_text(encoding="utf-8")
    )
    raw_columns, raw_names = _selected_raw_columns(
        cache_names,
        importance_model,
        args.feature_limit,
    )
    semantic_columns = _semantic_columns(cache_names)
    decision_starts, decision_ends = _ranges(groups)
    validation_decisions = np.flatnonzero(splits == "validation")
    test_decisions = np.flatnonzero(splits == "test")
    validation_rows = np.concatenate([
        np.arange(decision_starts[index], decision_ends[index])
        for index in validation_decisions
    ])
    test_rows = np.concatenate([
        np.arange(decision_starts[index], decision_ends[index])
        for index in test_decisions
    ])
    validation_raw = features[validation_rows]
    test_raw = features[test_rows]
    validation_x, validation_y, names, _ = _decision_rows(
        validation_scores,
        validation_labels,
        validation_groups,
        validation_raw,
        raw_columns,
        semantic_columns,
        blend_weights,
    )
    test_x, test_y, _, _ = _decision_rows(
        test_scores,
        test_labels,
        test_groups,
        test_raw,
        raw_columns,
        semantic_columns,
        blend_weights,
    )
    model_count = len(model_names)
    validation_episodes = episode_ids[validation_decisions]
    ordered = np.unique(validation_episodes)
    ordered.sort()
    cut = max(1, int(len(ordered) * 0.60))
    early = set(ordered[:cut].tolist())
    gate_train = np.flatnonzero(np.asarray([
        episode in early for episode in validation_episodes
    ]))
    gate_validation = np.flatnonzero(np.asarray([
        episode not in early for episode in validation_episodes
    ]))
    configs = [
        {
            "leaves": 7,
            "max_depth": 3,
            "minimum": 80,
            "learning_rate": 0.02,
            "column_fraction": 0.65,
            "reg_alpha": 1.0,
            "reg_lambda": 8.0,
        },
        {
            "leaves": 15,
            "max_depth": 4,
            "minimum": 60,
            "learning_rate": 0.02,
            "column_fraction": 0.70,
            "reg_alpha": 1.0,
            "reg_lambda": 8.0,
        },
        {
            "leaves": 31,
            "max_depth": 5,
            "minimum": 50,
            "learning_rate": 0.015,
            "column_fraction": 0.70,
            "reg_alpha": 1.5,
            "reg_lambda": 10.0,
        },
    ]
    experiments = []
    for config in configs:
        gate = _fit(
            validation_x,
            validation_y,
            gate_train,
            model_count,
            config,
            iterations=1200,
            validation_decisions=gate_validation,
        )
        late_rows = (
            gate_validation[:, None] * model_count
            + np.arange(model_count)[None, :]
        ).reshape(-1)
        row = {
            **config,
            "best_iteration": int(gate.best_iteration_ or 1200),
            "late_validation_top1": _gate_accuracy(
                gate.predict(validation_x[late_rows]),
                validation_y[late_rows],
                model_count,
            ),
        }
        experiments.append(row)
        print(json.dumps(row), flush=True)
    selected = max(
        experiments,
        key=lambda row: (
            row["late_validation_top1"],
            -row["leaves"],
        ),
    )
    final = _fit(
        validation_x,
        validation_y,
        np.arange(len(validation_groups), dtype=np.int64),
        model_count,
        selected,
        iterations=selected["best_iteration"],
    )
    test_gate_scores = final.predict(test_x)
    gate_top1 = _gate_accuracy(test_gate_scores, test_y, model_count)
    fixed_test = np.sum(
        blend_weights[:, None] * np.stack(test_scores),
        axis=0,
    )
    oracle_top1 = float(np.mean(
        test_y.reshape(-1, model_count).max(axis=1) == 1
    ))
    report = {
        "model_order": model_names,
        "raw_features": raw_names,
        "gate_features": len(names),
        "gate_train_decisions": int(len(gate_train)),
        "gate_validation_decisions": int(len(gate_validation)),
        "experiments": experiments,
        "selected": selected,
        "test_top1": gate_top1,
        "fixed_blend_reference": _accuracy(
            fixed_test,
            test_labels,
            test_groups,
        ),
        "test_oracle_any_model": oracle_top1,
        "target_top1": 0.9,
        "target_met": gate_top1 >= 0.9,
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
