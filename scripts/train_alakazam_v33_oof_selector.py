"""Train the leakage-free v33 OOF candidate selector.

The original v32 gate learned only from the 5,038-decision validation split.
This experiment instead creates episode-grouped out-of-fold predictions for
every Yushin training decision, then learns one candidate ranker from:

* six base-policy scores;
* within-decision ranks, margins, and votes; and
* a compact subset of the original board/action features.

Validation selects the selector capacity.  Test is evaluated once after that
selection.  Runtime artifacts are marked enabled only when validation clears
the frozen v32 six-model reference by ``--minimum-improvement``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_alakazam_v31_teacher as teacher  # noqa: E402
from ml.core.distill import compact_booster  # noqa: E402


BASE_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "lambda_standard",
        "objective": "lambdarank",
        "iterations": 314,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 40,
        "colsample_bytree": 0.88,
        "seed": 741,
        "categorical": True,
    },
    {
        "name": "lambda_small_leaf",
        "objective": "lambdarank",
        "iterations": 484,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "min_child_samples": 20,
        "colsample_bytree": 0.95,
        "seed": 19,
        "categorical": True,
    },
    {
        "name": "lambda_large_leaf",
        "objective": "lambdarank",
        "iterations": 499,
        "learning_rate": 0.025,
        "num_leaves": 255,
        "min_child_samples": 55,
        "colsample_bytree": 0.80,
        "seed": 1086,
        "categorical": True,
    },
    {
        "name": "lambda_numeric_ids",
        "objective": "lambdarank",
        "iterations": 520,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 35,
        "colsample_bytree": 0.88,
        "seed": 305,
        "categorical": False,
    },
    {
        "name": "rank_xendcg",
        "objective": "rank_xendcg",
        "iterations": 327,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 35,
        "colsample_bytree": 0.88,
        "seed": 743,
        "categorical": True,
    },
    {
        "name": "lambda_recency",
        "objective": "lambdarank",
        "iterations": 692,
        "learning_rate": 0.025,
        "num_leaves": 255,
        "min_child_samples": 55,
        "colsample_bytree": 0.80,
        "seed": 1091,
        "categorical": True,
        "recency_floor": 0.25,
        "recency_power": 2.0,
    },
)

SELECTOR_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "selector_31",
        "objective": "lambdarank",
        "iterations": 800,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 45,
        "colsample_bytree": 0.78,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
    },
    {
        "name": "selector_63",
        "objective": "lambdarank",
        "iterations": 800,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.8,
        "reg_lambda": 7.0,
    },
    {
        "name": "selector_127",
        "objective": "lambdarank",
        "iterations": 800,
        "learning_rate": 0.018,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 60,
        "colsample_bytree": 0.82,
        "reg_alpha": 1.0,
        "reg_lambda": 9.0,
    },
)

SUMMARY_FEATURES: tuple[str, ...] = (
    "model_score_mean",
    "model_score_std",
    "model_score_min",
    "model_score_max",
    "model_score_range",
    "model_rank_mean",
    "model_rank_min",
    "model_vote_count",
    "model_vote_fraction",
    "model_unique_top_count",
)


def _ranges(groups: np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(groups, dtype=np.int64)
    ends = np.cumsum(values, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | list[int],
) -> float:
    starts, ends = _ranges(groups)
    return float(np.mean([
        labels[start + int(np.argmax(scores[start:end]))] == 1
        for start, end in zip(starts, ends)
    ]))


def _oracle(
    score_sets: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | list[int],
) -> float:
    starts, ends = _ranges(groups)
    return float(np.mean([
        any(
            labels[start + int(np.argmax(scores[start:end]))] == 1
            for scores in score_sets
        )
        for start, end in zip(starts, ends)
    ]))


def _normalize(
    scores: np.ndarray,
    groups: np.ndarray | list[int],
) -> np.ndarray:
    result = np.asarray(scores, dtype=np.float32).copy()
    starts, ends = _ranges(groups)
    for start, end in zip(starts, ends):
        values = result[start:end]
        result[start:end] = (
            (values - float(values.mean()))
            / max(float(values.std()), 1e-5)
        )
    return result


def _tree_importance(model: dict[str, Any]) -> dict[str, float]:
    counts = np.zeros(len(model["feature_names"]), dtype=np.float64)
    stack = list(model["trees"])
    while stack:
        node = stack.pop()
        if "v" in node:
            continue
        counts[int(node["f"])] += 1.0
        stack.extend((node["l"], node["r"]))
    return {
        name: float(counts[index])
        for index, name in enumerate(model["feature_names"])
    }


def _raw_feature_columns(
    feature_names: list[str],
    importance_model: dict[str, Any],
    limit: int,
) -> tuple[np.ndarray, list[str]]:
    importance = _tree_importance(importance_model)
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
        "has_alakazam_anywhere",
        "needs_first_abra",
        "needs_stage2",
        "needs_attacker_energy",
        "dudunsparce_engine_count",
        "dunsparce_engine_count",
        "opp_has_grimmsnarl_ex",
        "opp_has_froslass",
        "current_powerful_hand_damage",
        "fallback_policy_score",
        "fallback_policy_score_gap",
        "fallback_policy_rank",
        "legacy_ranker_score",
        "legacy_ranker_score_gap",
        "legacy_ranker_rank",
        "v29_ranker_score",
        "v29_ranker_score_gap",
        "v29_ranker_rank",
    }
    order = sorted(
        range(len(feature_names)),
        key=lambda index: (
            feature_names[index] not in mandatory,
            -importance.get(feature_names[index], 0.0),
            index,
        ),
    )
    chosen = np.asarray(order[:limit], dtype=np.int64)
    return chosen, [feature_names[index] for index in chosen]


def _decision_recency(
    episode_ids: np.ndarray,
    decisions: np.ndarray,
    *,
    floor: float,
    power: float,
) -> np.ndarray:
    selected = episode_ids[decisions]
    ordered = np.unique(selected)
    ordered.sort()
    positions = {
        int(episode): index / max(len(ordered) - 1, 1)
        for index, episode in enumerate(ordered)
    }
    return np.asarray([
        floor + (1.0 - floor) * positions[int(episode)] ** power
        for episode in selected
    ], dtype=np.float32)


def _selected_data(
    arrays: dict[str, Any],
    decisions: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    x, y, weights, groups = teacher._select_decisions(arrays, decisions)
    if "recency_floor" not in config:
        return x, y, weights, groups
    multipliers = _decision_recency(
        arrays["episode_ids"],
        decisions,
        floor=float(config["recency_floor"]),
        power=float(config["recency_power"]),
    )
    starts, ends = _ranges(groups)
    adjusted = weights.copy()
    for multiplier, start, end in zip(multipliers, starts, ends):
        adjusted[start:end] *= multiplier
    return x, y, adjusted, groups


def _fit_base(
    arrays: dict[str, Any],
    feature_names: list[str],
    decisions: np.ndarray,
    config: dict[str, Any],
    *,
    n_jobs: int,
) -> lgb.LGBMRanker:
    x, y, weights, groups = _selected_data(arrays, decisions, config)
    categorical = [
        index
        for index, name in enumerate(feature_names)
        if name in teacher.BASE_CATEGORICAL or name.endswith("_id")
    ]
    model = lgb.LGBMRanker(
        objective=str(config["objective"]),
        metric="ndcg",
        n_estimators=int(config["iterations"]),
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["num_leaves"]),
        min_child_samples=int(config["min_child_samples"]),
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=float(config["colsample_bytree"]),
        reg_alpha=float(config.get("reg_alpha", 0.2)),
        reg_lambda=float(config.get("reg_lambda", 1.0)),
        random_state=int(config["seed"]),
        n_jobs=n_jobs,
        verbosity=-1,
    )
    model.fit(
        x,
        y,
        group=groups,
        sample_weight=weights,
        feature_name=feature_names,
        categorical_feature=(
            categorical if bool(config.get("categorical", True)) else []
        ),
    )
    return model


def _fold_assignments(
    episode_ids: np.ndarray,
    train_decisions: np.ndarray,
    folds: int,
) -> np.ndarray:
    episodes = np.unique(episode_ids[train_decisions])
    episodes.sort()
    episode_fold = {
        int(episode): index % folds
        for index, episode in enumerate(episodes)
    }
    return np.asarray([
        episode_fold[int(episode_ids[decision])]
        for decision in train_decisions
    ], dtype=np.int8)


def _put_fold_scores(
    target: np.ndarray,
    scores: np.ndarray,
    fold_local_decisions: np.ndarray,
    train_group_starts: np.ndarray,
    train_group_ends: np.ndarray,
) -> None:
    offset = 0
    for local_decision in fold_local_decisions:
        size = int(
            train_group_ends[local_decision]
            - train_group_starts[local_decision]
        )
        start = int(train_group_starts[local_decision])
        target[start:start + size] = scores[offset:offset + size]
        offset += size
    if offset != len(scores):
        raise RuntimeError(
            f"OOF score length mismatch: consumed={offset}, rows={len(scores)}"
        )


def _derived_feature_names(model_names: list[str]) -> list[str]:
    names: list[str] = []
    for name in model_names:
        names.extend((
            f"{name}__score",
            f"{name}__gap",
            f"{name}__rank",
            f"{name}__selected",
        ))
    names.extend(SUMMARY_FEATURES)
    return names


def _meta_features(
    raw_features: np.ndarray,
    raw_columns: np.ndarray,
    score_sets: np.ndarray,
    groups: np.ndarray | list[int],
) -> np.ndarray:
    model_count, row_count = score_sets.shape
    derived_count = model_count * 4 + len(SUMMARY_FEATURES)
    output = np.empty(
        (row_count, len(raw_columns) + derived_count),
        dtype=np.float32,
    )
    output[:, :len(raw_columns)] = raw_features[:, raw_columns]
    starts, ends = _ranges(groups)
    base = len(raw_columns)
    for start, end in zip(starts, ends):
        local = score_sets[:, start:end]
        ranks = np.argsort(
            np.argsort(-local, axis=1, kind="stable"),
            axis=1,
            kind="stable",
        ).astype(np.float32)
        tops = np.argmax(local, axis=1)
        votes = np.bincount(tops, minlength=end - start).astype(np.float32)
        for model_index in range(model_count):
            column = base + model_index * 4
            values = local[model_index]
            output[start:end, column] = values
            output[start:end, column + 1] = values - float(values.max())
            output[start:end, column + 2] = ranks[model_index]
            output[start:end, column + 3] = (
                np.arange(end - start) == tops[model_index]
            )
        summary = base + model_count * 4
        output[start:end, summary] = local.mean(axis=0)
        output[start:end, summary + 1] = local.std(axis=0)
        output[start:end, summary + 2] = local.min(axis=0)
        output[start:end, summary + 3] = local.max(axis=0)
        output[start:end, summary + 4] = (
            local.max(axis=0) - local.min(axis=0)
        )
        output[start:end, summary + 5] = ranks.mean(axis=0)
        output[start:end, summary + 6] = ranks.min(axis=0)
        output[start:end, summary + 7] = votes
        output[start:end, summary + 8] = votes / model_count
        output[start:end, summary + 9] = len(set(tops.tolist()))
    return output


def _selector_categorical(names: list[str]) -> list[int]:
    categorical = []
    for index, name in enumerate(names):
        raw_name = name.removeprefix("raw__")
        if (
            raw_name in teacher.BASE_CATEGORICAL
            or raw_name.endswith("_id")
        ):
            categorical.append(index)
    return categorical


def _fit_selector(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    train_groups: list[int],
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    validation_groups: list[int],
    feature_names: list[str],
    config: dict[str, Any],
    *,
    n_jobs: int,
) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective=str(config["objective"]),
        metric="ndcg",
        n_estimators=int(config["iterations"]),
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["num_leaves"]),
        max_depth=int(config["max_depth"]),
        min_child_samples=int(config["min_child_samples"]),
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=float(config["colsample_bytree"]),
        reg_alpha=float(config["reg_alpha"]),
        reg_lambda=float(config["reg_lambda"]),
        random_state=3300,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        group=train_groups,
        sample_weight=train_weights,
        feature_name=feature_names,
        categorical_feature=_selector_categorical(feature_names),
        eval_set=[(validation_x, validation_y)],
        eval_group=[validation_groups],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    return model


def _write_compact(
    model: lgb.LGBMRanker,
    path: Path,
    metadata: dict[str, Any],
) -> int:
    compact = compact_booster(model.booster_, "ranker")
    compact.update(metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return path.stat().st_size


def _split_rows(
    arrays: dict[str, Any],
    decisions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    return teacher._select_decisions(arrays, decisions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("importance_model", type=Path)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--raw-feature-limit", type=int, default=180)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--reference-validation",
        type=float,
        default=0.8021040095275903,
    )
    parser.add_argument("--minimum-improvement", type=float, default=0.002)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    with np.load(args.cache, allow_pickle=False) as cached:
        arrays: dict[str, Any] = {
            name: cached[name]
            for name in (
                "features",
                "labels",
                "weights",
                "groups",
                "episode_ids",
            )
        }
        splits = cached["splits"].astype(str)
        feature_names = cached["feature_names"].astype(str).tolist()

    train_decisions = np.flatnonzero(splits == "train")
    validation_decisions = np.flatnonzero(splits == "validation")
    test_decisions = np.flatnonzero(splits == "test")
    importance_model = json.loads(
        args.importance_model.read_text(encoding="utf-8")
    )
    raw_columns, raw_names = _raw_feature_columns(
        feature_names,
        importance_model,
        args.raw_feature_limit,
    )
    model_names = [str(config["name"]) for config in BASE_CONFIGS]
    assignments = _fold_assignments(
        arrays["episode_ids"],
        train_decisions,
        args.folds,
    )

    (
        train_raw,
        train_y,
        train_weights,
        train_groups,
    ) = _split_rows(arrays, train_decisions)
    validation_raw, validation_y, _, validation_groups = _split_rows(
        arrays,
        validation_decisions,
    )
    test_raw, test_y, _, test_groups = _split_rows(
        arrays,
        test_decisions,
    )
    train_starts, train_ends = _ranges(train_groups)
    oof_scores = np.full(
        (len(BASE_CONFIGS), len(train_y)),
        np.nan,
        dtype=np.float32,
    )
    validation_scores = np.empty(
        (len(BASE_CONFIGS), len(validation_y)),
        dtype=np.float32,
    )
    test_scores = np.empty(
        (len(BASE_CONFIGS), len(test_y)),
        dtype=np.float32,
    )
    base_reports: list[dict[str, Any]] = []
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    for model_index, config in enumerate(BASE_CONFIGS):
        fold_metrics = []
        for fold in range(args.folds):
            fold_local = np.flatnonzero(assignments == fold)
            fit_local = np.flatnonzero(assignments != fold)
            fit_decisions = train_decisions[fit_local]
            held_decisions = train_decisions[fold_local]
            model = _fit_base(
                arrays,
                feature_names,
                fit_decisions,
                config,
                n_jobs=args.n_jobs,
            )
            held_x, held_y, _, held_groups = _split_rows(
                arrays,
                held_decisions,
            )
            held_scores = _normalize(
                model.predict(held_x),
                held_groups,
            )
            _put_fold_scores(
                oof_scores[model_index],
                held_scores,
                fold_local,
                train_starts,
                train_ends,
            )
            fold_row = {
                "fold": fold,
                "fit_decisions": int(len(fit_decisions)),
                "holdout_decisions": int(len(held_decisions)),
                "top1": _accuracy(held_scores, held_y, held_groups),
            }
            fold_metrics.append(fold_row)
            print(json.dumps({
                "model": config["name"],
                **fold_row,
            }), flush=True)
            del model, held_x, held_y, held_scores
            gc.collect()
        if np.isnan(oof_scores[model_index]).any():
            raise RuntimeError(f"Incomplete OOF scores for {config['name']}")

        final_model = _fit_base(
            arrays,
            feature_names,
            train_decisions,
            config,
            n_jobs=args.n_jobs,
        )
        validation_scores[model_index] = _normalize(
            final_model.predict(validation_raw),
            validation_groups,
        )
        test_scores[model_index] = _normalize(
            final_model.predict(test_raw),
            test_groups,
        )
        artifact_name = f"selector_base_{config['name']}.json"
        artifact_bytes = _write_compact(
            final_model,
            args.artifact_dir / artifact_name,
            {
                "runtime_scope": "v33_oof_selector_base",
                "model_name": config["name"],
                "training_decisions": int(len(train_decisions)),
                "oof_folds": int(args.folds),
            },
        )
        base_row = {
            "name": config["name"],
            "config": config,
            "folds": fold_metrics,
            "oof_top1": _accuracy(
                oof_scores[model_index],
                train_y,
                train_groups,
            ),
            "validation_top1": _accuracy(
                validation_scores[model_index],
                validation_y,
                validation_groups,
            ),
            "artifact": artifact_name,
            "artifact_bytes": artifact_bytes,
        }
        base_reports.append(base_row)
        print(json.dumps(base_row), flush=True)
        del final_model
        gc.collect()

    selector_names = [
        *[f"raw__{name}" for name in raw_names],
        *_derived_feature_names(model_names),
    ]
    train_x = _meta_features(
        train_raw,
        raw_columns,
        oof_scores,
        train_groups,
    )
    validation_x = _meta_features(
        validation_raw,
        raw_columns,
        validation_scores,
        validation_groups,
    )
    del train_raw, validation_raw
    gc.collect()

    selector_trials: list[dict[str, Any]] = []
    selector_models: list[lgb.LGBMRanker] = []
    for config in SELECTOR_CONFIGS:
        selector = _fit_selector(
            train_x,
            train_y,
            train_weights,
            train_groups,
            validation_x,
            validation_y,
            validation_groups,
            selector_names,
            config,
            n_jobs=args.n_jobs,
        )
        scores = selector.predict(validation_x).astype(np.float32)
        row = {
            **config,
            "best_iteration": int(
                selector.best_iteration_ or config["iterations"]
            ),
            "validation_top1": _accuracy(
                scores,
                validation_y,
                validation_groups,
            ),
        }
        selector_trials.append(row)
        selector_models.append(selector)
        print(json.dumps(row), flush=True)
    selected_index = max(
        range(len(selector_trials)),
        key=lambda index: (
            selector_trials[index]["validation_top1"],
            -selector_trials[index]["num_leaves"],
        ),
    )
    selected_trial = selector_trials[selected_index]
    selected_selector = selector_models[selected_index]
    enabled = bool(
        selected_trial["validation_top1"]
        >= args.reference_validation + args.minimum_improvement
    )

    # Test is intentionally not transformed or scored until selector capacity
    # and the adoption decision have both been frozen on validation.
    test_x = _meta_features(
        test_raw,
        raw_columns,
        test_scores,
        test_groups,
    )
    test_selector_scores = selected_selector.predict(test_x).astype(np.float32)
    test_top1 = _accuracy(test_selector_scores, test_y, test_groups)
    selector_bytes = _write_compact(
        selected_selector,
        args.artifact_dir / "selector_model.json",
        {
            "runtime_scope": "v33_oof_candidate_selector",
            "enabled": enabled,
            "model_order": model_names,
            "base_artifacts": [
                row["artifact"] for row in base_reports
            ],
            "raw_feature_names": raw_names,
            "derived_feature_names": _derived_feature_names(model_names),
            "oof_folds": int(args.folds),
            "reference_validation": float(args.reference_validation),
            "minimum_improvement": float(args.minimum_improvement),
            "validation_top1": float(
                selected_trial["validation_top1"]
            ),
            "test_top1": float(test_top1),
        },
    )

    report = {
        "teacher": "Yushin Ito rank 3",
        "method": "episode_grouped_oof_candidate_ranker",
        "folds": int(args.folds),
        "split_decisions": {
            "train": int(len(train_decisions)),
            "validation": int(len(validation_decisions)),
            "test": int(len(test_decisions)),
        },
        "base_models": base_reports,
        "oof_oracle_any_base": _oracle(
            oof_scores,
            train_y,
            train_groups,
        ),
        "validation_oracle_any_base": _oracle(
            validation_scores,
            validation_y,
            validation_groups,
        ),
        "test_oracle_any_base": _oracle(
            test_scores,
            test_y,
            test_groups,
        ),
        "raw_features": raw_names,
        "selector_features": len(selector_names),
        "selector_trials": selector_trials,
        "selected_selector": selected_trial,
        "reference_validation": float(args.reference_validation),
        "minimum_improvement": float(args.minimum_improvement),
        "adoption_threshold": float(
            args.reference_validation + args.minimum_improvement
        ),
        "selector_enabled": enabled,
        "test_top1": test_top1,
        "selector_artifact_bytes": selector_bytes,
        "target_top1": 0.9,
        "target_met": bool(test_top1 >= 0.9),
        "test_policy": (
            "evaluated once after validation froze selector capacity "
            "and adoption"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.scores_output is not None:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.scores_output,
            oof_scores=oof_scores,
            validation_scores=validation_scores,
            test_scores=test_scores,
            train_labels=train_y,
            validation_labels=validation_y,
            test_labels=test_y,
            train_groups=np.asarray(train_groups),
            validation_groups=np.asarray(validation_groups),
            test_groups=np.asarray(test_groups),
            model_names=np.asarray(model_names),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
