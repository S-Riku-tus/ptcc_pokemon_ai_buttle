from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .distill_model import tree_score, write_compact_model
from .evaluate_offline import evaluate_scores, frequency_scores, handwritten_scores
from .feature_engineering import FEATURE_COLUMNS, STATE_FEATURES
from .train_policy import train_neural_ranker


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["episode_id", "decision_id", "candidate_index"], kind="stable").reset_index(drop=True)


def _groups(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("decision_id", sort=False).size().astype(int).tolist()


def train_lgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    seed: int,
    estimators: int = 180,
    weights: np.ndarray | None = None,
) -> lgb.LGBMRanker:
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=estimators,
        learning_rate=0.045,
        num_leaves=31,
        max_depth=7,
        min_child_samples=30,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        deterministic=True,
        verbosity=-1,
    )
    fit_kwargs: dict[str, Any] = {
        "X": train[features].fillna(0),
        "y": train["selected"].astype(int),
        "group": _groups(train),
        "eval_set": [(validation[features].fillna(0), validation["selected"].astype(int))],
        "eval_group": [_groups(validation)],
        "eval_at": [1, 3, 5],
        "callbacks": [lgb.early_stopping(25, verbose=False)],
    }
    fit_kwargs["sample_weight"] = weights if weights is not None else train["teacher_weight"].to_numpy()
    model.fit(**fit_kwargs)
    return model


def train_all(
    processed_dir: Path, models_dir: Path, reports_dir: Path, seed: int = 741,
    confidence_threshold: float = 0.65,
) -> dict[str, Any]:
    candidates = _ordered(pd.read_parquet(processed_dir / "legal_candidate_dataset.parquet"))
    decisions = pd.read_parquet(processed_dir / "decision_dataset.parquet")
    split_map = decisions.set_index("decision_id")["split_time"].to_dict()
    candidates["split_time"] = candidates["decision_id"].map(split_map)
    train = _ordered(candidates[candidates["split_time"] == "train"])
    validation = _ordered(candidates[candidates["split_time"] == "validation"])
    test = _ordered(candidates[candidates["split_time"] == "test"])
    test_decisions = decisions[decisions["split_time"] == "test"].copy()
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    baseline_scores = {
        "first_legal": -test["candidate_index"].to_numpy(dtype=float),
        "action_frequency": frequency_scores(train, test),
        "handwritten": handwritten_scores(test),
    }
    all_metrics: dict[str, Any] = {}
    predictions = test[["decision_id", "candidate_index"]].copy()
    for name, scores in baseline_scores.items():
        metrics, _ = evaluate_scores(test, test_decisions, scores, confidence_threshold)
        all_metrics[name] = metrics
        predictions[name] = scores

    started = time.perf_counter()
    ranker = train_lgbm(train, validation, FEATURE_COLUMNS, seed)
    lgbm_scores = ranker.predict(test[FEATURE_COLUMNS].fillna(0))
    inference_ms = (time.perf_counter() - started) * 1000.0
    metrics, detail = evaluate_scores(test, test_decisions, lgbm_scores, confidence_threshold)
    metrics["train_and_test_seconds"] = inference_ms / 1000.0
    all_metrics["lightgbm_ranker"] = metrics
    predictions["lightgbm_ranker"] = lgbm_scores
    ranker.booster_.save_model(str(models_dir / "ranker_model.txt"))
    compact = write_compact_model(ranker.booster_, models_dir / "ranker_model.json", "ranker")
    sample_count = min(1000, len(test))
    compact_scores = np.asarray([
        tree_score(row.tolist(), compact)
        for row in test.iloc[:sample_count][FEATURE_COLUMNS].fillna(0).to_numpy(dtype=float)
    ])
    distill_max_error = float(np.max(np.abs(compact_scores - lgbm_scores[:sample_count]))) if sample_count else 0.0

    neural_scores, neural_info = train_neural_ranker(
        train, validation, test, FEATURE_COLUMNS, models_dir / "neural_ranker.json", seed=seed
    )
    neural_metrics, _ = evaluate_scores(test, test_decisions, neural_scores, confidence_threshold)
    neural_metrics.update(neural_info)
    all_metrics["small_neural_ranker"] = neural_metrics
    predictions["small_neural_ranker"] = neural_scores

    decision_train = decisions[decisions["split_time"] == "train"]
    decision_validation = decisions[decisions["split_time"] == "validation"]
    decision_test = decisions[decisions["split_time"] == "test"]
    value_model = lgb.LGBMRegressor(
        objective="regression", n_estimators=140, learning_rate=0.04, num_leaves=15,
        max_depth=5, min_child_samples=30, reg_lambda=1.0, random_state=seed,
        n_jobs=1, deterministic=True, verbosity=-1,
    )
    value_model.fit(
        decision_train[STATE_FEATURES].fillna(0), decision_train["value_target"],
        sample_weight=decision_train["teacher_weight"],
        eval_set=[(decision_validation[STATE_FEATURES].fillna(0), decision_validation["value_target"])],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    value_prediction = np.clip(value_model.predict(decision_test[STATE_FEATURES].fillna(0)), 0, 1)
    value_metrics = {
        "rmse": float(mean_squared_error(decision_test["value_target"], value_prediction) ** 0.5),
        "brier": float(np.mean((decision_test["value_target"].to_numpy() - value_prediction) ** 2)),
        "accuracy_at_0_5": float(np.mean((value_prediction >= 0.5) == (decision_test["value_target"].to_numpy() >= 0.5))),
    }
    value_model.booster_.save_model(str(models_dir / "value_model.txt"))
    write_compact_model(value_model.booster_, models_dir / "value_model.json", "value")

    ablations: list[dict[str, Any]] = []
    variants = {
        "unweighted": (train, np.ones(len(train))),
        "wins_only": (
            _ordered(train[train["decision_id"].isin(set(decision_train[decision_train["outcome"] == "win"]["decision_id"]))]),
            None,
        ),
        "exclude_unique_legal": (_ordered(train[train["option_count"] > 1]), None),
    }
    weight_table = pd.read_parquet(processed_dir / "expert_weights.parquet").set_index("decision_id")
    no_post = train["decision_id"].map(
        (weight_table["rank_outcome_weight"] * weight_table["importance_weight"] * weight_table["agreement_weight"]).to_dict()
    ).fillna(1.0).to_numpy()
    no_agreement = train["decision_id"].map(
        (weight_table["rank_outcome_weight"] * weight_table["importance_weight"] * weight_table["post_action_quality_weight"]).to_dict()
    ).fillna(1.0).to_numpy()
    variants["no_post_action_quality"] = (train, no_post)
    variants["no_agreement"] = (train, no_agreement)
    for name, (variant_train, variant_weights) in variants.items():
        if variant_train.empty:
            continue
        model = train_lgbm(
            variant_train, validation, FEATURE_COLUMNS, seed + len(ablations) + 1,
            estimators=90,
            weights=variant_weights,
        )
        scores = model.predict(test[FEATURE_COLUMNS].fillna(0))
        variant_metrics, _ = evaluate_scores(test, test_decisions, scores, confidence_threshold)
        ablations.append({"name": name, "status": "evaluated", **{
            key: variant_metrics[key] for key in ("semantic_top1", "top3", "mrr", "weighted_log_loss")
        }})
    ablations.extend([
        {"name": "rank_weight_on_off", "status": "not_identifiable", "reason": "all trainable episodes are rank-1"},
        {"name": "majkel_vs_multiple_top", "status": "not_available", "reason": "other bundles lack observations/actions/legal candidates"},
        {"name": "deck_type_input_on_off", "status": "not_identifiable", "reason": "all trainable episodes use one exact deck"},
        {"name": "value_head_on_off", "status": "reported_separately", **value_metrics},
        {"name": "safety_rules_on_off", "status": "deferred_to_golden_and_battle_evaluation"},
        {"name": "fallback_on_off", "status": "deferred_to_hybrid_evaluation"},
    ])

    predictions.to_csv(reports_dir / "test_predictions.csv", index=False)
    detail.to_csv(reports_dir / "lightgbm_decision_results.csv", index=False)
    summary = {
        "seed": seed,
        "features": FEATURE_COLUMNS,
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "models": all_metrics,
        "value_head": value_metrics,
        "ablations": ablations,
        "distillation_max_abs_error": distill_max_error,
        "confidence_threshold": confidence_threshold,
    }
    (reports_dir / "offline_evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (reports_dir / "action_type_metrics.json").write_text(
        json.dumps(all_metrics["lightgbm_ranker"]["by_action_type"], indent=2), encoding="utf-8"
    )
    (reports_dir / "ablation_results.json").write_text(json.dumps(ablations, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1]
    parser.add_argument("--processed", default=str(base / "data_processed"))
    parser.add_argument("--models", default=str(base / "models"))
    parser.add_argument("--reports", default=str(base / "reports"))
    parser.add_argument("--seed", type=int, default=741)
    args = parser.parse_args()
    summary = train_all(Path(args.processed), Path(args.models), Path(args.reports), args.seed)
    print(json.dumps({name: {
        key: value for key, value in metrics.items() if key in {"semantic_top1", "top3", "mrr", "weighted_log_loss"}
    } for name, metrics in summary["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
