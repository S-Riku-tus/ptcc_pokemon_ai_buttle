from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .dataset import feature_columns
from .evaluate import calibrate_action_thresholds, evaluate_scores, fit_temperature, write_metrics
from .splits import Split, make_splits

CATEGORICAL = ["action_type", "select_type", "select_context", "option_type", "candidate_card_id", "candidate_attack_id", "candidate_target_id", "self_active_id", "opp_active_id", "stadium_id"]


def _prepare(frame: pd.DataFrame, features: list[str], category_maps: dict[str, dict[str, int]] | None = None) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    x = frame[features].copy()
    maps = {} if category_maps is None else {k: dict(v) for k, v in category_maps.items()}
    for column in features:
        if column == "action_type":
            values = x[column].astype(str)
            if column not in maps:
                maps[column] = {value: i for i, value in enumerate(sorted(values.unique()))}
            x[column] = values.map(maps[column]).fillna(-1).astype("int32")
        elif column in CATEGORICAL:
            x[column] = pd.to_numeric(x[column], errors="coerce").fillna(-1).astype("int32")
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce").fillna(-1).astype("float32")
    return x, maps


def _groups(frame: pd.DataFrame) -> list[int]:
    return frame.groupby("decision_id", sort=False, observed=True).size().astype(int).tolist()


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["decision_id", "candidate_index"]).reset_index(drop=True)


def train_one(train_rows: pd.DataFrame, params: dict[str, Any] | None = None, weight_column: str = "sample_weight") -> tuple[lgb.LGBMRanker, list[str], dict[str, dict[str, int]]]:
    ordered = _ordered(train_rows)
    features = feature_columns(ordered)
    x, maps = _prepare(ordered, features)
    model_params = {
        "objective": "lambdarank", "metric": "ndcg", "n_estimators": 80, "learning_rate": 0.05,
        "num_leaves": 31, "min_child_samples": 40, "subsample": 0.85, "colsample_bytree": 0.85,
        "reg_alpha": 0.1, "reg_lambda": 0.5, "random_state": 741, "n_jobs": 4, "verbosity": -1,
    }
    model_params.update(params or {})
    model = lgb.LGBMRanker(**model_params)
    categorical = [column for column in CATEGORICAL if column in features]
    model.fit(
        x, ordered["label"].astype(int), group=_groups(ordered),
        sample_weight=ordered[weight_column].astype(float),
        categorical_feature=categorical,
    )
    return model, features, maps


def score(model: lgb.LGBMRanker, rows: pd.DataFrame, features: list[str], maps: dict[str, dict[str, int]]) -> tuple[pd.DataFrame, np.ndarray]:
    ordered = _ordered(rows)
    x, _ = _prepare(ordered, features, maps)
    return ordered, model.predict(x)



def compare_legacy_vs_expanded(rows: pd.DataFrame, decisions: pd.DataFrame, reports: Path) -> dict[str, Any]:
    """Compare the old singular-only corpus with the expanded corpus on the same future Majkel test."""
    latest = decisions[decisions["submission_id"] == 54662660].copy()
    if latest.empty:
        return {}
    episodes = sorted(latest["episode_id"].unique())
    cutoff = episodes[max(1, int(len(episodes) * 0.8)) - 1]
    test_ids = set(latest.loc[latest["episode_id"] > cutoff, "decision_id"])
    legacy_train_ids = set(latest.loc[latest["episode_id"] <= cutoff, "decision_id"])
    expanded_train_ids = set(decisions.loc[~decisions["decision_id"].isin(test_ids), "decision_id"])
    test_rows = rows[rows["decision_id"].isin(test_ids)]
    results = {}
    for name, train_ids in (("legacy_singular_only", legacy_train_ids), ("expanded_all_teams", expanded_train_ids)):
        train_rows = rows[rows["decision_id"].isin(train_ids)]
        model, features, maps = train_one(train_rows)
        ordered, predictions = score(model, test_rows, features, maps)
        metrics, _ = evaluate_scores(ordered, predictions)
        metrics.update({"train_decisions": len(train_ids), "test_decisions": len(test_ids), "test_episode_cutoff": int(cutoff)})
        results[name] = metrics
    write_metrics(reports / "legacy_vs_expanded.json", results)
    return results


def _fit_calibration_split(train_decisions: set[str]) -> tuple[set[str], set[str]]:
    import hashlib
    calibration = {
        decision_id for decision_id in train_decisions
        if int(hashlib.sha1(decision_id.encode()).hexdigest(), 16) % 10 == 0
    }
    if not calibration:
        calibration = set(sorted(train_decisions)[-max(1, len(train_decisions) // 10):])
    fit = set(train_decisions) - calibration
    return fit, calibration


def run_training(rows: pd.DataFrame, decisions: pd.DataFrame, artifact_dir: str | Path, report_dir: str | Path) -> dict[str, Any]:
    artifacts, reports = Path(artifact_dir), Path(report_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    splits = make_splits(decisions)
    all_metrics: dict[str, Any] = {}
    action_thresholds: dict[str, float] = {}
    time_temperature = 1.0

    for split in splits:
        fit_ids, calibration_ids = _fit_calibration_split(split.train_decisions)
        fit_rows = rows[rows["decision_id"].isin(fit_ids)]
        calibration_rows = rows[rows["decision_id"].isin(calibration_ids)]
        test_rows = rows[rows["decision_id"].isin(split.test_decisions)]
        model, features, maps = train_one(fit_rows)
        calibration_ordered, calibration_scores = score(model, calibration_rows, features, maps)
        temperature = fit_temperature(calibration_ordered, calibration_scores)
        ordered, scores = score(model, test_rows, features, maps)
        metrics, per_decision = evaluate_scores(ordered, scores, temperature=temperature)
        metrics.update({
            "description": split.description,
            "fit_decisions": len(fit_ids),
            "calibration_decisions": len(calibration_ids),
            "test_decisions": len(split.test_decisions),
        })
        all_metrics[split.name] = metrics
        per_decision.to_csv(reports / f"{split.name}_predictions.csv", index=False)
        write_metrics(reports / f"{split.name}_metrics.json", metrics)
        if split.name == "time_holdout":
            time_temperature = temperature
            action_thresholds = calibrate_action_thresholds(per_decision)

    # Ablations on the valid within-submission chronological split. Ranking
    # metrics are invariant to temperature, while ECE uses the same held-out
    # time-split temperature for comparability.
    time_split = next(split for split in splits if split.name == "time_holdout")
    train_rows = rows[rows["decision_id"].isin(time_split.train_decisions)].copy()
    test_rows = rows[rows["decision_id"].isin(time_split.test_decisions)].copy()
    decision_index = decisions.set_index("decision_id")
    ablations: dict[str, Any] = {}
    variants = {
        "full": train_rows["sample_weight"],
        "uniform": pd.Series(1.0, index=train_rows.index),
        "no_deck_distance": train_rows["sample_weight"] / train_rows["decision_id"].map(decision_index["deck_weight"]).fillna(1.0),
        "no_rank_outcome": train_rows["sample_weight"] / (
            train_rows["decision_id"].map(decision_index["rank_weight"]).fillna(1.0)
            * train_rows["decision_id"].map(decision_index["outcome_weight"]).fillna(1.0)
        ),
    }
    for name, weights in variants.items():
        temp = train_rows.copy()
        temp["ablation_weight"] = weights.astype(float)
        model, features, maps = train_one(temp, weight_column="ablation_weight")
        ordered, scores = score(model, test_rows, features, maps)
        metrics, _ = evaluate_scores(ordered, scores, temperature=time_temperature)
        ablations[name] = metrics
    write_metrics(reports / "ablation_metrics.json", ablations)

    # Final model on all aligned data; confidence calibration is inherited from
    # the strict time holdout, not fitted on the final training labels.
    final_model, features, maps = train_one(rows)
    joblib.dump(final_model, artifacts / "ranker.joblib")
    final_model.booster_.save_model(str(artifacts / "ranker.txt"))
    schema = {
        "feature_columns": features,
        "category_maps": maps,
        "temperature": float(time_temperature),
        "fallback_probability": 0.55,
        "fallback_margin": 0.12,
        "action_type_thresholds": action_thresholds,
        "legal_option_only": True,
    }
    (artifacts / "model_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metrics(reports / "offline_evaluation.json", all_metrics)
    legacy_comparison = compare_legacy_vs_expanded(rows, decisions, reports)
    return {"splits": all_metrics, "ablations": ablations, "legacy_comparison": legacy_comparison, "schema": schema}
