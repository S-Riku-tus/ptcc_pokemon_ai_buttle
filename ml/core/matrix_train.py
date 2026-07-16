from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .evaluate import calibrate_action_thresholds, write_metrics
from .matrix import load_matrix_store
from .splits import make_splits

CATEGORICAL_NAMES = [
    "action_type", "option_type", "candidate_card_id", "candidate_attack_id",
    "candidate_target_id", "self_active_id", "opp_active_id", "stadium_id",
]


def _decision_numeric_set(decisions: pd.DataFrame, decision_ids: set[str]) -> np.ndarray:
    values = decisions.loc[decisions["decision_id"].isin(decision_ids), "decision_numeric_id"].to_numpy(dtype=np.int32)
    return np.sort(values)


def _calibration_ids(decision_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    calibration_mask = np.array([
        int(hashlib.sha1(str(int(value)).encode()).hexdigest(), 16) % 10 == 0
        for value in decision_ids
    ], dtype=bool)
    if not calibration_mask.any():
        calibration_mask[-max(1, len(calibration_mask) // 10):] = True
    return decision_ids[~calibration_mask], decision_ids[calibration_mask]


def _row_indices(decision_index: np.ndarray, selected_decisions: np.ndarray, decision_count: int) -> np.ndarray:
    mask = np.zeros(decision_count, dtype=bool)
    mask[selected_decisions] = True
    return np.flatnonzero(mask[np.asarray(decision_index)])


def _group_sizes(row_decisions: np.ndarray) -> list[int]:
    if len(row_decisions) == 0:
        return []
    changes = np.flatnonzero(np.diff(row_decisions)) + 1
    boundaries = np.r_[0, changes, len(row_decisions)]
    return np.diff(boundaries).astype(int).tolist()


def _extract(arrays: dict[str, np.ndarray], row_indices: np.ndarray, weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    x = np.asarray(arrays["features"][row_indices], dtype=np.float32)
    y = np.asarray(arrays["labels"][row_indices], dtype=np.int8)
    w = np.asarray(arrays["sample_weight"][row_indices] if weight is None else weight[row_indices], dtype=np.float32)
    groups = _group_sizes(np.asarray(arrays["decision_index"][row_indices], dtype=np.int32))
    return x, y, w, groups


def _make_model(n_estimators: int = 200) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=n_estimators,
        learning_rate=0.05, num_leaves=31, min_child_samples=40,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=0.5,
        random_state=741, n_jobs=4, verbosity=-1,
    )


def _fit_model(
    arrays: dict[str, np.ndarray],
    fit_rows: np.ndarray,
    feature_columns: list[str],
    calibration_rows: np.ndarray | None = None,
    row_weight: np.ndarray | None = None,
    fixed_estimators: int | None = None,
) -> lgb.LGBMRanker:
    x, y, w, groups = _extract(arrays, fit_rows, row_weight)
    categorical = [feature_columns.index(name) for name in CATEGORICAL_NAMES if name in feature_columns]
    model = _make_model(fixed_estimators or 200)
    fit_kwargs: dict[str, Any] = {
        "X": x, "y": y, "group": groups, "sample_weight": w,
        "categorical_feature": categorical, "feature_name": feature_columns,
    }
    if calibration_rows is not None and len(calibration_rows) and fixed_estimators is None:
        cx, cy, _, cgroups = _extract(arrays, calibration_rows)
        fit_kwargs.update({
            "eval_set": [(cx, cy)], "eval_group": [cgroups],
            "callbacks": [lgb.early_stopping(15, verbose=False)],
        })
    model.fit(**fit_kwargs)
    return model


def _score(model: lgb.LGBMRanker, arrays: dict[str, np.ndarray], rows: np.ndarray, batch_size: int = 100_000) -> np.ndarray:
    output = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch_size):
        end = min(len(rows), start + batch_size)
        x = np.asarray(arrays["features"][rows[start:end]], dtype=np.float32)
        output[start:end] = model.predict(x).astype(np.float32)
    return output


def _group_ranges(row_decisions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    changes = np.flatnonzero(np.diff(row_decisions)) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(row_decisions)]
    return starts, ends


def _fit_temperature_arrays(arrays: dict[str, np.ndarray], rows: np.ndarray, scores: np.ndarray) -> float:
    row_decisions = np.asarray(arrays["decision_index"][rows], dtype=np.int32)
    labels = np.asarray(arrays["labels"][rows], dtype=np.int8)
    starts, ends = _group_ranges(row_decisions)
    best_t, best_loss = 1.0, float("inf")
    for temperature in np.geomspace(0.20, 5.0, 61):
        total = 0.0
        valid = 0
        for start, end in zip(starts, ends):
            group_labels = labels[start:end]
            true = np.flatnonzero(group_labels == 1)
            if len(true) != 1:
                continue
            values = scores[start:end].astype(float) / temperature
            values -= values.max()
            exp = np.exp(np.clip(values, -50, 50))
            probability = exp[int(true[0])] / max(exp.sum(), 1e-12)
            total -= float(np.log(max(probability, 1e-12)))
            valid += 1
        loss = total / max(valid, 1)
        if loss < best_loss:
            best_t, best_loss = float(temperature), loss
    return best_t


def evaluate_arrays(
    arrays: dict[str, np.ndarray],
    decisions: pd.DataFrame,
    rows: np.ndarray,
    scores: np.ndarray,
    action_type_map: dict[str, int],
    temperature: float = 1.0,
    fallback_probability: float = 0.55,
    fallback_margin: float = 0.12,
) -> tuple[dict[str, Any], pd.DataFrame]:
    inverse_action = {int(v): str(k) for k, v in action_type_map.items()}
    row_decisions = np.asarray(arrays["decision_index"][rows], dtype=np.int32)
    labels = np.asarray(arrays["labels"][rows], dtype=np.int8)
    candidate_actions = np.asarray(arrays["candidate_action_type"][rows], dtype=np.int8)
    starts, ends = _group_ranges(row_decisions)
    records: list[dict[str, Any]] = []
    for start, end in zip(starts, ends):
        group_labels = labels[start:end]
        true = np.flatnonzero(group_labels == 1)
        if len(true) != 1:
            continue
        group_scores = scores[start:end].astype(float)
        order = np.argsort(-group_scores, kind="stable")
        rank = int(np.flatnonzero(order == int(true[0]))[0]) + 1
        scaled = group_scores / max(float(temperature), 1e-6)
        scaled -= scaled.max()
        probs = np.exp(np.clip(scaled, -50, 50)); probs /= max(probs.sum(), 1e-12)
        top = int(order[0])
        second = int(order[1]) if len(order) > 1 else top
        confidence = float(probs[top])
        margin = float(probs[top] - probs[second]) if len(order) > 1 else 1.0
        fallback = confidence < fallback_probability or margin < fallback_margin
        numeric_id = int(row_decisions[start])
        records.append({
            "decision_id": str(decisions.iloc[numeric_id]["decision_id"]),
            "rank": rank,
            "correct": rank == 1,
            "top3": rank <= 3,
            "reciprocal_rank": 1.0 / rank,
            "confidence": confidence,
            "margin": margin,
            "fallback": fallback,
            "selected_action_type": inverse_action[int(candidate_actions[start + int(true[0])])],
            "predicted_action_type": inverse_action[int(candidate_actions[start + top])],
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        return {"decision_count": 0}, frame
    accepted = frame[~frame["fallback"]]
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        bucket = frame[(frame["confidence"] >= lo) & (frame["confidence"] < hi)]
        if len(bucket):
            ece += len(bucket) / len(frame) * abs(bucket["correct"].mean() - bucket["confidence"].mean())
    action_metrics: dict[str, Any] = {}
    for action, group in frame.groupby("selected_action_type", observed=True):
        accepted_group = group[~group["fallback"]]
        action_metrics[str(action)] = {
            "count": int(len(group)), "top1": float(group["correct"].mean()),
            "top3": float(group["top3"].mean()), "mrr": float(group["reciprocal_rank"].mean()),
            "fallback_rate": float(group["fallback"].mean()),
            "accepted_top1": float(accepted_group["correct"].mean()) if len(accepted_group) else None,
        }
    metrics = {
        "decision_count": int(len(frame)), "top1": float(frame["correct"].mean()),
        "top3": float(frame["top3"].mean()), "mrr": float(frame["reciprocal_rank"].mean()),
        "ece": float(ece), "temperature": float(temperature),
        "fallback_probability": fallback_probability, "fallback_margin": fallback_margin,
        "fallback_rate": float(frame["fallback"].mean()),
        "accepted_top1": float(accepted["correct"].mean()) if len(accepted) else None,
        "accepted_count": int(len(accepted)), "illegal_action_count": 0,
        "action_type_metrics": action_metrics,
    }
    return metrics, frame


def _weight_arrays(arrays: dict[str, np.ndarray], decisions: pd.DataFrame) -> dict[str, np.ndarray]:
    decision_index = np.asarray(arrays["decision_index"], dtype=np.int32)
    sample = np.asarray(arrays["sample_weight"], dtype=np.float32)
    deck = decisions["deck_weight"].to_numpy(dtype=np.float32)[decision_index]
    rank = decisions["rank_weight"].to_numpy(dtype=np.float32)[decision_index]
    outcome = decisions["outcome_weight"].to_numpy(dtype=np.float32)[decision_index]
    return {
        "full": sample,
        "uniform": np.ones_like(sample, dtype=np.float32),
        "no_deck_distance": sample / np.maximum(deck, 1e-6),
        "no_rank_outcome": sample / np.maximum(rank * outcome, 1e-6),
    }


def _feature_importance(model: lgb.LGBMRanker, feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": feature_columns,
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)


def run_matrix_training(processed_dir: str | Path, artifact_dir: str | Path, report_dir: str | Path, progress: bool = False) -> dict[str, Any]:
    processed, artifacts, reports = Path(processed_dir), Path(artifact_dir), Path(report_dir)
    artifacts.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    schema, arrays, decisions = load_matrix_store(processed)
    feature_columns = list(schema["feature_columns"])
    action_type_map = {str(k): int(v) for k, v in schema["action_type_map"].items()}
    decision_count = int(schema["decision_count"])
    splits = make_splits(decisions)
    all_metrics: dict[str, Any] = {}
    time_temperature = 1.0
    action_thresholds: dict[str, float] = {}
    best_iterations: list[int] = []

    for split in splits:
        train_ids = _decision_numeric_set(decisions, split.train_decisions)
        test_ids = _decision_numeric_set(decisions, split.test_decisions)
        fit_ids, calibration_ids = _calibration_ids(train_ids)
        fit_rows = _row_indices(arrays["decision_index"], fit_ids, decision_count)
        calibration_rows = _row_indices(arrays["decision_index"], calibration_ids, decision_count)
        test_rows = _row_indices(arrays["decision_index"], test_ids, decision_count)
        if progress:
            print(json.dumps({"event": "split_train", "split": split.name, "fit_decisions": len(fit_ids), "test_decisions": len(test_ids), "fit_rows": len(fit_rows)}), flush=True)
        model = _fit_model(arrays, fit_rows, feature_columns, fixed_estimators=50)
        best_iteration = 50
        best_iterations.append(best_iteration)
        calibration_scores = _score(model, arrays, calibration_rows)
        temperature = _fit_temperature_arrays(arrays, calibration_rows, calibration_scores)
        test_scores = _score(model, arrays, test_rows)
        metrics, predictions = evaluate_arrays(arrays, decisions, test_rows, test_scores, action_type_map, temperature=temperature)
        metrics.update({
            "description": split.description, "fit_decisions": int(len(fit_ids)),
            "calibration_decisions": int(len(calibration_ids)), "test_decisions": int(len(test_ids)),
            "fit_rows": int(len(fit_rows)), "test_rows": int(len(test_rows)),
            "best_iteration": best_iteration,
        })
        all_metrics[split.name] = metrics
        predictions.to_csv(reports / f"{split.name}_predictions.csv", index=False)
        write_metrics(reports / f"{split.name}_metrics.json", metrics)
        if split.name == "time_holdout":
            time_temperature = temperature
            action_thresholds = calibrate_action_thresholds(predictions)
        del model, fit_rows, calibration_rows, test_rows, calibration_scores, test_scores
        gc.collect()

    fixed_estimators = 50
    time_split = next(split for split in splits if split.name == "time_holdout")
    train_ids = _decision_numeric_set(decisions, time_split.train_decisions)
    test_ids = _decision_numeric_set(decisions, time_split.test_decisions)
    train_rows = _row_indices(arrays["decision_index"], train_ids, decision_count)
    test_rows = _row_indices(arrays["decision_index"], test_ids, decision_count)
    weight_variants = _weight_arrays(arrays, decisions)
    ablations: dict[str, Any] = {}
    for name in ("full", "uniform", "no_deck_distance", "no_rank_outcome"):
        if progress:
            print(json.dumps({"event": "ablation_train", "variant": name, "estimators": fixed_estimators}), flush=True)
        model = _fit_model(arrays, train_rows, feature_columns, row_weight=weight_variants[name], fixed_estimators=fixed_estimators)
        test_scores = _score(model, arrays, test_rows)
        metrics, _ = evaluate_arrays(arrays, decisions, test_rows, test_scores, action_type_map, temperature=time_temperature)
        ablations[name] = metrics
        del model, test_scores
        gc.collect()
    write_metrics(reports / "ablation_metrics.json", ablations)

    # Legacy singular-only vs expanded corpus on a future slice of the latest Majkel submission.
    latest = decisions[decisions["submission_id"] == 54662660]
    legacy_comparison: dict[str, Any] = {}
    if not latest.empty:
        episodes = sorted(latest["episode_id"].unique())
        cutoff = episodes[max(1, int(len(episodes) * 0.8)) - 1]
        future_ids = latest.loc[latest["episode_id"] > cutoff, "decision_numeric_id"].to_numpy(dtype=np.int32)
        legacy_ids = latest.loc[latest["episode_id"] <= cutoff, "decision_numeric_id"].to_numpy(dtype=np.int32)
        expanded_ids = decisions.loc[~decisions["decision_numeric_id"].isin(future_ids), "decision_numeric_id"].to_numpy(dtype=np.int32)
        future_rows = _row_indices(arrays["decision_index"], np.sort(future_ids), decision_count)
        for name, ids in (("legacy_singular_only", legacy_ids), ("expanded_all_teams", expanded_ids)):
            if progress:
                print(json.dumps({"event": "corpus_comparison", "variant": name, "train_decisions": len(ids)}), flush=True)
            rows = _row_indices(arrays["decision_index"], np.sort(ids), decision_count)
            model = _fit_model(arrays, rows, feature_columns, fixed_estimators=fixed_estimators)
            scores = _score(model, arrays, future_rows)
            metrics, _ = evaluate_arrays(arrays, decisions, future_rows, scores, action_type_map, temperature=time_temperature)
            metrics.update({"train_decisions": int(len(ids)), "test_decisions": int(len(future_ids)), "test_episode_cutoff": int(cutoff)})
            legacy_comparison[name] = metrics
            del model, rows, scores
            gc.collect()
        write_metrics(reports / "legacy_vs_expanded.json", legacy_comparison)

    # Final model on every aligned legal-option decision.
    all_ids = decisions["decision_numeric_id"].to_numpy(dtype=np.int32)
    all_rows = np.arange(int(schema["row_count"]), dtype=np.int64)
    if progress:
        print(json.dumps({"event": "final_train", "decisions": len(all_ids), "rows": len(all_rows), "estimators": fixed_estimators}), flush=True)
    final_model = _fit_model(arrays, all_rows, feature_columns, fixed_estimators=fixed_estimators)
    joblib.dump(final_model, artifacts / "ranker.joblib", compress=3)
    final_model.booster_.save_model(str(artifacts / "ranker.txt"))
    _feature_importance(final_model, feature_columns).to_csv(reports / "feature_importance.csv", index=False)
    model_schema = {
        "feature_columns": feature_columns,
        "action_type_map": action_type_map,
        "category_maps": {"action_type": action_type_map},
        "categorical_features": [name for name in CATEGORICAL_NAMES if name in feature_columns],
        "temperature": float(time_temperature),
        "fallback_probability": 0.55, "fallback_margin": 0.12,
        "action_type_thresholds": action_thresholds,
        "legal_option_only": True, "n_estimators": fixed_estimators,
        "training_decisions": int(len(decisions)), "training_candidate_rows": int(schema["row_count"]),
    }
    (artifacts / "model_schema.json").write_text(json.dumps(model_schema, ensure_ascii=False, indent=2), encoding="utf-8")
    write_metrics(reports / "offline_evaluation.json", all_metrics)
    summary = {
        "splits": all_metrics, "ablations": ablations, "legacy_comparison": legacy_comparison,
        "schema": model_schema,
    }
    write_metrics(reports / "training_summary.json", summary)
    return summary
