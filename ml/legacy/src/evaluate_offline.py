from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -50, 50))
    return exp / max(exp.sum(), 1e-12)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    return float(sum(value * weight for value, weight in zip(values, weights)) / total) if total else 0.0


def evaluate_scores(
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    scores: Iterable[float],
    confidence_threshold: float = 0.58,
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = candidates.copy()
    frame["score"] = np.asarray(list(scores), dtype=np.float64)
    decision_meta = decisions.set_index("decision_id")
    records: list[dict[str, Any]] = []
    for decision_id, group in frame.groupby("decision_id", sort=False):
        group = group.sort_values("candidate_index", kind="stable")
        group_scores = group["score"].to_numpy(dtype=np.float64)
        order = np.argsort(-group_scores, kind="stable")
        probabilities = softmax(group_scores)
        selected_positions = np.flatnonzero(group["selected"].to_numpy(dtype=bool))
        selected_semantic = set(group.iloc[selected_positions]["semantic_action_key"])
        predicted = group.iloc[int(order[0])]
        semantic_hits = [
            group.iloc[int(position)]["semantic_action_key"] in selected_semantic for position in order
        ]
        exact_top1 = bool(predicted["selected"])
        semantic_top1 = bool(semantic_hits[0])
        first_hit = next((index + 1 for index, hit in enumerate(semantic_hits) if hit), len(group) + 1)
        target = np.zeros(len(group), dtype=np.float64)
        if len(selected_positions):
            target[selected_positions] = 1.0 / len(selected_positions)
        log_loss = -float(np.sum(target * np.log(np.clip(probabilities, 1e-12, 1.0))))
        meta = decision_meta.loc[decision_id]
        selected_type = str(group.iloc[int(selected_positions[0])]["action_type"]) if len(selected_positions) else "none"
        record = {
            "decision_id": decision_id,
            "episode_id": int(group.iloc[0]["episode_id"]),
            "exact_top1": exact_top1,
            "semantic_top1": semantic_top1,
            "top3": any(semantic_hits[:3]),
            "top5": any(semantic_hits[:5]),
            "mrr": 1.0 / first_hit,
            "log_loss": log_loss,
            "confidence": float(probabilities[int(order[0])]),
            "low_confidence": float(probabilities[int(order[0])]) < confidence_threshold,
            "selected_action_type": selected_type,
            "predicted_action_type": str(predicted["action_type"]),
            "teacher_weight": float(meta.get("teacher_weight", 1.0)),
            "unique_legal": bool(meta.get("unique_legal", False)),
            "high_importance": bool(meta.get("high_importance", False)),
            "opponent": str(meta.get("opponent", "unknown")),
            "go_first": bool(meta.get("go_first", False)),
            "outcome": str(meta.get("outcome", "unknown")),
            "rank": int(meta.get("rank", 0) or 0),
            "predicted_candidate_index": int(predicted["candidate_index"]),
            "selected_candidate_indices": json.dumps(group.iloc[selected_positions]["candidate_index"].astype(int).tolist()),
        }
        records.append(record)
    result_frame = pd.DataFrame(records)
    weights = result_frame["teacher_weight"].tolist()
    metrics = {
        "decisions": len(result_frame),
        "exact_top1": _weighted_mean(result_frame["exact_top1"].astype(float).tolist(), weights),
        "semantic_top1": _weighted_mean(result_frame["semantic_top1"].astype(float).tolist(), weights),
        "top3": _weighted_mean(result_frame["top3"].astype(float).tolist(), weights),
        "top5": _weighted_mean(result_frame["top5"].astype(float).tolist(), weights),
        "mrr": _weighted_mean(result_frame["mrr"].tolist(), weights),
        "weighted_log_loss": _weighted_mean(result_frame["log_loss"].tolist(), weights),
        "mean_confidence": _weighted_mean(result_frame["confidence"].tolist(), weights),
        "low_confidence_rate": _weighted_mean(result_frame["low_confidence"].astype(float).tolist(), weights),
    }
    non_unique = result_frame[~result_frame["unique_legal"]]
    metrics["semantic_top1_without_unique"] = (
        np.average(non_unique["semantic_top1"], weights=non_unique["teacher_weight"])
        if len(non_unique) else 0.0
    )
    calibration_bins = []
    ece = 0.0
    total_weight = result_frame["teacher_weight"].sum()
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (result_frame["confidence"] >= lower) & (
            result_frame["confidence"] < upper if upper < 1.0 else result_frame["confidence"] <= upper
        )
        bucket = result_frame[mask]
        if bucket.empty:
            continue
        bucket_weight = bucket["teacher_weight"].sum()
        accuracy = float(np.average(bucket["semantic_top1"], weights=bucket["teacher_weight"]))
        confidence = float(np.average(bucket["confidence"], weights=bucket["teacher_weight"]))
        ece += bucket_weight / max(total_weight, 1e-12) * abs(accuracy - confidence)
        calibration_bins.append({
            "lower": round(float(lower), 1), "upper": round(float(upper), 1),
            "count": len(bucket), "accuracy": accuracy, "confidence": confidence,
        })
    metrics["ece"] = float(ece)
    metrics["calibration_bins"] = calibration_bins

    def breakdown(column: str) -> dict[str, Any]:
        output = {}
        for value, group in result_frame.groupby(column, dropna=False):
            output[str(value)] = {
                "count": len(group),
                "semantic_top1": float(np.average(group["semantic_top1"], weights=group["teacher_weight"])),
                "top3": float(np.average(group["top3"], weights=group["teacher_weight"])),
            }
        return output

    metrics["by_action_type"] = breakdown("selected_action_type")
    metrics["by_matchup"] = breakdown("opponent")
    metrics["by_go_first"] = breakdown("go_first")
    metrics["by_outcome"] = breakdown("outcome")
    metrics["by_rank"] = breakdown("rank")
    metrics["by_importance"] = breakdown("high_importance")
    return metrics, result_frame


def frequency_scores(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    selected = train[train["selected"]]
    key_counts = selected["semantic_action_key"].value_counts().to_dict()
    context_counts = selected["context_code"].value_counts().to_dict()
    return np.asarray([
        math.log1p(key_counts.get(key, 0)) + 0.05 * math.log1p(context_counts.get(context, 0))
        for key, context in zip(target["semantic_action_key"], target["context_code"])
    ])


def handwritten_scores(frame: pd.DataFrame) -> np.ndarray:
    score = np.zeros(len(frame), dtype=np.float64)
    score += frame["ko_possible"].to_numpy() * 100.0
    score += frame["is_attack"].to_numpy() * 28.0
    score += frame["is_evolve"].to_numpy() * 15.0
    score += frame["is_energy"].to_numpy() * 9.0
    score += frame["is_ability"].to_numpy() * 7.0
    score += frame["is_boss"].to_numpy() * 12.0
    score += frame["is_hammer"].to_numpy() * 8.0
    score -= frame["is_end"].to_numpy() * 8.0
    score -= frame["candidate_index"].to_numpy() * 1e-3
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="CSV with decision_id,candidate_index,score")
    parser.add_argument("--processed", default=str(Path(__file__).resolve().parents[1] / "data_processed"))
    args = parser.parse_args()
    processed = Path(args.processed)
    candidates = pd.read_parquet(processed / "legal_candidate_dataset.parquet")
    decisions = pd.read_parquet(processed / "decision_dataset.parquet")
    scores = pd.read_csv(args.scores)
    merged = candidates.merge(scores, on=["decision_id", "candidate_index"], how="inner")
    metrics, _ = evaluate_scores(merged, decisions, merged["score"])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

