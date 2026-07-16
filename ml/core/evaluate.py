from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -50, 50))
    return exp / max(exp.sum(), 1e-12)


def fit_temperature(frame: pd.DataFrame, scores: np.ndarray) -> float:
    """Fit scalar temperature on a held-out calibration set by grouped NLL."""
    work = frame[["decision_id", "label"]].copy()
    work["score"] = np.asarray(scores, dtype=float)
    groups: list[tuple[np.ndarray, int]] = []
    for _, group in work.groupby("decision_id", sort=False, observed=True):
        labels = group["label"].to_numpy(int)
        positives = np.flatnonzero(labels == 1)
        if len(positives) == 1:
            groups.append((group["score"].to_numpy(float), int(positives[0])))
    if not groups:
        return 1.0
    best_t, best_loss = 1.0, float("inf")
    for temperature in np.geomspace(0.20, 5.0, 61):
        loss = 0.0
        for values, true_index in groups:
            probs = _softmax(values / temperature)
            loss -= float(np.log(max(probs[true_index], 1e-12)))
        loss /= len(groups)
        if loss < best_loss:
            best_loss, best_t = loss, float(temperature)
    return best_t


def evaluate_scores(
    frame: pd.DataFrame,
    scores: np.ndarray,
    fallback_probability: float = 0.55,
    fallback_margin: float = 0.12,
    temperature: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = ["decision_id", "label", "candidate_index"]
    if "selected_action_type" in frame.columns:
        columns.append("selected_action_type")
    if "action_type" in frame.columns:
        columns.append("action_type")
    work = frame[columns].copy()
    work["score"] = np.asarray(scores, dtype=float)
    records = []
    for decision_id, group in work.groupby("decision_id", sort=False, observed=True):
        group = group.sort_values("score", ascending=False).reset_index(drop=True)
        true_positions = np.flatnonzero(group["label"].to_numpy() == 1)
        if len(true_positions) != 1:
            continue
        rank = int(true_positions[0]) + 1
        probs = _softmax(group["score"].to_numpy(float) / max(float(temperature), 1e-6))
        confidence = float(probs[0])
        margin = float(probs[0] - probs[1]) if len(probs) > 1 else 1.0
        correct = rank == 1
        fallback = confidence < fallback_probability or margin < fallback_margin
        true_row = group.loc[true_positions[0]]
        records.append({
            "decision_id": decision_id,
            "rank": rank,
            "correct": correct,
            "top3": rank <= 3,
            "reciprocal_rank": 1.0 / rank,
            "confidence": confidence,
            "margin": margin,
            "fallback": fallback,
            "selected_action_type": str(true_row.get("selected_action_type", true_row.get("action_type", "unknown"))),
            "predicted_action_type": str(group.loc[0, "action_type"]) if "action_type" in group.columns else "unknown",
        })
    per_decision = pd.DataFrame(records)
    if per_decision.empty:
        return {"decision_count": 0}, per_decision
    accepted = per_decision[~per_decision["fallback"]]
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        bucket = per_decision[(per_decision["confidence"] >= lo) & (per_decision["confidence"] < hi)]
        if len(bucket):
            ece += len(bucket) / len(per_decision) * abs(bucket["correct"].mean() - bucket["confidence"].mean())
    action_metrics = {}
    for action, group in per_decision.groupby("selected_action_type", observed=True):
        action_metrics[action] = {
            "count": int(len(group)),
            "top1": float(group["correct"].mean()),
            "top3": float(group["top3"].mean()),
            "mrr": float(group["reciprocal_rank"].mean()),
            "fallback_rate": float(group["fallback"].mean()),
            "accepted_top1": float(group.loc[~group["fallback"], "correct"].mean()) if (~group["fallback"]).any() else None,
        }
    metrics = {
        "decision_count": int(len(per_decision)),
        "top1": float(per_decision["correct"].mean()),
        "top3": float(per_decision["top3"].mean()),
        "mrr": float(per_decision["reciprocal_rank"].mean()),
        "ece": float(ece),
        "temperature": float(temperature),
        "fallback_probability": float(fallback_probability),
        "fallback_margin": float(fallback_margin),
        "fallback_rate": float(per_decision["fallback"].mean()),
        "accepted_top1": float(accepted["correct"].mean()) if len(accepted) else None,
        "accepted_count": int(len(accepted)),
        # The model only ranks options emitted by the engine, so illegal actions
        # are structurally impossible in offline scoring and the hybrid wrapper.
        "illegal_action_count": 0,
        "action_type_metrics": action_metrics,
    }
    return metrics, per_decision


def calibrate_action_thresholds(per_decision: pd.DataFrame, target_accuracy: float = 0.75, min_samples: int = 25) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    if per_decision.empty or "predicted_action_type" not in per_decision:
        return thresholds
    for action, group in per_decision.groupby("predicted_action_type", observed=True):
        if len(group) < min_samples:
            continue
        chosen = 0.55
        for threshold in np.arange(0.35, 0.91, 0.02):
            accepted = group[group["confidence"] >= threshold]
            if len(accepted) >= max(10, int(0.2 * len(group))) and accepted["correct"].mean() >= target_accuracy:
                chosen = float(round(threshold, 2))
                break
        thresholds[str(action)] = chosen
    return thresholds


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
