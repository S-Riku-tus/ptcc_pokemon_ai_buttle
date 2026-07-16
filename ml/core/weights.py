from __future__ import annotations

import math

import pandas as pd

FOCUS_TYPES = {"boss", "energy", "retreat", "xerosic", "hammer"}


def rank_weight(rank: float) -> float:
    # Rank is useful teacher-quality metadata, but should not overpower the
    # state/action signal. Range: about 0.90..1.10.
    return 0.90 + 0.20 * math.exp(-(max(1.0, rank) - 1.0) / 20.0)


def deck_weight(distance: float) -> float:
    # Keep distant Alakazam variants useful while modestly preferring the
    # Majkel reference. Range: about 0.80..1.00.
    return 0.80 + 0.20 * math.exp(-max(0.0, distance) / 8.0)


def outcome_weight(win: bool, loss: bool) -> float:
    return 1.05 if win else (0.95 if loss else 0.90)


def add_decision_weights(decisions: pd.DataFrame) -> pd.DataFrame:
    decisions = decisions.copy()
    decisions["rank_weight"] = decisions["rank"].astype(float).map(rank_weight)
    decisions["deck_weight"] = decisions["majkel_distance"].fillna(20).astype(float).map(deck_weight)
    decisions["outcome_weight"] = [outcome_weight(bool(w), bool(l)) for w, l in zip(decisions["target_win"], decisions["target_loss"])]
    decisions["action_balance_weight"] = decisions["selected_action_type"].map(lambda x: 1.08 if x in FOCUS_TYPES else 1.0)
    raw = (
        decisions["rank_weight"] * decisions["deck_weight"] * decisions["outcome_weight"]
        * decisions["seat_confidence"].astype(float) * decisions["alignment_confidence"].astype(float)
        * decisions["action_balance_weight"]
    ).clip(0.65, 1.35)
    decisions["sample_weight"] = raw / max(float(raw.mean()), 1e-8)
    return decisions


def add_training_weights(rows: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = add_decision_weights(decisions)
    weight_map = decisions.set_index("decision_id")["sample_weight"]
    rows = rows.copy()
    rows["sample_weight"] = rows["decision_id"].map(weight_map).astype(float)
    return rows, decisions
