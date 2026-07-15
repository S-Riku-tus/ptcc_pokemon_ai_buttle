from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import load_card_metadata, load_config, resolve_workspace_path
from .feature_engineering import (
    FEATURE_COLUMNS, action_type, candidate_features, exact_option_key, semantic_key,
    state_features, tactical_fingerprint,
)
from .replay_io import ReplayRef, aligned_decisions, future_log_types, load_replay


def _base_weight(rank: int, outcome: str) -> float:
    if rank <= 5:
        return 1.0 if outcome == "win" else 0.60
    if rank <= 15:
        return 0.90 if outcome == "win" else 0.50
    return 0.75 if outcome == "win" else 0.35


def _importance(selected_types: list[str], option_count: int) -> float:
    if option_count <= 1:
        return 0.25
    important = {"attack", "boss", "hammer", "xerosic", "retreat", "energy", "evolve"}
    return 1.25 if any(value in important for value in selected_types) else 1.0


def _post_quality(selected_types: list[str], log_types: set[int], has_attack_option: bool) -> float:
    quality = 1.0
    attacked = 15 in log_types
    damage_or_ko = 16 in log_types or 17 in log_types
    if attacked:
        quality += 0.10
    if damage_or_ko:
        quality += 0.10
    if "end" in selected_types and has_attack_option and not attacked:
        quality -= 0.25
    return min(1.25, max(0.60, quality))


def _assign_time_splits(decisions: pd.DataFrame, train_fraction: float, validation_fraction: float) -> pd.Series:
    episodes = sorted(decisions["episode_id"].unique())
    train_end = max(1, int(len(episodes) * train_fraction))
    validation_end = max(train_end + 1, int(len(episodes) * (train_fraction + validation_fraction)))
    mapping = {}
    for index, episode in enumerate(episodes):
        mapping[episode] = "train" if index < train_end else "validation" if index < validation_end else "test"
    return decisions["episode_id"].map(mapping)


def _assign_opponent_splits(decisions: pd.DataFrame) -> pd.Series:
    opponents = sorted(value for value in decisions["opponent"].dropna().unique() if value)
    holdout = opponents[-1] if opponents else ""
    return decisions["opponent"].map(lambda value: "test" if value == holdout else "train")


def build_dataset(
    config: dict[str, Any], processed_dir: Path, manifest: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if manifest is None:
        manifest = pd.read_csv(processed_dir / "episode_manifest.csv")
    cards, attacks = load_card_metadata()
    decisions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    fingerprint_votes: dict[str, Counter[str]] = defaultdict(Counter)
    decision_fingerprints: dict[str, str] = {}
    option_schema = Counter()

    usable = manifest[manifest["usable"].fillna(False).astype(bool)]
    for manifest_row in usable.to_dict("records"):
        ref = ReplayRef(
            resolve_workspace_path(manifest_row["source_zip"]),
            manifest_row["source_member"], int(manifest_row["episode_id"]),
            int(manifest_row["target_seat"]), str(manifest_row.get("log_acquired_at") or ""), "",
        )
        replay = load_replay(ref)
        outcome = str(manifest_row["outcome"])
        value_target = 1.0 if outcome == "win" else 0.0 if outcome == "loss" else 0.5
        for step, observation, selected_indices, _ in aligned_decisions(replay, ref.target_seat or 0, shift=1):
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            if not options:
                continue
            if not selected_indices and int(select.get("minCount") or 0) == 0:
                selected_indices = [len(options)]
                options.append({"type": "__none__"})
            decision_id = f"{ref.episode_id}:{ref.target_seat}:{step}"
            state = state_features(observation)
            identities: list[dict[str, Any]] = []
            candidate_feature_rows: list[dict[str, float]] = []
            for candidate_index, raw_option in enumerate(options):
                option = raw_option if isinstance(raw_option, dict) else {"type": str(raw_option)}
                option_schema.update(option.keys())
                features, identity = candidate_features(observation, option, candidate_index, cards, attacks)
                identities.append(identity)
                candidate_feature_rows.append(features)
            selected_keys = [semantic_key(identities[index]) for index in selected_indices]
            selected_types = [action_type(identities[index]) for index in selected_indices]
            fingerprint = tactical_fingerprint(state, identities[selected_indices[0]] if selected_indices else identities[0])
            for key in selected_keys:
                fingerprint_votes[fingerprint][key] += 1
            decision_fingerprints[decision_id] = fingerprint
            future_types = future_log_types(replay, ref.target_seat or 0, step)
            has_attack_option = any(action_type(identity) == "attack" for identity in identities)
            base = _base_weight(int(manifest_row["rank"] or 50), outcome)
            importance = _importance(selected_types, len(options))
            post_quality = _post_quality(selected_types, future_types, has_attack_option)
            preliminary = base * importance * post_quality
            preliminary = float(np.clip(
                preliminary, config["teacher_weight_min"], config["teacher_weight_max"]
            ))
            decision_row: dict[str, Any] = {
                "decision_id": decision_id,
                "episode_id": ref.episode_id,
                "step": step,
                "agent_id": ref.target_seat,
                "team": manifest_row["team"],
                "rank": manifest_row["rank"],
                "rating": manifest_row.get("rating"),
                "submission_id": manifest_row["submission_id"],
                "deck_hash": manifest_row["deck_hash"],
                "deck_type": manifest_row["deck_type"],
                "opponent": manifest_row["opponent"],
                "turn": state["turn"],
                "go_first": state["go_first"],
                "selected_candidate_ids": json.dumps(selected_indices, separators=(",", ":")),
                "selected_semantic_keys": json.dumps(selected_keys, separators=(",", ":")),
                "outcome": outcome,
                "value_target": value_target,
                "teacher_weight_preliminary": preliminary,
                "option_count": len(options),
                "unique_legal": len(options) == 1,
                "high_importance": importance > 1.0,
                "tactical_fingerprint": fingerprint,
            }
            decision_row.update(state)
            decisions.append(decision_row)
            for candidate_index, (features, identity, raw_option) in enumerate(zip(candidate_feature_rows, identities, options)):
                selected = candidate_index in selected_indices
                candidate_row: dict[str, Any] = {
                    "decision_id": decision_id,
                    "episode_id": ref.episode_id,
                    "candidate_index": candidate_index,
                    "action_type": action_type(identity),
                    "card_id": identity["card_id"],
                    "attack_id": identity["attack_id"],
                    "skill_id": identity["skill_id"],
                    "target_card_id": identity["target_card_id"],
                    "source_zone": identity["source_area"],
                    "target_zone": identity["target_area"],
                    "selected": selected,
                    "semantic_action_key": semantic_key(identity),
                    "exact_action_key": exact_option_key(raw_option if isinstance(raw_option, dict) else {"type": str(raw_option)}),
                    "teacher_weight_preliminary": preliminary,
                }
                candidate_row.update(state)
                candidate_row.update(features)
                candidates.append(candidate_row)
            weights.append({
                "decision_id": decision_id,
                "rank_outcome_weight": base,
                "data_quality_weight": 1.0,
                "importance_weight": importance,
                "post_action_quality_weight": post_quality,
                "agreement_weight": 1.0,
                "teacher_weight": preliminary,
            })

    decision_df = pd.DataFrame(decisions)
    candidate_df = pd.DataFrame(candidates)
    weight_df = pd.DataFrame(weights)
    if decision_df.empty:
        raise RuntimeError("No aligned decisions were produced")

    agreement_by_decision: dict[str, float] = {}
    soft_labels: dict[tuple[str, str], float] = {}
    for decision_id, fingerprint in decision_fingerprints.items():
        votes = fingerprint_votes[fingerprint]
        total = sum(votes.values())
        agreement = max(votes.values()) / total if total else 1.0
        agreement_by_decision[decision_id] = 0.85 + 0.30 * agreement
        for key, count in votes.items():
            soft_labels[(decision_id, key)] = count / total if total else 0.0
    weight_df["agreement_weight"] = weight_df["decision_id"].map(agreement_by_decision).fillna(1.0)
    weight_df["teacher_weight"] = (
        weight_df["rank_outcome_weight"] * weight_df["data_quality_weight"]
        * weight_df["importance_weight"] * weight_df["post_action_quality_weight"]
        * weight_df["agreement_weight"]
    ).clip(config["teacher_weight_min"], config["teacher_weight_max"])
    final_weights = dict(zip(weight_df["decision_id"], weight_df["teacher_weight"]))
    decision_df["teacher_weight"] = decision_df["decision_id"].map(final_weights)
    candidate_df["teacher_weight"] = candidate_df["decision_id"].map(final_weights)
    candidate_df["soft_label"] = [
        soft_labels.get((decision_id, key), float(selected))
        for decision_id, key, selected in zip(
            candidate_df["decision_id"], candidate_df["semantic_action_key"], candidate_df["selected"]
        )
    ]
    decision_df["split_time"] = _assign_time_splits(
        decision_df, config["time_train_fraction"], config["time_validation_fraction"]
    )
    split_map = dict(zip(decision_df["decision_id"], decision_df["split_time"]))
    candidate_df["split_time"] = candidate_df["decision_id"].map(split_map)
    weight_df["split_time"] = weight_df["decision_id"].map(split_map)
    decision_df["split_opponent"] = _assign_opponent_splits(decision_df)
    opponent_map = dict(zip(decision_df["decision_id"], decision_df["split_opponent"]))
    candidate_df["split_opponent"] = candidate_df["decision_id"].map(opponent_map)
    decision_df["split_team"] = "not_available_single_teacher"
    decision_df["split_submission"] = "not_available_single_submission"
    decision_df["split_deck"] = "not_available_single_variant"
    for column in ("split_team", "split_submission", "split_deck"):
        mapping = dict(zip(decision_df["decision_id"], decision_df[column]))
        candidate_df[column] = candidate_df["decision_id"].map(mapping)

    processed_dir.mkdir(parents=True, exist_ok=True)
    decision_df.to_parquet(processed_dir / "decision_dataset.parquet", index=False)
    candidate_df.to_parquet(processed_dir / "legal_candidate_dataset.parquet", index=False)
    weight_df.to_parquet(processed_dir / "expert_weights.parquet", index=False)
    stats = {
        "decision_count": len(decision_df),
        "candidate_count": len(candidate_df),
        "episode_count": int(decision_df["episode_id"].nunique()),
        "selected_candidate_count": int(candidate_df["selected"].sum()),
        "unique_legal_decisions": int(decision_df["unique_legal"].sum()),
        "split_counts": decision_df["split_time"].value_counts().to_dict(),
        "outcome_counts": decision_df.groupby("outcome")["episode_id"].nunique().to_dict(),
        "option_schema": dict(option_schema),
        "feature_columns": FEATURE_COLUMNS,
        "leakage_guards": {
            "opponent_hand_cards_read": False,
            "visualize_frames_used_for_features": False,
            "future_logs_used_as_features": False,
            "outcome_used_as_policy_feature": False,
            "split_unit": "episode",
        },
    }
    (processed_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return decision_df, candidate_df, weight_df, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--processed", default=str(Path(__file__).resolve().parents[1] / "data_processed"))
    args = parser.parse_args()
    _, _, _, stats = build_dataset(load_config(args.config), Path(args.processed))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
