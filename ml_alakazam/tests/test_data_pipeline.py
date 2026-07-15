from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ml_alakazam.src.common import deck_hash
from ml_alakazam.src.feature_engineering import state_features


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data_processed"


def test_usable_manifest_is_unique_complete_and_aligned():
    manifest = pd.read_csv(PROCESSED / "episode_manifest.csv")
    usable = manifest[manifest["usable"] == True]
    assert len(usable) == 162
    assert usable["episode_id"].is_unique
    assert (usable["normal_end"] == True).all()
    assert not usable["timeout"].astype(bool).any()
    assert (usable["decision_count"] > 0).all()


def test_timeout_duplicate_and_missing_replay_are_excluded():
    manifest = pd.read_csv(PROCESSED / "episode_manifest.csv")
    assert not manifest.loc[manifest["timeout"] == True, "usable"].astype(bool).any()
    assert not manifest.loc[manifest["duplicate"] == True, "usable"].astype(bool).any()
    assert not manifest.loc[manifest["replay_available"] == False, "usable"].astype(bool).any()


def test_deck_hashes_match_canonical_deck():
    manifest = pd.read_csv(PROCESSED / "episode_manifest.csv")
    for row in manifest[manifest["usable"] == True].to_dict("records"):
        assert deck_hash(json.loads(row["deck_60"])) == row["deck_hash"]


def test_each_decision_has_a_selected_legal_candidate():
    decisions = pd.read_parquet(PROCESSED / "decision_dataset.parquet")
    candidates = pd.read_parquet(PROCESSED / "legal_candidate_dataset.parquet")
    selected = candidates.groupby("decision_id")["selected"].sum()
    assert len(decisions) == 11438
    assert (selected >= 1).all()
    assert set(selected.index) == set(decisions["decision_id"])


def test_episode_split_has_no_leakage():
    decisions = pd.read_parquet(PROCESSED / "decision_dataset.parquet")
    assert decisions.groupby("episode_id")["split_time"].nunique().max() == 1
    assert not decisions["decision_id"].duplicated().any()


def test_opponent_private_hand_cards_never_change_features():
    base = {
        "current": {
            "yourIndex": 0,
            "firstPlayer": 0,
            "turn": 2,
            "players": [
                {"hand": [{"id": 741}], "handCount": 1, "active": [], "bench": [], "prize": [], "deckCount": 40},
                {"hand": [{"id": 999999}], "handCount": 7, "active": [], "bench": [], "prize": [], "deckCount": 40},
            ],
        },
        "select": {"type": 0, "context": 0, "option": [{"type": 14}], "minCount": 1, "maxCount": 1},
    }
    changed = json.loads(json.dumps(base))
    changed["current"]["players"][1]["hand"] = [{"id": 123456}, {"id": 654321}]
    assert state_features(base) == state_features(changed)


def test_parquet_artifacts_are_readable_and_nonempty():
    for name in ("decision_dataset.parquet", "legal_candidate_dataset.parquet", "expert_weights.parquet"):
        path = PROCESSED / name
        assert path.stat().st_size > 0
        assert not pd.read_parquet(path).empty

