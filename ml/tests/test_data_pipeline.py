from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from ml.core.features import LEAKAGE_DENYLIST, state_features
from ml.core.replay_io import replay_refs


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "ml" / "alakazam" / "processed"


def test_manifest_recovers_plural_and_singular_replay_layouts():
    stats = json.loads((PROCESSED / "manifest_stats.json").read_text())
    assert stats["legacy_singular_replay_count"] == 164
    assert stats["plural_replays_recovered"] >= 1894
    assert stats["full_replay_count"] >= 2058
    assert stats["replay_layouts_supported"] == ["replay", "replays"]


def test_replay_refs_accepts_both_layouts(tmp_path):
    path = tmp_path / "both.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a/replay/episode_12345678.json", "{}")
        archive.writestr("b/replays/episode_87654321.json", "{}")
    refs = replay_refs(path)
    assert {(ref.episode_id, ref.path_variant) for ref in refs} == {
        (12345678, "replay"), (87654321, "replays")
    }


def test_manifest_seats_and_decks_are_auditable():
    manifest = pd.read_csv(PROCESSED / "episode_manifest.csv")
    usable = manifest[manifest["usable_manifest"] == True]
    assert len(usable) >= 2074
    assert usable["target_seat"].isin([0, 1]).all()
    assert usable["seat_confidence"].ge(0.88).all()
    assert usable["deck_size"].eq(60).all()
    assert usable["alakazam_line"].astype(bool).all()
    assert usable["deck_hash"].nunique() == 8
    assert usable["target_team"].nunique() >= 19


def test_dataset_alignment_and_legal_candidate_integrity():
    stats = json.loads((PROCESSED / "dataset_stats.json").read_text())
    decisions = pd.read_csv(PROCESSED / "decisions.csv")
    assert stats["usable_decision_count"] >= 95254
    assert stats["candidate_row_count"] >= 1097481
    assert stats["unresolved_decision_count"] == 1
    assert stats["alignment_rate"] > 0.9999
    assert len(decisions) >= 95254
    assert decisions["decision_id"].is_unique
    assert decisions["candidate_count"].ge(2).all()


def test_all_holdout_dimensions_are_available():
    decisions = pd.read_csv(PROCESSED / "decisions.csv")
    for column in ("split_time", "split_team", "split_submission", "split_deck"):
        assert column in decisions
        assert {"train", "test"}.issubset(set(decisions[column]))
    assert decisions["target_team"].nunique() > 1
    assert decisions["submission_id"].nunique() > 1
    assert decisions["deck_hash"].nunique() > 1


def test_policy_feature_names_are_leakage_safe():
    stats = json.loads((PROCESSED / "dataset_stats.json").read_text())
    assert not (set(stats["feature_columns"]) & LEAKAGE_DENYLIST)
    assert stats["policy_feature_provenance"] == "observation_t_and_legal_option_only"
    assert stats["label_provenance"] == "same_seat_action_from_replay_step_t_plus_1"


def test_opponent_private_hand_ids_do_not_change_features():
    current = {
        "yourIndex": 0, "firstPlayer": 0, "turn": 2,
        "players": [
            {"hand": [{"id": 741}], "handCount": 1, "active": [], "bench": [], "prize": [], "deckCount": 40},
            {"hand": [{"id": 999999}], "handCount": 7, "active": [], "bench": [], "prize": [], "deckCount": 40},
        ],
    }
    changed = json.loads(json.dumps(current))
    changed["players"][1]["hand"] = [{"id": 123456}, {"id": 654321}]
    assert state_features(current) == state_features(changed)


def test_compressed_candidate_dataset_is_present_and_nonempty():
    path = PROCESSED / "dataset_rows.csv.gz"
    assert path.stat().st_size > 1_000_000
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline()
        first = handle.readline()
    assert "decision_id" in header and "label" in header
    assert first
