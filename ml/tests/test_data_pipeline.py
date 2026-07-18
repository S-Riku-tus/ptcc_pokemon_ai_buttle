from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from ml.core.features import LEAKAGE_DENYLIST, option_features, state_features
from ml.core.manifest import _deduplicate_usable_trajectories, _resolved_target_team
from ml.core.replay_io import replay_refs, zip_metadata


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


def test_full_bundle_metadata_recovers_team_rank_and_submission_seat(tmp_path):
    path = tmp_path / "rank03_example_sub54762014_full.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle/submission.json", json.dumps({
            "submission_id": 54762014,
            "leaderboard_rank": "3",
            "team_name": "Expert Team",
        }))
        archive.writestr("bundle/episodes.json", json.dumps({
            "episodes": [{
                "episode_id": 86306463,
                "agent_0_submission_id": "11111111",
                "agent_1_submission_id": "54762014",
            }]
        }))
        archive.writestr(
            "bundle/manifest.csv",
            "episode_id,detected_submission_agent_index\n86306463,\n",
        )
    meta = zip_metadata(path)
    assert meta["submission_id"] == 54762014
    assert meta["rank"] == 3
    assert meta["team_name"] == "Expert Team"
    assert meta["source_manifest"][86306463]["detected_submission_agent_index"] == "1"


def test_combined_rank_zip_resolves_metadata_per_submission_bundle(tmp_path):
    path = tmp_path / "rank23_combined_full.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for bundle, submission_id, episode_id, team_name, seat in (
            ("rank23_a_sub54770001", 54770001, 86300001, "Team A", 0),
            ("rank23_b_sub54770002", 54770002, 86300002, "Team B", 1),
        ):
            archive.writestr(f"{bundle}/submission.json", json.dumps({
                "submission_id": submission_id,
                "leaderboard_rank": "23",
                "team_name": team_name,
            }))
            archive.writestr(
                f"{bundle}/manifest.csv",
                "episode_id,detected_submission_agent_index\n"
                f"{episode_id},{seat}\n",
            )
            archive.writestr(
                f"{bundle}/episodes/{episode_id}/replay/episode_{episode_id}.json",
                "{}",
            )

    refs = replay_refs(path)
    metadata = {ref.episode_id: zip_metadata(path, ref.member) for ref in refs}
    assert metadata[86300001]["submission_id"] == 54770001
    assert metadata[86300001]["team_name"] == "Team A"
    assert metadata[86300001]["source_manifest"][86300001]["detected_submission_agent_index"] == "0"
    assert metadata[86300002]["submission_id"] == 54770002
    assert metadata[86300002]["team_name"] == "Team B"
    assert metadata[86300002]["source_manifest"][86300002]["detected_submission_agent_index"] == "1"


def test_duplicate_trajectory_rows_prefer_stronger_seat_evidence():
    frame = pd.DataFrame([
        {
            "trajectory_id": "54762014:86306463:1", "usable_manifest": True,
            "seat_confidence": 0.995, "zip_name": "older.zip",
            "episode_id": 86306463, "target_seat": 1,
        },
        {
            "trajectory_id": "54762014:86306463:1", "usable_manifest": True,
            "seat_confidence": 1.0, "zip_name": "full.zip",
            "episode_id": 86306463, "target_seat": 1,
        },
    ])
    deduplicated, removed = _deduplicate_usable_trajectories(frame)
    assert removed == 1
    assert len(deduplicated) == 1
    assert deduplicated.iloc[0]["zip_name"] == "full.zip"


def test_target_team_falls_back_to_observed_seat_name():
    assert _resolved_target_team("", ["opponent", "v5 teacher"], 1) == "v5 teacher"
    assert _resolved_target_team("configured", ["opponent", "observed"], 1) == "configured"


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


def test_attack_continuity_features_separate_active_from_backup_attacker():
    current = {
        "yourIndex": 0,
        "firstPlayer": 0,
        "turn": 6,
        "players": [
            {
                "hand": [], "handCount": 0, "prize": [], "deckCount": 30,
                "active": [{"id": 140, "energies": []}],
                "bench": [{"id": 743, "energies": [5]}],
            },
            {
                "hand": None, "handCount": 5, "prize": [], "deckCount": 30,
                "active": [{"id": 140, "energies": []}], "bench": [],
            },
        ],
    }
    state = state_features(current)
    assert state["has_ready_alakazam"] == 1
    assert state["has_ready_active_alakazam"] == 0
    assert state["has_ready_backup_alakazam"] == 1
    assert state["active_is_fezandipiti"] == 1
    assert state["active_is_setup_wall"] == 1

    retreat = option_features(current, {"type": 0, "context": 0}, {"type": 12})
    end = option_features(current, {"type": 0, "context": 0}, {"type": 14})
    assert retreat["retreat_to_ready_backup_value"] == 1
    assert end["end_ready_active_alakazam_penalty"] == 0


def test_compressed_candidate_dataset_is_present_and_nonempty():
    path = PROCESSED / "dataset_rows.csv.gz"
    assert path.stat().st_size > 1_000_000
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline()
        first = handle.readline()
    assert "decision_id" in header and "label" in header
    assert first
