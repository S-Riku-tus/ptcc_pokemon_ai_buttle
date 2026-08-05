from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_grimmsnarl_v2_corpus import (
    LEGACY_DOWNLOAD_STATUSES,
    select_training_index,
    write_selection_manifest,
)
from scripts.train_grimmsnarl_v2_teacher import Corpus


DECK = "same-deck"


def row(
    *,
    team: int,
    submission: int,
    episode: int,
    status: str = "success",
    deck: str = DECK,
    seat: int = 0,
) -> dict[str, object]:
    return {
        "download_status": status,
        "deck_hash": deck,
        "team_id": team,
        "submission_id": submission,
        "episode_id": episode,
        "seat_index": seat,
    }


def select(index: pd.DataFrame, **overrides) -> pd.DataFrame:
    options = {
        "deck_hash_value": DECK,
        "excluded_teams": set(),
        "accepted_download_statuses": set(LEGACY_DOWNLOAD_STATUSES),
        "limit_per_team": 0,
        "latest_per_team": 0,
    }
    options.update(overrides)
    return select_training_index(index, **options)


def test_default_selection_preserves_legacy_success_only_behavior() -> None:
    index = pd.DataFrame(
        [
            row(team=1, submission=10, episode=3, status="skipped_existing"),
            row(team=1, submission=10, episode=2),
            row(team=2, submission=20, episode=1, deck="other"),
            row(team=3, submission=30, episode=4),
        ]
    )

    selected = select(index, excluded_teams={3})

    assert selected["episode_id"].tolist() == [2]


def test_validated_cache_and_latest_cap_are_opt_in() -> None:
    index = pd.DataFrame(
        [
            row(team=1, submission=10, episode=1),
            row(team=1, submission=10, episode=2),
            row(team=1, submission=11, episode=3, status="skipped_existing"),
            row(team=2, submission=20, episode=4),
            row(team=2, submission=20, episode=5),
            # Duplicate relation must not consume a slot in the team cap.
            row(team=2, submission=20, episode=5),
        ]
    )

    selected = select(
        index,
        accepted_download_statuses={"success", "skipped_existing"},
        latest_per_team=2,
    )

    assert selected[["team_id", "episode_id"]].values.tolist() == [
        [1, 2],
        [1, 3],
        [2, 4],
        [2, 5],
    ]


def test_oldest_and_latest_caps_cannot_be_combined() -> None:
    index = pd.DataFrame([row(team=1, submission=10, episode=1)])

    with pytest.raises(ValueError, match="exclusive"):
        select(index, limit_per_team=1, latest_per_team=1)


def test_selection_manifest_is_exact_and_hashed(tmp_path) -> None:
    selected = pd.DataFrame(
        [
            row(team=1, submission=10, episode=2),
            row(team=2, submission=20, episode=3),
        ]
    )
    path = tmp_path / "selection.csv"

    digest = write_selection_manifest(selected, path)

    restored = pd.read_csv(path)
    pd.testing.assert_frame_equal(restored, selected)
    assert len(digest) == 64
    assert digest == write_selection_manifest(selected, path)


def test_trainer_loads_legacy_corpus_without_submission_ids(tmp_path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        features=np.zeros((2, 1), dtype=np.float32),
        labels=np.asarray([1, 0], dtype=np.int8),
        groups=np.asarray([2], dtype=np.int32),
        splits=np.asarray(["train"]),
        episode_ids=np.asarray([100], dtype=np.int64),
        team_ids=np.asarray([10], dtype=np.int64),
        turns=np.asarray([1], dtype=np.int16),
        contexts=np.asarray([0], dtype=np.int16),
        won=np.asarray([1], dtype=np.int8),
        teacher_action_types=np.asarray([0], dtype=np.int16),
        feature_names=np.asarray(["feature"]),
        categorical=np.asarray([], dtype=str),
    )

    corpus = Corpus(path)

    assert corpus.submission_ids.tolist() == [-1]
