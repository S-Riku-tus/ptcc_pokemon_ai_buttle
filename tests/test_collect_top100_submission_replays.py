from __future__ import annotations

from scripts.collect_top100_submission_replays import (
    merge_existing_collection_indexes,
    merge_rows_by_key,
    merge_submission_rows,
    select_requested_submissions,
    write_csv,
)


def test_merge_rows_by_key_normalizes_numeric_keys() -> None:
    merged = merge_rows_by_key(
        [{"submission_id": "10", "status": "old"}],
        [{"submission_id": 10, "status": "new"}],
        ("submission_id",),
    )

    assert merged == [{"submission_id": 10, "status": "new"}]


def test_merge_submission_rows_keeps_success_on_transient_failure() -> None:
    merged = merge_submission_rows(
        [{"submission_id": "10", "status": "success"}],
        [{"submission_id": 10, "status": "failed"}],
    )

    assert merged == [{"submission_id": "10", "status": "success"}]


def test_select_requested_submissions_restricts_and_validates_ids() -> None:
    submissions = [{"submission_id": "10"}, {"submission_id": "20"}]

    assert select_requested_submissions(submissions, {20}) == [
        {"submission_id": "20"}
    ]

    try:
        select_requested_submissions(submissions, {30})
    except ValueError as exc:
        assert "[30]" in str(exc)
    else:
        raise AssertionError("missing submission ID was not rejected")


def test_merge_existing_collection_indexes_appends_and_deduplicates(tmp_path) -> None:
    indexes_dir = tmp_path / "indexes"
    indexes_dir.mkdir()

    write_csv(
        indexes_dir / "submissions.csv",
        [
            {"submission_id": "10", "status": "success"},
            {"submission_id": "20", "status": "success"},
        ],
        ["submission_id", "status"],
    )
    write_csv(
        indexes_dir / "episodes.csv",
        [
            {
                "submission_id": "10",
                "episode_id": "100",
                "download_status": "old",
            },
            {
                "submission_id": "20",
                "episode_id": "200",
                "download_status": "old",
            },
        ],
        ["submission_id", "episode_id", "download_status"],
    )
    write_csv(
        indexes_dir / "replay_index.csv",
        [
            {
                "submission_id": "10",
                "episode_id": "100",
                "download_status": "old",
            }
        ],
        ["submission_id", "episode_id", "download_status"],
    )
    write_csv(
        indexes_dir / "failures.csv",
        [
            {
                "scope": "submission",
                "submission_id": "10",
                "episode_id": "",
                "error": "keep me",
            },
            {
                "scope": "episode",
                "submission_id": "20",
                "episode_id": "200",
                "error": "stale",
            },
        ],
        ["scope", "submission_id", "episode_id", "error"],
    )

    submissions, episodes, replay_index, failures, details = (
        merge_existing_collection_indexes(
            indexes_dir,
            submission_rows=[{"submission_id": 20, "status": "success"}],
            episode_rows=[
                {
                    "submission_id": 20,
                    "episode_id": 200,
                    "download_status": "new",
                },
                {
                    "submission_id": 20,
                    "episode_id": 201,
                    "download_status": "new",
                },
            ],
            replay_index_rows=[
                {
                    "submission_id": 20,
                    "episode_id": 201,
                    "download_status": "new",
                }
            ],
            failure_rows=[],
            touched_submission_ids={20},
        )
    )

    assert {str(row["submission_id"]) for row in submissions} == {"10", "20"}
    assert {
        (str(row["submission_id"]), str(row["episode_id"])) for row in episodes
    } == {("10", "100"), ("20", "200"), ("20", "201")}
    refreshed = next(
        row
        for row in episodes
        if str(row["submission_id"]) == "20" and str(row["episode_id"]) == "200"
    )
    assert refreshed["download_status"] == "new"
    assert {
        (str(row["submission_id"]), str(row["episode_id"]))
        for row in replay_index
    } == {("10", "100"), ("20", "201")}
    assert [row["error"] for row in failures] == ["keep me"]
    assert details == {
        "existing_submission_rows": 2,
        "existing_episode_rows": 2,
        "incoming_submission_rows": 1,
        "incoming_episode_rows": 2,
        "merged_submission_rows": 2,
        "merged_episode_rows": 3,
    }
