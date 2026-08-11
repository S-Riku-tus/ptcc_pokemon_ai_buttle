"""Freeze the refreshed high-rated Grimmsnarl replays for v19 training.

The public log fetcher stores one replay below each episode directory.  This
script converts those three isolated run directories into the immutable
selection manifest consumed by ``build_grimmsnarl_v2_corpus.py``.  No network
access is performed and no hidden game information is added to the features.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    55138264: (16422241, "Sixth Sense", 1113.7),
    55177269: (16452116, "Raihan", 1151.0),
    55187358: (16561259, "kd", 1116.3),
}
DECK_HASH = "9714ab5c3996f6cc"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for submission_id, (team_id, team_name, score) in SOURCES.items():
        run = (
            ROOT / "data" / "runs" / "grimmsnarl"
            / f"v18_teacher_{submission_id}"
        )
        with (run / "episodes.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            episodes = {
                int(row["episode_id"]): row for row in csv.DictReader(handle)
            }
        with (run / "manifest.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            manifest = list(csv.DictReader(handle))

        for item in manifest:
            episode_id = int(item["episode_id"])
            episode = episodes.get(episode_id)
            if episode is None or episode.get("state") != "COMPLETED":
                continue
            seat = int(item["detected_submission_agent_index"])
            replay = (
                run / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            ).resolve()
            if not replay.is_file():
                continue
            rows.append({
                "leaderboard_rank": "",
                "team_id": team_id,
                "team_name": team_name,
                "submission_id": submission_id,
                "submission_score": score,
                "leaderboard_submission_id": submission_id,
                "episode_id": episode_id,
                "episode_state": episode["state"],
                "episode_type": episode["episode_type"],
                "created_at": episode["create_time"],
                "ended_at": episode["end_time"],
                "seat_index": seat,
                "opponent_submission_id": episode[f"agent_{1-seat}_submission_id"],
                "opponent_team_name": "",
                "agent_0_submission_id": episode["agent_0_submission_id"],
                "agent_1_submission_id": episode["agent_1_submission_id"],
                "deck_hash": DECK_HASH,
                "matched_deck_card_ids": 60,
                "replay_path": str(replay),
                "log_paths": "",
                "download_status": "success",
                "error": "",
                "downloaded_at": "",
            })

    rows.sort(key=lambda row: (int(row["episode_id"]), int(row["seat_index"])))
    if not rows:
        raise SystemExit("no refreshed replays found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {len(rows)} trajectories from {len(SOURCES)} teachers to "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
