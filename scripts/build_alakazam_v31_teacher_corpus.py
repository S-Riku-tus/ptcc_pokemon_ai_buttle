"""Build an accessible v31 teacher index including recovered Rmy replays."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _existing_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # The current_top rows point to an expired temporary extraction. The two
    # full archives are workspace-backed and remain authoritative.
    return [
        dict(row)
        for row in rows
        if row["source_cohort"] in {"majkel_full", "yushin_full"}
        and Path(row["storage_path"]).exists()
    ]


def _recovered_rows(
    root: Path,
    *,
    leaderboard_rank: int,
    team_name: str,
    submission_id: int,
    source_cohort: str,
) -> list[dict[str, Any]]:
    manifest_path = root / "manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    rows = []
    for row in manifest:
        if row["replay_status"] not in {"downloaded", "skipped_existing"}:
            continue
        seat = row["detected_submission_agent_index"]
        if seat not in {"0", "1"}:
            continue
        episode_id = int(row["episode_id"])
        replay = (
            root
            / "episodes"
            / str(episode_id)
            / "replay"
            / f"episode_{episode_id}.json"
        )
        if not replay.exists():
            continue
        rows.append({
            "episode_id": episode_id,
            "seat_index": int(seat),
            "leaderboard_rank": leaderboard_rank,
            "team_name": team_name,
            "submission_id": submission_id,
            "source_cohort": source_cohort,
            "storage_type": "file",
            "storage_path": str(root.resolve()),
            "replay_path": str(replay.relative_to(root)),
            "teacher_priority": 1.0 / math.sqrt(leaderboard_rank),
            "created_at": "",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v30-index", type=Path, required=True)
    parser.add_argument("--rmy-root", type=Path, required=True)
    parser.add_argument("--majkel-previous-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _existing_rows(args.v30_index)
    rows.extend(_recovered_rows(
        args.rmy_root,
        leaderboard_rank=4,
        team_name="Rmy",
        submission_id=54750312,
        source_cohort="rmy_recovered",
    ))
    if args.majkel_previous_root is not None:
        rows.extend(_recovered_rows(
            args.majkel_previous_root,
            leaderboard_rank=2,
            team_name="Majkel1337",
            submission_id=54618168,
            source_cohort="majkel_previous",
        ))
    deduplicated: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        deduplicated[(int(row["episode_id"]), int(row["seat_index"]))] = row
    final = sorted(
        deduplicated.values(),
        key=lambda row: (
            row["source_cohort"],
            int(row["episode_id"]),
            int(row["seat_index"]),
        ),
    )
    fieldnames = [
        "episode_id",
        "seat_index",
        "leaderboard_rank",
        "team_name",
        "submission_id",
        "source_cohort",
        "storage_type",
        "storage_path",
        "replay_path",
        "teacher_priority",
        "created_at",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final)
    report = {
        "output": str(args.output.resolve()),
        "trajectories": len(final),
        "cohorts": dict(Counter(row["source_cohort"] for row in final)),
        "teams": dict(Counter(row["team_name"] for row in final)),
        "deck_hash": "cc38cb450b86770a",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
