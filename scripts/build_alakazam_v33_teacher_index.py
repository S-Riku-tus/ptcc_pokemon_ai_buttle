"""Collect every locally available trajectory of one teacher submission.

v32 trained on the 1,000 Yushin Ito games in the 2026-07-26 archive. Two more
archives of the *same* submission sit in the workspace and were never used:
179 games in the 2026-07-17 top-20 pull and 100 games in the 2026-07-24
current-top pull. Every one of them is chronologically older than the v32
validation and test episodes, so they can enter training while the frozen
holdout stays bit-identical and directly comparable with v32.

Seats are resolved exactly: from the archive's own ``episodes.json`` when it
has one, otherwise from a previously validated teacher index. Validation
self-play games contribute both seats because both were played by the teacher.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

EPISODE_RE = re.compile(r"episode_(\d+)\.json$")


def _seat_lookup_from_index(
    path: Path, submission_id: int
) -> dict[int, list[int]]:
    lookup: dict[int, list[int]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["submission_id"]) != submission_id:
                continue
            lookup.setdefault(int(row["episode_id"]), []).append(
                int(row["seat_index"])
            )
    return lookup


def _seat_lookup_from_archive(
    archive: zipfile.ZipFile, submission_id: int
) -> dict[int, list[int]]:
    lookup: dict[int, list[int]] = {}
    for name in archive.namelist():
        if not name.endswith("episodes.json"):
            continue
        payload = json.loads(archive.read(name))
        for episode in payload.get("episodes") or []:
            seats = [
                seat for seat in (0, 1)
                if str(episode.get(f"agent_{seat}_submission_id"))
                == str(submission_id)
            ]
            if seats:
                lookup[int(episode["episode_id"])] = seats
    return lookup


def _archive_rows(
    archive_path: Path,
    *,
    cohort: str,
    submission_id: int,
    team_name: str,
    leaderboard_rank: int,
    fallback_seats: dict[int, list[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        seats_by_episode = _seat_lookup_from_archive(archive, submission_id)
        seat_source = "archive_episodes_json"
        if not seats_by_episode:
            seats_by_episode = fallback_seats
            seat_source = "validated_teacher_index"
        for name in archive.namelist():
            match = EPISODE_RE.search(name)
            if match is None or "/replay" not in name:
                continue
            episode_id = int(match.group(1))
            for seat in seats_by_episode.get(episode_id, []):
                rows.append({
                    "episode_id": episode_id,
                    "seat_index": seat,
                    "leaderboard_rank": leaderboard_rank,
                    "team_name": team_name,
                    "submission_id": submission_id,
                    "source_cohort": cohort,
                    "storage_type": "zip",
                    "storage_path": str(archive_path.resolve()),
                    "replay_path": name,
                    "teacher_priority": 1.0 / math.sqrt(leaderboard_rank),
                    "created_at": "",
                    "seat_source": seat_source,
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-id", type=int, default=54773249)
    parser.add_argument("--team-name", default="Yushin Ito")
    parser.add_argument("--leaderboard-rank", type=int, default=3)
    parser.add_argument(
        "--archive", action="append", default=[], metavar="COHORT:PATH",
        required=True,
    )
    parser.add_argument(
        "--seat-index", action="append", default=[], type=Path,
        help="Previously validated teacher index CSVs used as a seat source.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fallback_seats: dict[int, list[int]] = {}
    for index_path in args.seat_index:
        for episode_id, seats in _seat_lookup_from_index(
            index_path, args.submission_id
        ).items():
            fallback_seats.setdefault(episode_id, []).extend(seats)

    rows: list[dict[str, Any]] = []
    for spec in args.archive:
        cohort, _, path = spec.partition(":")
        found = _archive_rows(
            Path(path),
            cohort=cohort,
            submission_id=args.submission_id,
            team_name=args.team_name,
            leaderboard_rank=args.leaderboard_rank,
            fallback_seats=fallback_seats,
        )
        rows.extend(found)
        print(f"{cohort}: {len(found)} trajectories from {Path(path).name}")

    deduplicated: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates = 0
    for row in rows:
        key = (int(row["episode_id"]), int(row["seat_index"]))
        if key in deduplicated:
            duplicates += 1
            continue
        deduplicated[key] = row
    final = sorted(
        deduplicated.values(),
        key=lambda r: (int(r["episode_id"]), int(r["seat_index"])),
    )

    fieldnames = [
        "episode_id", "seat_index", "leaderboard_rank", "team_name",
        "submission_id", "source_cohort", "storage_type", "storage_path",
        "replay_path", "teacher_priority", "created_at", "seat_source",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final)

    episodes = sorted({int(row["episode_id"]) for row in final})
    report = {
        "output": str(args.output.resolve()),
        "submission_id": args.submission_id,
        "team_name": args.team_name,
        "trajectories": len(final),
        "unique_episodes": len(episodes),
        "duplicate_episode_seats_removed": duplicates,
        "cohorts": dict(Counter(row["source_cohort"] for row in final)),
        "seat_sources": dict(Counter(row["seat_source"] for row in final)),
        "episode_id_min": episodes[0] if episodes else None,
        "episode_id_max": episodes[-1] if episodes else None,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
