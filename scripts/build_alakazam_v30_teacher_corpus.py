"""Build a deduplicated, coherent teacher index for Alakazam v30.

v29 only trained on the small current-top snapshot.  v30 combines that
snapshot with the two 1,000-game full expert bundles while keeping replay
storage external to the index.  Episode/seat duplicates are removed before
any split is assigned.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.manifest import build_manifest  # noqa: E402


DECK_HASH = "cc38cb450b86770a"
CURRENT_RANKS = {2, 3, 5, 6, 8}


def _current_rows(current_root: Path) -> list[dict[str, Any]]:
    index_path = current_root / "indexes" / "replay_index.csv"
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for row in source:
        rank = int(row["leaderboard_rank"])
        if rank not in CURRENT_RANKS or row["deck_hash"] != DECK_HASH:
            continue
        rows.append({
            "episode_id": int(row["episode_id"]),
            "seat_index": int(row["seat_index"]),
            "leaderboard_rank": rank,
            "team_name": row["team_name"],
            "submission_id": int(row["submission_id"]),
            "source_cohort": "current_top",
            "storage_type": "file",
            "storage_path": str(current_root.resolve()),
            "replay_path": row["replay_path"],
            "teacher_priority": 1.0 / (rank ** 0.5),
            "created_at": row.get("created_at", ""),
        })
    return rows


def _archive_rows(
    manifest: pd.DataFrame,
    *,
    zip_name: str,
    cohort: str,
    rank: int,
    team_name: str,
    submission_id: int,
) -> list[dict[str, Any]]:
    selected = manifest[
        manifest["zip_name"].eq(zip_name)
        & manifest["usable_manifest"].eq(True)
        & manifest["deck_hash"].eq(DECK_HASH)
    ]
    rows = []
    for row in selected.itertuples(index=False):
        rows.append({
            "episode_id": int(row.episode_id),
            "seat_index": int(row.target_seat),
            "leaderboard_rank": rank,
            "team_name": team_name,
            "submission_id": submission_id,
            "source_cohort": cohort,
            "storage_type": "zip",
            "storage_path": str(Path(row.zip_path).resolve()),
            "replay_path": str(row.replay_path),
            "teacher_priority": 1.0 / (rank ** 0.5),
            "created_at": "",
        })
    return rows


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    audit_dir = output.parent / "archive_manifest"
    manifest, manifest_stats, _ = build_manifest(
        [args.majkel_zip.resolve(), args.yushin_zip.resolve()],
        audit_dir,
    )

    rows = _current_rows(args.current_root.resolve())
    rows.extend(_archive_rows(
        manifest,
        zip_name=args.majkel_zip.name,
        cohort="majkel_full",
        rank=2,
        team_name="Majkel1337",
        submission_id=54662660,
    ))
    rows.extend(_archive_rows(
        manifest,
        zip_name=args.yushin_zip.name,
        cohort="yushin_full",
        rank=3,
        team_name="Yushin Ito",
        submission_id=54773249,
    ))

    # A replay from the current snapshot can also be present in a full bundle.
    # The choice itself is identical, so keep the row with the highest teacher
    # priority and then the current snapshot on a tie.
    cohort_order = {"current_top": 2, "majkel_full": 1, "yushin_full": 0}
    rows.sort(
        key=lambda row: (
            row["teacher_priority"],
            cohort_order[row["source_cohort"]],
        ),
        reverse=True,
    )
    deduplicated: dict[tuple[int, int], dict[str, Any]] = {}
    duplicate_rows = 0
    for row in rows:
        key = (row["episode_id"], row["seat_index"])
        if key in deduplicated:
            duplicate_rows += 1
            continue
        deduplicated[key] = row
    final_rows = sorted(
        deduplicated.values(),
        key=lambda row: (
            row["source_cohort"],
            row["episode_id"],
            row["seat_index"],
        ),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
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
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    cohorts = Counter(row["source_cohort"] for row in final_rows)
    teams = Counter(row["team_name"] for row in final_rows)
    report = {
        "output": str(output),
        "deck_hash": DECK_HASH,
        "rows_before_deduplication": len(rows),
        "duplicate_episode_seats_removed": duplicate_rows,
        "teacher_trajectories": len(final_rows),
        "cohorts": dict(cohorts),
        "teams": dict(teams),
        "archive_manifest_stats": manifest_stats,
    }
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--majkel-zip", type=Path, required=True)
    parser.add_argument("--yushin-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
