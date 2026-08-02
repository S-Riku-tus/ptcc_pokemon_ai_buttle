"""Narrow a leaderboard submission CSV down to one deck archetype.

Takes the ``teams.csv`` written by
``scripts/analyze_top50_deck_distribution.py`` and the
``public_submissions_top*.csv`` written by
``scripts/fetch_kaggle_top100_snapshot.py``, and emits a submission CSV in
the same schema holding only the teams whose archetype label matches.

The result is a drop-in ``--input`` for
``scripts/collect_top100_submission_replays.py``, so a bulk replay pull only
touches the teams that actually play the target deck.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUBMISSION_ID_COLUMN = "public_submission_id"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archetype",
        required=True,
        help="Case-insensitive substring of the archetype label, e.g. Grimmsnarl.",
    )
    parser.add_argument(
        "--teams",
        type=Path,
        default=ROOT / "data" / "kaggle_top50_meta" / "analysis" / "teams.csv",
        help="teams.csv from analyze_top50_deck_distribution.py.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            ROOT
            / "data"
            / "kaggle_top100"
            / "latest"
            / "public_submissions_top50.csv"
        ),
        help="Leaderboard submission CSV to filter.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV. Default: <teams dir>/submissions_<archetype>.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    needle = args.archetype.strip().lower()

    teams = read_csv(args.teams)
    selected = [
        team for team in teams if needle in team.get("archetype", "").lower()
    ]
    if not selected:
        labels = sorted({team.get("archetype", "") for team in teams})
        raise SystemExit(
            f"No archetype label contains {args.archetype!r}. Known labels:\n  "
            + "\n  ".join(labels)
        )

    wanted = {team["submission_id"].strip() for team in selected}

    source_rows = read_csv(args.source)
    if not source_rows:
        raise SystemExit(f"No rows in {args.source}")
    fieldnames = list(source_rows[0])
    if SUBMISSION_ID_COLUMN not in fieldnames:
        raise SystemExit(
            f"{args.source} has no {SUBMISSION_ID_COLUMN} column; "
            f"columns are {fieldnames}"
        )

    kept = [
        row
        for row in source_rows
        if str(row.get(SUBMISSION_ID_COLUMN, "")).strip() in wanted
    ]

    output = args.output or (
        args.teams.parent / f"submissions_{needle.replace(' ', '_')}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    found = {str(row.get(SUBMISSION_ID_COLUMN, "")).strip() for row in kept}
    for team in sorted(selected, key=lambda item: int(item["rank"] or 0)):
        mark = " " if team["submission_id"].strip() in found else "!"
        print(
            f"{mark} rank {team['rank']:>3}  {team['submission_id']:>10}  "
            f"{team['team_name']}"
        )

    missing = wanted - found
    print()
    print(f"Teams matching {args.archetype!r}: {len(selected)}")
    print(f"Rows written: {len(kept)} -> {output}")
    if missing:
        print(f"Not present in {args.source}: {sorted(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
