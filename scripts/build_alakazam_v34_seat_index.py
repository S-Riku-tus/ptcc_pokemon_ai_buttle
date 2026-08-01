"""Turn a fetched run's episodes.csv into a seat-index CSV the v33 indexer reads.

``build_alakazam_v33_teacher_index.py`` resolves which seat the teacher played
from an ``episodes.json`` embedded in the archive, and falls back to a
previously validated teacher index CSV when the archive has none.
``fetch_submission_logs.py`` writes ``episodes.csv`` instead, so this converts
one into the fallback format (``submission_id``, ``episode_id``,
``seat_index``).

Self-play episodes where the same submission holds both seats contribute both
seats, matching the v33 indexer's treatment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.run_dir / "episodes.csv"
    rows = list(csv.DictReader(
        source.read_text(encoding="utf-8-sig").splitlines()
    ))

    out: list[dict[str, int]] = []
    for row in rows:
        for seat in (0, 1):
            if str(row.get(f"agent_{seat}_submission_id")) != str(
                args.submission_id
            ):
                continue
            out.append({
                "submission_id": args.submission_id,
                "episode_id": int(row["episode_id"]),
                "seat_index": seat,
            })

    out.sort(key=lambda r: (r["episode_id"], r["seat_index"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["submission_id", "episode_id", "seat_index"]
        )
        writer.writeheader()
        writer.writerows(out)

    episodes = {r["episode_id"] for r in out}
    both = len(out) - len(episodes)
    print(
        f"{source}: {len(rows)} episodes -> {len(out)} teacher seats "
        f"over {len(episodes)} episodes ({both} self-play second seats)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
