"""Rating against games played, so short runs are compared at equal length.

Every Kaggle submission restarts at 600 and the per-game increment decays as
the submission accumulates games, so a 34-game run and a 57-game run of the
same code do not land on the same number.  Plotting rating against game index
puts v27's 853 next to what each champion run was worth after the same number
of games, which is the only honest way to read a truncated run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / "experiments/grimmsnarl_ml_v27/version_games.csv"
CHECKPOINTS = (10, 20, 25, 30, 34, 38, 45, 50, 57)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/rating_trajectory.json",
    )
    args = parser.parse_args()

    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not raw["our_rating_after"]:
            continue
        runs[raw["version"]].append({
            "time": raw["create_time"],
            "after": float(raw["our_rating_after"]),
            "before": float(raw["our_rating_before"]) if raw["our_rating_before"] else None,
            "won": int(raw["won"]),
            "opponent_rating": (
                float(raw["opponent_rating"]) if raw["opponent_rating"] else None
            ),
        })
    for rows in runs.values():
        rows.sort(key=lambda r: r["time"])

    order = sorted(runs, key=lambda v: runs[v][0]["time"])
    header = f"{'game':>6}" + "".join(f"{v:>9}" for v in order)
    print("rating after N games\n")
    print(header)
    print("-" * len(header))
    payload: dict[str, Any] = {"checkpoints": {}}
    for point in CHECKPOINTS:
        line = f"{point:>6}"
        payload["checkpoints"][point] = {}
        for version in order:
            rows = runs[version]
            value = rows[point - 1]["after"] if len(rows) >= point else None
            payload["checkpoints"][point][version] = value
            line += f"{'-':>9}" if value is None else f"{value:>9.1f}"
        print(line)
    line = f"{'final':>6}"
    for version in order:
        line += f"{runs[version][-1]['after']:>9.1f}"
    print(line)
    line = f"{'games':>6}"
    for version in order:
        line += f"{len(runs[version]):>9d}"
    print(line)

    print("\nmean opponent rating over the same first N games\n")
    print(header)
    print("-" * len(header))
    for point in CHECKPOINTS:
        line = f"{point:>6}"
        for version in order:
            rows = [
                r for r in runs[version][:point]
                if r["opponent_rating"] is not None
            ]
            line += (
                f"{'-':>9}" if len(runs[version]) < point
                else f"{sum(r['opponent_rating'] for r in rows) / len(rows):>9.1f}"
            )
        print(line)

    print("\nper-game rating increment, by game index (all runs pooled)")
    buckets: dict[int, list[float]] = defaultdict(list)
    for rows in runs.values():
        for index, row in enumerate(rows, 1):
            if row["before"] is not None:
                buckets[(index - 1) // 5 * 5 + 1].append(abs(row["after"] - row["before"]))
    for start in sorted(buckets):
        values = buckets[start]
        print(f"  games {start:>3}-{start + 4:<3} n={len(values):>4} "
              f"mean |delta| {sum(values) / len(values):6.2f}")

    print("\nwhere each run stopped relative to its own climb:")
    for version in order:
        rows = runs[version]
        tail = rows[-5:]
        drift = tail[-1]["after"] - tail[0]["after"]
        print(
            f"  {version:<7} games {len(rows):>3}  final {rows[-1]['after']:7.1f}  "
            f"last-5 drift {drift:+7.1f}  "
            f"last-5 mean |delta| "
            f"{sum(abs(r['after'] - (r['before'] or r['after'])) for r in tail) / len(tail):5.2f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
