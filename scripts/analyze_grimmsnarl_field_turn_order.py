"""Field baseline: how much does the same 60-card deck lose by going second?

Our own second-seat deficit only means something against the deck's own
first/second split, because going second is intrinsically worse for everybody.
This reads the archived same-deck field games and reports the split per rating
band, so the submitted agent's gap can be compared with the field's rather than
with 50%.

Only a prefix of each replay is read: ``rewards`` sits in the header and
``firstPlayer`` resolves on the coin flip a few steps in, so a full parse is
not needed and the whole archive scans in about a minute.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.replay_io import extract_fast_header_from_bytes  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"
FIRST_PLAYER_RE = re.compile(rb'"firstPlayer"\s*:\s*([01])\b')


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963985
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) * z
    return [
        round((centre - margin) / denominator, 4),
        round((centre + margin) / denominator, 4),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument("--prefix-bytes", type=int, default=3_000_000)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v20"
        / "field_turn_order.json",
    )
    args = parser.parse_args()

    index = args.data_root / "indexes" / "episodes.csv"
    bands: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    per_pilot: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"first": [0, 0], "second": [0, 0]}
    )
    ratings: dict[int, float] = {}
    read = skipped = 0

    for row in csv.DictReader(index.open(encoding="utf-8-sig")):
        if row.get("download_status") != "success":
            continue
        if row.get("deck_hash") != OUR_DECK_HASH:
            continue
        if row.get("episode_type") != "EPISODE_TYPE_PUBLIC":
            continue
        path = args.data_root / row["replay_path"].replace("\\", "/")
        if not path.exists():
            continue
        seat = int(row["seat_index"])
        with path.open("rb") as handle:
            raw = handle.read(args.prefix_bytes)
        match = FIRST_PLAYER_RE.search(raw)
        header = extract_fast_header_from_bytes(raw)
        rewards = header.get("rewards") or [None, None]
        if match is None or rewards[seat] is None:
            skipped += 1
            continue
        other = rewards[1 - seat]
        won = int(rewards[seat] > (other if other is not None else 0))
        order = "first" if int(match.group(1)) == seat else "second"

        team = int(row["team_id"])
        try:
            ratings[team] = float(row["submission_score"])
        except (TypeError, ValueError):
            pass
        rating = ratings.get(team)
        band = (
            "unknown" if rating is None
            else ">=1100" if rating >= 1100
            else "1000-1100" if rating >= 1000
            else "<1000"
        )
        bands[(band, order)][0] += won
        bands[(band, order)][1] += 1
        bands[("all", order)][0] += won
        bands[("all", order)][1] += 1
        per_pilot[team][order][0] += won
        per_pilot[team][order][1] += 1
        read += 1

    payload = {
        "replays_read": read,
        "skipped": skipped,
        "bands": {
            f"{band}|{order}": {
                "wins": wins,
                "games": games,
                "win_rate": round(wins / games, 4) if games else None,
                "wilson95": wilson(wins, games),
            }
            for (band, order), (wins, games) in sorted(bands.items())
        },
        "per_pilot": {
            str(team): {
                "rating": ratings.get(team),
                "first": counts["first"],
                "second": counts["second"],
                "gap": (
                    round(
                        counts["first"][0] / counts["first"][1]
                        - counts["second"][0] / counts["second"][1],
                        4,
                    )
                    if counts["first"][1] >= 20 and counts["second"][1] >= 20
                    else None
                ),
            }
            for team, counts in sorted(per_pilot.items())
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"replays read: {read} (skipped {skipped})")
    for key, value in payload["bands"].items():
        print(
            f"  {key:20s} {value['wins']:4d}/{value['games']:4d} "
            f"= {value['win_rate']}  {value['wilson95']}"
        )
    gaps = [
        (info["rating"], info["gap"], team)
        for team, info in payload["per_pilot"].items()
        if info["gap"] is not None and info["rating"] is not None
    ]
    print("\nper-pilot first-minus-second gap (>=20 games each side):")
    for rating, gap, team in sorted(gaps):
        print(f"  {team:>10s} rating {rating:7.1f}  gap {gap:+.4f}")
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
