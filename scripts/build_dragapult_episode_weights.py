"""Build an auditable teacher-episode weight sidecar for Dragapult training.

The primary mode gives a modest extra weight to trajectories that sustained a
minimum number of Phantom Dives.  It weights an entire trajectory, never an
individual candidate, and the generated file records support and win rates so
the selection pressure is visible in the experiment report.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

PHANTOM_DIVE = 154
OPT_ATTACK = 13


def phantom_dives(replay: dict[str, Any], seat: int) -> int:
    steps = replay.get("steps") or []
    count = 0
    for index, pair in enumerate(steps[:-1]):
        payload = pair[seat]
        if payload.get("status") != "ACTIVE":
            continue
        observation = payload.get("observation") or {}
        select = observation.get("select")
        if not isinstance(select, dict):
            continue
        action = steps[index + 1][seat].get("action")
        if not isinstance(action, list) or len(action) != 1:
            continue
        options = select.get("option") or []
        chosen = int(action[0])
        if not 0 <= chosen < len(options):
            continue
        option = options[chosen]
        if (
            int(option.get("type", -1)) == OPT_ATTACK
            and int(option.get("attackId", -1)) == PHANTOM_DIVE
        ):
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-dives", type=int, default=4)
    parser.add_argument("--weight", type=float, default=1.25)
    args = parser.parse_args()
    if args.min_dives < 1:
        parser.error("--min-dives must be positive")
    if args.weight <= 0:
        parser.error("--weight must be positive")

    seen: set[tuple[int, int]] = set()
    weights: dict[str, float] = {}
    histogram: Counter[int] = Counter()
    wins = boosted_wins = 0
    total = boosted = 0
    for row in csv.DictReader(
        args.teacher_index.read_text(encoding="utf-8-sig").splitlines()
    ):
        episode_id = int(row["episode_id"])
        seat = int(row["seat_index"])
        key = (episode_id, seat)
        if key in seen:
            continue
        seen.add(key)
        path = Path(row["replay_path"])
        if not path.is_absolute():
            path = args.teacher_index.parent.parent / path
        replay = json.loads(path.read_text(encoding="utf-8"))
        dives = phantom_dives(replay, seat)
        rewards = replay.get("rewards") or [0, 0]
        won = int(rewards[seat] > rewards[1 - seat])
        total += 1
        wins += won
        histogram[dives] += 1
        if dives >= args.min_dives:
            boosted += 1
            boosted_wins += won
            weights[f"{episode_id}:{seat}"] = args.weight

    report = {
        "kind": "minimum_phantom_dives",
        "min_dives": args.min_dives,
        "boost_weight": args.weight,
        "total_trajectories": total,
        "boosted_trajectories": boosted,
        "boosted_share": round(boosted / max(1, total), 4),
        "all_win_rate": round(wins / max(1, total), 4),
        "boosted_win_rate": round(boosted_wins / max(1, boosted), 4),
        "dive_histogram": {str(key): value for key, value in sorted(histogram.items())},
        "weights": weights,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "weights"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
