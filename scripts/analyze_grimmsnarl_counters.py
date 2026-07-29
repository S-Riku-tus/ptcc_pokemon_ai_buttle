"""What actually beats Marnie's Grimmsnarl ex on the ladder?

Reads the archived top-50 Grimmsnarl submissions under
``data/runs/leaderboard_top50/grimmsnarl/`` and reports the Grimmsnarl
pilot's win rate split by opponent archetype, so our own 25% is scored
against the field rather than against a guess.

Usage: python scripts/analyze_grimmsnarl_counters.py [--min-games 8]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "runs" / "leaderboard_top50" / "grimmsnarl"

CARDS: dict[int, dict[str, Any]] = {
    c["cardId"]: c
    for c in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}


def archetype(deck: list[int]) -> str:
    pokes = Counter(
        cid for cid in deck
        if CARDS.get(cid) and CARDS[cid]["cardType"] == 0
    )
    if not pokes:
        return "unknown"

    def key(item: tuple[int, int]) -> tuple:
        cid, count = item
        card = CARDS[cid]
        return (
            card["stage2"], card["megaEx"] or card["ex"],
            card["stage1"], count, card["hp"],
        )

    return CARDS[max(pokes.items(), key=key)[0]]["name"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-games", type=int, default=8)
    args = parser.parse_args()

    per_opponent: defaultdict[str, list[bool]] = defaultdict(list)
    total = 0

    for zip_path in sorted(ARCHIVE.glob("*.zip")):
        submission_id = int(zip_path.stem.split("_")[-1])
        with ZipFile(zip_path) as zf:
            seats: dict[str, int] = {}
            for entry in zf.namelist():
                if not entry.endswith("episodes.csv"):
                    continue
                text = zf.read(entry).decode("utf-8-sig")
                for row in csv.DictReader(io.StringIO(text)):
                    for seat in (0, 1):
                        if str(row.get(f"agent_{seat}_submission_id")) == str(
                            submission_id
                        ):
                            seats[str(row["episode_id"])] = seat

            for entry in zf.namelist():
                if "/replay/" not in entry or not entry.endswith(".json"):
                    continue
                episode_id = Path(entry).stem.replace("episode_", "")
                seat = seats.get(episode_id)
                if seat is None:
                    continue
                replay = json.loads(zf.read(entry).decode("utf-8"))
                steps = replay.get("steps") or []
                if len(steps) < 2:
                    continue
                opp = 1 - seat
                action = steps[1][opp].get("action")
                if not (isinstance(action, list) and len(action) == 60):
                    continue
                reward = steps[-1][seat].get("reward")
                if reward is None:
                    continue
                per_opponent[archetype(action)].append(reward > 0)
                total += 1

        print(f"read {zip_path.name}")

    print(f"\n=== Grimmsnarl pilots (top-50 archive): {total} games ===")
    overall = [w for rows in per_opponent.values() for w in rows]
    print(f"overall Grimmsnarl win rate: {sum(overall)}/{len(overall)} "
          f"({sum(overall) / len(overall) * 100:.1f}%)\n")
    print(f"{'opponent archetype':34s} {'n':>5} {'grimm WR':>9} "
          f"{'opp WR':>8}")
    for name, rows in sorted(
        per_opponent.items(), key=lambda kv: -len(kv[1])
    ):
        if len(rows) < args.min_games:
            continue
        rate = sum(rows) / len(rows)
        print(f"{name:34s} {len(rows):>5} {rate * 100:>8.1f}% "
              f"{(1 - rate) * 100:>7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
