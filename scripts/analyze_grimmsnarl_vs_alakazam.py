"""How do the field's Alakazam pilots beat Grimmsnarl, and how do we differ?

Pulls every Alakazam-vs-Grimmsnarl game out of the archived top-50 Grimmsnarl
submissions, then compares the Alakazam side's deck lists and Powerful Hand
cadence between the games Alakazam won and the games it lost. Our own v31
list is printed alongside so deck deltas are visible.

Usage: python scripts/analyze_grimmsnarl_vs_alakazam.py [--our-deck PATH]
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

ALAKAZAM = 743
TURN_START, ATTACK = 2, 15
ALAKAZAM_ATTACK = 1072  # Powerful Hand


def name(card_id: int) -> str:
    return CARDS.get(card_id, {}).get("name", f"#{card_id}")


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


def count_powerful_hands(zf: ZipFile, episode: str, agent: int) -> int:
    entry = (
        f"{Path(zf.namelist()[0]).parts[0]}/episodes/{episode}"
        f"/agent_{agent}/agent_{agent}_observation_logs.json"
    )
    try:
        raw = zf.read(entry)
    except KeyError:
        return -1
    logs: list[dict[str, Any]] = []
    previous = None
    for item in json.loads(raw.decode("utf-8"))["entries"]:
        blob = json.dumps(item["logs"], sort_keys=True)
        if blob == previous:
            continue
        previous = blob
        logs.extend(item["logs"])
    return sum(
        1 for log in logs
        if log.get("type") == ATTACK
        and log.get("attackId") == ALAKAZAM_ATTACK
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--our-deck",
        type=Path,
        default=ROOT / "agents" / "alakazam" / "alakazam_ml_v31" / "deck.csv",
    )
    args = parser.parse_args()

    games: list[dict[str, Any]] = []

    for zip_path in sorted(ARCHIVE.glob("*.zip")):
        submission_id = int(zip_path.stem.split("_")[-1])
        with ZipFile(zip_path) as zf:
            seats: dict[str, int] = {}
            for entry in zf.namelist():
                if entry.endswith("episodes.csv"):
                    text = zf.read(entry).decode("utf-8-sig")
                    for row in csv.DictReader(io.StringIO(text)):
                        for seat in (0, 1):
                            if str(
                                row.get(f"agent_{seat}_submission_id")
                            ) == str(submission_id):
                                seats[str(row["episode_id"])] = seat

            for entry in zf.namelist():
                if "/replay/" not in entry or not entry.endswith(".json"):
                    continue
                episode = Path(entry).stem.replace("episode_", "")
                grimm_seat = seats.get(episode)
                if grimm_seat is None:
                    continue
                replay = json.loads(zf.read(entry).decode("utf-8"))
                steps = replay.get("steps") or []
                if len(steps) < 2:
                    continue
                zam_seat = 1 - grimm_seat
                deck = steps[1][zam_seat].get("action")
                if not (isinstance(deck, list) and len(deck) == 60):
                    continue
                if archetype(deck) != "Alakazam":
                    continue
                reward = steps[-1][zam_seat].get("reward")
                if reward is None:
                    continue
                games.append({
                    "episode": episode,
                    "zam_won": reward > 0,
                    "deck": deck,
                    "powerful_hands": count_powerful_hands(
                        zf, episode, zam_seat
                    ),
                })
        print(f"read {zip_path.name}")

    wins = [g for g in games if g["zam_won"]]
    losses = [g for g in games if not g["zam_won"]]
    print(f"\n=== Alakazam vs Grimmsnarl in the archive: {len(games)} games, "
          f"Alakazam won {len(wins)} ({len(wins) / len(games) * 100:.1f}%) ===")

    valid = [g for g in games if g["powerful_hands"] >= 0]
    print(f"\n--- Powerful Hands per game (n={len(valid)}) ---")
    buckets: defaultdict[int, list[bool]] = defaultdict(list)
    for game in valid:
        buckets[min(game["powerful_hands"], 5)].append(game["zam_won"])
    for count in sorted(buckets):
        rows = buckets[count]
        label = f"{count}+" if count == 5 else str(count)
        print(f"  {label:>3} attacks: {sum(rows):3d}/{len(rows):3d} "
              f"({sum(rows) / len(rows) * 100:5.1f}%)")
    for label, rows in (("win", wins), ("loss", losses)):
        vals = [g["powerful_hands"] for g in rows if g["powerful_hands"] >= 0]
        if vals:
            print(f"  mean when Alakazam {label:4s}: "
                  f"{sum(vals) / len(vals):.2f}")

    # Deck comparison: distinct lists, and win rate per list.
    print("\n--- distinct Alakazam lists faced (>=5 games) ---")
    by_list: defaultdict[tuple, list[bool]] = defaultdict(list)
    for game in games:
        by_list[tuple(sorted(game["deck"]))].append(game["zam_won"])

    ours = Counter(
        int(x) for x in args.our_deck.read_text(encoding="utf-8").split()
    )
    ranked = sorted(by_list.items(), key=lambda kv: -len(kv[1]))
    for deck, rows in ranked:
        if len(rows) < 5:
            continue
        counts = Counter(deck)
        diff_add = counts - ours
        diff_cut = ours - counts
        print(f"\n  n={len(rows):3d}  win rate "
              f"{sum(rows) / len(rows) * 100:5.1f}%  "
              f"({'IDENTICAL to ours' if not diff_add and not diff_cut else 'differs'})")
        if diff_add:
            print("      they run more: " + ", ".join(
                f"+{n} {name(c)}" for c, n in sorted(diff_add.items())))
        if diff_cut:
            print("      we run more:   " + ", ".join(
                f"+{n} {name(c)}" for c, n in sorted(diff_cut.items())))

    same = [
        rows for deck, rows in by_list.items()
        if Counter(deck) == ours
    ]
    flat = [w for rows in same for w in rows]
    if flat:
        print(f"\n--- games where their Alakazam list is EXACTLY ours ---")
        print(f"  {sum(flat)}/{len(flat)} "
              f"({sum(flat) / len(flat) * 100:.1f}%) won by Alakazam")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
