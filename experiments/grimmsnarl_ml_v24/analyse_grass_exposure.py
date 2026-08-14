"""Price the Grass problem against the board we have to climb through.

`top40_weakness.json` shows the top 40 splits cleanly by Grass *count*, not by
"contains a Grass card": a Dragapult list carries 1-2 Grass Pokemon as tech,
while the Ogerpon/Hydrapple shell carries 16-18 and attacks with them.  Only
the second kind gets the x2 on all three of our bodies.

This groups every deck we have met and every top-40 deck by that count,
reports our record against each group, and slot-weights the top 40 to give the
expected win rate against the field standing between us and 1100.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents/grimmsnarl/grimmsnarl_ml_v22"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor/cg/cards.json").read_text(encoding="utf-8"))
}
TOP40 = ROOT / "experiments/grimmsnarl_ml_v24/top40_decks_20260814.csv"
GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_v24_games.csv"
RUNS = ROOT / "data/runs/grimmsnarl"
CACHE = ROOT / "data/kaggle_top100_current/replays/probe_20260814"
OUT = ROOT / "experiments/grimmsnarl_ml_v24/grass_exposure.json"
GRASS = 1
# Share, not count: the pure Teal Mask Ogerpon ex list runs *four* Pokemon and
# 56 Trainers, so any absolute threshold files the deck that beats us 1-8 as
# "tech".  What matters is whether the deck attacks with Grass.
ATTACKER_THRESHOLD = 0.5


def grass_count(deck: list[int]) -> int:
    return sum(
        1 for card_id in deck
        if (CARDS.get(card_id, {}).get("cardType") == 0
            and CARDS.get(card_id, {}).get("energyType") == GRASS)
    )


def pokemon_count(deck: list[int]) -> int:
    return sum(1 for c in deck if CARDS.get(c, {}).get("cardType") == 0)


def grass_share(deck: list[int]) -> float:
    total = pokemon_count(deck)
    return grass_count(deck) / total if total else 0.0


def decks_from_our_runs() -> dict[str, list[int]]:
    """Both 60-card lists are in step 1 of every replay we own."""
    out: dict[str, list[int]] = {}
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir():
            continue
        for path in run_dir.glob("episodes/*/replay/*.json"):
            try:
                replay = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            steps = replay.get("steps") or []
            if len(steps) < 2:
                continue
            for seat in (0, 1):
                action = (steps[1][seat] or {}).get("action")
                if isinstance(action, list) and len(action) == 60:
                    cards = [int(v) for v in action]
                    out.setdefault(deck_hash(cards), cards)
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    decks = decks_from_our_runs()
    for path in sorted(CACHE.glob("episode_*.json")):
        replay = json.loads(path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        if len(steps) < 2:
            continue
        for seat in (0, 1):
            action = (steps[1][seat] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                cards = [int(v) for v in action]
                decks.setdefault(deck_hash(cards), cards)
    print(f"deck lists resolved: {len(decks)}")

    record: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    family: dict[str, Counter] = defaultdict(Counter)
    strong: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not raw["version"].startswith(("v22", "v24")):
            continue
        h = raw["opponent_deck_hash"]
        won = int(raw["won"] == "True")
        record[h][0] += 1
        record[h][1] += won
        family[h][raw["opponent_family"]] += 1
        try:
            if float(raw["opponent_rating"]) >= 950:
                strong[h][0] += 1
                strong[h][1] += won
        except (TypeError, ValueError):
            pass

    def group(h: str) -> str:
        deck = decks.get(h)
        if deck is None:
            return "unknown"
        share = grass_share(deck)
        if share >= ATTACKER_THRESHOLD:
            return "Grass attacker deck"
        if share > 0:
            return "Grass tech only"
        return "no Grass"

    print("\n=== our 281 pooled games, grouped by the opponent's Grass count ===")
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    buckets_strong: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for h, (n, wins) in record.items():
        buckets[group(h)][0] += n
        buckets[group(h)][1] += wins
        buckets_strong[group(h)][0] += strong[h][0]
        buckets_strong[group(h)][1] += strong[h][1]
    for label in ("Grass attacker deck", "Grass tech only", "no Grass", "unknown"):
        n, wins = buckets[label]
        if not n:
            continue
        low, high = wilson(wins, n)
        sn, sw = buckets_strong[label]
        print(f"  {label:<22} n={n:>3}  {wins:>3}-{n - wins:<3} {wins / n:.3f} "
              f"[{low:.3f},{high:.3f}]   opp>=950: "
              f"{sw}-{sn - sw} " + (f"{sw / sn:.3f}" if sn else "-"))

    print("\n=== every Grass-attacker deck we have met ===")
    for h, (n, wins) in sorted(record.items(), key=lambda kv: -kv[1][0]):
        if group(h) != "Grass attacker deck":
            continue
        print(f"  {h}  grass={grass_count(decks[h]):>2}/{pokemon_count(decks[h]):<2} n={n:>2}  "
              f"{wins}-{n - wins}  {family[h].most_common(1)[0][0]}")

    top40 = list(csv.DictReader(TOP40.open(encoding="utf-8-sig")))
    slots = Counter(r["deck_hash"] for r in top40)
    total = sum(slots.values())
    print(f"\n=== the top 40 by group (slots out of {total}) ===")
    slot_groups: Counter = Counter()
    slot_ranks: dict[str, list[int]] = defaultdict(list)
    for row in top40:
        g = group(row["deck_hash"])
        slot_groups[g] += 1
        slot_ranks[g].append(int(row["rank"]))
    for label, count in slot_groups.most_common():
        n, wins = buckets[label]
        rate = wins / n if n else None
        print(f"  {label:<22} slots {count:>2} ({count / total:.0%})  "
              f"ranks {sorted(slot_ranks[label])}")
        print(f"  {'':<22} our rate {rate:.3f} (n={n})" if rate is not None else "")

    expected = sum(
        (slot_groups[label] / total) * (buckets[label][1] / buckets[label][0])
        for label in slot_groups if buckets[label][0]
    )
    covered = sum(
        slot_groups[label] / total for label in slot_groups if buckets[label][0])
    print(f"\ntop-40 slot-weighted expected win rate: {expected / covered:.4f} "
          f"(covering {covered:.0%} of slots)")

    for target in (0.30, 0.40, 0.50):
        base = buckets["Grass attacker deck"]
        share = slot_groups["Grass attacker deck"] / total
        gain = share * (target - base[1] / base[0])
        print(f"  if the Grass-attacker cell went from "
              f"{base[1] / base[0]:.3f} to {target:.2f}: "
              f"{gain:+.4f} win rate against the top 40")

    OUT.write_text(json.dumps({
        "threshold": ATTACKER_THRESHOLD,
        "our_record_by_group": {k: v for k, v in buckets.items()},
        "strong_band_by_group": {k: v for k, v in buckets_strong.items()},
        "top40_slots_by_group": dict(slot_groups),
        "top40_ranks_by_group": {k: sorted(v) for k, v in slot_ranks.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
