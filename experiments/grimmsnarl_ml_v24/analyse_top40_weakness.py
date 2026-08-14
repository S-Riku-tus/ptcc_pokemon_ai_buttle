"""Is the top-40 field built on the type our whole line is weak to?

Marnie's Impidimp, Morgrem and Grimmsnarl ex all carry ``weakness == 1``.  The
enum is read off the card file rather than assumed: Teal Mask Ogerpon ex is a
Grass Pokemon with ``weakness == 2`` (Fire), Hearthflame is Fire with 3
(Water), Wellspring is Water with 4 (Lightning) and Cornerstone is Fighting
with 1 - so 1 is Grass, and Grass damage doubles against every body we play.

That is already the accepted explanation for Teal Mask Ogerpon ex holding this
list to ~0.20 for the entire field.  If the decks holding the top of the
current board are also Grass, the ceiling on this 60 is a card-pool fact and no
ranker or planner work reaches it.  If they are not, the Ogerpon cell is a
mid-ladder tax and the top-40 deficit has to be explained some other way.

For every top-40 deck this decodes the cached probe replay's 60 card IDs and
reports the Pokemon energy types alongside our record against that hash.
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

import ml_features as mf  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor/cg/cards.json").read_text(encoding="utf-8"))
}
TOP40 = ROOT / "experiments/grimmsnarl_ml_v24/top40_decks_20260814.csv"
GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_v24_games.csv"
CACHE = ROOT / "data/kaggle_top100_current/replays/probe_20260814"
OUT = ROOT / "experiments/grimmsnarl_ml_v24/top40_weakness.json"

ENERGY = {
    1: "Grass", 2: "Fire", 3: "Water", 4: "Lightning", 5: "Psychic",
    6: "Fighting", 7: "Darkness", 8: "Metal", 9: "Dragon", 0: "Colorless",
}
OUR_LINE = [mf.IMPIDIMP_ID, mf.MORGREM_ID, mf.GRIMMSNARL_EX_ID]


def energy_name(value: Any) -> str:
    return ENERGY.get(value, str(value))


def profile(deck: list[int]) -> dict[str, Any]:
    """Energy types of the deck's Pokemon, and its biggest attackers."""
    types: Counter = Counter()
    headline: list[str] = []
    for card_id, count in Counter(deck).items():
        card = CARDS.get(card_id)
        if not card or card.get("cardType") != 0:
            continue
        types[energy_name(card.get("energyType"))] += count
        if card.get("ex") or card.get("megaEx") or card.get("stage2"):
            headline.append(
                f"{card.get('name')}"
                f"[{energy_name(card.get('energyType'))},"
                f"{card.get('hp')}hp]")
    return {"energy_types": dict(types.most_common()), "headline": headline}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== our line ===")
    for card_id in OUR_LINE:
        card = CARDS[card_id]
        print(f"  {card['name']:<26} hp {card['hp']:>4}  "
              f"energy {energy_name(card.get('energyType')):<10} "
              f"weakness {energy_name(card.get('weakness'))} (x2)")
    print()

    cached: dict[str, list[int]] = {}
    for path in sorted(CACHE.glob("episode_*.json")):
        replay = json.loads(path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        if len(steps) < 2:
            continue
        for seat in (0, 1):
            action = (steps[1][seat] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                cards = [int(v) for v in action]
                cached.setdefault(deck_hash(cards), cards)

    record: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if raw["version"].startswith(("v22", "v24")):
            record[raw["opponent_deck_hash"]][0] += 1
            record[raw["opponent_deck_hash"]][1] += int(raw["won"] == "True")

    top40 = list(csv.DictReader(TOP40.open(encoding="utf-8-sig")))
    slots = Counter(r["deck_hash"] for r in top40)
    rows: list[dict[str, Any]] = []

    print(f"{'deck_hash':<20}{'slots':>6}{'best':>7}{'n':>4}{'wr':>7}"
          f"{'grass':>7}  headline Pokemon")
    for h, n_slots in slots.most_common():
        best = max(float(r["leaderboard_score"]) for r in top40
                   if r["deck_hash"] == h)
        deck = cached.get(h)
        prof = profile(deck) if deck else None
        n, wins = record.get(h, [0, 0])
        grass = prof["energy_types"].get("Grass") if prof else None
        rows.append({
            "deck_hash": h, "slots": n_slots, "best_rating": best,
            "our_games": n, "our_wins": wins, "grass_pokemon": grass,
            "decoded": prof is not None, **(prof or {}),
        })
        print(
            f"{h:<20}{n_slots:>6}{best:>7.0f}{n:>4}"
            f"{(f'{wins / n:.3f}' if n else '-'):>7}"
            f"{(str(grass) if grass is not None else '?'):>7}  "
            f"{' '.join((prof or {}).get('headline', []))[:70]}")

    decoded = [r for r in rows if r["decoded"]]
    known_slots = sum(r["slots"] for r in decoded)
    grass_slots = sum(r["slots"] for r in decoded if (r["grass_pokemon"] or 0) > 0)
    print(f"\ndecoded {known_slots}/40 top-40 slots; "
          f"{grass_slots} of them ({grass_slots / known_slots:.0%}) "
          f"run Grass Pokemon (x2 into every body we play)")

    print("\n=== the Grass decks we have actually met ===")
    for raw_hash, (n, wins) in sorted(record.items(), key=lambda kv: -kv[1][0]):
        deck = cached.get(raw_hash)
        if deck is None or n < 2:
            continue
        prof = profile(deck)
        if prof["energy_types"].get("Grass"):
            print(f"  {raw_hash}  n={n:>3} {wins}-{n - wins} "
                  f"{wins / n:.3f}  {' '.join(prof['headline'])[:60]}")

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
