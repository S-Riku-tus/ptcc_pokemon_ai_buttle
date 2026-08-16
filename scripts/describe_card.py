"""Print a card's full record: HP, type, weakness, abilities and attacks.

A matchup finding is only actionable once the opposing card's actual text is on
the table, so this resolves ids through cards.json, attacks.json and
abilities.json in one place.

Usage:
  python scripts/describe_card.py 678 676 675
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def table(filename: str, key: str) -> dict[int, dict]:
    path = ROOT / "vendor" / "cg" / filename
    if not path.exists():
        return {}
    return {
        int(entry[key]): entry
        for entry in json.loads(path.read_text(encoding="utf-8"))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", type=int, nargs="+")
    args = parser.parse_args()

    cards = table("cards.json", "cardId")
    attacks = table("attacks.json", "attackId")
    abilities = table("abilities.json", "abilityId")

    for card_id in args.ids:
        card = cards.get(card_id)
        if card is None:
            print(f"\n{card_id}: not found")
            continue
        print(f"\n=== {card_id} {card.get('name')}")
        for key, value in card.items():
            if key in ("attackIds", "abilityIds", "attacks", "abilities"):
                continue
            print(f"  {key}: {value}")
        for key in ("attackIds", "attacks"):
            for attack_id in card.get(key) or []:
                entry = attacks.get(int(attack_id)) or {}
                print(f"  ATTACK {attack_id}: {json.dumps(entry, ensure_ascii=False)}")
        for key in ("abilityIds", "abilities"):
            for ability_id in card.get(key) or []:
                entry = abilities.get(int(ability_id)) or {}
                print(f"  ABILITY {ability_id}: {json.dumps(entry, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
