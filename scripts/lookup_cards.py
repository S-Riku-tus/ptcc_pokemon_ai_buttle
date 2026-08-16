"""Resolve card ids from names, or names from ids, against vendor/cg/cards.json.

Every analysis in this repo hard-codes card ids, and a wrong id fails silently
as a zero count rather than an error.  This is the check.

Usage:
  python scripts/lookup_cards.py --name "Ultra Ball" --name Poffin
  python scripts/lookup_cards.py --id 119 --id 121
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--id", type=int, action="append", default=[])
    args = parser.parse_args()

    cards = json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8"))
    for card in cards:
        card_id = int(card["cardId"])
        name = str(card.get("name") or "")
        if card_id in args.id or any(
            needle.lower() in name.lower() for needle in args.name
        ):
            print(f"{card_id:>6}  {name:36} {card.get('kind', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
