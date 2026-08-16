"""Find a main-phase option list containing EVOLVE and dump it verbatim.

Evolution options carry no obvious target field, so which body an option would
evolve has to be established from the data before any analysis relies on it.

Usage:
  python scripts/peek_evolve_options.py <replay.json> --seat 1 --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}
OPT_EVOLVE = 9


def name(card_id):
    try:
        return str(CARDS.get(int(card_id), {}).get("name") or card_id)
    except (TypeError, ValueError):
        return str(card_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    shown = 0
    for pair in replay.get("steps") or []:
        payload = pair[args.seat]
        observation = payload.get("observation") or {}
        select = (observation or {}).get("select") or {}
        options = select.get("option") or []
        if not any(int(option.get("type", -1)) == OPT_EVOLVE
                   for option in options):
            continue
        current = observation.get("current") or {}
        players = current.get("players") or [{}, {}]
        your = int(current.get("yourIndex", args.seat))
        mine = players[your]
        active = mine.get("active") or []
        if isinstance(active, dict):
            active = [active]
        print("\n=== hand: " + ", ".join(
            f"{position}:{name(card.get('id'))}"
            for position, card in enumerate(mine.get("hand") or [])))
        print("    active: " + ", ".join(
            f"{name(card.get('id'))}#{card.get('serial')}" for card in active))
        print("    bench:  " + ", ".join(
            f"{position}:{name(card.get('id'))}#{card.get('serial')}"
            for position, card in enumerate(mine.get("bench") or [])))
        for position, option in enumerate(options):
            if int(option.get("type", -1)) == OPT_EVOLVE:
                print(f"    option[{position}] {json.dumps(option)}")
        shown += 1
        if shown >= args.limit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
