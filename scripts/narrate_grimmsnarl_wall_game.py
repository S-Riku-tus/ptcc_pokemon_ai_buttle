"""Turn-by-turn narrative of one stored game from our seat.

Written for the wall matchups (Crustle / Cornerstone Mask Ogerpon ex) where the
aggregate counters say we spend turns for zero damage but not what the turn was
spent on.  Prints, per own turn, the Active pair, the prize counts, the deck
counts and every action we took with the card it was taken with.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor", ROOT / "scripts",
             ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v25"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402

NAMES: dict[int, str] = {}


def load_names() -> None:
    cards = json.loads((ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8"))
    for card in cards:
        NAMES[int(card["cardId"])] = card["name"]


def name(card_id: int) -> str:
    return NAMES.get(int(card_id), f"#{card_id}")


def body(card: dict[str, Any]) -> str:
    return f"{name(int(card.get('id', -1)))}({card.get('hp')})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--seat", type=int, required=True)
    parser.add_argument("--from-turn", type=int, default=0)
    parser.add_argument("--to-turn", type=int, default=10_000)
    args = parser.parse_args()

    load_names()
    path = (ROOT / args.run / "episodes" / args.episode / "replay"
            / f"episode_{args.episode}.json")
    replay = json.loads(path.read_text(encoding="utf-8"))
    steps = replay.get("steps") or []
    seat = args.seat
    print(f"episode {args.episode} seat {seat} rewards {replay.get('rewards')}")

    shown = -1
    for index, step in enumerate(steps[:-1]):
        record = step[seat] if seat < len(step) else None
        if not record or record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        current = observation.get("current") or {}
        if not options or not current.get("players"):
            continue
        turn = int(current.get("turn", -1))
        if not args.from_turn <= turn <= args.to_turn:
            continue
        us = current["players"][seat]
        them = current["players"][1 - seat]
        if turn != shown:
            shown = turn
            ours = mf._cards(us, "active")
            theirs = mf._cards(them, "active")
            print(
                f"\n--- turn {turn} | us {body(ours[0]) if ours else '-'} "
                f"vs {body(theirs[0]) if theirs else '-'} | "
                f"prizes {len(us.get('prize') or [])}-{len(them.get('prize') or [])} | "
                f"deck {us.get('deckCount')}-{them.get('deckCount')} | "
                f"their bench {[body(c) for c in mf._cards(them, 'bench')]} | "
                f"our bench {[body(c) for c in mf._cards(us, 'bench')]}"
            )
        action = (steps[index + 1][seat] or {}).get("action")
        picked = [int(v) for v in action
                  if isinstance(v, int) and 0 <= int(v) < len(options)] \
            if isinstance(action, list) else []
        for choice in picked:
            option = options[choice]
            try:
                kind = mf.action_type(current, option, select)
            except Exception:  # noqa: BLE001
                kind = "?"
            card = mf.candidate_card(current, option, select) or {}
            target = mf.candidate_target(current, option) or {}
            bits = [kind]
            if card:
                bits.append(name(int(card.get("id", -1))))
            if target:
                bits.append(f"-> {name(int(target.get('id', -1)))}")
            if kind == "attack":
                bits.append(f"attackId={option.get('attackId')}")
            print("    " + " ".join(bits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
