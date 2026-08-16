"""Turn-by-turn trace of one game from our seat.

Aggregate counters say a game went wrong; they do not say what the board looked
like when it did.  This prints, for each of our own turns: our Active and its
Energy, our bench, the opponent's Active, prizes, and every main-phase action we
took, so a single loss can be read like a game log.

Usage:
  python scripts/trace_dragapult_game.py \
      data/submissions/submission_55550682_dragapult_v2 93603391
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}
ATTACKS = {
    int(attack["attackId"]): attack
    for attack in json.loads(
        (ROOT / "vendor" / "cg" / "attacks.json").read_text(encoding="utf-8")
    )
}
COLOR = {1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D", 8: "M"}
TYPE_NAME = {
    7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 11: "DISCARD",
    12: "RETREAT", 13: "ATTACK", 14: "END", 3: "CARD", 0: "NUMBER",
}
AREA = {2: "hand", 3: "discard", 4: "active", 5: "bench", 7: "stadium"}


def name(card_id: Any) -> str:
    try:
        return str(CARDS.get(int(card_id), {}).get("name") or card_id)
    except (TypeError, ValueError):
        return str(card_id)


def describe(card: Any) -> str:
    if not isinstance(card, dict):
        return "-"
    energies = "".join(COLOR.get(int(value), "?")
                       for value in (card.get("energies") or []))
    damage = ""
    hp, max_hp = card.get("hp"), card.get("maxHp")
    if isinstance(hp, int) and isinstance(max_hp, int) and hp < max_hp:
        damage = f" [{hp}/{max_hp}]"
    return f"{name(card.get('id'))}{'(' + energies + ')' if energies else ''}{damage}"


def label(observation: dict[str, Any], option: dict[str, Any]) -> str:
    option_type = int(option.get("type", -1))
    prefix = TYPE_NAME.get(option_type, f"T{option_type}")
    if option_type == 13:
        attack_id = int(option.get("attackId", -1))
        return f"ATTACK {ATTACKS.get(attack_id, {}).get('name', attack_id)}"
    if option_type in (14, 12):
        return prefix
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    mine = players[your] if your in (0, 1) else {}
    index = int(option.get("index", -1))
    if option_type == 10:
        area = int(option.get("area", -1))
        zone = {
            2: mine.get("hand"), 3: mine.get("discard"), 4: mine.get("active"),
            5: mine.get("bench"), 7: current.get("stadium"),
        }.get(area) or []
        card = zone[index] if isinstance(zone, list) and 0 <= index < len(zone) else {}
        return f"ABILITY {name((card or {}).get('id'))} @{AREA.get(area, area)}"
    hand = mine.get("hand") or []
    card = hand[index] if 0 <= index < len(hand) else {}
    return f"{prefix} {name((card or {}).get('id'))}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("episode", type=int)
    parser.add_argument("--seat", type=int)
    parser.add_argument("--hand", action="store_true",
                        help="Also print our hand at the start of each turn.")
    args = parser.parse_args()

    seat = args.seat
    if seat is None:
        for row in csv.DictReader(
            (args.run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()
        ):
            if int(row["episode_id"]) == args.episode:
                seat = int(row["detected_submission_agent_index"])
                break
    if seat is None:
        raise SystemExit("seat not found in manifest; pass --seat")

    path = (args.run / "episodes" / str(args.episode) / "replay"
            / f"episode_{args.episode}.json")
    replay = json.loads(path.read_text(encoding="utf-8"))
    steps = replay.get("steps") or []
    rewards = replay.get("rewards") or [0, 0]
    print(f"episode {args.episode}  our seat {seat}  rewards {rewards}")

    seen: set[int] = set()
    own_turn = 0
    for index, pair in enumerate(steps):
        payload = pair[seat]
        observation = payload.get("observation") or {}
        if not isinstance(observation, dict):
            continue
        current = observation.get("current") or {}
        players = current.get("players") or []
        if len(players) != 2:
            continue
        your = int(current.get("yourIndex", seat))
        mine, theirs = players[your], players[1 - your]
        turn = current.get("turn")
        if payload.get("status") != "ACTIVE":
            continue
        if isinstance(turn, int) and turn not in seen:
            seen.add(turn)
            own_turn += 1
            active = mine.get("active") or []
            if isinstance(active, dict):
                active = [active]
            their_active = theirs.get("active") or []
            if isinstance(their_active, dict):
                their_active = [their_active]
            bench = ", ".join(describe(card) for card in (mine.get("bench") or []))
            print(f"\n--- own turn {own_turn} (shared turn {turn})  "
                  f"prizes {len(mine.get('prize') or [])}"
                  f"-{len(theirs.get('prize') or [])}  "
                  f"deck {len(mine.get('deck') or [])}")
            print(f"    us   {describe(active[0]) if active else '-':30} "
                  f"bench: {bench}")
            print(f"    them {describe(their_active[0]) if their_active else '-':30} "
                  f"bench: "
                  f"{', '.join(describe(card) for card in (theirs.get('bench') or []))}")
            stadium = current.get("stadium")
            if stadium:
                print(f"    stadium {describe(stadium if isinstance(stadium, dict) else {})}")
            if args.hand:
                print("    hand "
                      + ", ".join(name(card.get("id"))
                                  for card in (mine.get("hand") or [])
                                  if isinstance(card, dict)))

        select = observation.get("select")
        if select is None:
            continue
        action = (steps[index + 1][seat].get("action")
                  if index + 1 < len(steps) else None)
        if not isinstance(action, list) or len(action) != 1:
            continue
        options = select.get("option") or []
        picked = int(action[0])
        if not 0 <= picked < len(options):
            continue
        context = int(select.get("context", -1))
        if context != 0:
            continue
        print(f"      -> {label(observation, options[picked])}"
              f"   (of {len(options)}: "
              f"{'; '.join(label(observation, option) for option in options[:8])}"
              f"{' ...' if len(options) > 8 else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
