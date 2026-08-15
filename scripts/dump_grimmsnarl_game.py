"""Turn-by-turn account of one stored episode from our seat.

Written for the v27 loss autopsy: the aggregate tables say which cells are
losing, and this says what actually happened inside a named game - what was on
each board at the start of every one of our turns, what we did with the turn,
and what the opponent did with theirs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "agents/grimmsnarl/grimmsnarl_ml_v22"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402

NAMES = {
    7: "DarkEnergy", 646: "Impidimp", 647: "Morgrem", 648: "GrimmsnarlEX",
    860: "Snorunt", 104: "Froslass", 112: "Munkidori", 1079: "RareCandy",
    1080: "UnfairStamp", 1086: "Poffin", 1097: "NightStretcher",
    1122: "Pokegear", 1137: "ToolScrapper", 1152: "PokePad", 1182: "Boss",
    1219: "Petrel", 1227: "Lillie", 1231: "Dawn", 1259: "SpikemuthGym",
}


def name(card: dict[str, Any]) -> str:
    ident = int(card.get("id", -1))
    label = NAMES.get(ident) or (card.get("name") or f"#{ident}")
    energy = mf._dark_energy_count(card) if ident else 0
    damage = card.get("damage") or 0
    extras = []
    if energy:
        extras.append(f"E{energy}")
    if damage:
        extras.append(f"dmg{damage}")
    return f"{label}({','.join(extras)})" if extras else str(label)


def side(player: dict[str, Any]) -> str:
    active = mf._cards(player, "active")
    bench = mf._cards(player, "bench")
    prize = player.get("prize")
    prizes = len(prize) if isinstance(prize, list) else prize
    return (
        f"A={name(active[0]) if active else '-'} "
        f"B=[{', '.join(name(c) for c in bench)}] "
        f"prizes={prizes} deck={player.get('deckCount')} "
        f"hand={len(player.get('hand') or []) if isinstance(player.get('hand'), list) else '?'}"
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--full", action="store_true",
                        help="Print every decision, not just turn summaries.")
    args = parser.parse_args()

    run = args.run if args.run.is_absolute() else ROOT / args.run
    seat = None
    for row in csv.DictReader((run / "manifest.csv").open(encoding="utf-8-sig")):
        if int(row["episode_id"]) == args.episode:
            seat = int(row["detected_submission_agent_index"])
    if seat is None:
        print("episode not in manifest")
        return 1
    path = run / "episodes" / str(args.episode) / "replay" / f"episode_{args.episode}.json"
    replay = json.loads(path.read_text(encoding="utf-8"))
    steps = replay["steps"]
    print(f"episode {args.episode} seat {seat} rewards {replay.get('rewards')}")

    seen_turn = None
    for index, step in enumerate(steps[:-1]):
        for actor in (0, 1):
            if actor >= len(step) or actor >= len(steps[index + 1]):
                continue
            record = step[actor] or {}
            if record.get("status") != "ACTIVE":
                continue
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            current = observation.get("current") or {}
            players = current.get("players") or []
            options = list(select.get("option") or [])
            if len(players) < 2 or not options:
                continue
            turn = int(current.get("turn", -1))
            action = (steps[index + 1][actor] or {}).get("action")
            picked = [
                int(v) for v in action
                if isinstance(v, int) and 0 <= int(v) < len(options)
            ] if isinstance(action, list) else []
            if turn != seen_turn:
                seen_turn = turn
                mover = "US" if actor == seat else "THEM"
                print(f"\n--- turn {turn} ({mover} to act), "
                      f"firstPlayer={current.get('firstPlayer')} ---")
                print(f"  us:   {side(players[seat])}")
                print(f"  them: {side(players[1 - seat])}")
            for choice in picked:
                option = options[choice]
                try:
                    kind = mf.action_type(current, option, select)
                except Exception:  # noqa: BLE001
                    kind = "?"
                card = mf.candidate_card(current, option, select) or {}
                who = "US " if actor == seat else "opp"
                if not args.full and kind in ("?",) and who == "opp":
                    continue
                label = NAMES.get(int(card.get("id", -1)), card.get("name", ""))
                attack = option.get("attackId")
                print(
                    f"    [{who}] ctx={select.get('context')} {kind:<9}"
                    f" {label or ''}"
                    f"{f' attack={attack}' if attack else ''}"
                )
    final = steps[-1]
    print("\n--- final ---")
    for actor in (0, 1):
        rec = final[actor] or {}
        print(f"  seat {actor}: status={rec.get('status')} reward={rec.get('reward')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
