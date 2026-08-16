"""Run local games between two agent directories and measure development.

Win rate from a local arena is close to useless for ranking versions: the
native shuffle cannot be seeded, so two runs of the identical agent against the
identical opponent have swung by 30 points.  What *is* measurable here is a
within-game quantity - the own-turn on which a Dragapult first holds both
Phantom Dive colours - because it is one number per game rather than one bit,
and it is exactly the quantity the v2 features were built to change.

Usage:
  python scripts/arena_dragapult.py \
      --a agents/dragapult/dragapult_ml_v2 \
      --b agents/dragapult/dragapult_ml_v1 \
      --games 40 --report experiments/dragapult_ml_v2/arena.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loader import load_dir_agent_module  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

FIRE, PSYCHIC = 2, 5
DREEPY, DRAKLOAK, DRAGAPULT = 119, 120, 121
PHANTOM_DIVE = 154
OPT_ATTACK = 13
MAX_STEPS = 6000


def phantom_ready(player: dict[str, Any]) -> bool:
    bodies = (player.get("active") or []) + (player.get("bench") or [])
    for card in bodies:
        if not isinstance(card, dict) or int(card.get("id", -1)) != DRAGAPULT:
            continue
        energies = [int(v) for v in card.get("energies") or []]
        if FIRE in energies and PSYCHIC in energies:
            return True
    return False


def play(modules: list[Any], decks: list[list[int]]) -> dict[str, Any] | None:
    for module in modules:
        module.diag_reset()
    observation, _ = battle_start(list(decks[0]), list(decks[1]))
    if observation is None:
        return None
    own_turns: list[list[int]] = [[], []]
    ready_turn: list[int | None] = [None, None]
    dive_turn: list[int | None] = [None, None]
    exceptions = Counter()
    try:
        for _ in range(MAX_STEPS):
            select = observation.get("select")
            current = observation.get("current") or {}
            if select is None or int(current.get("result", -1)) >= 0:
                break
            seat = int(current.get("yourIndex", 0))
            players = current.get("players") or [{}, {}]
            turn = int(current.get("turn") or 0)
            if int(select.get("context", -1)) == 0 and turn not in own_turns[seat]:
                own_turns[seat].append(turn)
            if ready_turn[seat] is None and phantom_ready(players[seat]):
                ready_turn[seat] = len(own_turns[seat]) or 1
            try:
                action = modules[seat].agent(observation)
            except Exception as error:  # noqa: BLE001
                exceptions[f"seat{seat}:{type(error).__name__}"] += 1
                action = list(range(int(select.get("minCount") or 0)))
            if not isinstance(action, list):
                action = list(range(int(select.get("minCount") or 0)))
            options = select.get("option") or []
            if dive_turn[seat] is None:
                for index in action:
                    if not isinstance(index, int) or not 0 <= index < len(options):
                        continue
                    option = options[index]
                    if (int(option.get("type", -1)) == OPT_ATTACK
                            and int(option.get("attackId", -1)) == PHANTOM_DIVE):
                        dive_turn[seat] = len(own_turns[seat]) or 1
            try:
                observation = battle_select(action)
            except Exception as error:  # noqa: BLE001
                exceptions[f"seat{seat}:select:{type(error).__name__}"] += 1
                fallback = list(range(max(1, int(select.get("minCount") or 0))))
                observation = battle_select(fallback)
        result = int((observation.get("current") or {}).get("result", -1))
    finally:
        battle_finish()
    return {
        "result": result,
        "own_turns": [len(values) for values in own_turns],
        "phantom_ready_own_turn": ready_turn,
        "phantom_dive_own_turn": dive_turn,
        "exceptions": dict(exceptions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    module_a = load_dir_agent_module(args.a.resolve())
    module_b = load_dir_agent_module(args.b.resolve())
    deck_a = list(module_a.MY_DECK)
    deck_b = list(module_b.MY_DECK)

    games: list[dict[str, Any]] = []
    for index in range(args.games):
        # Alternate seats so the first-player advantage is shared evenly.
        a_seat = index % 2
        modules = [module_a, module_b] if a_seat == 0 else [module_b, module_a]
        decks = [deck_a, deck_b] if a_seat == 0 else [deck_b, deck_a]
        outcome = play(modules, decks)
        if outcome is None:
            continue
        outcome["a_seat"] = a_seat
        outcome["a_won"] = int(outcome["result"] == a_seat)
        games.append(outcome)
        print(f"[{index + 1}/{args.games}] result={outcome['result']} "
              f"a_seat={a_seat} a_won={outcome['a_won']} "
              f"ready={outcome['phantom_ready_own_turn']} "
              f"dive={outcome['phantom_dive_own_turn']}", flush=True)

    def side(key: str, is_a: bool) -> list[int]:
        values = []
        for game in games:
            seat = game["a_seat"] if is_a else 1 - game["a_seat"]
            value = game[key][seat]
            if value is not None:
                values.append(value)
        return values

    def describe(name: str, is_a: bool) -> dict[str, Any]:
        ready = side("phantom_ready_own_turn", is_a)
        dive = side("phantom_dive_own_turn", is_a)
        return {
            "agent": name,
            "games": len(games),
            "phantom_ready_rate": round(len(ready) / len(games), 3) if games else 0.0,
            "phantom_ready_mean_own_turn": (
                round(statistics.mean(ready), 3) if ready else None
            ),
            "phantom_ready_median_own_turn": (
                statistics.median(ready) if ready else None
            ),
            "phantom_dive_rate": round(len(dive) / len(games), 3) if games else 0.0,
            "phantom_dive_mean_own_turn": (
                round(statistics.mean(dive), 3) if dive else None
            ),
        }

    wins = sum(game["a_won"] for game in games)
    report = {
        "a": str(args.a), "b": str(args.b), "games": len(games),
        "a_record": f"{wins}-{len(games) - wins}",
        "a_win_rate": round(wins / len(games), 4) if games else 0.0,
        "a_stats": describe(str(args.a.name), True),
        "b_stats": describe(str(args.b.name), False),
        "exceptions": {
            key: value
            for game in games for key, value in game["exceptions"].items()
        },
        "detail": games,
    }
    print(json.dumps(
        {k: v for k, v in report.items() if k != "detail"},
        ensure_ascii=False, indent=2,
    ))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
