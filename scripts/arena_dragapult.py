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
OPT_PLAY = 7
OPT_CARD = 3
AREA_ACTIVE, AREA_BENCH = 4, 5
CTX_COUNTER, CTX_COUNTER_ANY = 13, 14
TRACKED = {
    1121: "ultra_ball", 1086: "poffin", 1152: "poke_pad",
    1097: "night_stretcher", 1120: "crushing_hammer",
    1227: "lillie", 1198: "crispin", 1182: "boss",
}
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
    # The teachers' win rate is a function of how many Phantom Dives a game
    # yields, so the count matters as much as whether one ever landed.
    dive_count = [0, 0]
    # Board width, not just the route: the teachers keep 1.56 Dragapult ex in
    # play on average and a one-attacker policy caps the Phantom Dive count.
    max_pult = [0, 0]
    max_line = [0, 0]
    card_counts: list[Counter] = [Counter(), Counter()]
    phantom_targets: list[Counter] = [Counter(), Counter()]
    spread_targets: list[Counter] = [Counter(), Counter()]
    remaining_prizes = [6, 6]
    prizes_started = False
    exceptions = Counter()
    try:
        for _ in range(MAX_STEPS):
            select = observation.get("select")
            current = observation.get("current") or {}
            if select is None or int(current.get("result", -1)) >= 0:
                break
            seat = int(current.get("yourIndex", 0))
            players = current.get("players") or [{}, {}]
            prize_zones = [players[index].get("prize") for index in (0, 1)]
            if all(isinstance(zone, list) for zone in prize_zones):
                lengths = [len(zone) for zone in prize_zones]
                if prizes_started:
                    remaining_prizes = lengths
                elif lengths == [6, 6]:
                    prizes_started = True
                    remaining_prizes = lengths
            turn = int(current.get("turn") or 0)
            if int(select.get("context", -1)) == 0 and turn not in own_turns[seat]:
                own_turns[seat].append(turn)
            if ready_turn[seat] is None and phantom_ready(players[seat]):
                ready_turn[seat] = len(own_turns[seat]) or 1
            bodies = (
                (players[seat].get("active") or [])
                + (players[seat].get("bench") or [])
            )
            ids = [
                int(card.get("id", -1)) for card in bodies
                if isinstance(card, dict)
            ]
            max_pult[seat] = max(max_pult[seat], ids.count(DRAGAPULT))
            max_line[seat] = max(max_line[seat], sum(
                1 for value in ids if value in (DREEPY, DRAKLOAK, DRAGAPULT)
            ))
            try:
                action = modules[seat].agent(observation)
            except Exception as error:  # noqa: BLE001
                exceptions[f"seat{seat}:{type(error).__name__}"] += 1
                action = list(range(int(select.get("minCount") or 0)))
            if not isinstance(action, list):
                action = list(range(int(select.get("minCount") or 0)))
            options = select.get("option") or []
            for index in action:
                if not isinstance(index, int) or not 0 <= index < len(options):
                    continue
                option = options[index]
                if (int(option.get("type", -1)) == OPT_ATTACK
                        and int(option.get("attackId", -1)) == PHANTOM_DIVE):
                    dive_count[seat] += 1
                    if dive_turn[seat] is None:
                        dive_turn[seat] = len(own_turns[seat]) or 1
                    opponent_active = players[1 - seat].get("active") or []
                    if opponent_active and isinstance(opponent_active[0], dict):
                        phantom_targets[seat][
                            str(int(opponent_active[0].get("id", -1)))
                        ] += 1
                if (
                    int(select.get("context", -1)) in (CTX_COUNTER, CTX_COUNTER_ANY)
                    and int(option.get("type", -1)) == OPT_CARD
                ):
                    player_index = int(option.get("playerIndex", seat))
                    area = int(option.get("area", -1))
                    zone_name = (
                        "active" if area == AREA_ACTIVE
                        else "bench" if area == AREA_BENCH
                        else ""
                    )
                    zone = (
                        players[player_index].get(zone_name) or []
                        if zone_name and player_index in (0, 1)
                        else []
                    )
                    slot = int(option.get("index", -1))
                    if player_index != seat and 0 <= slot < len(zone):
                        target = zone[slot]
                        if isinstance(target, dict):
                            spread_targets[seat][
                                str(int(target.get("id", -1)))
                            ] += 1
                if (int(option.get("type", -1)) == OPT_PLAY
                        and int(select.get("context", -1)) == 0):
                    hand = players[seat].get("hand") or []
                    slot = int(option.get("index", -1))
                    if 0 <= slot < len(hand):
                        card_id = int(hand[slot].get("id", -1))
                        if card_id in TRACKED:
                            card_counts[seat][TRACKED[card_id]] += 1
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
        "phantom_dive_count": dive_count,
        "max_dragapult": max_pult,
        "max_line": max_line,
        # Censored by the engine's final transition, identically to the
        # teacher replay analysis.  This is still useful for comparing prize
        # conversion across arena candidates.
        "prizes_taken": [6 - value for value in remaining_prizes],
        "card_counts": [dict(counter) for counter in card_counts],
        "phantom_targets": [dict(counter) for counter in phantom_targets],
        "spread_targets": [dict(counter) for counter in spread_targets],
        "exceptions": dict(exceptions),
    }


def load_deck(module: Any, agent_dir: Path) -> list[int]:
    """Read either an exported ``MY_DECK`` or the standard deck.csv bundle."""
    exported = getattr(module, "MY_DECK", None)
    if exported is not None:
        deck = [int(card_id) for card_id in exported]
    else:
        path = agent_dir / "deck.csv"
        deck = [
            int(line.strip())
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    if len(deck) != 60:
        raise ValueError(f"{agent_dir} has {len(deck)} cards, expected 60")
    return deck


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    module_a = load_dir_agent_module(args.a.resolve())
    module_b = load_dir_agent_module(args.b.resolve())
    deck_a = load_deck(module_a, args.a)
    deck_b = load_deck(module_b, args.b)

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

    def counts(is_a: bool) -> dict[str, Any]:
        values = [
            game["phantom_dive_count"][
                game["a_seat"] if is_a else 1 - game["a_seat"]
            ]
            for game in games
        ]
        total = len(values) or 1
        return {
            "mean": round(statistics.mean(values), 3) if values else 0.0,
            "p_zero": round(sum(1 for v in values if v == 0) / total, 3),
            "p_one": round(sum(1 for v in values if v == 1) / total, 3),
            "p_two_plus": round(sum(1 for v in values if v >= 2) / total, 3),
            "p_four_plus": round(sum(1 for v in values if v >= 4) / total, 3),
            "implied_win_rate": round(
                sum(1 for v in values if v == 0) / total * 0.108
                + sum(1 for v in values if v == 1) / total * 0.231
                + sum(1 for v in values if v >= 2) / total * 0.731, 3),
        }

    def board(is_a: bool, key: str) -> float:
        values = [
            game[key][game["a_seat"] if is_a else 1 - game["a_seat"]]
            for game in games
        ]
        return round(statistics.mean(values), 3) if values else 0.0

    def cards(is_a: bool) -> dict[str, float]:
        total: Counter = Counter()
        for game in games:
            seat = game["a_seat"] if is_a else 1 - game["a_seat"]
            total.update(game["card_counts"][seat])
        return {
            key: round(value / max(1, len(games)), 3)
            for key, value in sorted(total.items())
        }

    def pooled(is_a: bool, key: str) -> dict[str, float]:
        total: Counter = Counter()
        for game in games:
            seat = game["a_seat"] if is_a else 1 - game["a_seat"]
            total.update(game[key][seat])
        return {
            name: round(value / max(1, len(games)), 3)
            for name, value in sorted(total.items())
        }

    def prize_mean(is_a: bool, own: bool) -> float:
        values = []
        for game in games:
            seat = game["a_seat"] if is_a else 1 - game["a_seat"]
            if not own:
                seat = 1 - seat
            values.append(game["prizes_taken"][seat])
        return round(statistics.mean(values), 3) if values else 0.0

    def dive_conditionals(is_a: bool) -> dict[str, dict[str, float | int | None]]:
        rows: dict[str, dict[str, float | int | None]] = {}
        buckets = {
            "zero": lambda value: value == 0,
            "one": lambda value: value == 1,
            "two_plus": lambda value: value >= 2,
            "four_plus": lambda value: value >= 4,
        }
        for name, keep in buckets.items():
            block = []
            for game in games:
                seat = game["a_seat"] if is_a else 1 - game["a_seat"]
                if keep(game["phantom_dive_count"][seat]):
                    won = game["result"] == seat
                    block.append(int(won))
            rows[name] = {
                "games": len(block),
                "win_rate": round(sum(block) / len(block), 3) if block else None,
            }
        return rows

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
            # The teachers win 0.108 / 0.231 / 0.731 of games with 0 / 1 / 2+
            # Phantom Dives, so this distribution is the strength estimate the
            # unseedable head-to-head record cannot give.
            "phantom_dive_counts": counts(is_a),
            "max_dragapult_mean": board(is_a, "max_dragapult"),
            "max_line_mean": board(is_a, "max_line"),
            "prizes_taken_mean_censored": prize_mean(is_a, True),
            "prizes_conceded_mean_censored": prize_mean(is_a, False),
            "win_rate_by_dive_bucket": dive_conditionals(is_a),
            "cards_per_game": cards(is_a),
            "phantom_targets_per_game": pooled(is_a, "phantom_targets"),
            "spread_counters_per_game": pooled(is_a, "spread_targets"),
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
