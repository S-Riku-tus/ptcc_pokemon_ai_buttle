"""Measure the v15 attack-access guard against v14 on the same boards.

Two questions, one script.

1. **Footprint.** Drive a game with v14 and ask v15 for an answer on every
   identical board. With ``GRIMMSNARL_ROUTE_DISABLE=1`` the answer must be
   v14's on every decision; with the guard on, the differences are the change.
2. **Effect.** Play v15 against v14 with swapped seats and record the primary
   KPI the v14 autopsy asked for - the turn of the *first Shadow Bullet*, not
   the first attack - plus the guard's own counters.

    python scripts/probe_grimmsnarl_v15_route.py --arena-games 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_loader import load_dir_agent_module  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

CHAMPION = ROOT / "agents/grimmsnarl/grimmsnarl_ml_v14"
CHALLENGER = ROOT / "agents/grimmsnarl/grimmsnarl_ml_v15"
SHADOW_BULLET_ID = 937
MAIN_CONTEXT = 0
ATTACK_OPTION = 13


def load(agent_dir: Path, *, route_disabled: bool = False):
    previous = os.environ.get("GRIMMSNARL_ROUTE_DISABLE")
    if route_disabled:
        os.environ["GRIMMSNARL_ROUTE_DISABLE"] = "1"
    else:
        os.environ.pop("GRIMMSNARL_ROUTE_DISABLE", None)
    try:
        return load_dir_agent_module(agent_dir)
    finally:
        if previous is None:
            os.environ.pop("GRIMMSNARL_ROUTE_DISABLE", None)
        else:
            os.environ["GRIMMSNARL_ROUTE_DISABLE"] = previous


def deck_of(agent_dir: Path) -> list[int]:
    text = (agent_dir / "deck.csv").read_text(encoding="utf-8-sig")
    return [int(value) for value in text.split()]


def action_shape(observation: dict, index: int) -> str:
    select = observation.get("select") or {}
    options = select.get("option") or []
    context = int(select.get("context", -1))
    if not 0 <= index < len(options):
        return f"ctx{context}:invalid"
    option = options[index]
    kind = option.get("type")
    if kind == ATTACK_OPTION:
        return f"ctx{context}:attack:{option.get('attackId')}"
    return f"ctx{context}:type{kind}"


def own_turn(current: dict, seat: int) -> int:
    turn = int(current.get("turn", 0) or 0)
    first = int(current.get("firstPlayer", -1))
    if first < 0:
        return turn
    return (turn + 1) // 2 if first == seat else turn // 2


def single(action) -> int | None:
    if (
        isinstance(action, list)
        and len(action) == 1
        and isinstance(action[0], int)
    ):
        return action[0]
    return None


# ----- 1. same-board footprint ----------------------------------------------
def footprint(games: int, *, route_disabled: bool) -> dict:
    driver = load(CHAMPION).agent
    shadow = load(CHALLENGER, route_disabled=route_disabled)
    # Teacher-force the mirrored agent: it answers v14's boards, but its own
    # intra-turn history advances with the action the game actually played, so a
    # single difference cannot drift every later feature column.
    if getattr(shadow, "_RANKER", None) is not None:
        shadow._RANKER.teacher_forced = True
    deck = deck_of(CHAMPION)
    totals = {
        "games": 0,
        "decisions": 0,
        "single_pick_decisions": 0,
        "differences": 0,
        "by_shape": Counter(),
    }
    for _ in range(games):
        observation, start = battle_start(deck, deck)
        if observation is None:
            raise RuntimeError(f"battle_start failed: {start}")
        try:
            for _step in range(6000):
                current = observation["current"]
                if current["result"] >= 0:
                    break
                played = driver(observation)
                mirrored = shadow.agent(observation)
                totals["decisions"] += 1
                left, right = single(played), single(mirrored)
                if left is not None and right is not None:
                    totals["single_pick_decisions"] += 1
                    if left != right:
                        totals["differences"] += 1
                        totals["by_shape"][
                            f"{action_shape(observation, left)}"
                            f" -> {action_shape(observation, right)}"
                        ] += 1
                if left is not None:
                    shadow.observe_external(observation, left)
                observation = battle_select(list(played))
        finally:
            battle_finish()
        totals["games"] += 1
    totals["by_shape"] = dict(totals["by_shape"])
    totals["guard"] = shadow.diag_snapshot().get("attack_access", {})
    return totals


# ----- 2. head to head with the first-Shadow KPI -----------------------------
def arena(games: int) -> dict:
    challenger = load(CHALLENGER)
    champion = load(CHAMPION)
    modules = {"v15": challenger, "v14": champion}
    deck = deck_of(CHALLENGER)
    record = {name: {"wins": 0, "losses": 0, "draws": 0} for name in modules}
    turns: dict[str, list[int]] = defaultdict(list)
    attacks: dict[str, Counter] = {name: Counter() for name in modules}
    errors: Counter = Counter()

    for game in range(games):
        seats = ["v15", "v14"] if game % 2 == 0 else ["v14", "v15"]
        agents = [modules[name].agent for name in seats]
        first_shadow: dict[str, int | None] = {name: None for name in seats}
        observation, start = battle_start(deck, deck)
        if observation is None:
            raise RuntimeError(f"battle_start failed: {start}")
        result = -1
        try:
            for _step in range(6000):
                current = observation["current"]
                if current["result"] >= 0:
                    result = int(current["result"])
                    break
                seat = int(current["yourIndex"])
                name = seats[seat]
                try:
                    action = agents[seat](observation)
                except Exception as error:  # noqa: BLE001
                    errors[f"{name}:{type(error).__name__}"] += 1
                    result = 1 - seat
                    break
                index = single(action)
                select = observation.get("select") or {}
                if (
                    index is not None
                    and int(select.get("context", -1)) == MAIN_CONTEXT
                ):
                    option = (select.get("option") or [])[index]
                    if option.get("type") == ATTACK_OPTION:
                        attack_id = int(option.get("attackId", -1))
                        attacks[name][attack_id] += 1
                        if (
                            attack_id == SHADOW_BULLET_ID
                            and first_shadow[name] is None
                        ):
                            first_shadow[name] = own_turn(current, seat)
                try:
                    observation = battle_select(list(action))
                except Exception:  # noqa: BLE001
                    errors[f"{name}:illegal"] += 1
                    result = 1 - seat
                    break
        finally:
            battle_finish()
        for seat, name in enumerate(seats):
            if result == -1:
                record[name]["draws"] += 1
            elif result == seat:
                record[name]["wins"] += 1
            else:
                record[name]["losses"] += 1
            if first_shadow[name] is not None:
                turns[name].append(first_shadow[name])

    summary = {"games": games, "record": record, "errors": dict(errors)}
    for name in modules:
        series = turns[name]
        summary[name] = {
            "games_with_shadow_bullet": len(series),
            "mean_first_shadow_own_turn": (
                round(sum(series) / len(series), 3) if series else None
            ),
            "shadow_by_own_turn_2": (
                round(sum(1 for t in series if t <= 2) / games, 3)
            ),
            "shadow_by_own_turn_3": (
                round(sum(1 for t in series if t <= 3) / games, 3)
            ),
            "first_shadow_turn_4_plus": (
                round(sum(1 for t in series if t >= 4) / games, 3)
            ),
            "attacks": {str(k): v for k, v in sorted(attacks[name].items())},
            "diag": modules[name].diag_snapshot().get("attack_access", {}),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--footprint-games", type=int, default=4)
    parser.add_argument("--arena-games", type=int, default=20)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-arena", action="store_true")
    args = parser.parse_args()

    out: dict = {}
    if args.footprint_games:
        out["footprint_route_off"] = footprint(
            args.footprint_games, route_disabled=True
        )
        out["footprint_route_on"] = footprint(
            args.footprint_games, route_disabled=False
        )
    if args.arena_games and not args.skip_arena:
        out["arena"] = arena(args.arena_games)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
