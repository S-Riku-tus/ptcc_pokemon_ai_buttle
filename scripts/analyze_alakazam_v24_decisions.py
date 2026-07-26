#!/usr/bin/env python3
"""Audit the v24 target-ordering and Dudunsparce-pivot failure classes.

The existing ladder audit measures broad attack and Boss frequencies.  This
script keeps the units closer to the reported failures:

* an Active single-prizer KO versus a Boss KO on a benched ex;
* an un-KO-able Active ex versus a lower-HP, same-card benched ex;
* Boss-to-Munkidori decisions while Grimmsnarl ex is visible;
* low-deck Active Dudunsparce states where attach -> retreat -> Alakazam KO is
  legal.

When ``--agent-dir`` is supplied, the agent's own deterministic KO planner is
also evaluated on each relevant MAIN state.  This exposes whether a failure is
target ranking, route feasibility, or action sequencing.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loader import load_dir_agent  # noqa: E402
from cg.api import all_card_data  # noqa: E402

import analyze_alakazam_ladder_strategy as ladder  # noqa: E402


ALAKAZAM = 743
DUDUNSPARCE = 66
MUNKIDORI = 112
GRIMMSNARL_EX = 648
BOSS_ORDERS = 1182
ENERGY_IDS = {5, 13, 19}
ENRICHING_ENERGY = 13
MAIN = 0
PLAY = 7
ATTACK = 13
RETREAT = 12
ACTIVE = 4
BENCH = 5

CARD = {int(card.cardId): card for card in all_card_data()}


def _prizes(card: dict[str, Any] | None) -> int:
    data = CARD.get(int((card or {}).get("id", -1)))
    if data is None:
        return 1
    if bool(getattr(data, "megaEx", False)):
        return 3
    return 2 if bool(getattr(data, "ex", False)) else 1


def _is_ex(card: dict[str, Any] | None) -> bool:
    return _prizes(card) >= 2


def _energy_count(card: dict[str, Any] | None) -> int:
    return len((card or {}).get("energies") or [])


def _serial(card: dict[str, Any] | None) -> int:
    return int((card or {}).get("serial", -1))


def _card_row(card: dict[str, Any] | None) -> dict[str, Any] | None:
    if not card:
        return None
    return {
        "id": int(card.get("id", -1)),
        "serial": _serial(card),
        "hp": int(card.get("hp") or 0),
        "max_hp": int(card.get("maxHp") or card.get("hp") or 0),
        "prizes": _prizes(card),
        "energy": _energy_count(card),
    }


def _seat_by_episode(run_dir: Path, submission_id: int) -> dict[int, int]:
    seats: dict[int, int] = {}
    with (run_dir / "episodes.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            episode_id = int(row["episode_id"])
            if int(row["agent_0_submission_id"]) == submission_id:
                seats[episode_id] = 0
            elif int(row["agent_1_submission_id"]) == submission_id:
                seats[episode_id] = 1
    return seats


def _iter_replays(run_dir: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    for path in sorted((run_dir / "episodes").glob("*/replay/*.json")):
        yield int(path.parents[1].name), json.loads(path.read_text(encoding="utf-8"))


def _option_card(
    current: dict[str, Any], seat: int, option: dict[str, Any]
) -> dict[str, Any] | None:
    return ladder._action_card(current, seat, option)


def _option_target(
    current: dict[str, Any], option: dict[str, Any]
) -> dict[str, Any] | None:
    return ladder._target_card(current, option)


def _semantic(
    current: dict[str, Any], seat: int, option: dict[str, Any]
) -> dict[str, Any]:
    card = _option_card(current, seat, option)
    target = _option_target(current, option)
    return {
        "type": int(option.get("type", -1)),
        "card_id": int((card or {}).get("id", -1)),
        "attack_id": int(option.get("attackId", -1)),
        "target_id": int((target or {}).get("id", -1)),
        "target_serial": _serial(target),
        "in_play_area": int(option.get("inPlayArea", option.get("area", -1))),
        "in_play_index": int(option.get("inPlayIndex", -1)),
    }


def _legal_boss(
    current: dict[str, Any], seat: int, options: list[dict[str, Any]]
) -> bool:
    return any(
        int(option.get("type", -1)) == PLAY
        and int((_option_card(current, seat, option) or {}).get("id", -1))
        == BOSS_ORDERS
        for option in options
    )


def _ready_alakazam(card: dict[str, Any] | None) -> bool:
    return bool(
        card
        and int(card.get("id", -1)) == ALAKAZAM
        and _energy_count(card) >= 1
    )


def _attach_to_active_dudun(
    current: dict[str, Any], seat: int, option: dict[str, Any]
) -> bool:
    card = _option_card(current, seat, option)
    return bool(
        int((card or {}).get("id", -1)) in ENERGY_IDS
        and int(option.get("inPlayArea", -1)) == ACTIVE
        and int(option.get("inPlayIndex", -1)) == 0
    )


def _dudun_ability(
    current: dict[str, Any], option: dict[str, Any]
) -> bool:
    target = _option_target(current, option)
    return int((target or {}).get("id", -1)) == DUDUNSPARCE


def _post_attach_hand(hand: int, energy_id: int) -> int:
    return hand + 3 if energy_id == ENRICHING_ENERGY else max(0, hand - 1)


def _plan_row(policy: Any, target: Any) -> dict[str, Any]:
    plan = policy._ko_route_plan(target)
    return {
        "target_id": int(getattr(target, "id", -1)),
        "target_serial": int(getattr(target, "serial", -1)),
        "reachable": bool(plan.get("reachable")),
        "ko": bool(plan.get("ko")),
        "winning": bool(plan.get("winning")),
        "damage": int(plan.get("damage", 0)),
        "hand": int(plan.get("hand", 0)),
        "required_hand": int(plan.get("required_hand", 0)),
        "actions": sorted(plan.get("actions") or []),
        "next_actions": sorted(plan.get("next_actions") or []),
        "action_count": int(plan.get("action_count", 0)),
        "deck_cost": int(plan.get("deck_cost", 0)),
        "priority": int(policy._target_priority_score(target)),
    }


def _policy_snapshot(
    module: Any | None, observation: dict[str, Any]
) -> dict[str, Any]:
    if module is None:
        return {}
    try:
        policy_module = module.fallback_policy
        parsed = policy_module.to_observation_class(observation)
        policy = policy_module.AlakazamPolicy(parsed)
        targets = [
            target
            for target in policy.opponent.active + policy.opponent.bench
            if target is not None
        ]
        plans = [_plan_row(policy, target) for target in targets]
        chosen = policy._chosen_ko_plan()
        return {
            "plans": plans,
            "chosen_target_id": (
                int(getattr(chosen.get("target"), "id", -1)) if chosen else None
            ),
            "chosen_target_serial": (
                int(getattr(chosen.get("target"), "serial", -1)) if chosen else None
            ),
            "chosen_actions": sorted(chosen.get("actions") or []) if chosen else [],
            "chosen_next_actions": (
                sorted(chosen.get("next_actions") or []) if chosen else []
            ),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _main_state(
    observation: dict[str, Any],
    *,
    episode_id: int,
    seat: int,
    step_index: int,
    selected: dict[str, Any],
    options: list[dict[str, Any]],
    policy_module: Any | None,
) -> dict[str, Any]:
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    me, opponent = players[seat], players[1 - seat]
    own_active = (ladder._cards(me, "active") or [None])[0]
    opponent_active = (ladder._cards(opponent, "active") or [None])[0]
    opponent_bench = ladder._cards(opponent, "bench")
    hand = int(me.get("handCount") or len(me.get("hand") or []))
    can_powerful_hand = _ready_alakazam(own_active) and any(
        int(option.get("type", -1)) == ATTACK for option in options
    )
    active_damage = 20 * hand if can_powerful_hand else 0
    boss_damage = 20 * max(0, hand - 1) if can_powerful_hand else 0
    attach_options = [
        option
        for option in options
        if _attach_to_active_dudun(current, seat, option)
    ]
    bench_alakazam = [
        card for card in ladder._cards(me, "bench") if _ready_alakazam(card)
    ]
    pivot_kos = []
    if int((own_active or {}).get("id", -1)) == DUDUNSPARCE and opponent_active:
        for option in attach_options:
            energy = _option_card(current, seat, option)
            post_hand = _post_attach_hand(hand, int((energy or {}).get("id", -1)))
            if bench_alakazam and 20 * post_hand >= int(opponent_active.get("hp") or 0):
                pivot_kos.append(
                    {
                        "energy_id": int((energy or {}).get("id", -1)),
                        "post_attach_hand": post_hand,
                        "damage": 20 * post_hand,
                    }
                )
    ex_upgrades = []
    active_prizes = _prizes(opponent_active)
    active_ko = bool(
        opponent_active
        and active_damage >= int(opponent_active.get("hp") or 0) > 0
    )
    for target in opponent_bench:
        if not _is_ex(target):
            continue
        target_hp = int(target.get("hp") or 0)
        target_ko = boss_damage >= target_hp > 0
        if target_ko and (
            not active_ko or _prizes(target) > active_prizes
        ):
            ex_upgrades.append(_card_row(target))
    same_ex_lower_hp = [
        _card_row(target)
        for target in opponent_bench
        if opponent_active
        and _is_ex(opponent_active)
        and int(target.get("id", -1)) == int(opponent_active.get("id", -2))
        and active_damage < int(opponent_active.get("hp") or 0)
        and boss_damage >= int(target.get("hp") or 0) > 0
    ]
    return {
        "episode_id": episode_id,
        "step_index": step_index,
        "turn": int(current.get("turn", -1)),
        "turn_action_count": int(current.get("turnActionCount", -1)),
        "hand": hand,
        "deck": int(me.get("deckCount") or len(me.get("deck") or [])),
        "own_prizes": len(me.get("prize") or []),
        "opp_prizes": len(opponent.get("prize") or []),
        "own_active": _card_row(own_active),
        "own_bench": [_card_row(card) for card in ladder._cards(me, "bench")],
        "opp_active": _card_row(opponent_active),
        "opp_bench": [_card_row(card) for card in opponent_bench],
        "selected": _semantic(current, seat, selected),
        "legal_boss": _legal_boss(current, seat, options),
        "active_damage": active_damage,
        "boss_damage": boss_damage,
        "active_ko": active_ko,
        "ex_upgrades": ex_upgrades,
        "same_ex_lower_hp": same_ex_lower_hp,
        "active_dudun_low_deck_pivot_ko": bool(
            int((own_active or {}).get("id", -1)) == DUDUNSPARCE
            and int(me.get("deckCount") or 0) <= 7
            and pivot_kos
        ),
        "pivot_kos": pivot_kos,
        "dudun_ability_offered": any(
            _dudun_ability(current, option) for option in options
        ),
        "retreat_offered": any(
            int(option.get("type", -1)) == RETREAT for option in options
        ),
        "policy": _policy_snapshot(policy_module, observation),
    }


def analyze_run(
    run_dir: Path,
    submission_id: int,
    policy_module: Any | None,
) -> dict[str, Any]:
    seats = _seat_by_episode(run_dir, submission_id)
    games = 0
    wins = 0
    main_states: list[dict[str, Any]] = []
    turn_actions: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    boss_targets: dict[tuple[int, int], dict[str, Any]] = {}
    grim_games: set[int] = set()

    for episode_id, replay in _iter_replays(run_dir):
        seat = seats.get(episode_id)
        if seat is None:
            continue
        games += 1
        reward = (replay.get("rewards") or [0, 0])[seat]
        wins += int(reward is not None and reward > 0)
        decks = ladder._initial_decks(replay)
        if any(
            int(card.get("id", -1)) == GRIMMSNARL_EX
            for card in (decks[1 - seat] if len(decks) == 2 else [])
            if isinstance(card, dict)
        ):
            grim_games.add(episode_id)

        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or step[seat].get("status") != "ACTIVE":
                continue
            observation = (step[seat].get("observation") or {})
            select = observation.get("select") or {}
            selected, options = ladder._selected_option(steps, step_index, seat)
            if selected is None:
                continue
            current = observation.get("current") or {}
            turn = int(current.get("turn", -1))
            if ladder._effect_card_id(select) == BOSS_ORDERS:
                target = _option_target(current, selected)
                if target is not None:
                    boss_targets[(episode_id, turn)] = {
                        "target": _card_row(target),
                        "step_index": step_index,
                    }
            if (
                int(select.get("type", -1)) != ladder.MAIN_SELECT_TYPE
                or int(select.get("context", -1)) != ladder.MAIN_SELECT_CONTEXT
            ):
                continue
            row = _main_state(
                observation,
                episode_id=episode_id,
                seat=seat,
                step_index=step_index,
                selected=selected,
                options=options,
                policy_module=policy_module,
            )
            main_states.append(row)
            turn_actions[(episode_id, turn)].append(
                {
                    "step_index": step_index,
                    "turn_action_count": row["turn_action_count"],
                    "hand": row["hand"],
                    "deck": row["deck"],
                    **row["selected"],
                }
            )

    by_turn: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in main_states:
        by_turn[(row["episode_id"], row["turn"])].append(row)

    upgrade_turns = []
    same_ex_turns = []
    dudun_traps = []
    grim_munk_boss = []
    for key, states in by_turn.items():
        episode_id, turn = key
        target_event = boss_targets.get(key)
        target_id = int(((target_event or {}).get("target") or {}).get("id", -1))
        played_boss = target_event is not None
        state_with_upgrades = max(states, key=lambda row: len(row["ex_upgrades"]))
        if state_with_upgrades["ex_upgrades"]:
            upgrade_serials = {
                int(row["serial"]) for row in state_with_upgrades["ex_upgrades"]
            }
            upgrade_turns.append(
                {
                    **state_with_upgrades,
                    "played_boss": played_boss,
                    "boss_target": (target_event or {}).get("target"),
                    "selected_upgrade": bool(
                        target_event
                        and int(target_event["target"]["serial"]) in upgrade_serials
                    ),
                    "turn_actions": turn_actions[key],
                }
            )
        state_with_same_ex = max(states, key=lambda row: len(row["same_ex_lower_hp"]))
        if state_with_same_ex["same_ex_lower_hp"]:
            serials = {
                int(row["serial"]) for row in state_with_same_ex["same_ex_lower_hp"]
            }
            same_ex_turns.append(
                {
                    **state_with_same_ex,
                    "played_boss": played_boss,
                    "boss_target": (target_event or {}).get("target"),
                    "selected_lower_hp_copy": bool(
                        target_event
                        and int(target_event["target"]["serial"]) in serials
                    ),
                    "turn_actions": turn_actions[key],
                }
            )
        trap_states = [
            row for row in states if row["active_dudun_low_deck_pivot_ko"]
        ]
        if trap_states:
            actions = turn_actions[key]
            dudun_traps.append(
                {
                    **trap_states[0],
                    "selected_attach_to_active": any(
                        event["card_id"] in ENERGY_IDS
                        and event["in_play_area"] == ACTIVE
                        and event["in_play_index"] == 0
                        for event in actions
                    ),
                    "selected_retreat": any(
                        event["type"] == RETREAT for event in actions
                    ),
                    "turn_actions": actions,
                }
            )
        if episode_id in grim_games and target_id == MUNKIDORI:
            boss_play_state = next(
                (
                    row
                    for row in states
                    if row["selected"]["type"] == PLAY
                    and row["selected"]["card_id"] == BOSS_ORDERS
                ),
                states[0],
            )
            grim_visible = [
                card
                for card in [boss_play_state["opp_active"], *boss_play_state["opp_bench"]]
                if card and card["id"] == GRIMMSNARL_EX
            ]
            grim_plans = [
                plan
                for plan in boss_play_state["policy"].get("plans", [])
                if plan["target_id"] == GRIMMSNARL_EX
            ]
            munk_plans = [
                plan
                for plan in boss_play_state["policy"].get("plans", [])
                if plan["target_id"] == MUNKIDORI
            ]
            grim_munk_boss.append(
                {
                    **boss_play_state,
                    "boss_target": (target_event or {}).get("target"),
                    "grimmsnarl_visible": grim_visible,
                    "grimmsnarl_plans": grim_plans,
                    "munkidori_plans": munk_plans,
                    "turn_actions": turn_actions[key],
                }
            )

    summary = {
        "games": games,
        "wins": wins,
        "win_rate": wins / games if games else None,
        "main_states": len(main_states),
        "grimmsnarl_games": len(grim_games),
        "ex_upgrade_turns": len(upgrade_turns),
        "ex_upgrade_boss_plays": sum(row["played_boss"] for row in upgrade_turns),
        "ex_upgrade_selected": sum(
            row["selected_upgrade"] for row in upgrade_turns
        ),
        "same_ex_lower_hp_turns": len(same_ex_turns),
        "same_ex_lower_hp_selected": sum(
            row["selected_lower_hp_copy"] for row in same_ex_turns
        ),
        "grim_munk_boss_plays": len(grim_munk_boss),
        "low_deck_dudun_pivot_ko_turns": len(dudun_traps),
        "low_deck_dudun_attach_selected": sum(
            row["selected_attach_to_active"] for row in dudun_traps
        ),
        "low_deck_dudun_retreat_selected": sum(
            row["selected_retreat"] for row in dudun_traps
        ),
    }
    return {
        "run_dir": str(run_dir),
        "submission_id": submission_id,
        "summary": summary,
        "ex_upgrade_examples": upgrade_turns[:30],
        "same_ex_lower_hp_examples": same_ex_turns[:30],
        "grim_munk_boss_examples": grim_munk_boss[:30],
        "low_deck_dudun_pivot_examples": dudun_traps[:30],
        "boss_target_counts": dict(
            Counter(
                int(event["target"]["id"])
                for event in boss_targets.values()
                if event.get("target")
            ).most_common()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--submission-id", required=True, type=int)
    parser.add_argument("--agent-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    module = None
    if args.agent_dir:
        _, _, module = load_dir_agent(args.agent_dir.resolve())
    report = analyze_run(
        args.run_dir.resolve(),
        args.submission_id,
        module,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
