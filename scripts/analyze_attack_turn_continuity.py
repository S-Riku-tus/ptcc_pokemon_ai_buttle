"""Measure attack setup and continuity from a Kaggle full-replay bundle.

The champion/challenger ``attack_turn_rate`` uses every engine turn on which an
agent made a decision.  This audit also reports MAIN-only and attack-opportunity
denominators so setup latency can be separated from avoidable skipped attacks.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile


ATTACK_OPTION_TYPE = 13
END_OPTION_TYPE = 14
MAIN_SELECT_TYPE = 0
MAIN_SELECT_CONTEXT = 0


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _selected_option(
    steps: list[list[dict[str, Any]]], step_index: int, seat: int
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    state = steps[step_index][seat]
    observation = state.get("observation") or {}
    options = ((observation.get("select") or {}).get("option") or [])
    if step_index + 1 >= len(steps) or seat >= len(steps[step_index + 1]):
        return None, options
    action = steps[step_index + 1][seat].get("action")
    if not isinstance(action, list) or len(action) != 1:
        return None, options
    index = action[0]
    if not isinstance(index, int) or not 0 <= index < len(options):
        return None, options
    return options[index], options


def _target_seat(replay: dict[str, Any], target_team: str) -> int | None:
    info = replay.get("info") or {}
    names = info.get("TeamNames") or [x.get("Name") for x in info.get("Agents") or []]
    exact = [index for index, name in enumerate(names) if str(name) == target_team]
    if len(exact) == 1:
        return exact[0]
    folded = [
        index for index, name in enumerate(names)
        if str(name).strip().casefold() == target_team.strip().casefold()
    ]
    return folded[0] if len(folded) == 1 else None


def analyze_replay(replay: dict[str, Any], target_team: str) -> dict[str, Any] | None:
    steps = replay.get("steps") or []
    seat = _target_seat(replay, target_team)
    if seat is None:
        return None

    acting_turns: set[int] = set()
    main_turns: set[int] = set()
    attack_turns: set[int] = set()
    opportunity_turns: set[int] = set()
    attackable_end_turns: set[int] = set()
    active_ids_by_turn: dict[int, Counter[int]] = defaultdict(Counter)
    main_actions_by_turn: dict[int, Counter[str]] = defaultdict(Counter)

    for step_index in range(max(0, len(steps) - 1)):
        if seat >= len(steps[step_index]):
            continue
        state = steps[step_index][seat]
        if state.get("status") != "ACTIVE":
            continue
        observation = state.get("observation") or {}
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        turn = int(current.get("turn", -1))
        if turn < 0:
            continue
        acting_turns.add(turn)
        if int(select.get("type", -1)) != MAIN_SELECT_TYPE:
            continue
        if int(select.get("context", -1)) != MAIN_SELECT_CONTEXT:
            continue

        main_turns.add(turn)
        player = (current.get("players") or [{}, {}])[seat]
        active = (player.get("active") or [None])[0]
        if isinstance(active, dict) and active.get("id") is not None:
            active_ids_by_turn[turn][int(active["id"])] += 1

        selected, options = _selected_option(steps, step_index, seat)
        offered_attack = any(int(option.get("type", -1)) == ATTACK_OPTION_TYPE for option in options)
        if offered_attack:
            opportunity_turns.add(turn)
        if selected is None:
            main_actions_by_turn[turn]["unresolved"] += 1
            continue
        selected_type = int(selected.get("type", -1))
        main_actions_by_turn[turn][str(selected_type)] += 1
        if selected_type == ATTACK_OPTION_TYPE:
            attack_turns.add(turn)
        elif selected_type == END_OPTION_TYPE and offered_attack:
            attackable_end_turns.add(turn)

    ordered_main = sorted(main_turns)
    first_attack = min(attack_turns) if attack_turns else None
    post_first_turns = {turn for turn in main_turns if first_attack is not None and turn >= first_attack}
    missed_opportunities = opportunity_turns - attack_turns
    post_first_missed = (post_first_turns & opportunity_turns) - attack_turns
    non_opportunity_post_first = post_first_turns - opportunity_turns
    active_on_non_opportunity = Counter()
    for turn in non_opportunity_post_first:
        if active_ids_by_turn[turn]:
            active_on_non_opportunity[active_ids_by_turn[turn].most_common(1)[0][0]] += 1

    rewards = replay.get("rewards") or [None, None]
    reward = rewards[seat] if seat < len(rewards) else None
    return {
        "episode_id": int((replay.get("info") or {}).get("EpisodeId", 0)),
        "seat": seat,
        "won": True if reward is not None and reward > 0 else False,
        "finished": reward is not None,
        "acting_turns": len(acting_turns),
        "main_turns": len(main_turns),
        "attack_turns": len(attack_turns),
        "attack_opportunity_turns": len(opportunity_turns),
        "missed_attack_opportunity_turns": len(missed_opportunities),
        "attackable_end_turns": len(attackable_end_turns),
        "first_attack_engine_turn": first_attack,
        "main_turns_before_first_attack": (
            sum(turn < first_attack for turn in ordered_main) if first_attack is not None else len(ordered_main)
        ),
        "post_first_main_turns": len(post_first_turns),
        "post_first_attack_turns": len(post_first_turns & attack_turns),
        "post_first_opportunity_turns": len(post_first_turns & opportunity_turns),
        "post_first_missed_opportunity_turns": len(post_first_missed),
        "post_first_no_attack_option_turns": len(non_opportunity_post_first),
        "post_first_no_attack_option_active_ids": dict(active_on_non_opportunity),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sums = Counter()
    active_ids = Counter()
    for row in rows:
        for key, value in row.items():
            if key in {"episode_id", "seat", "won", "finished", "first_attack_engine_turn", "post_first_no_attack_option_active_ids"}:
                continue
            if isinstance(value, int):
                sums[key] += value
        active_ids.update({int(key): value for key, value in row["post_first_no_attack_option_active_ids"].items()})

    games = len(rows)
    games_with_attack = sum(row["attack_turns"] > 0 for row in rows)
    first_attack_values = [
        row["main_turns_before_first_attack"] + 1 for row in rows if row["attack_turns"] > 0
    ]
    finished = sum(row["finished"] for row in rows)
    wins = sum(row["won"] for row in rows if row["finished"])
    return {
        "games": games,
        "finished_games": finished,
        "wins": wins,
        "win_rate": _ratio(wins, finished),
        "games_with_attack": games_with_attack,
        "games_with_attack_rate": _ratio(games_with_attack, games),
        "avg_first_attack_own_main_turn": (
            sum(first_attack_values) / len(first_attack_values) if first_attack_values else None
        ),
        "attacks_per_game": _ratio(sums["attack_turns"], games),
        "all_acting_turn_attack_rate": _ratio(sums["attack_turns"], sums["acting_turns"]),
        "main_turn_attack_rate": _ratio(sums["attack_turns"], sums["main_turns"]),
        "attack_opportunity_conversion_rate": _ratio(
            sums["attack_turns"], sums["attack_opportunity_turns"]
        ),
        "post_first_main_turn_attack_rate": _ratio(
            sums["post_first_attack_turns"], sums["post_first_main_turns"]
        ),
        "post_first_opportunity_conversion_rate": _ratio(
            sums["post_first_attack_turns"], sums["post_first_opportunity_turns"]
        ),
        "totals": dict(sums),
        "post_first_no_attack_option_active_ids": dict(active_ids.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--target-team", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    unresolved_team = 0
    with ZipFile(args.zip_path) as zf:
        replay_members = sorted(
            name for name in zf.namelist()
            if "/replay/episode_" in name and name.endswith(".json")
        )
        for member in replay_members:
            replay = json.loads(zf.read(member))
            row = analyze_replay(replay, args.target_team)
            if row is None:
                unresolved_team += 1
            else:
                rows.append(row)

    report = {
        "source_zip": str(args.zip_path),
        "target_team": args.target_team,
        "replay_count": len(rows),
        "unresolved_team_replays": unresolved_team,
        "aggregate": aggregate(rows),
        "episodes": rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.summary_only:
        print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, ensure_ascii=False, indent=2))
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
