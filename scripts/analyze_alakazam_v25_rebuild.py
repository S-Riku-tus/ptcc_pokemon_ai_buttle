#!/usr/bin/env python3
"""Compare high-impact engine routes in an Alakazam ladder run.

This audit was added after the first v25 improved one-step teacher agreement
but regressed live rating.  It deliberately measures trajectory-producing
actions instead: Dudunsparce cycles, Fezandipiti setup/attacks, and the exact
targets of Cruel Arrow.  Every aggregate is also split by the recorded game
outcome so a frequent expert action is not automatically treated as causal.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


AREA_KEYS = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench"}
PLAY = 7
ENERGY = 8
ABILITY = 10
ATTACK = 13
MAIN = 0
CRUEL_ARROW = 183
DUDUNSPARCE = 66
FEZANDIPITI_EX = 140
ALAKAZAM_LINE = {741, 742, 743}


def _seats(run_dir: Path, submission_id: int) -> dict[int, int]:
    result: dict[int, int] = {}
    with (run_dir / "episodes.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            episode_id = int(row["episode_id"])
            if int(row["agent_0_submission_id"]) == submission_id:
                result[episode_id] = 0
            elif int(row["agent_1_submission_id"]) == submission_id:
                result[episode_id] = 1
    return result


def _selected(steps: list[Any], step_index: int, seat: int) -> int | None:
    if step_index + 1 >= len(steps) or seat >= len(steps[step_index + 1]):
        return None
    action = steps[step_index + 1][seat].get("action")
    if not isinstance(action, list) or len(action) != 1:
        return None
    return int(action[0])


def _card(
    obs: dict[str, Any],
    option: dict[str, Any],
    *,
    in_play: bool = False,
) -> dict[str, Any]:
    current = obs.get("current") or {}
    players = current.get("players") or []
    player = int(option.get("playerIndex", current.get("yourIndex", 0)))
    if not 0 <= player < len(players):
        return {}
    area_key = "inPlayArea" if in_play else "area"
    index_key = "inPlayIndex" if in_play else "index"
    area = int(option.get(area_key, -1))
    index = int(option.get(index_key, -1))
    key = AREA_KEYS.get(area)
    if not in_play and key is None and int(option.get("type", -1)) in (7, 8, 9):
        key = "hand"
    cards = players[player].get(key, []) if key else []
    return cards[index] or {} if 0 <= index < len(cards) else {}


def _outcome(steps: list[Any], seat: int) -> str:
    final = steps[-1]
    own = final[seat] if seat < len(final) else {}
    other = final[1 - seat] if 1 - seat < len(final) else {}
    own_reward = own.get("reward")
    other_reward = other.get("reward")
    if own_reward is None or other_reward is None:
        return "unknown"
    if own_reward > other_reward:
        return "win"
    if own_reward < other_reward:
        return "loss"
    return "draw"


def _archetypes(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["episode_id"]): str(row.get("opponent_archetype") or "Unknown")
        for row in report.get("episodes") or []
    }


def _next_cruel_arrow_target(
    steps: list[Any],
    step_index: int,
    seat: int,
) -> dict[str, Any] | None:
    for next_index in range(step_index + 1, min(step_index + 6, len(steps) - 1)):
        record = steps[next_index][seat] if seat < len(steps[next_index]) else {}
        obs = (record or {}).get("observation") or {}
        select = obs.get("select") or {}
        if (record or {}).get("status") != "ACTIVE" or not select:
            continue
        effect = select.get("effect") or {}
        if int(select.get("context", -1)) != 15 or int(effect.get("id", -1)) != FEZANDIPITI_EX:
            continue
        selected = _selected(steps, next_index, seat)
        options = select.get("option") or []
        if selected is None or not 0 <= selected < len(options):
            return None
        return _card(obs, options[selected])
    return None


def analyze(
    run_dir: Path,
    submission_id: int,
    ladder_report: Path | None,
) -> dict[str, Any]:
    seats = _seats(run_dir, submission_id)
    archetypes = _archetypes(ladder_report)
    game_outcomes: Counter[str] = Counter()
    games_with: dict[str, set[int]] = {
        "dudun_ability": set(),
        "fez_play": set(),
        "fez_attach": set(),
        "fez_attack": set(),
    }
    counts: Counter[str] = Counter()
    outcome_counts: dict[str, Counter[str]] = {
        key: Counter() for key in games_with
    }
    matchup_counts: dict[str, Counter[str]] = {
        key: Counter() for key in games_with
    }
    dudun_deck_bins: Counter[str] = Counter()
    dudun_hand_bins: Counter[str] = Counter()
    dudun_locations: Counter[str] = Counter()
    fez_attach_source_ids: Counter[int] = Counter()
    fez_attach_before_energy: Counter[int] = Counter()
    fez_attack_target_ids: Counter[int] = Counter()
    fez_attack_target_hp: Counter[int] = Counter()
    fez_attack_ko = 0
    fez_attack_two_prize_ko = 0
    examples: dict[str, list[dict[str, Any]]] = {
        "low_deck_dudun": [],
        "fez_attach": [],
        "fez_attack": [],
    }

    replay_paths = sorted((run_dir / "episodes").glob("*/replay/*.json"))
    for replay_path in replay_paths:
        episode_id = int(replay_path.parents[1].name)
        seat = seats.get(episode_id)
        if seat is None:
            continue
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        outcome = _outcome(steps, seat)
        game_outcomes[outcome] += 1
        matchup = archetypes.get(episode_id, "Unknown")

        for step_index, step in enumerate(steps[:-1]):
            record = step[seat] if seat < len(step) else {}
            obs = (record or {}).get("observation") or {}
            select = obs.get("select") or {}
            if (record or {}).get("status") != "ACTIVE" or not select:
                continue
            selected = _selected(steps, step_index, seat)
            options = select.get("option") or []
            if selected is None or not 0 <= selected < len(options):
                continue
            option = options[selected]
            option_type = int(option.get("type", -1))
            current = obs.get("current") or {}
            players = current.get("players") or [{}, {}]
            own = players[seat]
            card = _card(obs, option)
            card_id = int(card.get("id", -1))
            context = int(select.get("context", -1))
            turn = int(current.get("turn", -1))
            deck = int(own.get("deckCount", 0))
            hand = int(own.get("handCount", 0))

            if context == MAIN and option_type == PLAY and card_id == FEZANDIPITI_EX:
                key = "fez_play"
                counts[key] += 1
                games_with[key].add(episode_id)
                outcome_counts[key][outcome] += 1
                matchup_counts[key][matchup] += 1

            if context == MAIN and option_type == ABILITY and card_id == DUDUNSPARCE:
                key = "dudun_ability"
                counts[key] += 1
                games_with[key].add(episode_id)
                outcome_counts[key][outcome] += 1
                matchup_counts[key][matchup] += 1
                dudun_locations[
                    "active" if int(option.get("area", -1)) == 4 else "bench"
                ] += 1
                dudun_deck_bins[
                    "0-4" if deck <= 4 else "5-8" if deck <= 8
                    else "9-12" if deck <= 12 else "13+"
                ] += 1
                dudun_hand_bins[
                    "0-5" if hand <= 5 else "6-9" if hand <= 9
                    else "10-12" if hand <= 12 else "13+"
                ] += 1
                if deck <= 12 and len(examples["low_deck_dudun"]) < 80:
                    examples["low_deck_dudun"].append(
                        {
                            "episode_id": episode_id,
                            "outcome": outcome,
                            "matchup": matchup,
                            "turn": turn,
                            "deck": deck,
                            "hand": hand,
                            "location": (
                                "active" if int(option.get("area", -1)) == 4 else "bench"
                            ),
                            "own_prizes": len(own.get("prize") or []),
                        }
                    )

            if context == MAIN and option_type == ABILITY and card_id == FEZANDIPITI_EX:
                counts["fez_ability"] += 1

            if context == MAIN and option_type == ENERGY:
                target = _card(obs, option, in_play=True)
                if int(target.get("id", -1)) == FEZANDIPITI_EX:
                    key = "fez_attach"
                    counts[key] += 1
                    games_with[key].add(episode_id)
                    outcome_counts[key][outcome] += 1
                    matchup_counts[key][matchup] += 1
                    fez_attach_source_ids[card_id] += 1
                    before_energy = len(target.get("energies") or [])
                    fez_attach_before_energy[before_energy] += 1
                    if len(examples["fez_attach"]) < 80:
                        opponent = players[1 - seat]
                        examples["fez_attach"].append(
                            {
                                "episode_id": episode_id,
                                "outcome": outcome,
                                "matchup": matchup,
                                "turn": turn,
                                "deck": deck,
                                "hand": hand,
                                "source_id": card_id,
                                "before_energy": before_energy,
                                "own_prizes": len(own.get("prize") or []),
                                "opp_prizes": len(opponent.get("prize") or []),
                                "has_ready_alakazam": any(
                                    int(pokemon.get("id", -1)) in ALAKAZAM_LINE
                                    and len(pokemon.get("energies") or []) >= 1
                                    for pokemon in (
                                        (own.get("active") or []) + (own.get("bench") or [])
                                    )
                                    if isinstance(pokemon, dict)
                                ),
                            }
                        )

            if option_type == ATTACK and int(option.get("attackId", -1)) == CRUEL_ARROW:
                key = "fez_attack"
                counts[key] += 1
                games_with[key].add(episode_id)
                outcome_counts[key][outcome] += 1
                matchup_counts[key][matchup] += 1
                target = _next_cruel_arrow_target(steps, step_index, seat)
                if target is not None:
                    target_id = int(target.get("id", -1))
                    target_hp = int(target.get("hp", 0))
                    target_max_hp = int(target.get("maxHp", target_hp))
                    fez_attack_target_ids[target_id] += 1
                    fez_attack_target_hp[
                        min(500, ((target_hp + 9) // 10) * 10)
                    ] += 1
                    ko = target_hp <= 100
                    fez_attack_ko += int(ko)
                    # Known ex/Mega state is not available without the card table;
                    # printed HP >= 170 is a conservative public proxy.
                    fez_attack_two_prize_ko += int(ko and target_max_hp >= 170)
                    if len(examples["fez_attack"]) < 100:
                        examples["fez_attack"].append(
                            {
                                "episode_id": episode_id,
                                "outcome": outcome,
                                "matchup": matchup,
                                "turn": turn,
                                "target_id": target_id,
                                "target_hp": target_hp,
                                "target_max_hp": target_max_hp,
                                "ko": ko,
                                "deck": deck,
                                "hand": hand,
                            }
                        )

    games = sum(game_outcomes.values())
    return {
        "run_dir": str(run_dir),
        "submission_id": submission_id,
        "games": games,
        "game_outcomes": dict(game_outcomes),
        "counts": dict(counts),
        "per_game": {
            key: counts[key] / games if games else 0.0
            for key in games_with
        },
        "games_with_action": {
            key: len(value) for key, value in games_with.items()
        },
        "action_outcomes": {
            key: dict(value) for key, value in outcome_counts.items()
        },
        "action_matchups": {
            key: dict(value.most_common()) for key, value in matchup_counts.items()
        },
        "dudun": {
            "locations": dict(dudun_locations),
            "deck_bins": dict(dudun_deck_bins),
            "hand_bins": dict(dudun_hand_bins),
        },
        "fez_attach": {
            "source_ids": dict(fez_attach_source_ids),
            "before_energy": dict(fez_attach_before_energy),
        },
        "fez_attack": {
            "target_ids": dict(fez_attack_target_ids.most_common()),
            "target_hp_bins": dict(sorted(fez_attack_target_hp.items())),
            "ko_count": fez_attack_ko,
            "ko_rate": fez_attack_ko / counts["fez_attack"] if counts["fez_attack"] else 0.0,
            "two_prize_ko_proxy_count": fez_attack_two_prize_ko,
        },
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--submission-id", required=True, type=int)
    parser.add_argument("--ladder-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.run_dir.resolve(),
        args.submission_id,
        args.ladder_report.resolve() if args.ladder_report else None,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
