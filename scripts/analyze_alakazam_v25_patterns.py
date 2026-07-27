#!/usr/bin/env python3
"""Audit the v25 decision classes against one recorded Alakazam run.

The report focuses on three high-impact choices that are not safely learned by
the current MAIN-action model: Fezandipiti exposure in the mirror, post-KO
promotion, and Boss targets on Grimmsnarl/Froslass boards.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent


AREA_KEYS = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench"}
ALAKAZAM_LINE = {741, 742, 743}
GRIM_SIGNATURE = {104, 112, 646, 647, 648, 860}


def _seats(run_dir: Path, submission_id: int) -> dict[int, int]:
    seats: dict[int, int] = {}
    with (run_dir / "episodes.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            episode_id = int(row["episode_id"])
            if int(row["agent_0_submission_id"]) == submission_id:
                seats[episode_id] = 0
            elif int(row["agent_1_submission_id"]) == submission_id:
                seats[episode_id] = 1
    return seats


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
    if in_play:
        area = int(option.get("inPlayArea", -1))
        index = int(option.get("inPlayIndex", -1))
    else:
        area = int(option.get("area", -1))
        index = int(option.get("index", -1))
    key = AREA_KEYS.get(area)
    if not in_play and key is None and int(option.get("type", -1)) in (7, 8, 9):
        key = "hand"
    cards = players[player].get(key, []) if key else []
    return cards[index] or {} if 0 <= index < len(cards) else {}


def _board_ids(player: dict[str, Any]) -> set[int]:
    return {
        int(card["id"])
        for card in (player.get("active") or []) + (player.get("bench") or [])
        if isinstance(card, dict) and "id" in card
    }


def _selected(
    steps: list[Any],
    step_index: int,
    seat: int,
) -> int | None:
    if step_index + 1 >= len(steps) or seat >= len(steps[step_index + 1]):
        return None
    action = steps[step_index + 1][seat].get("action")
    if not isinstance(action, list) or len(action) != 1:
        return None
    return int(action[0])


def analyze(run_dir: Path, agent_dir: Path, submission_id: int) -> dict[str, Any]:
    _, _, module = load_dir_agent(agent_dir)
    policy_module = module.fallback_policy
    seats = _seats(run_dir, submission_id)
    fez_mirror: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    grim_boss: list[dict[str, Any]] = []

    for replay_path in sorted((run_dir / "episodes").glob("*/replay/*.json")):
        episode_id = int(replay_path.parents[1].name)
        seat = seats.get(episode_id)
        if seat is None:
            continue
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        turn_events: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        decision_rows: list[tuple[int, dict[str, Any], dict[str, Any], int]] = []

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
            card_id = int(_card(obs, option).get("id", -1))
            turn = int((obs.get("current") or {}).get("turn", -1))
            turn_events[turn].append((int(option.get("type", -1)), card_id))
            decision_rows.append((step_index, obs, option, selected))

        for step_index, obs, option, selected in decision_rows:
            current = obs.get("current") or {}
            players = current.get("players") or [{}, {}]
            own = players[seat]
            opponent = players[1 - seat]
            own_ids = _board_ids(own)
            opponent_ids = _board_ids(opponent)
            select = obs.get("select") or {}
            context = int(select.get("context", -1))
            effect_id = int((select.get("effect") or {}).get("id", -1))
            option_type = int(option.get("type", -1))
            source = _card(obs, option)
            selected_id = int(source.get("id", -1))
            turn = int(current.get("turn", -1))

            if (
                context == 0
                and option_type == 7
                and selected_id == 140
                and opponent_ids & ALAKAZAM_LINE
            ):
                used_ability = any(
                    action_type == 10 and card_id == 140
                    for action_type, card_id in turn_events[turn]
                )
                fez_mirror.append(
                    {
                        "episode_id": episode_id,
                        "turn": turn,
                        "used_flip_the_script": used_ability,
                        "hand": int(own.get("handCount", 0)),
                        "deck": int(own.get("deckCount", 0)),
                        "own_prizes": len(own.get("prize") or []),
                        "opp_prizes": len(opponent.get("prize") or []),
                        "own_board": sorted(own_ids),
                        "opp_board": sorted(opponent_ids),
                        "opp_hand": int(opponent.get("handCount", 0)),
                    }
                )

            if context == 4:
                try:
                    parsed = policy_module.to_observation_class(obs)
                    policy = policy_module.AlakazamPolicy(parsed)
                    parsed_option = parsed.select.option[selected]
                    card = policy_module.get_card(
                        parsed,
                        parsed_option.area,
                        parsed_option.index,
                        parsed_option.playerIndex,
                    )
                    attacks_next = policy._promotion_attacks_next_turn(card)
                    ko_risk = policy._opponent_can_ko_target_next_turn(card)
                    score = policy._score_ko_promotion(card)
                except Exception:
                    attacks_next = ko_risk = False
                    score = None
                offered_ids = [
                    int(_card(obs, candidate).get("id", -1))
                    for candidate in (select.get("option") or [])
                ]
                promotions.append(
                    {
                        "episode_id": episode_id,
                        "turn": turn,
                        "selected_id": selected_id,
                        "attacks_next_turn": bool(attacks_next),
                        "visible_ko_risk": bool(ko_risk),
                        "v24_score": score,
                        "offered_ids": offered_ids,
                        "hand_ids": [int(card["id"]) for card in own.get("hand") or []],
                        "energy_attached": bool(current.get("energyAttached")),
                        "opp_active": next(iter(opponent_ids), -1),
                    }
                )

            if context == 3 and effect_id == 1182 and opponent_ids & GRIM_SIGNATURE:
                try:
                    parsed = policy_module.to_observation_class(obs)
                    policy = policy_module.AlakazamPolicy(parsed)
                    scores = [
                        float(policy._score(candidate))
                        for candidate in parsed.select.option
                    ]
                except Exception:
                    scores = [None] * len(select.get("option") or [])
                offered = [
                    {
                        "id": int(_card(obs, candidate).get("id", -1)),
                        "score": scores[index],
                    }
                    for index, candidate in enumerate(select.get("option") or [])
                ]
                grim_boss.append(
                    {
                        "episode_id": episode_id,
                        "turn": turn,
                        "selected_id": selected_id,
                        "offered": offered,
                    }
                )

    promotion_counts = Counter(row["selected_id"] for row in promotions)
    promotion_routes = Counter(
        (row["selected_id"], row["attacks_next_turn"], row["visible_ko_risk"])
        for row in promotions
    )
    return {
        "run_dir": str(run_dir),
        "agent_dir": str(agent_dir),
        "submission_id": submission_id,
        "summary": {
            "mirror_fez_plays": len(fez_mirror),
            "mirror_fez_with_ability": sum(
                row["used_flip_the_script"] for row in fez_mirror
            ),
            "promotion_counts": dict(promotion_counts.most_common()),
            "promotion_route_counts": [
                {
                    "selected_id": key[0],
                    "attacks_next_turn": key[1],
                    "visible_ko_risk": key[2],
                    "count": count,
                }
                for key, count in promotion_routes.most_common()
            ],
            "grim_boss_counts": dict(
                Counter(row["selected_id"] for row in grim_boss).most_common()
            ),
        },
        "mirror_fez_examples": fez_mirror[:100],
        "promotion_examples": promotions[:200],
        "grim_boss_examples": grim_boss[:200],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--submission-id", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.run_dir.resolve(),
        args.agent_dir.resolve(),
        args.submission_id,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
