"""Audit concrete v10 ladder failure modes that motivate Alakazam ML v11.

The collector layout used by ``collect_top100_submission_replays.py`` stores
replays under ``replays/`` and seat ownership in ``indexes/episodes.csv``.
This script intentionally works on that layout so a submitted agent can be
audited without repackaging its files first.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ABRA = 741
KADABRA = 742
ALAKAZAM = 743
DUNSPARCE = 305
DUDUNSPARCE = 66
RARE_CANDY = 1079
PSYCHIC_ENERGIES = {5, 19}

ACTIVE = 4
BENCH = 5
MAIN = 0
SETUP_ACTIVE = 1
ACTIVATE = 43
CARD = 3
PLAY = 7
EVOLVE = 9
ABILITY = 10
ATTACK = 13
END = 14
YES = 1


def _card_at(player: dict[str, Any], area: int, index: int) -> dict[str, Any] | None:
    key = {2: "hand", ACTIVE: "active", BENCH: "bench"}.get(area)
    cards = player.get(key) or [] if key else []
    if isinstance(index, int) and 0 <= index < len(cards):
        card = cards[index]
        return card if isinstance(card, dict) else None
    return None


def _selected(
    steps: list[list[dict[str, Any]]], step_index: int, seat: int
) -> tuple[int | None, dict[str, Any] | None]:
    if step_index + 1 >= len(steps) or seat >= len(steps[step_index + 1]):
        return None, None
    action = steps[step_index + 1][seat].get("action")
    options = (((steps[step_index][seat].get("observation") or {}).get("select") or {})
               .get("option") or [])
    if not isinstance(action, list) or len(action) != 1 or not isinstance(action[0], int):
        return None, None
    index = action[0]
    if not 0 <= index < len(options):
        return index, None
    return index, options[index]


def _option_card(player: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    option_type = int(option.get("type", -1))
    if option_type in {PLAY, EVOLVE}:
        return _card_at(player, 2, int(option.get("index", -1)))
    if option_type == ABILITY:
        return _card_at(player, int(option.get("area", -1)), int(option.get("index", -1)))
    if option_type == CARD:
        return _card_at(player, int(option.get("area", -1)), int(option.get("index", -1)))
    return None


def _won(replay: dict[str, Any], seat: int) -> bool:
    rewards = replay.get("rewards") or []
    return seat < len(rewards) and rewards[seat] is not None and rewards[seat] > 0


def _seat_map(root: Path, submission_id: int) -> dict[int, int]:
    result: dict[int, int] = {}
    with (root / "indexes" / "episodes.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            episode_id = int(row["episode_id"])
            if int(row["agent_0_submission_id"]) == submission_id:
                result[episode_id] = 0
            elif int(row["agent_1_submission_id"]) == submission_id:
                result[episode_id] = 1
    return result


def _board(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [card for card in (player.get("active") or []) + (player.get("bench") or [])
            if isinstance(card, dict)]


def _energy_count(card: dict[str, Any]) -> int:
    return len(card.get("energies") or [])


def analyze(root: Path, submission_id: int) -> dict[str, Any]:
    seat_map = _seat_map(root, submission_id)
    opening_rows: list[dict[str, Any]] = []
    candy_rows: list[dict[str, Any]] = []
    draw_choice_rows: list[dict[str, Any]] = []
    declined_activate_rows: list[dict[str, Any]] = []
    game_rows: list[dict[str, Any]] = []

    for replay_path in sorted((root / "replays").glob("episode_*.json")):
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        episode_id = int((replay.get("info") or {}).get("EpisodeId", 0))
        seat = seat_map.get(episode_id)
        if seat is None:
            continue
        won = _won(replay, seat)
        steps = replay.get("steps") or []
        main_actions = Counter()
        min_energyless_deck: int | None = None

        for step_index in range(max(0, len(steps) - 1)):
            if seat >= len(steps[step_index]):
                continue
            state = steps[step_index][seat]
            if state.get("status") != "ACTIVE":
                continue
            observation = state.get("observation") or {}
            current = observation.get("current") or {}
            players = current.get("players") or [{}, {}]
            if seat >= len(players):
                continue
            player = players[seat]
            opponent = players[1 - seat]
            select = observation.get("select") or {}
            context = int(select.get("context", -1))
            options = select.get("option") or []
            selected_index, selected = _selected(steps, step_index, seat)
            selected_type = int((selected or {}).get("type", -1))
            selected_card = _option_card(player, selected or {})
            selected_card_id = int((selected_card or {}).get("id", -1))
            turn = int(current.get("turn", -1))
            hand = [card for card in (player.get("hand") or []) if isinstance(card, dict)]
            hand_ids = [int(card.get("id", -1)) for card in hand]
            deck_count = int(player.get("deckCount") or 0)
            board = _board(player)
            attacker_energyless = any(
                int(card.get("id", -1)) == ALAKAZAM and _energy_count(card) == 0 for card in board
            )
            if attacker_energyless and not any(card_id in PSYCHIC_ENERGIES for card_id in hand_ids):
                min_energyless_deck = deck_count if min_energyless_deck is None else min(
                    min_energyless_deck, deck_count
                )

            if context == SETUP_ACTIVE:
                option_cards = [_option_card(player, option) for option in options]
                option_ids = [int((card or {}).get("id", -1)) for card in option_cards]
                if ABRA in option_ids and DUNSPARCE in option_ids:
                    opening_rows.append({
                        "episode_id": episode_id,
                        "won": won,
                        "selected_card_id": selected_card_id,
                        "option_card_ids": option_ids,
                    })

            if context != MAIN:
                if context == ACTIVATE and deck_count <= 10:
                    yes_offered = any(int(option.get("type", -1)) == YES for option in options)
                    if yes_offered and selected_type != YES:
                        declined_activate_rows.append({
                            "episode_id": episode_id,
                            "won": won,
                            "turn": turn,
                            "deck_count": deck_count,
                            "hand_count": int(player.get("handCount") or len(hand)),
                            "psychic_energy_in_hand": sum(card_id in PSYCHIC_ENERGIES for card_id in hand_ids),
                            "energyless_alakazam": attacker_energyless,
                            "selected_type": selected_type,
                        })
                continue

            main_actions[str(selected_type)] += 1
            option_rows = []
            for index, option in enumerate(options):
                card = _option_card(player, option)
                option_rows.append((index, int(option.get("type", -1)), int((card or {}).get("id", -1))))

            candy_indices = [index for index, kind, card_id in option_rows
                             if kind == PLAY and card_id == RARE_CANDY]
            kadabra_indices = [index for index, kind, card_id in option_rows
                               if kind == EVOLVE and card_id == KADABRA]
            if candy_indices and kadabra_indices and ALAKAZAM in hand_ids:
                opp_active = next((card for card in (opponent.get("active") or [])
                                   if isinstance(card, dict)), None)
                active = next((card for card in (player.get("active") or [])
                               if isinstance(card, dict)), None)
                candy_rows.append({
                    "episode_id": episode_id,
                    "won": won,
                    "turn": turn,
                    "deck_count": deck_count,
                    "hand_count": int(player.get("handCount") or len(hand)),
                    "active_id": int((active or {}).get("id", -1)),
                    "active_energy": _energy_count(active or {}),
                    "opponent_active_id": int((opp_active or {}).get("id", -1)),
                    "opponent_active_hp": int((opp_active or {}).get("hp", 0)),
                    "selected": "rare_candy" if selected_index in candy_indices else (
                        "kadabra" if selected_index in kadabra_indices else str(selected_type)
                    ),
                    "rare_candy_options": candy_indices,
                    "kadabra_options": kadabra_indices,
                })

            ability_indices = []
            for index, kind, card_id in option_rows:
                if kind == ABILITY and card_id == DUDUNSPARCE:
                    ability_indices.append(index)
            if ability_indices and deck_count <= 10 and attacker_energyless \
                    and not any(card_id in PSYCHIC_ENERGIES for card_id in hand_ids):
                draw_choice_rows.append({
                    "episode_id": episode_id,
                    "won": won,
                    "turn": turn,
                    "deck_count": deck_count,
                    "hand_count": int(player.get("handCount") or len(hand)),
                    "selected": "dudunsparce" if selected_index in ability_indices else (
                        "end" if selected_type == END else str(selected_type)
                    ),
                    "ability_options": ability_indices,
                    "attack_offered": any(int(option.get("type", -1)) == ATTACK for option in options),
                })

        game_rows.append({
            "episode_id": episode_id,
            "won": won,
            "min_deck_while_energyless": min_energyless_deck,
            "main_actions": dict(main_actions),
        })

    def _summary(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(Counter(str(row[key]) for row in rows))

    return {
        "submission_id": submission_id,
        "games": len(game_rows),
        "wins": sum(row["won"] for row in game_rows),
        "opening_abra_and_dunsparce": {
            "count": len(opening_rows),
            "selected": _summary(opening_rows, "selected_card_id"),
            "rows": opening_rows,
        },
        "rare_candy_vs_kadabra": {
            "count": len(candy_rows),
            "selected": _summary(candy_rows, "selected"),
            "rows": candy_rows,
        },
        "late_energyless_dudunsparce_main_choices": {
            "count": len(draw_choice_rows),
            "selected": _summary(draw_choice_rows, "selected"),
            "rows": draw_choice_rows,
        },
        "declined_low_deck_activate_prompts": {
            "count": len(declined_activate_rows),
            "rows": declined_activate_rows,
        },
        "energyless_games": {
            "count": sum(row["min_deck_while_energyless"] is not None for row in game_rows),
            "losses": sum(
                row["min_deck_while_energyless"] is not None and not row["won"] for row in game_rows
            ),
            "at_deck_10_or_less": sum(
                row["min_deck_while_energyless"] is not None
                and row["min_deck_while_energyless"] <= 10 for row in game_rows
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.root, args.submission_id)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
