"""Audit missed attacks and Mist/Hammer decisions in a packaged ladder run."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


SUBMISSION_DEFAULT = 54868871

ALAKAZAM = 743
KADABRA = 742
BOSS_ORDERS = 1182
ENHANCED_HAMMER = 1081
MIST_ENERGY = 11
PSYCHIC_ENERGIES = {5, 19}

HAND = 2
ACTIVE = 4
BENCH = 5

# Official 1.32 replay JSON encodes MAIN as 0. The local cg enum used by the
# policy is intentionally not imported so this audit can run on saved logs.
MAIN = 0
PLAY = 7
ENERGY = 8
# Replay schema 1.32 uses 8 for the complete MAIN attachment action and 9 for
# evolution. The in-process cg enum used by unit tests has a separate ATTACH=9.
ATTACH = 8
EVOLVE = 9
ATTACK = 13
END = 14


def _seat_map(run_dir: Path, submission_id: int) -> dict[int, int]:
    result: dict[int, int] = {}
    with (run_dir / "episodes.csv").open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            episode_id = int(row["episode_id"])
            if int(row["agent_0_submission_id"]) == submission_id:
                result[episode_id] = 0
            elif int(row["agent_1_submission_id"]) == submission_id:
                result[episode_id] = 1
    return result


def _selected(steps: list[list[dict[str, Any]]], index: int, seat: int) -> dict[str, Any] | None:
    if index + 1 >= len(steps) or seat >= len(steps[index + 1]):
        return None
    action = steps[index + 1][seat].get("action")
    options = (((steps[index][seat].get("observation") or {}).get("select") or {})
               .get("option") or [])
    if not isinstance(action, list) or len(action) != 1 or not isinstance(action[0], int):
        return None
    selected_index = action[0]
    if not 0 <= selected_index < len(options):
        return None
    return options[selected_index]


def _card_at(player: dict[str, Any], area: int, index: int) -> dict[str, Any] | None:
    key = {HAND: "hand", ACTIVE: "active", BENCH: "bench"}.get(area)
    cards = (player.get(key) or []) if key else []
    if 0 <= index < len(cards) and isinstance(cards[index], dict):
        return cards[index]
    return None


def _option_card(player: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    if int(option.get("type", -1)) in {PLAY, ENERGY, ATTACH, EVOLVE}:
        return _card_at(player, HAND, int(option.get("index", -1)))
    return None


def _card_ids(cards: list[Any] | None) -> list[int]:
    return [int(card.get("id", -1)) for card in (cards or []) if isinstance(card, dict)]


def _energy_ids(pokemon: dict[str, Any] | None) -> list[int]:
    return _card_ids((pokemon or {}).get("energyCards") or [])


def analyze(run_dir: Path, submission_id: int) -> dict[str, Any]:
    seats = _seat_map(run_dir, submission_id)
    missed_attack_rows: list[dict[str, Any]] = []
    missed_attach_rows: list[dict[str, Any]] = []
    diverted_attach_rows: list[dict[str, Any]] = []
    hammer_rows: list[dict[str, Any]] = []
    attack_offered = 0
    attack_chosen = 0
    main_end = 0

    for replay_path in sorted((run_dir / "episodes").glob("*/replay/episode_*.json")):
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        episode_id = int((replay.get("info") or {}).get("EpisodeId", 0))
        seat = seats.get(episode_id)
        if seat is None:
            continue
        won = (replay.get("rewards") or [0, 0])[seat] > 0
        steps = replay.get("steps") or []
        pending_hammer: dict[str, Any] | None = None
        mist_seen_public = False

        for index in range(max(0, len(steps) - 1)):
            if seat >= len(steps[index]):
                continue
            state = steps[index][seat]
            if state.get("status") != "ACTIVE":
                continue
            observation = state.get("observation") or {}
            current = observation.get("current") or {}
            select = observation.get("select") or {}
            players = current.get("players") or []
            if seat >= len(players):
                continue
            me = players[seat]
            opponent = players[1 - seat]
            selected = _selected(steps, index, seat)
            if selected is None:
                continue

            context = int(select.get("context", -1))
            options = select.get("option") or []
            selected_type = int(selected.get("type", -1))
            selected_card = _option_card(me, selected)
            selected_card_id = int((selected_card or {}).get("id", -1))
            active = next(iter(me.get("active") or []), None)
            opp_active = next(iter(opponent.get("active") or []), None)
            hand_ids = _card_ids(me.get("hand"))
            public_opponent_cards = list(opponent.get("discard") or [])
            for pokemon in (opponent.get("active") or []) + (opponent.get("bench") or []):
                if isinstance(pokemon, dict):
                    public_opponent_cards.extend(pokemon.get("energyCards") or [])
            mist_seen_public = mist_seen_public or MIST_ENERGY in _card_ids(public_opponent_cards)

            # Hammer target selection follows the PLAY decision. Record the actual
            # energy selected via energyIndex, not merely the target Pokemon.
            effect = select.get("effect") or select.get("contextCard") or {}
            if int(effect.get("id", -1)) == ENHANCED_HAMMER:
                owner = _card_at(
                    opponent if int(selected.get("playerIndex", -1)) == 1 - seat else me,
                    int(selected.get("area", selected.get("inPlayArea", -1))),
                    int(selected.get("index", selected.get("inPlayIndex", -1))),
                )
                energy_index = selected.get("energyIndex")
                energies = (owner or {}).get("energyCards") or []
                target_energy = (
                    energies[energy_index]
                    if isinstance(energy_index, int) and 0 <= energy_index < len(energies)
                    else None
                )
                row = pending_hammer or {"episode_id": episode_id, "won": won}
                row.update({
                    "turn": int(current.get("turn", -1)),
                    "target_pokemon_id": int((owner or {}).get("id", -1)),
                    "target_energy_id": int((target_energy or {}).get("id", -1)),
                    "mist_seen_before": mist_seen_public,
                    "mist_attached_anywhere": any(
                        MIST_ENERGY in _energy_ids(pokemon)
                        for pokemon in (opponent.get("active") or []) + (opponent.get("bench") or [])
                        if isinstance(pokemon, dict)
                    ),
                })
                hammer_rows.append(row)
                pending_hammer = None

            if context != MAIN:
                continue

            offered_types = [int(option.get("type", -1)) for option in options]
            active_psychic_attach_options = [
                option for option in options
                if int(option.get("type", -1)) in {ENERGY, ATTACH}
                and int((_option_card(me, option) or {}).get("id", -1)) in PSYCHIC_ENERGIES
                and int(option.get("inPlayArea", -1)) == ACTIVE
                and int(option.get("inPlayIndex", -1)) == 0
            ]
            has_attack = ATTACK in offered_types
            if has_attack:
                attack_offered += 1
            if selected_type == ATTACK:
                attack_chosen += 1
            if selected_type == PLAY and selected_card_id == ENHANCED_HAMMER:
                pending_hammer = {
                    "episode_id": episode_id,
                    "won": won,
                    "play_turn": int(current.get("turn", -1)),
                }
            if (selected_type in {ENERGY, ATTACH}
                    and int((active or {}).get("id", -1)) in {ALAKAZAM, KADABRA}
                    and not _energy_ids(active)
                    and active_psychic_attach_options):
                selected_target = _card_at(
                    me,
                    int(selected.get("inPlayArea", -1)),
                    int(selected.get("inPlayIndex", -1)),
                )
                selected_is_attack_route = (
                    selected_card_id in PSYCHIC_ENERGIES
                    and int(selected.get("inPlayArea", -1)) == ACTIVE
                    and int(selected.get("inPlayIndex", -1)) == 0
                )
                if not selected_is_attack_route:
                    diverted_attach_rows.append({
                        "episode_id": episode_id,
                        "won": won,
                        "turn": int(current.get("turn", -1)),
                        "active_id": int((active or {}).get("id", -1)),
                        "selected_energy_id": selected_card_id,
                        "selected_target_id": int((selected_target or {}).get("id", -1)),
                        "hand_ids": hand_ids,
                    })

            if selected_type != END:
                continue
            main_end += 1
            row = {
                "episode_id": episode_id,
                "won": won,
                "turn": int(current.get("turn", -1)),
                "turn_action_count": int(current.get("turnActionCount", -1)),
                "active_id": int((active or {}).get("id", -1)),
                "active_energy_ids": _energy_ids(active),
                "opp_active_id": int((opp_active or {}).get("id", -1)),
                "opp_active_hp": int((opp_active or {}).get("hp", 0)),
                "opp_active_energy_ids": _energy_ids(opp_active),
                "opp_bench": [
                    {
                        "id": int(pokemon.get("id", -1)),
                        "hp": int(pokemon.get("hp", 0)),
                        "energy_ids": _energy_ids(pokemon),
                    }
                    for pokemon in (opponent.get("bench") or [])
                    if isinstance(pokemon, dict)
                ],
                "hand_count": int(me.get("handCount", len(hand_ids))),
                "hand_ids": hand_ids,
                "attack_ids_offered": [
                    int(option.get("attackId", -1))
                    for option in options if int(option.get("type", -1)) == ATTACK
                ],
                "boss_play_offered": any(
                    int(option.get("type", -1)) == PLAY
                    and int((_option_card(me, option) or {}).get("id", -1)) == BOSS_ORDERS
                    for option in options
                ),
                "hammer_play_offered": any(
                    int(option.get("type", -1)) == PLAY
                    and int((_option_card(me, option) or {}).get("id", -1)) == ENHANCED_HAMMER
                    for option in options
                ),
            }
            if has_attack:
                missed_attack_rows.append(row)

            attach_options = []
            for option in options:
                if int(option.get("type", -1)) not in {ENERGY, ATTACH}:
                    continue
                source = _option_card(me, option)
                if int((source or {}).get("id", -1)) not in PSYCHIC_ENERGIES:
                    continue
                if (int(option.get("inPlayArea", -1)) == ACTIVE
                        and int(option.get("inPlayIndex", -1)) == 0):
                    attach_options.append(int((source or {}).get("id", -1)))
            if (int((active or {}).get("id", -1)) in {ALAKAZAM, KADABRA}
                    and not _energy_ids(active)
                    and not current.get("energyAttached") and attach_options):
                row["active_attach_options"] = attach_options
                missed_attach_rows.append(row)

    return {
        "submission_id": submission_id,
        "games": len(seats),
        "attack_offered_main_decisions": attack_offered,
        "attack_chosen_main_decisions": attack_chosen,
        "main_end_decisions": main_end,
        "end_with_attack_offered": {
            "count": len(missed_attack_rows),
            "rows": missed_attack_rows,
        },
        "end_with_active_attach_available": {
            "count": len(missed_attach_rows),
            "rows": missed_attach_rows,
        },
        "attack_route_diverted_by_other_attachment": {
            "count": len(diverted_attach_rows),
            "rows": diverted_attach_rows,
        },
        "hammer_targets": {
            "count": len(hammer_rows),
            "by_energy_id": dict(Counter(str(row["target_energy_id"]) for row in hammer_rows)),
            "mist_present_but_other_targeted": sum(
                row["mist_attached_anywhere"] and row["target_energy_id"] != MIST_ENERGY
                for row in hammer_rows
            ),
            "mist_seen_but_other_targeted": sum(
                row["mist_seen_before"] and row["target_energy_id"] != MIST_ENERGY
                for row in hammer_rows
            ),
            "rows": hammer_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--submission-id", type=int, default=SUBMISSION_DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.run_dir, args.submission_id)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
