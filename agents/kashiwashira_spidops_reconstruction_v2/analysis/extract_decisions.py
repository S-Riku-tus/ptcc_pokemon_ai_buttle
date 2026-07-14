#!/usr/bin/env python3
"""Extract choice-level evidence used by the Spidops clean-room policy."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

ROCKET_IDS = {400, 401, 414, 431, 434}


def load_index(root: Path) -> tuple[dict[str, Any], dict[str, int]]:
    meta_path = next(iter(glob.glob(str(root / "**" / "episodes.json"), recursive=True)), None)
    if not meta_path:
        raise FileNotFoundError("episodes.json not found")
    meta = json.load(open(meta_path, encoding="utf-8"))
    sid = str(meta["submission_id"])
    index: dict[str, int] = {}
    for episode in meta["episodes"]:
        eid = str(episode["episode_id"])
        if str(episode.get("agent_0_submission_id")) == sid:
            index[eid] = 0
        elif str(episode.get("agent_1_submission_id")) == sid:
            index[eid] = 1
    return meta, index


def card_from_option(obs: dict[str, Any], option: dict[str, Any], player_index: int) -> int | None:
    current = obs.get("current") or {}
    player = (current.get("players") or [{}, {}])[player_index]
    area = option.get("area")
    index = option.get("index")
    if not isinstance(index, int):
        return None
    if area == 1:
        cards = (obs.get("select") or {}).get("deck") or []
    elif area == 2:
        cards = player.get("hand") or []
    elif area == 3:
        cards = player.get("discard") or []
    elif area == 4:
        cards = player.get("active") or []
    elif area == 5:
        cards = player.get("bench") or []
    elif area == 12:
        cards = current.get("looking") or []
    else:
        return None
    if index < 0 or index >= len(cards):
        return None
    return (cards[index] or {}).get("id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    meta, target_index = load_index(args.root)
    setup_active = Counter()
    setup_bench = Counter()
    first_choice = Counter()
    mewtwo_bonus = Counter()
    mewtwo_discarded_energy = Counter()
    giovanni = Counter()
    bangle = Counter()

    for replay_path in glob.glob(str(args.root / "**" / "replay" / "episode_*.json"), recursive=True):
        eid = os.path.basename(replay_path).replace("episode_", "").replace(".json", "")
        if eid not in target_index:
            continue
        ti = target_index[eid]
        replay = json.load(open(replay_path, encoding="utf-8"))
        steps = replay["steps"]

        seen_select = set()
        got_active = False
        for j, step in enumerate(steps[:-1]):
            obs = step[ti].get("observation") or {}
            select = obs.get("select") or {}
            if not select:
                continue
            signature = json.dumps(select, sort_keys=True, separators=(",", ":"))
            is_new_selection = signature not in seen_select
            seen_select.add(signature)
            next_action = steps[j + 1][ti].get("action") or []
            context = select.get("context")

            if is_new_selection and context == 41 and next_action:
                chosen = next_action[0]
                options = select.get("option") or []
                if 0 <= chosen < len(options):
                    first_choice["yes" if options[chosen].get("type") == 1 else "no"] += 1

            if is_new_selection and context in (1, 2):
                chosen_cards = []
                options = select.get("option") or []
                for chosen in next_action:
                    if 0 <= chosen < len(options):
                        cid = card_from_option(obs, options[chosen], ti)
                        if cid is not None:
                            chosen_cards.append(cid)
                if context == 1 and chosen_cards and not got_active:
                    setup_active[chosen_cards[0]] += 1
                    got_active = True
                elif context == 2:
                    setup_bench.update(chosen_cards)

            effect = select.get("effect") or {}
            if effect.get("id") == 431 and any(
                option.get("energyIndex") is not None for option in (select.get("option") or [])
            ):
                next_obs = steps[j + 1][ti].get("observation") or {}
                next_logs = next_obs.get("logs") or []
                damage = sum(
                    -int(log.get("value", 0))
                    for log in next_logs
                    if log.get("playerIndex") != ti
                    and log.get("type") == 16
                    and int(log.get("value", 0)) < 0
                    and not log.get("putDamageCounter")
                )
                if damage:
                    selected = len(next_action)
                    mewtwo_bonus[f"selected_{selected}"] += 1
                    mewtwo_bonus[f"damage_{damage}"] += 1
                    current = obs.get("current") or {}
                    player = (current.get("players") or [{}, {}])[ti]
                    for chosen in next_action:
                        selection_options = select.get("option") or []
                        if not (0 <= chosen < len(selection_options)):
                            continue
                        option = selection_options[chosen]
                        area = option.get("area")
                        pokemon_index = option.get("index")
                        energy_index = option.get("energyIndex")
                        pokemon_list = (
                            player.get("active") or [] if area == 4 else player.get("bench") or []
                        )
                        if not isinstance(pokemon_index, int) or not isinstance(energy_index, int):
                            continue
                        if not (0 <= pokemon_index < len(pokemon_list)):
                            continue
                        energies = (pokemon_list[pokemon_index] or {}).get("energyCards") or []
                        if 0 <= energy_index < len(energies):
                            mewtwo_discarded_energy[(energies[energy_index] or {}).get("id")] += 1

        # Build a de-duplicated action event stream for Giovanni and Brave Bangle evidence.
        events = []
        previous_signature = None
        previous_state = None
        for step in steps:
            obs = step[ti].get("observation") or {}
            current = obs.get("current") or {}
            logs = obs.get("logs") or []
            signature = json.dumps(logs, sort_keys=True, separators=(",", ":"))
            if signature == previous_signature:
                if current:
                    previous_state = current
                continue
            previous_signature = signature
            events.append((current.get("turn"), logs, previous_state))
            if current:
                previous_state = current

        for i, (turn, logs, pre_state) in enumerate(events):
            if any(
                log.get("playerIndex") == ti and log.get("type") == 10 and log.get("cardId") == 1218
                for log in logs
            ):
                giovanni["uses"] += 1
                attacked = False
                ko = False
                for next_turn, next_logs, _ in events[i + 1 :]:
                    if next_turn != turn:
                        break
                    attacked |= any(
                        log.get("playerIndex") == ti and log.get("type") == 15 for log in next_logs
                    )
                    ko |= any(
                        log.get("playerIndex") != ti
                        and log.get("type") == 6
                        and log.get("fromArea") == 4
                        and log.get("toArea") == 3
                        for log in next_logs
                    )
                giovanni["same_turn_attack"] += int(attacked)
                giovanni["same_turn_ko"] += int(ko)

            attacks = [
                log
                for log in logs
                if log.get("playerIndex") == ti
                and log.get("type") == 15
                and log.get("cardId") == 401
                and log.get("attackId") == 560
            ]
            if len(attacks) == 1 and pre_state:
                me = (pre_state.get("players") or [{}, {}])[ti]
                active = (me.get("active") or [None])[0]
                board = (me.get("active") or []) + (me.get("bench") or [])
                rocket_count = sum(1 for pokemon in board if pokemon and pokemon.get("id") in ROCKET_IDS)
                has_bangle = any((tool or {}).get("id") == 1175 for tool in (active or {}).get("tools") or [])
                damage = sum(
                    -int(log.get("value", 0))
                    for log in logs
                    if log.get("playerIndex") != ti
                    and log.get("type") == 16
                    and int(log.get("value", 0)) < 0
                    and not log.get("putDamageCounter")
                )
                if has_bangle and damage:
                    bangle["attacks"] += 1
                    if damage == 30 * rocket_count + 30:
                        bangle["observed_plus_30"] += 1
                    elif damage == 30 * rocket_count:
                        bangle["observed_no_bonus"] += 1

    result = {
        "submission_id": meta["submission_id"],
        "setup_active": dict(setup_active),
        "setup_bench": dict(setup_bench),
        "resolved_go_first_choices": dict(first_choice),
        "mewtwo_bonus": dict(mewtwo_bonus),
        "mewtwo_discarded_energy": dict(mewtwo_discarded_energy),
        "giovanni": dict(giovanni),
        "brave_bangle_spidops": dict(bangle),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
