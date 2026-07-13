#!/usr/bin/env python3
"""Reproduce the replay metrics used to design the clean-room policy.

Usage:
    python analysis/extract_metrics.py LEADER_EXTRACTED_DIR COMPARISON_EXTRACTED_DIR OUTPUT_DIR

The replay observation log is cumulative for stretches of a game. The script de-duplicates
consecutive identical log arrays before counting actions. For energy attachment source analysis,
it locates the energy serial in the immediately preceding public state (hand vs discard).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

CARD_NAMES = {
    1: "Basic {G} Energy",
    5: "Basic {P} Energy",
    15: "Team Rocket's Energy",
    400: "Team Rocket's Tarountula",
    401: "Team Rocket's Spidops",
    414: "Team Rocket's Articuno",
    431: "Team Rocket's Mewtwo ex",
    434: "Team Rocket's Mimikyu",
    1094: "Bug Catching Set",
    1121: "Ultra Ball",
    1134: "Team Rocket's Transceiver",
    1152: "Poké Pad",
    1159: "Hero’s Cape",
    1175: "Brave Bangle",
    1216: "Team Rocket's Ariana",
    1217: "Team Rocket's Archer",
    1218: "Team Rocket's Giovanni",
    1220: "Team Rocket's Proton",
    1227: "Lillie's Determination",
    1257: "Team Rocket's Factory",
}
ATTACK_NAMES = {559: "Tarountula attack", 560: "Spidops attack", 608: "Mewtwo ex attack", 154: "Mimikyu attack"}


def average(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def load_episode_index(root: Path) -> tuple[dict[str, Any], dict[str, int]]:
    files = glob.glob(str(root / "**" / "episodes.json"), recursive=True)
    if not files:
        raise FileNotFoundError(f"episodes.json not found under {root}")
    meta = json.load(open(files[0], encoding="utf-8"))
    submission_id = str(meta["submission_id"])
    target_index: dict[str, int] = {}
    for episode in meta["episodes"]:
        eid = str(episode["episode_id"])
        if str(episode.get("agent_0_submission_id")) == submission_id:
            target_index[eid] = 0
        elif str(episode.get("agent_1_submission_id")) == submission_id:
            target_index[eid] = 1
    return meta, target_index


def locate_serial(state: dict[str, Any] | None, player_index: int, serial: int | None) -> str:
    if not state or serial is None:
        return "unknown"
    players = state.get("players") or []
    if player_index >= len(players):
        return "unknown"
    player = players[player_index]
    for area in ("hand", "discard", "deck"):
        for card in player.get(area) or []:
            if card and card.get("serial") == serial:
                return area
    for area in ("active", "bench"):
        for pokemon in player.get(area) or []:
            for card in (pokemon or {}).get("energyCards") or []:
                if card and card.get("serial") == serial:
                    return "attached"
    return "unknown"


def analyze(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Counter, Counter, Counter]:
    meta, target_index = load_episode_index(root)
    games: list[dict[str, Any]] = []
    card_use = Counter()
    attachment_matrix = Counter()
    damage_clusters = Counter()

    for replay_path in glob.glob(str(root / "**" / "replay" / "episode_*.json"), recursive=True):
        episode_id = os.path.basename(replay_path).replace("episode_", "").replace(".json", "")
        if episode_id not in target_index:
            continue
        ti = target_index[episode_id]
        replay = json.load(open(replay_path, encoding="utf-8"))
        reward = replay["rewards"][ti]
        game: dict[str, Any] = {
            "episode_id": int(episode_id),
            "win": reward > 0,
            "first": None,
            "attacks": 0,
            "tarountula_attacks": 0,
            "spidops_attacks": 0,
            "mewtwo_attacks": 0,
            "mewtwo_bonus_uses": 0,
            "mewtwo_bonus_discards": 0,
            "mimikyu_attacks": 0,
            "spidops_evolutions": 0,
            "spidops_accelerations": 0,
            "board_t1": None,
            "board_t2": None,
            "board_t3": None,
            "final_deck_zero": False,
        }
        prev_log_signature = None
        prev_state = None
        last_board_by_own_turn: dict[int, int] = {}

        for step_index, step in enumerate(replay["steps"]):
            record = step[ti]
            obs = record.get("observation") or {}
            current = obs.get("current") or None
            select = obs.get("select") or {}
            if current:
                first_player = current.get("firstPlayer")
                if first_player in (0, 1):
                    game["first"] = first_player == ti
                turn = current.get("turn")
                if isinstance(turn, int) and turn > 0 and select.get("context") == 0:
                    own_turn = (turn + 1) // 2
                    player = (current.get("players") or [{}, {}])[ti]
                    last_board_by_own_turn[own_turn] = len(player.get("active") or []) + len(player.get("bench") or [])

            logs = obs.get("logs") or []
            signature = json.dumps(logs, sort_keys=True, separators=(",", ":"))
            if signature != prev_log_signature:
                prev_log_signature = signature
                attacks = [l for l in logs if l.get("playerIndex") == ti and l.get("type") == 15]
                for log in logs:
                    if log.get("playerIndex") != ti:
                        continue
                    log_type = log.get("type")
                    card_id = log.get("cardId")
                    if log_type == 10 and card_id is not None:
                        card_use[("play_or_ability", card_id)] += 1
                    elif log_type == 12 and card_id == 401:
                        game["spidops_evolutions"] += 1
                    elif log_type == 15:
                        game["attacks"] += 1
                        if card_id == 400:
                            game["tarountula_attacks"] += 1
                        elif card_id == 401:
                            game["spidops_attacks"] += 1
                        elif card_id == 431:
                            game["mewtwo_attacks"] += 1
                        elif card_id == 434:
                            game["mimikyu_attacks"] += 1
                    elif log_type == 11:
                        source = locate_serial(prev_state, ti, log.get("serial"))
                        attachment_matrix[(card_id, log.get("cardIdTarget"), source)] += 1
                        if card_id == 1 and source == "discard" and log.get("cardIdTarget") == 401:
                            game["spidops_accelerations"] += 1
                if len(attacks) == 1:
                    attack_id = attacks[0].get("attackId")
                    total_damage = sum(
                        -int(l.get("value", 0))
                        for l in logs
                        if l.get("playerIndex") != ti
                        and l.get("type") == 16
                        and int(l.get("value", 0)) < 0
                        and not l.get("putDamageCounter")
                    )
                    if total_damage:
                        damage_clusters[(attack_id, total_damage)] += 1
            # Mewtwo ex resolves its optional bonus in a separate selection after the attack:
            # discard 0-2 Benched Energy, then deal 160 + 60 per selected Energy. The action for
            # this selection is stored on the next replay step, so read it together with the next
            # step's damage/discard logs. Ignore repeated waiting observations with zero damage.
            effect = select.get("effect") or {}
            options = select.get("option") or []
            if (
                effect.get("id") == 431
                and any(option.get("energyIndex") is not None for option in options)
                and step_index + 1 < len(replay["steps"])
            ):
                next_record = replay["steps"][step_index + 1][ti]
                next_action = next_record.get("action") or []
                next_logs = (next_record.get("observation") or {}).get("logs") or []
                resolved_damage = sum(
                    -int(log.get("value", 0))
                    for log in next_logs
                    if log.get("playerIndex") != ti
                    and log.get("type") == 16
                    and int(log.get("value", 0)) < 0
                    and not log.get("putDamageCounter")
                )
                if resolved_damage:
                    selected = len(next_action)
                    game["mewtwo_bonus_uses"] += int(selected > 0)
                    game["mewtwo_bonus_discards"] += selected
                    damage_clusters[(608, resolved_damage)] += 1

            if current:
                prev_state = current

        for turn_number in (1, 2, 3):
            game[f"board_t{turn_number}"] = last_board_by_own_turn.get(turn_number)
        final_current = (replay["steps"][-1][ti].get("observation") or {}).get("current") or {}
        players = final_current.get("players") or []
        if not game["win"] and ti < len(players):
            game["final_deck_zero"] = players[ti].get("deckCount") == 0
        games.append(game)

    def subset(win: bool | None = None) -> list[dict[str, Any]]:
        return games if win is None else [g for g in games if g["win"] is win]

    def metric(key: str, win: bool | None = None) -> float:
        values = [float(g[key]) for g in subset(win) if g.get(key) is not None]
        return average(values)

    summary = {
        "submission_id": meta["submission_id"],
        "metadata_episode_count": meta["episode_count"],
        "available_replays": len(games),
        "wins": sum(g["win"] for g in games),
        "losses": sum(not g["win"] for g in games),
        "win_rate": average([float(g["win"]) for g in games]),
        "first_games": sum(g["first"] is True for g in games),
        "first_win_rate": average([float(g["win"]) for g in games if g["first"] is True]),
        "second_win_rate": average([float(g["win"]) for g in games if g["first"] is False]),
        "attacks_per_game": metric("attacks"),
        "tarountula_attacks_per_game": metric("tarountula_attacks"),
        "spidops_attacks_per_game": metric("spidops_attacks"),
        "mewtwo_attacks_per_game": metric("mewtwo_attacks"),
        "mewtwo_bonus_uses_per_game": metric("mewtwo_bonus_uses"),
        "mewtwo_bonus_discards_per_game": metric("mewtwo_bonus_discards"),
        "mimikyu_attacks_per_game": metric("mimikyu_attacks"),
        "spidops_evolutions_per_game": metric("spidops_evolutions"),
        "spidops_accelerations_per_game": metric("spidops_accelerations"),
        "board_t1": metric("board_t1"),
        "board_t2": metric("board_t2"),
        "board_t3": metric("board_t3"),
        "winning_game": {
            "attacks_per_game": metric("attacks", True),
            "spidops_accelerations_per_game": metric("spidops_accelerations", True),
            "board_t1": metric("board_t1", True),
            "board_t2": metric("board_t2", True),
            "board_t3": metric("board_t3", True),
        },
        "losing_game": {
            "attacks_per_game": metric("attacks", False),
            "spidops_accelerations_per_game": metric("spidops_accelerations", False),
            "board_t1": metric("board_t1", False),
            "board_t2": metric("board_t2", False),
            "board_t3": metric("board_t3", False),
        },
        "losses_with_final_deck_count_zero": sum(g["final_deck_zero"] for g in games),
    }
    return summary, games, card_use, attachment_matrix, damage_clusters


def write_outputs(label: str, result: tuple, output_dir: Path) -> None:
    summary, games, card_use, attachment_matrix, damage_clusters = result
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{label}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(output_dir / f"{label}_games.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(games[0].keys()))
        writer.writeheader()
        writer.writerows(games)
    with open(output_dir / f"{label}_card_usage.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "card_id", "card_name", "count", "per_game"])
        for (category, card_id), count in sorted(card_use.items(), key=lambda x: (-x[1], x[0])):
            writer.writerow([category, card_id, CARD_NAMES.get(card_id, ""), count, count / len(games)])
    with open(output_dir / f"{label}_attachments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_card_id", "source_name", "target_card_id", "target_name", "source_area", "count", "per_game"])
        for (source_id, target_id, source_area), count in sorted(attachment_matrix.items(), key=lambda x: -x[1]):
            writer.writerow([
                source_id,
                CARD_NAMES.get(source_id, ""),
                target_id,
                CARD_NAMES.get(target_id, ""),
                source_area,
                count,
                count / len(games),
            ])
    with open(output_dir / f"{label}_attack_damage.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["attack_id", "attack_name", "observed_damage", "count"])
        for (attack_id, damage), count in sorted(damage_clusters.items(), key=lambda x: (x[0][0], -x[1])):
            writer.writerow([attack_id, ATTACK_NAMES.get(attack_id, ""), damage, count])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("leader_root", type=Path)
    parser.add_argument("comparison_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    write_outputs("leader", analyze(args.leader_root), args.output_dir)
    write_outputs("comparison", analyze(args.comparison_root), args.output_dir)


if __name__ == "__main__":
    main()
