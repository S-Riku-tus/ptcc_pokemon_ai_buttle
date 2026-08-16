"""Audit concrete Dragapult decisions in a downloaded Kaggle ladder run.

This report deliberately focuses on the small set of decisions that can ruin a
Dragapult game even when aggregate teacher agreement looks healthy: typed
energy, evolution pace, Phantom Dive access, premature END, and unsafe Boss.

Usage:
  python scripts/analyze_dragapult_live.py \
      data/submissions/submission_55545828_dragapult_v1 \
      --report experiments/dragapult_ml_v1/live_55545828_audit.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover - portable fallback
    orjson = None


ROOT = Path(__file__).resolve().parents[1]
CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}
ATTACKS = {
    int(attack["attackId"]): attack
    for attack in json.loads(
        (ROOT / "vendor" / "cg" / "attacks.json").read_text(encoding="utf-8")
    )
}

FIRE, PSYCHIC, DARK = 2, 5, 7
MUNKIDORI, DREEPY, DRAKLOAK, DRAGAPULT = 112, 119, 120, 121
BOSS = 1182
PHANTOM_DIVE = 154

MAIN = 0
TO_HAND = 7
DAMAGE_COUNTER = {13, 14, 15}

PLAY, ATTACH, EVOLVE = 7, 8, 9
RETREAT, ATTACK, END = 12, 13, 14

AREA_HAND, AREA_ACTIVE, AREA_BENCH = 2, 4, 5


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw)


def card_name(card_id: int | None) -> str:
    if card_id is None:
        return "unknown"
    return str(CARDS.get(int(card_id), {}).get("name") or card_id)


def attack_name(attack_id: int | None) -> str:
    if attack_id is None:
        return "unknown"
    return str(ATTACKS.get(int(attack_id), {}).get("name") or attack_id)


def in_play(player: dict[str, Any]) -> list[dict[str, Any]]:
    return list(player.get("active") or []) + list(player.get("bench") or [])


def card_at(
    current: dict[str, Any], select: dict[str, Any], option: dict[str, Any], seat: int
) -> dict[str, Any] | None:
    option_type = int(option.get("type", -1))
    area = int(option.get("area", -1))
    index = int(option.get("index", -1))
    player_index = int(option.get("playerIndex", seat))
    if option_type in (PLAY, ATTACH, EVOLVE):
        area, player_index = AREA_HAND, seat
    players = current.get("players") or [{}, {}]
    owner = players[player_index] if player_index in (0, 1) else {}
    zones = {
        1: select.get("deck") or [],
        AREA_HAND: owner.get("hand") or [],
        3: owner.get("discard") or [],
        AREA_ACTIVE: owner.get("active") or [],
        AREA_BENCH: owner.get("bench") or [],
        6: owner.get("prize") or [],
        7: current.get("stadium") or [],
        12: current.get("looking") or [],
    }
    zone = zones.get(area, [])
    if not isinstance(zone, list) or not 0 <= index < len(zone):
        return None
    value = zone[index]
    return value if isinstance(value, dict) else None


def target_at(
    current: dict[str, Any], option: dict[str, Any], seat: int
) -> dict[str, Any] | None:
    area = int(option.get("inPlayArea", -1))
    index = int(option.get("inPlayIndex", -1))
    players = current.get("players") or [{}, {}]
    owner = players[seat]
    zone = owner.get("active") if area == AREA_ACTIVE else owner.get("bench")
    if not isinstance(zone, list) or not 0 <= index < len(zone):
        return None
    value = zone[index]
    return value if isinstance(value, dict) else None


def archetype(deck: list[int]) -> str:
    pokemon = Counter(
        card_id
        for card_id in deck
        if CARDS.get(card_id, {}).get("cardType") == 0
    )
    if not pokemon:
        return "unknown"

    def key(item: tuple[int, int]) -> tuple[int, int, int, int, int]:
        card_id, count = item
        card = CARDS[card_id]
        return (
            int(bool(card.get("stage2"))),
            int(bool(card.get("megaEx") or card.get("ex"))),
            int(bool(card.get("stage1"))),
            count,
            int(card.get("hp") or 0),
        )

    return card_name(max(pokemon.items(), key=key)[0])


def action_label(
    current: dict[str, Any], select: dict[str, Any], option: dict[str, Any], seat: int
) -> tuple[str, int | None, int | None]:
    option_type = int(option.get("type", -1))
    card = card_at(current, select, option, seat)
    target = target_at(current, option, seat)
    card_id = int(card["id"]) if card and card.get("id") is not None else None
    target_id = int(target["id"]) if target and target.get("id") is not None else None
    if option_type == ATTACH:
        return "attach", card_id, target_id
    if option_type == EVOLVE:
        return "evolve", card_id, target_id
    if option_type == PLAY:
        return "play", card_id, None
    if option_type == ATTACK:
        return "attack", int(option.get("attackId", -1)), None
    if option_type == RETREAT:
        return "retreat", None, None
    if option_type == END:
        return "end", None, None
    return f"option_{option_type}", card_id, target_id


def analyse_episode(path: Path, seat: int) -> dict[str, Any]:
    replay = load(path)
    steps = replay.get("steps") or []
    decks: list[list[int]] = [[], []]
    if len(steps) > 1:
        for player in (0, 1):
            action = steps[1][player].get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[player] = [int(value) for value in action]

    rewards = replay.get("rewards") or [None, None]
    result = "draw"
    if rewards[seat] != rewards[1 - seat]:
        result = "win" if rewards[seat] > rewards[1 - seat] else "loss"

    events: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    first_attack_turn: int | None = None
    first_phantom_turn: int | None = None
    min_prizes = 6
    max_turn = 0

    for step_index, pair in enumerate(steps):
        payload = pair[seat]
        # Kaggle carries the previous action forward on the inactive seat.
        # Counting those snapshots would duplicate almost every decision.
        if payload.get("status") != "ACTIVE":
            continue
        observation = payload.get("observation") or {}
        current = observation.get("current")
        select = observation.get("select")
        # In the Kaggle replay schema, steps[i + 1].action is the response to
        # steps[i].observation.  Pairing an action with the observation stored
        # in the same step silently assigns choices to the wrong board.
        action = (
            steps[step_index + 1][seat].get("action")
            if step_index + 1 < len(steps) else None
        )
        if not isinstance(current, dict) or not isinstance(select, dict):
            continue
        if not isinstance(action, list) or not action or len(action) == 60:
            continue

        turn = int(current.get("turn") or 0)
        max_turn = max(max_turn, turn)
        players = current.get("players") or [{}, {}]
        mine = players[seat]
        visible_prizes = len(mine.get("prize") or [])
        if visible_prizes:
            min_prizes = min(min_prizes, visible_prizes)
        options = select.get("option") or []
        context = int(select.get("context", -1))

        for selected_index in action:
            if not isinstance(selected_index, int) or not 0 <= selected_index < len(options):
                continue
            option = options[selected_index]
            kind, subject, target_id = action_label(current, select, option, seat)
            event: dict[str, Any] = {
                "step": step_index,
                "turn": turn,
                "context": context,
                "kind": kind,
                "subject": subject,
                "subject_name": (
                    attack_name(subject) if kind == "attack" else card_name(subject)
                ),
                "target": target_id,
                "target_name": card_name(target_id),
            }

            if kind == "attach":
                stats["attachments"] += 1
                target = target_at(current, option, seat) or {}
                before = [int(value) for value in target.get("energies") or []]
                event["target_energy_before"] = before
                if subject in before:
                    stats["duplicate_color_attachments"] += 1
                    event["warning"] = "duplicate_energy_color"
                if target_id in (DREEPY, DRAKLOAK, DRAGAPULT) and subject in (FIRE, PSYCHIC):
                    stats["route_attachments"] += 1
                    if subject not in before:
                        stats["useful_route_attachments"] += 1
                if target_id == MUNKIDORI and subject == DARK:
                    stats["munkidori_dark_attachments"] += 1

            elif kind == "evolve":
                stats[f"evolve_{subject}"] += 1
            elif kind == "attack":
                stats["attacks"] += 1
                first_attack_turn = turn if first_attack_turn is None else first_attack_turn
                if subject == PHANTOM_DIVE:
                    stats["phantom_dives"] += 1
                    first_phantom_turn = turn if first_phantom_turn is None else first_phantom_turn
            elif kind == "end":
                attack_available = any(
                    int(candidate.get("type", -1)) == ATTACK for candidate in options
                )
                if attack_available:
                    stats["end_with_attack_available"] += 1
                    event["warning"] = "end_with_attack_available"
            elif kind == "play" and subject == BOSS:
                active = (mine.get("active") or [{}])[0]
                active_energy = set(int(value) for value in active.get("energies") or [])
                opponent = players[1 - seat]
                ko_bench = any(
                    0 < int(card.get("hp") or 0) <= 200
                    for card in opponent.get("bench") or []
                )
                safe = (
                    int(active.get("id", -1)) == DRAGAPULT
                    and {FIRE, PSYCHIC}.issubset(active_energy)
                    and ko_bench
                )
                if not safe:
                    stats["unsafe_boss"] += 1
                    event["warning"] = "boss_without_active_phantom_ko"

            if context == TO_HAND and subject is not None:
                stats[f"searched_{subject}"] += 1
            if context in DAMAGE_COUNTER:
                stats["spread_selections"] += 1
            events.append(event)

    return {
        "episode_id": int(replay.get("info", {}).get("EpisodeId") or path.stem.split("_")[-1]),
        "seat": seat,
        "result": result,
        "reward": rewards[seat],
        "opponent_archetype": archetype(decks[1 - seat]),
        "max_turn": max_turn,
        "prizes_taken": 6 - min_prizes,
        "first_attack_turn": first_attack_turn,
        "first_phantom_turn": first_phantom_turn,
        "stats": dict(stats),
        "warnings": [event for event in events if event.get("warning")],
        "key_events": [
            event for event in events
            if event["kind"] in {"attach", "evolve", "attack", "retreat"}
            or event.get("warning")
        ],
    }


def summarise(games: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    for game in games:
        totals.update(game["stats"])
    wins = sum(game["result"] == "win" for game in games)
    phantom_turns = [
        game["first_phantom_turn"]
        for game in games if game["first_phantom_turn"] is not None
    ]
    return {
        "games": len(games),
        "record": f"{wins}-{len(games) - wins}",
        "win_rate": wins / len(games) if games else 0.0,
        "games_with_phantom_dive": len(phantom_turns),
        "first_phantom_turn_mean": (
            sum(phantom_turns) / len(phantom_turns) if phantom_turns else None
        ),
        "games_without_prize": sum(game["prizes_taken"] == 0 for game in games),
        "decision_totals": dict(totals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument(
        "--teacher-index", type=Path,
        help="Exact-deck episodes.csv to audit instead of a downloaded run.",
    )
    parser.add_argument(
        "--split-report", type=Path,
        help="With --teacher-index, keep only the chronological test split.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    games = []
    source: str
    if args.teacher_index:
        rows = list(csv.DictReader(
            args.teacher_index.read_text(encoding="utf-8-sig").splitlines()
        ))
        boundaries: dict[str, list[int]] = {}
        if args.split_report:
            boundaries = load(args.split_report).get("split_boundaries") or {}
        seen: set[tuple[str, int]] = set()
        for row in rows:
            episode_id = str(row["episode_id"])
            seat = int(row["seat_index"])
            key = (episode_id, seat)
            if key in seen:
                continue
            seen.add(key)
            boundary = boundaries.get(str(row.get("team_id")))
            if boundary and int(episode_id) <= int(boundary[1]):
                continue
            replay_path = Path(row["replay_path"])
            if not replay_path.is_absolute():
                replay_path = args.teacher_index.parent.parent / replay_path
            games.append(analyse_episode(replay_path, seat))
        source = str(args.teacher_index)
    else:
        if args.run is None:
            parser.error("run or --teacher-index is required")
        rows = list(csv.DictReader(
            (args.run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()
        ))
        for row in rows:
            episode_id = str(row["episode_id"])
            seat = int(row["detected_submission_agent_index"])
            replay_path = (
                args.run / "episodes" / episode_id / "replay"
                / f"episode_{episode_id}.json"
            )
            games.append(analyse_episode(replay_path, seat))
        source = str(args.run)

    report = {"run": source, "summary": summarise(games), "games": games}
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    for game in games:
        print(json.dumps({
            "episode": game["episode_id"],
            "result": game["result"],
            "opponent": game["opponent_archetype"],
            "max_turn": game["max_turn"],
            "prizes_taken": game["prizes_taken"],
            "first_phantom": game["first_phantom_turn"],
            "warnings": [warning["warning"] for warning in game["warnings"]],
        }, ensure_ascii=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
