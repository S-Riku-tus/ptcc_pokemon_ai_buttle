"""Compare a local Alakazam policy with actions recorded by top public agents."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent


AREA_KEYS = {
    1: "deck",
    2: "hand",
    3: "discard",
    4: "active",
    5: "bench",
}


def _card_id(observation: dict, option: dict) -> int | None:
    current = observation.get("current") or {}
    players = current.get("players") or []
    player_index = option.get("playerIndex", current.get("yourIndex", 0))
    area = AREA_KEYS.get(option.get("area"))
    if area is None and option.get("type") in (7, 8, 9):
        area = "hand"
    index = option.get("index")
    if area is None or not isinstance(index, int) or not isinstance(player_index, int):
        return None
    if not (0 <= player_index < len(players)):
        return None
    cards = players[player_index].get(area) or []
    if not (0 <= index < len(cards)) or cards[index] is None:
        return None
    return cards[index].get("id")


def _label(observation: dict, action: list[int]) -> str:
    select = observation.get("select") or {}
    options = select.get("option") or []
    labels = []
    for selected in action:
        if not (0 <= selected < len(options)):
            labels.append("OUT")
            continue
        option = options[selected]
        option_type = option.get("type")
        card_id = _card_id(observation, option)
        attack_id = option.get("attackId")
        if card_id is not None:
            labels.append(f"{option_type}:card:{card_id}")
        elif attack_id is not None:
            labels.append(f"{option_type}:attack:{attack_id}")
        else:
            labels.append(str(option_type))
    return "+".join(labels) or "empty"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--ranks", default="2,3,5,8")
    parser.add_argument("--deck-hash", default="cc38cb450b86770a")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    ranks = {int(value) for value in args.ranks.split(",") if value}
    index_path = args.run_root / "indexes" / "replay_index.csv"
    with index_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["leaderboard_rank"]) in ranks
            and row["deck_hash"] == args.deck_hash
        ]

    agent, _, _ = load_dir_agent(args.agent_dir.resolve())
    decisions = 0
    differences = 0
    contexts: Counter[str] = Counter()
    teacher: Counter[str] = Counter()
    local: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    by_rank: Counter[tuple[int, str]] = Counter()
    focus_states: list[dict] = []

    for row in rows:
        agent({"select": None})
        replay = json.loads(
            (args.run_root / row["replay_path"]).read_text(encoding="utf-8")
        )
        seat = int(row["seat_index"])
        rank = int(row["leaderboard_rank"])
        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            record = step[seat]
            observation = record.get("observation") or {}
            select = observation.get("select")
            if select is None or record.get("status") != "ACTIVE":
                continue
            # Kaggle stores the action chosen from this observation in the
            # following replay step.
            recorded = list(steps[step_index + 1][seat].get("action") or [])
            if len(recorded) == 60:
                continue
            predicted = list(agent(observation))
            decisions += 1
            if predicted == recorded:
                continue
            differences += 1
            context = str(select.get("context"))
            teacher_label = _label(observation, recorded)
            local_label = _label(observation, predicted)
            contexts[context] += 1
            teacher[teacher_label] += 1
            local[local_label] += 1
            pairs[(teacher_label, local_label)] += 1
            by_rank[(rank, context)] += 1
            if teacher_label in {
                "7:card:1086",
                "7:card:1184",
                "7:card:1197",
                "7:card:1266",
            }:
                current = observation.get("current") or {}
                players = current.get("players") or []
                own = players[seat] if seat < len(players) else {}
                opponent = players[1 - seat] if len(players) == 2 else {}
                focus_states.append({
                    "episode_id": int(row["episode_id"]),
                    "rank": rank,
                    "teacher": teacher_label,
                    "local": local_label,
                    "turn": current.get("turn"),
                    "hand_count": own.get("handCount"),
                    "deck_count": own.get("deckCount"),
                    "prize_count": len(own.get("prize") or []),
                    "hand_ids": [card.get("id") for card in own.get("hand") or []],
                    "discard_ids": [
                        card.get("id") for card in own.get("discard") or []
                    ],
                    "board_ids": [
                        card.get("id")
                        for card in (own.get("active") or []) + (own.get("bench") or [])
                        if card is not None
                    ],
                    "opponent_hand_count": opponent.get("handCount"),
                    "opponent_board_ids": [
                        card.get("id")
                        for card in (
                            (opponent.get("active") or [])
                            + (opponent.get("bench") or [])
                        )
                        if card is not None
                    ],
                })

    report = {
        "agent_dir": str(args.agent_dir),
        "ranks": sorted(ranks),
        "games": len(rows),
        "decisions": decisions,
        "differences": differences,
        "agreement": 1.0 - differences / decisions if decisions else None,
        "contexts": contexts.most_common(),
        "teacher_actions": teacher.most_common(80),
        "local_actions": local.most_common(80),
        "pairs": [
            {"teacher": pair[0], "local": pair[1], "count": count}
            for pair, count in pairs.most_common(120)
        ],
        "focus_states": focus_states,
        "differences_by_rank_context": [
            {"rank": key[0], "context": key[1], "count": count}
            for key, count in sorted(by_rank.items())
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(args.output),
            "games": report["games"],
            "decisions": report["decisions"],
            "differences": report["differences"],
            "agreement": report["agreement"],
        }, ensure_ascii=False, indent=2))
    else:
        print(rendered)


if __name__ == "__main__":
    main()
