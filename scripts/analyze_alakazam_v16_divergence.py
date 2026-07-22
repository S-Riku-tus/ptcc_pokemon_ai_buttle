"""Summarize action differences between Alakazam v16 and v15 trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from agent_loader import load_dir_agent
from local_arena import ROOT


def _option_label(observation: dict, action: list[int]) -> str:
    select = observation.get("select") or {}
    options = select.get("option") or []
    labels = []
    for index in action:
        option = options[index] if 0 <= index < len(options) else {}
        labels.append(str(option.get("type", "out_of_range")))
    return "+".join(labels) or "empty"


def _option_detail(observation: dict, action: list[int]) -> str:
    select = observation.get("select") or {}
    options = select.get("option") or []
    current = observation.get("current") or {}
    player_index = current.get("yourIndex", 0)
    players = current.get("players") or []
    player = players[player_index] if player_index < len(players) else {}
    hand = player.get("hand") or []
    details = []
    for selected in action:
        option = options[selected] if 0 <= selected < len(options) else {}
        detail = {key: option.get(key) for key in (
            "type", "attackId", "area", "index", "inPlayArea", "inPlayIndex"
        ) if option.get(key) is not None}
        if option.get("type") in (7, 8, 9) and option.get("index") is not None:
            index = option["index"]
            if 0 <= index < len(hand):
                detail["handCard"] = hand[index].get("id")
        details.append(detail)
    return json.dumps(details, sort_keys=True, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--source-agent", default="alakazam_ml_v16")
    args = parser.parse_args()

    v16_dir = ROOT / "agents" / "alakazam" / "alakazam_ml_v16"
    v15_dir = ROOT / "agents" / "alakazam" / "alakazam_ml_v15"
    v16, _, _ = load_dir_agent(v16_dir)
    v15, _, _ = load_dir_agent(v15_dir)

    previous_game = None
    total = 0
    differences = Counter()
    pairs = Counter()
    contexts = Counter()
    detail_pairs = Counter()
    with args.trajectory.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("agent_version") != args.source_agent:
                continue
            game_id = row.get("game_id")
            if game_id != previous_game:
                v16({"select": None})
                v15({"select": None})
                previous_game = game_id
            observation = row["observation"]
            action16 = list(v16(observation))
            action15 = list(v15(observation))
            total += 1
            if action16 == action15:
                continue
            select = observation.get("select") or {}
            label16 = _option_label(observation, action16)
            label15 = _option_label(observation, action15)
            differences[label16] += 1
            pairs[(label16, label15)] += 1
            detail_pairs[
                (_option_detail(observation, action16),
                 _option_detail(observation, action15))
            ] += 1
            contexts[str(select.get("context"))] += 1

    print(json.dumps({
        "decisions": total,
        "different": sum(differences.values()),
        "v16_types": differences.most_common(),
        "type_pairs": [(list(pair), count) for pair, count in pairs.most_common()],
        "detail_pairs": [
            (list(pair), count) for pair, count in detail_pairs.most_common(30)
        ],
        "contexts": contexts.most_common(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
