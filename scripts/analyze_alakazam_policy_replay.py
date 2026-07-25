#!/usr/bin/env python3
"""Compare an agent's one-step choices with recorded ladder decisions.

This is a teacher-forced replay: every decision is evaluated on the real
observation that occurred in the ladder game, even when the candidate policy
would have chosen a different preceding action.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _seat_by_episode(run_dir: Path, submission_id: int) -> dict[int, int]:
    result: dict[int, int] = {}
    with (run_dir / "episodes.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            episode_id = int(row["episode_id"])
            if int(row["agent_0_submission_id"]) == submission_id:
                result[episode_id] = 0
            elif int(row["agent_1_submission_id"]) == submission_id:
                result[episode_id] = 1
    return result


def _candidate_card(obs: dict[str, Any], option: dict[str, Any]) -> dict[str, Any]:
    current = obs.get("current") or {}
    player = int(option.get("playerIndex", current.get("yourIndex", 0)))
    players = current.get("players") or []
    if player < 0 or player >= len(players):
        return {}
    area = int(option.get("area", -1))
    index = int(option.get("index", -1))
    key = AREA_KEYS.get(area)
    if key is None and int(option.get("type", -1)) in (7, 8, 9):
        key = "hand"
    cards = players[player].get(key, []) if key else []
    if 0 <= index < len(cards):
        return cards[index] or {}
    return {}


def _semantic_label(
    obs: dict[str, Any], action: list[int]
) -> tuple[tuple[Any, ...], ...]:
    options = (obs.get("select") or {}).get("option") or []
    labels: list[tuple[Any, ...]] = []
    for selected in action:
        if not isinstance(selected, int) or not (0 <= selected < len(options)):
            labels.append(("OUT",))
            continue
        option = options[selected]
        card = _candidate_card(obs, option)
        labels.append(
            (
                int(option.get("type", -1)),
                int(card.get("id", -1)),
                int(option.get("attackId", -1)),
            )
        )
    return tuple(labels)


def _select_context(obs: dict[str, Any]) -> tuple[int, int]:
    select = obs.get("select") or {}
    effect = select.get("effect") or {}
    return int(select.get("context", -1)), int(effect.get("id", -1))


def _focus_label(obs: dict[str, Any], action: list[int]) -> str | None:
    select_type, effect_id = _select_context(obs)
    options = (obs.get("select") or {}).get("option") or []
    if not action or not isinstance(action[0], int) or not (0 <= action[0] < len(options)):
        return None
    option = options[action[0]]
    option_type = int(option.get("type", -1))
    card_id = int(_candidate_card(obs, option).get("id", -1))
    if select_type == 0 and option_type == 10 and card_id == 66:
        return "dudunsparce_draw"
    if select_type == 0 and option_type == 7 and card_id == 140:
        current = obs.get("current") or {}
        players = current.get("players") or [{}, {}]
        own_index = int(current.get("yourIndex", 0))
        opp_prizes = int(
            len(players[1 - own_index].get("prize") or [])
        )
        return f"fez_play_opp_prizes_{opp_prizes}"
    if select_type == 0 and option_type == 0:
        return "main_end"
    if select_type == 3 and effect_id == 741:
        return f"abra_switch_to_{card_id}"
    if select_type == 4:
        return f"ko_promote_{card_id}"
    if select_type == 3 and effect_id == 1182:
        return f"boss_target_{card_id}"
    return None


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    seat_by_episode = _seat_by_episode(run_dir, args.submission_id)
    agent, _, module = load_dir_agent(args.agent_dir.resolve())

    exact = 0
    semantic = 0
    decisions = 0
    recorded_focus: Counter[str] = Counter()
    predicted_focus: Counter[str] = Counter()
    changed_pairs: Counter[str] = Counter()

    for replay_path in sorted((run_dir / "episodes").glob("*/replay/*.json")):
        episode_id = int(replay_path.parents[1].name)
        seat = seat_by_episode.get(episode_id)
        if seat is None:
            continue
        agent({"select": None})
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            obs = (step[seat] or {}).get("observation") or {}
            record = step[seat] or {}
            if record.get("status") != "ACTIVE" or not obs.get("select"):
                continue
            next_step = steps[step_index + 1]
            if seat >= len(next_step):
                continue
            recorded = (next_step[seat] or {}).get("action")
            if not isinstance(recorded, list) or len(recorded) == 60:
                continue
            try:
                predicted = list(agent(obs))
            except Exception:
                continue
            decisions += 1
            exact += int(predicted == recorded)
            recorded_label = _semantic_label(obs, recorded)
            predicted_label = _semantic_label(obs, predicted)
            semantic += int(recorded_label == predicted_label)
            recorded_name = _focus_label(obs, recorded)
            predicted_name = _focus_label(obs, predicted)
            if recorded_name:
                recorded_focus[recorded_name] += 1
            if predicted_name:
                predicted_focus[predicted_name] += 1
            if recorded_label != predicted_label:
                context = _select_context(obs)
                pair = f"{context}: {recorded_label} -> {predicted_label}"
                changed_pairs[pair] += 1

    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "agent_dir": str(args.agent_dir.resolve()),
        "submission_id": args.submission_id,
        "decisions": decisions,
        "exact_agreement": exact / decisions if decisions else 0.0,
        "semantic_agreement": semantic / decisions if decisions else 0.0,
        "recorded_focus": dict(recorded_focus.most_common()),
        "predicted_focus": dict(predicted_focus.most_common()),
        "top_changed_pairs": changed_pairs.most_common(30),
    }
    snapshot = getattr(module, "diag_snapshot", None)
    if callable(snapshot):
        report["diag"] = snapshot()

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
