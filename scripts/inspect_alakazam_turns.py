#!/usr/bin/env python3
"""Inspect scored MAIN decisions for selected Alakazam ladder episodes.

The replay stores the action for one observation in the following step.  This
tool keeps that alignment, resolves every option to a semantic label, and
records the fallback policy score so a missed attack or promotion can be
audited without dumping the full replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent


AREA_KEYS = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench"}
MAIN_CONTEXT = 0


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


def _card(obs: dict[str, Any], option: dict[str, Any]) -> dict[str, Any]:
    current = obs.get("current") or {}
    players = current.get("players") or []
    player = int(option.get("playerIndex", current.get("yourIndex", 0)))
    if not 0 <= player < len(players):
        return {}
    key = AREA_KEYS.get(int(option.get("area", -1)))
    if key is None and int(option.get("type", -1)) in (7, 8, 9):
        key = "hand"
    cards = players[player].get(key, []) if key else []
    index = int(option.get("index", -1))
    return cards[index] or {} if 0 <= index < len(cards) else {}


def _target(obs: dict[str, Any], option: dict[str, Any]) -> dict[str, Any]:
    current = obs.get("current") or {}
    players = current.get("players") or []
    player = int(option.get("playerIndex", current.get("yourIndex", 0)))
    if not 0 <= player < len(players):
        return {}
    key = AREA_KEYS.get(int(option.get("inPlayArea", -1)))
    cards = players[player].get(key, []) if key else []
    index = int(option.get("inPlayIndex", -1))
    return cards[index] or {} if 0 <= index < len(cards) else {}


def _label(obs: dict[str, Any], option: dict[str, Any]) -> dict[str, Any]:
    source = _card(obs, option)
    target = _target(obs, option)
    return {
        "type": int(option.get("type", -1)),
        "card_id": int(source.get("id", -1)),
        "attack_id": int(option.get("attackId", -1)),
        "area": int(option.get("area", -1)),
        "target_id": int(target.get("id", -1)),
        "target_area": int(option.get("inPlayArea", -1)),
    }


def _pokemon(cards: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": int(card.get("id", -1)),
            "hp": int(card.get("hp", -1)),
            "energy": len(card.get("energies") or []),
            "serial": int(card.get("serial", -1)),
        }
        for card in cards
        if isinstance(card, dict)
    ]


def _state(obs: dict[str, Any], seat: int) -> dict[str, Any]:
    current = obs.get("current") or {}
    players = current.get("players") or [{}, {}]
    own = players[seat]
    opponent = players[1 - seat]
    return {
        "turn": int(current.get("turn", -1)),
        "hand": int(own.get("handCount", len(own.get("hand") or []))),
        "hand_ids": [int(card["id"]) for card in own.get("hand") or []],
        "deck": int(own.get("deckCount", -1)),
        "own_prizes": len(own.get("prize") or []),
        "opp_prizes": len(opponent.get("prize") or []),
        "own_active": _pokemon(own.get("active") or []),
        "own_bench": _pokemon(own.get("bench") or []),
        "opp_active": _pokemon(opponent.get("active") or []),
        "opp_bench": _pokemon(opponent.get("bench") or []),
    }


def _scores(module: Any, obs: dict[str, Any]) -> list[float]:
    policy_module = module.fallback_policy
    parsed = policy_module.to_observation_class(obs)
    policy = policy_module.AlakazamPolicy(parsed)
    return [float(policy._score(option)) for option in parsed.select.option]


def inspect(
    run_dir: Path,
    agent_dir: Path,
    submission_id: int,
    episode_ids: set[int],
) -> dict[str, Any]:
    _, _, module = load_dir_agent(agent_dir)
    seats = _seats(run_dir, submission_id)
    episodes: dict[str, Any] = {}
    for episode_id in sorted(episode_ids):
        seat = seats.get(episode_id)
        replay_path = (
            run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if seat is None or not replay_path.exists():
            continue
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        decisions = []
        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            record = step[seat] if seat < len(step) else {}
            obs = (record or {}).get("observation") or {}
            select = obs.get("select") or {}
            if (
                (record or {}).get("status") != "ACTIVE"
                or int(select.get("context", -1)) != MAIN_CONTEXT
            ):
                continue
            action = steps[step_index + 1][seat].get("action")
            if not isinstance(action, list) or len(action) != 1:
                continue
            options = select.get("option") or []
            selected = int(action[0])
            if not 0 <= selected < len(options):
                continue
            try:
                scores = _scores(module, obs)
            except Exception as exc:
                scores = [0.0] * len(options)
                score_error = f"{type(exc).__name__}: {exc}"
            else:
                score_error = None
            ranked = sorted(
                range(len(options)),
                key=lambda index: scores[index],
                reverse=True,
            )
            decisions.append(
                {
                    "step_index": step_index,
                    **_state(obs, seat),
                    "selected": _label(obs, options[selected]),
                    "selected_score": scores[selected],
                    "top_options": [
                        {
                            **_label(obs, options[index]),
                            "score": scores[index],
                        }
                        for index in ranked[:8]
                    ],
                    "score_error": score_error,
                }
            )
        episodes[str(episode_id)] = {
            "seat": seat,
            "reward": (replay.get("rewards") or [None, None])[seat],
            "decisions": decisions,
        }
    return {
        "run_dir": str(run_dir),
        "agent_dir": str(agent_dir),
        "submission_id": submission_id,
        "episodes": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--episodes", required=True, help="Comma-separated episode ids")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    episode_ids = {int(value) for value in args.episodes.split(",") if value.strip()}
    report = inspect(
        args.run_dir.resolve(),
        args.agent_dir.resolve(),
        args.submission_id,
        episode_ids,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
