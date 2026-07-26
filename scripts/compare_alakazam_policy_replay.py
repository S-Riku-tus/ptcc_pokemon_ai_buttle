#!/usr/bin/env python3
"""Teacher-forced comparison of two Alakazam policies on one ladder run."""

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
    if not 0 <= player < len(players):
        return {}
    area = int(option.get("area", -1))
    key = AREA_KEYS.get(area)
    if key is None and int(option.get("type", -1)) in (7, 8, 9):
        key = "hand"
    cards = players[player].get(key, []) if key else []
    index = int(option.get("index", -1))
    return cards[index] or {} if 0 <= index < len(cards) else {}


def _label(obs: dict[str, Any], action: list[int]) -> tuple[tuple[int, int, int], ...]:
    options = (obs.get("select") or {}).get("option") or []
    result = []
    for index in action:
        if not isinstance(index, int) or not 0 <= index < len(options):
            result.append((-999, -999, -999))
            continue
        option = options[index]
        result.append(
            (
                int(option.get("type", -1)),
                int(_candidate_card(obs, option).get("id", -1)),
                int(option.get("attackId", -1)),
            )
        )
    return tuple(result)


def _context(obs: dict[str, Any]) -> tuple[int, int]:
    select = obs.get("select") or {}
    effect = select.get("effect") or {}
    return int(select.get("context", -1)), int(effect.get("id", -1))


def _state_summary(obs: dict[str, Any], episode_id: int, seat: int) -> dict[str, Any]:
    current = obs.get("current") or {}
    players = current.get("players") or [{}, {}]
    own = players[seat]
    opponent = players[1 - seat]

    def ids(cards: list[Any]) -> list[int]:
        return [int(card["id"]) for card in cards if isinstance(card, dict) and "id" in card]

    return {
        "episode_id": episode_id,
        "turn": int(current.get("turn", -1)),
        "hand": int(own.get("handCount", len(own.get("hand") or []))),
        "deck": int(own.get("deckCount", len(own.get("deck") or []))),
        "own_prizes": len(own.get("prize") or []),
        "opp_prizes": len(opponent.get("prize") or []),
        "own_active": ids(own.get("active") or []),
        "own_bench": ids(own.get("bench") or []),
        "opp_active": ids(opponent.get("active") or []),
        "opp_bench": ids(opponent.get("bench") or []),
    }


def _policy_scores(main_module: Any, obs: dict[str, Any]) -> list[float] | None:
    try:
        policy_module = main_module.fallback_policy
        parsed = policy_module.to_observation_class(obs)
        policy = policy_module.AlakazamPolicy(parsed)
        return [float(policy._score(option)) for option in parsed.select.option]
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--submission-id", required=True, type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-replays",
        type=int,
        help="Deterministic sorted replay cap for a frozen, reproducible audit.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    seats = _seat_by_episode(run_dir, args.submission_id)
    baseline, _, baseline_module = load_dir_agent(args.baseline_dir.resolve())
    candidate, _, candidate_module = load_dir_agent(args.candidate_dir.resolve())
    total = exact = semantic = 0
    teacher_comparable = baseline_teacher = candidate_teacher = 0
    changed_teacher_comparable = changed_baseline_teacher = changed_candidate_teacher = 0
    changed_pairs: Counter[str] = Counter()
    changed_contexts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    replay_paths = sorted((run_dir / "episodes").glob("*/replay/*.json"))
    if args.max_replays is not None:
        replay_paths = replay_paths[: max(0, args.max_replays)]
    for replay_path in replay_paths:
        episode_id = int(replay_path.parents[1].name)
        seat = seats.get(episode_id)
        if seat is None:
            continue
        baseline({"select": None})
        candidate({"select": None})
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            record = step[seat] if seat < len(step) else {}
            obs = (record or {}).get("observation") or {}
            if (record or {}).get("status") != "ACTIVE" or not obs.get("select"):
                continue
            try:
                base_action = list(baseline(obs))
                cand_action = list(candidate(obs))
            except Exception:
                continue
            total += 1
            exact += int(base_action == cand_action)
            base_label = _label(obs, base_action)
            cand_label = _label(obs, cand_action)
            teacher_action = steps[step_index + 1][seat].get("action")
            teacher_label = None
            if isinstance(teacher_action, list):
                teacher_label = _label(obs, teacher_action)
                teacher_comparable += 1
                baseline_teacher += int(base_label == teacher_label)
                candidate_teacher += int(cand_label == teacher_label)
            semantic += int(base_label == cand_label)
            if base_label == cand_label:
                continue
            if teacher_label is not None:
                changed_teacher_comparable += 1
                changed_baseline_teacher += int(base_label == teacher_label)
                changed_candidate_teacher += int(cand_label == teacher_label)
            context = _context(obs)
            key = f"{context}: {base_label} -> {cand_label}"
            changed_pairs[key] += 1
            changed_contexts[str(context)] += 1
            if len(examples[key]) < 5:
                base_scores = _policy_scores(baseline_module, obs)
                cand_scores = _policy_scores(candidate_module, obs)
                ranked_scores = {}
                if base_scores is not None and cand_scores is not None:
                    option_labels = [
                        _label(obs, [index])[0]
                        for index in range(len(base_scores))
                    ]
                    ranked_scores = {
                        "baseline_top_scores": sorted(
                            zip(option_labels, base_scores),
                            key=lambda item: item[1],
                            reverse=True,
                        )[:8],
                        "candidate_top_scores": sorted(
                            zip(option_labels, cand_scores),
                            key=lambda item: item[1],
                            reverse=True,
                        )[:8],
                    }
                examples[key].append(
                    {
                        **_state_summary(obs, episode_id, seat),
                        "baseline": base_label,
                        "candidate": cand_label,
                        "teacher": teacher_label,
                        **ranked_scores,
                    }
                )

    top_pairs = changed_pairs.most_common(40)
    report = {
        "run_dir": str(run_dir),
        "baseline_dir": str(args.baseline_dir.resolve()),
        "candidate_dir": str(args.candidate_dir.resolve()),
        "replays_compared": len(replay_paths),
        "decisions": total,
        "exact_agreement": exact / total if total else 0.0,
        "semantic_agreement": semantic / total if total else 0.0,
        "semantic_changes": total - semantic,
        "teacher_comparable_decisions": teacher_comparable,
        "baseline_teacher_agreement": (
            baseline_teacher / teacher_comparable if teacher_comparable else None
        ),
        "candidate_teacher_agreement": (
            candidate_teacher / teacher_comparable if teacher_comparable else None
        ),
        "changed_teacher_comparable": changed_teacher_comparable,
        "changed_baseline_teacher_agreement": (
            changed_baseline_teacher / changed_teacher_comparable
            if changed_teacher_comparable
            else None
        ),
        "changed_candidate_teacher_agreement": (
            changed_candidate_teacher / changed_teacher_comparable
            if changed_teacher_comparable
            else None
        ),
        "changed_contexts": dict(changed_contexts.most_common()),
        "top_changed_pairs": top_pairs,
        "examples": {key: examples[key] for key, _ in top_pairs},
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
