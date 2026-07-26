"""Build and evaluate an interpretable Boss-target ranker from expert replays.

The full action ranker is a poor fit for this question: a replay contains only
the action that was taken, and most MAIN decisions are unrelated to target
choice.  Boss's Orders is different.  Its SWITCH sub-selection exposes every
legal benched target, so each play is a small ranking example with one selected
candidate and several legal alternatives.

This script extracts those ranking groups, measures the concrete
"Active single-prizer versus benched ex KO" opportunity, and fits a regularized
pairwise linear ranker.  The learned score is intended as a same-prize
tie-breaker behind hard legality/KO/prize-race gates, not as permission to
override them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from cg.api import all_attack, all_card_data  # noqa: E402

import analyze_alakazam_ladder_strategy as ladder  # noqa: E402


CARD = {int(card.cardId): card for card in all_card_data()}
ATTACK = {int(attack.attackId): attack for attack in all_attack()}

FEATURE_NAMES = [
    "prizes",
    "wins_game",
    "same_turn_ko",
    "is_ex",
    "is_mega_ex",
    "is_stage1",
    "is_stage2",
    "energy_count",
    "tool_count",
    "remaining_hp_100",
    "damage_taken_100",
    "retreat_gap",
    "attack_ready",
    "has_ability",
    "engine_ability",
    "protection_ability",
]


def _norm_text(value: Any) -> str:
    return str(value or "").replace("’", "'").lower()


def _meta(card: dict[str, Any] | None) -> Any:
    return CARD.get(int((card or {}).get("id", -1)))


def _prizes(card: dict[str, Any] | None) -> int:
    data = _meta(card)
    if data is None:
        return 1
    if bool(getattr(data, "megaEx", False)):
        return 3
    return 2 if bool(getattr(data, "ex", False)) else 1


def _energy_count(card: dict[str, Any] | None) -> int:
    return len((card or {}).get("energies") or [])


def _tool_count(card: dict[str, Any] | None) -> int:
    return len((card or {}).get("tools") or [])


def _attack_costs(data: Any) -> list[int]:
    costs = []
    for attack_id in getattr(data, "attacks", None) or []:
        attack = ATTACK.get(int(attack_id))
        if attack is not None:
            costs.append(len(getattr(attack, "energies", None) or []))
    return costs


def _candidate_features(
    card: dict[str, Any],
    *,
    own_prizes: int,
    available_damage: int,
) -> dict[str, float]:
    data = _meta(card)
    prizes = _prizes(card)
    hp = max(0, int(card.get("hp") or 0))
    max_hp = max(hp, int(card.get("maxHp") or hp))
    energy = _energy_count(card)
    skills = list(getattr(data, "skills", None) or []) if data is not None else []
    skill_text = " ".join(_norm_text(getattr(skill, "text", "")) for skill in skills)
    costs = _attack_costs(data) if data is not None else []
    min_cost = min(costs) if costs else 99
    engine_terms = (
        "draw ",
        "draws ",
        "search your deck",
        "damage counter",
        "attach ",
        "put into your hand",
    )
    protection_terms = ("prevent all effects", "prevent all damage", "can't use")
    return {
        "prizes": float(prizes),
        "wins_game": float(0 < own_prizes <= prizes and available_damage >= hp > 0),
        "same_turn_ko": float(available_damage >= hp > 0),
        "is_ex": float(bool(data is not None and getattr(data, "ex", False))),
        "is_mega_ex": float(bool(data is not None and getattr(data, "megaEx", False))),
        "is_stage1": float(bool(data is not None and getattr(data, "stage1", False))),
        "is_stage2": float(bool(data is not None and getattr(data, "stage2", False))),
        "energy_count": float(energy),
        "tool_count": float(_tool_count(card)),
        "remaining_hp_100": hp / 100.0,
        "damage_taken_100": max(0, max_hp - hp) / 100.0,
        "retreat_gap": float(
            max(0, int(getattr(data, "retreatCost", 0) or 0) - energy)
            if data is not None
            else 0
        ),
        "attack_ready": float(bool(costs and energy >= min_cost)),
        "has_ability": float(bool(skills)),
        "engine_ability": float(any(term in skill_text for term in engine_terms)),
        "protection_ability": float(any(term in skill_text for term in protection_terms)),
    }


def _available_damage(current: dict[str, Any], seat: int) -> int:
    players = current.get("players") or [{}, {}]
    me = players[seat]
    active = (ladder._cards(me, "active") or [None])[0]
    if not active or _energy_count(active) < 1:
        return 0
    if int(active.get("id", -1)) == ladder.ALAKAZAM:
        return 20 * int(me.get("handCount") or len(me.get("hand") or []))
    data = _meta(active)
    energy = _energy_count(active)
    best = 0
    for attack_id in getattr(data, "attacks", None) or []:
        attack = ATTACK.get(int(attack_id))
        if attack is None or len(getattr(attack, "energies", None) or []) > energy:
            continue
        best = max(best, int(getattr(attack, "damage", 0) or 0))
    return best


def _selected_index(
    steps: list[list[dict[str, Any]]], step_index: int, seat: int
) -> int | None:
    if step_index + 1 >= len(steps) or seat >= len(steps[step_index + 1]):
        return None
    action = steps[step_index + 1][seat].get("action")
    if not isinstance(action, list) or len(action) != 1:
        return None
    return int(action[0]) if isinstance(action[0], int) else None


def _teacher_boss_turns(
    replay: dict[str, Any], seat: int
) -> tuple[set[int], dict[int, int]]:
    turns: set[int] = set()
    targets: dict[int, int] = {}
    steps = replay.get("steps") or []
    for step_index in range(max(0, len(steps) - 1)):
        if seat >= len(steps[step_index]):
            continue
        state = steps[step_index][seat]
        if state.get("status") != "ACTIVE":
            continue
        observation = state.get("observation") or {}
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        selected, _ = ladder._selected_option(steps, step_index, seat)
        if selected is None:
            continue
        turn = int(current.get("turn", -1))
        if (
            int(select.get("context", -1)) == ladder.MAIN_SELECT_CONTEXT
            and int(selected.get("type", -1)) == ladder.PLAY
            and int((ladder._action_card(current, seat, selected) or {}).get("id", -1))
            == ladder.BOSS_ORDERS
        ):
            turns.add(turn)
        if ladder._effect_card_id(select) == ladder.BOSS_ORDERS:
            target = ladder._target_card(current, selected)
            if target is not None:
                targets[turn] = int(target.get("id", -1))
    return turns, targets


def _boss_target_groups(
    replay: dict[str, Any],
    seat: int,
    source_name: str,
) -> list[dict[str, Any]]:
    steps = replay.get("steps") or []
    episode_id = int((replay.get("info") or {}).get("EpisodeId", 0))
    rewards = replay.get("rewards") or [0, 0]
    reward = rewards[seat] if seat < len(rewards) else None
    won = bool(reward is not None and reward > 0)
    groups: list[dict[str, Any]] = []
    for step_index in range(max(0, len(steps) - 1)):
        if seat >= len(steps[step_index]):
            continue
        state = steps[step_index][seat]
        if state.get("status") != "ACTIVE":
            continue
        observation = state.get("observation") or {}
        select = observation.get("select") or {}
        if ladder._effect_card_id(select) != ladder.BOSS_ORDERS:
            continue
        current = observation.get("current") or {}
        players = current.get("players") or [{}, {}]
        if len(players) != 2:
            continue
        selected_index = _selected_index(steps, step_index, seat)
        options = list(select.get("option") or [])
        if selected_index is None or not 0 <= selected_index < len(options):
            continue
        damage = _available_damage(current, seat)
        own_prizes = len(players[seat].get("prize") or [])
        candidates = []
        for option_index, option in enumerate(options):
            target = ladder._target_card(current, option)
            if target is None:
                continue
            features = _candidate_features(
                target,
                own_prizes=own_prizes,
                available_damage=damage,
            )
            candidates.append(
                {
                    "option_index": option_index,
                    "card_id": int(target.get("id", -1)),
                    "hp": int(target.get("hp") or 0),
                    "selected": option_index == selected_index,
                    "features": features,
                }
            )
        if len(candidates) < 2 or sum(row["selected"] for row in candidates) != 1:
            continue
        groups.append(
            {
                "decision_id": f"{source_name}:{episode_id}:{step_index}",
                "source": source_name,
                "episode_id": episode_id,
                "turn": int(current.get("turn", -1)),
                "won": won,
                "available_damage": damage,
                "own_prizes": own_prizes,
                "candidates": candidates,
            }
        )
    return groups


def _ex_upgrade_opportunities(
    replay: dict[str, Any],
    seat: int,
    source_name: str,
) -> list[dict[str, Any]]:
    steps = replay.get("steps") or []
    episode_id = int((replay.get("info") or {}).get("EpisodeId", 0))
    boss_turns, target_by_turn = _teacher_boss_turns(replay, seat)
    rows_by_turn: dict[int, dict[str, Any]] = {}
    for step_index in range(max(0, len(steps) - 1)):
        if seat >= len(steps[step_index]):
            continue
        state = steps[step_index][seat]
        if state.get("status") != "ACTIVE":
            continue
        observation = state.get("observation") or {}
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        if (
            int(select.get("type", -1)) != ladder.MAIN_SELECT_TYPE
            or int(select.get("context", -1)) != ladder.MAIN_SELECT_CONTEXT
        ):
            continue
        players = current.get("players") or [{}, {}]
        if len(players) != 2:
            continue
        options = list(select.get("option") or [])
        legal_boss = any(
            int(option.get("type", -1)) == ladder.PLAY
            and int((ladder._action_card(current, seat, option) or {}).get("id", -1))
            == ladder.BOSS_ORDERS
            for option in options
        )
        offered_attack = any(
            int(option.get("type", -1)) == ladder.ATTACK for option in options
        )
        me, opponent = players[seat], players[1 - seat]
        attacker = (ladder._cards(me, "active") or [None])[0]
        active_target = (ladder._cards(opponent, "active") or [None])[0]
        if (
            not legal_boss
            or not offered_attack
            or not attacker
            or int(attacker.get("id", -1)) != ladder.ALAKAZAM
            or _energy_count(attacker) < 1
            or not active_target
        ):
            continue
        hand = int(me.get("handCount") or len(me.get("hand") or []))
        active_damage = 20 * hand
        boss_damage = 20 * max(0, hand - 1)
        active_prizes = _prizes(active_target)
        active_ko = active_damage >= int(active_target.get("hp") or 0) > 0
        own_prizes = len(me.get("prize") or [])
        if active_ko and active_prizes >= own_prizes > 0:
            continue
        upgrades = []
        for target in ladder._cards(opponent, "bench"):
            prizes = _prizes(target)
            hp = int(target.get("hp") or 0)
            if prizes >= 2 and boss_damage >= hp > 0 and (
                not active_ko or prizes > active_prizes
            ):
                upgrades.append(
                    {
                        "card_id": int(target.get("id", -1)),
                        "prizes": prizes,
                        "hp": hp,
                        "energy": _energy_count(target),
                    }
                )
        if not upgrades:
            continue
        turn = int(current.get("turn", -1))
        chosen_target_id = target_by_turn.get(turn)
        row = {
            "source": source_name,
            "episode_id": episode_id,
            "turn": turn,
            "active_id": int(active_target.get("id", -1)),
            "active_prizes": active_prizes,
            "active_hp": int(active_target.get("hp") or 0),
            "active_ko": active_ko,
            "boss_damage": boss_damage,
            "teacher_played_boss": turn in boss_turns,
            "teacher_target_id": chosen_target_id,
            "teacher_selected_upgrade": any(
                target["card_id"] == chosen_target_id for target in upgrades
            ),
            "upgrade_targets": upgrades,
        }
        old = rows_by_turn.get(turn)
        if old is None or len(upgrades) > len(old["upgrade_targets"]):
            rows_by_turn[turn] = row
    return list(rows_by_turn.values())


def _iter_source(path: Path) -> Iterable[dict[str, Any]]:
    return ladder._iter_directory(path) if path.is_dir() else ladder._iter_zip(path)


def _source_rows(
    path: Path,
    *,
    submission_id: int | None,
    team_name: str | None,
    source_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    groups = []
    upgrades = []
    counts = Counter()
    seats = ladder._submission_seats(path, submission_id) if submission_id else {}
    for replay in _iter_source(path):
        counts["replays_found"] += 1
        episode_id = int((replay.get("info") or {}).get("EpisodeId", 0))
        seat = seats.get(episode_id)
        if seat is None and team_name:
            seat = ladder._seat_for_team(replay, team_name)
        if seat is None:
            counts["unresolved"] += 1
            continue
        counts["replays_analyzed"] += 1
        groups.extend(_boss_target_groups(replay, seat, source_name))
        upgrades.extend(_ex_upgrade_opportunities(replay, seat, source_name))
    return groups, upgrades, dict(counts)


def _vector(candidate: dict[str, Any]) -> np.ndarray:
    features = candidate["features"]
    return np.asarray([float(features[name]) for name in FEATURE_NAMES], dtype=np.float64)


def _fit_pairwise(
    groups: list[dict[str, Any]],
    *,
    regularization: float = 0.08,
    iterations: int = 4000,
    learning_rate: float = 0.08,
) -> dict[str, Any]:
    diffs = []
    for group in groups:
        if group["available_damage"] <= 0:
            continue
        selected = next(row for row in group["candidates"] if row["selected"])
        selected_vector = _vector(selected)
        for candidate in group["candidates"]:
            if candidate["selected"]:
                continue
            diff = selected_vector - _vector(candidate)
            diffs.extend((diff, -diff))
    if not diffs:
        raise ValueError("no pairwise Boss-target examples")
    matrix = np.vstack(diffs)
    labels = np.tile(np.asarray([1.0, 0.0]), len(diffs) // 2)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = matrix / scale
    weights = np.zeros(normalized.shape[1], dtype=np.float64)
    for iteration in range(iterations):
        logits = np.clip(normalized @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = normalized.T @ (probabilities - labels) / len(labels)
        gradient += regularization * weights
        step = learning_rate / math.sqrt(1.0 + iteration / 400.0)
        weights -= step * gradient
    raw_weights = weights / scale
    logits = np.clip(matrix @ raw_weights, -30.0, 30.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    loss = float(
        np.mean(
            -labels * np.log(np.maximum(probabilities, 1e-12))
            - (1.0 - labels) * np.log(np.maximum(1.0 - probabilities, 1e-12))
        )
    )
    return {
        "feature_names": FEATURE_NAMES,
        "weights": raw_weights.tolist(),
        "regularization": regularization,
        "iterations": iterations,
        "pair_count": len(diffs) // 2,
        "training_logloss": loss,
    }


def _learned_score(candidate: dict[str, Any], model: dict[str, Any]) -> float:
    return float(
        np.dot(
            _vector(candidate),
            np.asarray(model["weights"], dtype=np.float64),
        )
    )


def _rule_score(candidate: dict[str, Any]) -> float:
    f = candidate["features"]
    return (
        f["wins_game"] * 1_000_000
        + f["same_turn_ko"] * 100_000
        + f["prizes"] * 20_000
        + f["protection_ability"] * 4_000
        + f["engine_ability"] * 2_500
        + f["attack_ready"] * 1_500
        + f["energy_count"] * 500
        + f["tool_count"] * 250
        + f["damage_taken_100"] * 100
    )


def _evaluate(
    groups: list[dict[str, Any]],
    model: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Counter[str]] = defaultdict(Counter)
    examples = []
    for group in groups:
        if group["available_damage"] <= 0:
            continue
        candidates = group["candidates"]
        selected = next(row for row in candidates if row["selected"])
        scopes = ["all"]
        if any(row["features"]["same_turn_ko"] for row in candidates):
            scopes.append("has_ko")
        if any(row["features"]["is_ex"] for row in candidates):
            scopes.append("has_ex")
        predictions = {
            "learned": max(
                candidates,
                key=lambda row: (
                    _learned_score(row, model),
                    -row["option_index"],
                ),
            ),
            "hard_prize_rule": max(
                candidates,
                key=lambda row: (_rule_score(row), -row["option_index"]),
            ),
        }
        for scope in scopes:
            for name, prediction in predictions.items():
                metrics[scope][f"{name}_correct"] += int(
                    prediction["option_index"] == selected["option_index"]
                )
            metrics[scope]["decisions"] += 1
        if (
            predictions["learned"]["option_index"] != selected["option_index"]
            or predictions["hard_prize_rule"]["option_index"] != selected["option_index"]
        ) and len(examples) < 20:
            examples.append(
                {
                    "decision_id": group["decision_id"],
                    "selected_card_id": selected["card_id"],
                    "learned_card_id": predictions["learned"]["card_id"],
                    "rule_card_id": predictions["hard_prize_rule"]["card_id"],
                    "candidates": [
                        {
                            "card_id": row["card_id"],
                            "selected": row["selected"],
                            "learned_score": _learned_score(row, model),
                            "rule_score": _rule_score(row),
                            **row["features"],
                        }
                        for row in candidates
                    ],
                }
            )
    rendered = {}
    for scope, values in metrics.items():
        decisions = int(values["decisions"])
        rendered[scope] = {
            "decisions": decisions,
            "learned_top1": (
                values["learned_correct"] / decisions if decisions else None
            ),
            "hard_prize_rule_top1": (
                values["hard_prize_rule_correct"] / decisions if decisions else None
            ),
        }
    return {"metrics": rendered, "examples": examples}


def _parse_source(value: str) -> tuple[Path, int | None, str | None, str]:
    parts = value.split("::")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "source must be PATH::SUBMISSION_ID or PATH::team=NAME[::LABEL]"
        )
    path = Path(parts[0])
    identity = parts[1]
    label = parts[2] if len(parts) == 3 else path.stem
    if identity.startswith("team="):
        return path, None, identity.removeprefix("team="), label
    try:
        return path, int(identity), None, label
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid source identity: {identity}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="PATH::SUBMISSION_ID or PATH::team=NAME[::LABEL]; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()

    groups = []
    upgrades = []
    source_counts = {}
    for raw_source in args.source:
        path, submission_id, team_name, label = _parse_source(raw_source)
        source_groups, source_upgrades, counts = _source_rows(
            path,
            submission_id=submission_id,
            team_name=team_name,
            source_name=label,
        )
        groups.extend(source_groups)
        upgrades.extend(source_upgrades)
        source_counts[label] = counts

    ordered = sorted(groups, key=lambda row: (row["source"], row["episode_id"]))
    cutoff = max(1, int(len(ordered) * 0.8))
    train_groups = ordered[:cutoff]
    test_groups = ordered[cutoff:]
    model = _fit_pairwise(train_groups)

    upgrade_by_source = {}
    for source, rows in sorted(
        (
            (source, [row for row in upgrades if row["source"] == source])
            for source in {row["source"] for row in upgrades}
        )
    ):
        upgrade_by_source[source] = {
            "opportunities": len(rows),
            "teacher_played_boss": sum(row["teacher_played_boss"] for row in rows),
            "teacher_played_boss_rate": (
                sum(row["teacher_played_boss"] for row in rows) / len(rows)
                if rows
                else None
            ),
            "teacher_selected_upgrade": sum(
                row["teacher_selected_upgrade"] for row in rows
            ),
            "teacher_selected_upgrade_rate": (
                sum(row["teacher_selected_upgrade"] for row in rows) / len(rows)
                if rows
                else None
            ),
        }

    report = {
        "sources": source_counts,
        "boss_target_decisions": len(groups),
        "boss_target_candidates": sum(len(group["candidates"]) for group in groups),
        "train_decisions": len(train_groups),
        "test_decisions": len(test_groups),
        "model": model,
        "train_evaluation": _evaluate(train_groups, model),
        "test_evaluation": _evaluate(test_groups, model),
        "evaluation_by_source": {
            source: _evaluate(
                [group for group in groups if group["source"] == source],
                model,
            )["metrics"]
            for source in sorted({group["source"] for group in groups})
        },
        "ex_upgrade_opportunities": {
            "total": len(upgrades),
            "teacher_played_boss": sum(row["teacher_played_boss"] for row in upgrades),
            "teacher_selected_upgrade": sum(
                row["teacher_selected_upgrade"] for row in upgrades
            ),
            "by_source": upgrade_by_source,
            "examples": upgrades[:40],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.model_output:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        args.model_output.write_text(
            json.dumps(
                {
                    "format": "pairwise_linear_target_ranker_v1",
                    "scope": "same-prize strategic tie-break only",
                    **model,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "boss_target_decisions": report["boss_target_decisions"],
                "test_evaluation": report["test_evaluation"]["metrics"],
                "ex_upgrade_opportunities": report["ex_upgrade_opportunities"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
