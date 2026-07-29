"""Rank-weighted teacher-forced imitation evaluation for Alakazam agents.

The historical imitation audit compared raw legal-option indexes.  That
penalises harmless choices such as selecting the second copy of the same card.
This evaluator reports three levels:

* exact: the recorded legal-option indexes match
* semantic: card/attack intent and in-play target state match
* intent: the same card or attack is selected, ignoring the target instance

Every observation is taken from the recorded expert trajectory, so a different
earlier candidate action cannot change the state being evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent


AREA_KEYS = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench"}
ACTION_NAMES = {
    0: "number",
    1: "type",
    2: "area",
    3: "card",
    7: "play",
    8: "energy",
    9: "evolve",
    10: "ability",
    12: "retreat",
    13: "attack",
    14: "end",
}


def _card_at(
    observation: dict[str, Any],
    player_index: int,
    area: int,
    index: int,
) -> dict[str, Any]:
    current = observation.get("current") or {}
    players = current.get("players") or []
    key = AREA_KEYS.get(area)
    if not 0 <= player_index < len(players) or key is None:
        return {}
    cards = players[player_index].get(key) or []
    if not 0 <= index < len(cards) or not isinstance(cards[index], dict):
        return {}
    return cards[index]


def _source_card(
    observation: dict[str, Any], option: dict[str, Any]
) -> dict[str, Any]:
    current = observation.get("current") or {}
    player_index = int(option.get("playerIndex", current.get("yourIndex", 0)))
    option_type = int(option.get("type", -1))
    area = int(option.get("area", -1))
    if option_type in (7, 8, 9):
        area = 2
    index = option.get("index")
    if not isinstance(index, int):
        return {}
    return _card_at(observation, player_index, area, index)


def _target_card(
    observation: dict[str, Any], option: dict[str, Any]
) -> dict[str, Any]:
    current = observation.get("current") or {}
    player_index = int(option.get("playerIndex", current.get("yourIndex", 0)))
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    if not isinstance(area, int) or not isinstance(index, int):
        return {}
    return _card_at(observation, player_index, area, index)


def _energy_count(card: dict[str, Any]) -> int:
    return len(card.get("energyCards") or card.get("energies") or [])


def _card_state(card: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Instance-independent state for strategically interchangeable targets."""
    return (
        int(card.get("id", -1)),
        int(card.get("hp", -1)),
        int(card.get("maxHp", -1)),
        _energy_count(card),
        len(card.get("tools") or []),
    )


def _atom(
    observation: dict[str, Any],
    selected: int,
    *,
    level: str,
) -> tuple[Any, ...]:
    options = (observation.get("select") or {}).get("option") or []
    if not isinstance(selected, int) or not 0 <= selected < len(options):
        return ("OUT",)
    option = options[selected]
    option_type = int(option.get("type", -1))
    source = _source_card(observation, option)
    attack_id = int(option.get("attackId", -1))
    base: tuple[Any, ...] = (
        option_type,
        int(source.get("id", -1)),
        attack_id,
    )
    if level == "intent":
        return base

    target = _target_card(observation, option)
    if target:
        return base + _card_state(target)
    # Ability and switch/promotion choices encode their in-play card in the
    # option's ordinary area/index fields rather than inPlayArea/inPlayIndex.
    if option_type in (3, 10, 12) and source:
        return base + _card_state(source)
    return base


def _label(
    observation: dict[str, Any], action: list[int], *, level: str
) -> tuple[tuple[Any, ...], ...]:
    atoms = [_atom(observation, index, level=level) for index in action]
    # Multi-select card order has no game meaning; compare it as a multiset.
    if len(atoms) > 1:
        atoms.sort(key=repr)
    return tuple(atoms)


def _display(label: tuple[tuple[Any, ...], ...]) -> str:
    parts = []
    for atom in label:
        if atom == ("OUT",):
            parts.append("OUT")
            continue
        option_type, card_id, attack_id = atom[:3]
        name = ACTION_NAMES.get(int(option_type), str(option_type))
        if int(card_id) >= 0:
            name += f":card:{card_id}"
        if int(attack_id) >= 0:
            name += f":attack:{attack_id}"
        if len(atom) > 3:
            name += f":target:{atom[3:]}"
        parts.append(name)
    return "+".join(parts) or "empty"


def _merge_counter(target: Counter[str], source: dict[str, int]) -> None:
    target.update({str(key): int(value) for key, value in source.items()})


def _evaluate_chunk(
    run_root: str,
    agent_dir: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    root = Path(run_root)
    agent, _, main_module = load_dir_agent(Path(agent_dir))
    runtime = getattr(main_module, "_RUNTIME", None)
    model = getattr(runtime, "model", None)
    # v29's residual model needs fallback and legacy scores constructed inside
    # its runtime.  The evaluator's simple raw-ranker counterfactual only
    # applies to pre-v29 observation/option-only models.
    if isinstance(model, dict) and "fallback_selected" in model.get(
        "feature_names", []
    ):
        model = None
    runtime_module = sys.modules.get(getattr(runtime.__class__, "__module__", ""))
    features_module = sys.modules.get("ml_features")
    counts: Counter[str] = Counter()
    by_rank: dict[str, Counter[str]] = defaultdict(Counter)
    by_context: dict[str, Counter[str]] = defaultdict(Counter)
    mismatch_pairs: Counter[str] = Counter()

    for row in rows:
        agent({"select": None})
        replay = json.loads((root / row["replay_path"]).read_text(encoding="utf-8"))
        seat = int(row["seat_index"])
        rank = int(row["leaderboard_rank"])
        weight = 1.0 / math.sqrt(max(1, rank))
        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step):
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
            select = observation.get("select")
            if select is None or record.get("status") != "ACTIVE":
                continue
            if seat >= len(steps[step_index + 1]):
                continue
            recorded = (steps[step_index + 1][seat] or {}).get("action")
            if not isinstance(recorded, list) or len(recorded) == 60:
                continue
            try:
                predicted = list(agent(observation))
            except Exception:
                counts["agent_errors"] += 1
                continue

            options = select.get("option") or []
            forced = len(options) <= 1
            exact = predicted == recorded
            teacher_semantic = _label(observation, recorded, level="semantic")
            local_semantic = _label(observation, predicted, level="semantic")
            semantic = teacher_semantic == local_semantic
            teacher_intent = _label(observation, recorded, level="intent")
            local_intent = _label(observation, predicted, level="intent")
            intent = teacher_intent == local_intent
            context = str(select.get("context", -1))

            metrics = {
                "decisions": 1,
                "exact": int(exact),
                "semantic": int(semantic),
                "intent": int(intent),
                "nontrivial": int(not forced),
                "nontrivial_exact": int(not forced and exact),
                "nontrivial_semantic": int(not forced and semantic),
                "nontrivial_intent": int(not forced and intent),
            }
            for key, value in metrics.items():
                counts[key] += value
                by_rank[str(rank)][key] += value
                by_context[context][key] += value
            for key in ("decisions", "exact", "semantic", "intent"):
                counts[f"weighted_{key}"] += weight * metrics[key]
                by_rank[str(rank)][f"weighted_{key}"] += weight * metrics[key]
                by_context[context][f"weighted_{key}"] += weight * metrics[key]
            if not semantic:
                pair = (
                    f"{_display(teacher_semantic)} -> "
                    f"{_display(local_semantic)}"
                )
                mismatch_pairs[pair] += 1

            is_ranker_scope = (
                isinstance(model, dict)
                and runtime_module is not None
                and features_module is not None
                and int(select.get("type", -1)) == 0
                and int(select.get("context", -1)) == 0
                and int(select.get("minCount") or 0) == 1
                and int(select.get("maxCount") or 0) == 1
                and len(options) >= 2
                and len(recorded) == 1
            )
            if not is_ranker_scope:
                continue
            try:
                current = observation.get("current") or {}
                action_map = {
                    str(key): int(value)
                    for key, value in (model.get("action_type_map") or {}).items()
                }
                rows = []
                for option in options:
                    feature = features_module.option_features(current, select, option)
                    action_type = str(feature.get("action_type") or "other")
                    feature["action_type"] = action_map.get(action_type, -1)
                    rows.append([
                        float(feature.get(name, -1))
                        for name in model["feature_names"]
                    ])
                scores = [
                    float(runtime_module._tree_score(row, model))
                    for row in rows
                ]
                order = sorted(
                    range(len(scores)),
                    key=lambda index: scores[index],
                    reverse=True,
                )
                probabilities = runtime_module._probabilities(
                    scores, float(model.get("temperature", 1.0))
                )
                confidence = float(probabilities[order[0]])
                second = float(probabilities[order[1]]) if len(order) > 1 else 0.0
                margin = confidence - second
                teacher_key = teacher_semantic
                ranked_keys = [
                    _label(observation, [index], level="semantic")
                    for index in order
                ]
                model_correct = ranked_keys[0] == teacher_key
                counts["ranker_decisions"] += 1
                counts["ranker_semantic_top1"] += int(model_correct)
                counts["ranker_semantic_top2"] += int(
                    teacher_key in ranked_keys[:2]
                )
                counts["ranker_semantic_top3"] += int(
                    teacher_key in ranked_keys[:3]
                )
                counts["ranker_fallback_oracle"] += int(
                    model_correct or semantic
                )
                for probability_threshold in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                    for margin_threshold in (0.0, 0.05, 0.1, 0.15, 0.2):
                        gate = (
                            confidence >= probability_threshold
                            and margin >= margin_threshold
                        )
                        key = (
                            f"ranker_gate_p{probability_threshold:.2f}"
                            f"_m{margin_threshold:.2f}"
                        )
                        counts[f"{key}_selected"] += int(gate)
                        counts[f"{key}_correct"] += int(
                            model_correct if gate else semantic
                        )
            except Exception:
                counts["ranker_errors"] += 1

    return {
        "counts": dict(counts),
        "by_rank": {key: dict(value) for key, value in by_rank.items()},
        "by_context": {key: dict(value) for key, value in by_context.items()},
        "mismatch_pairs": dict(mismatch_pairs),
    }


def _rates(counter: Counter[str]) -> dict[str, int | float | None]:
    decisions = counter["decisions"]
    nontrivial = counter["nontrivial"]
    weighted = float(counter["weighted_decisions"])
    return {
        "decisions": int(decisions),
        "exact": counter["exact"] / decisions if decisions else None,
        "semantic": counter["semantic"] / decisions if decisions else None,
        "intent": counter["intent"] / decisions if decisions else None,
        "nontrivial_decisions": int(nontrivial),
        "nontrivial_exact": (
            counter["nontrivial_exact"] / nontrivial if nontrivial else None
        ),
        "nontrivial_semantic": (
            counter["nontrivial_semantic"] / nontrivial if nontrivial else None
        ),
        "nontrivial_intent": (
            counter["nontrivial_intent"] / nontrivial if nontrivial else None
        ),
        "rank_weighted_exact": (
            counter["weighted_exact"] / weighted if weighted else None
        ),
        "rank_weighted_semantic": (
            counter["weighted_semantic"] / weighted if weighted else None
        ),
        "rank_weighted_intent": (
            counter["weighted_intent"] / weighted if weighted else None
        ),
        "agent_errors": int(counter["agent_errors"]),
    }


def _chunks(rows: list[dict[str, str]], count: int) -> Iterable[list[dict[str, str]]]:
    size = max(1, math.ceil(len(rows) / max(1, count)))
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--ranks", default="2,3,5,6,8")
    parser.add_argument("--deck-hash", default="cc38cb450b86770a")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
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

    aggregate: Counter[str] = Counter()
    by_rank: dict[str, Counter[str]] = defaultdict(Counter)
    by_context: dict[str, Counter[str]] = defaultdict(Counter)
    mismatch_pairs: Counter[str] = Counter()
    jobs = list(_chunks(rows, min(max(1, args.workers), len(rows) or 1)))
    if len(jobs) == 1:
        results = [
            _evaluate_chunk(
                str(args.run_root.resolve()),
                str(args.agent_dir.resolve()),
                jobs[0],
            )
        ]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
            results = list(
                executor.map(
                    _evaluate_chunk,
                    [str(args.run_root.resolve())] * len(jobs),
                    [str(args.agent_dir.resolve())] * len(jobs),
                    jobs,
                )
            )
    for result in results:
        for key, value in result["counts"].items():
            aggregate[key] += value
        for rank, values in result["by_rank"].items():
            for key, value in values.items():
                by_rank[rank][key] += value
        for context, values in result["by_context"].items():
            for key, value in values.items():
                by_context[context][key] += value
        _merge_counter(mismatch_pairs, result["mismatch_pairs"])

    report = {
        "run_root": str(args.run_root.resolve()),
        "agent_dir": str(args.agent_dir.resolve()),
        "deck_hash": args.deck_hash,
        "ranks": sorted(ranks),
        "games": len(rows),
        "metrics": _rates(aggregate),
        "by_rank": {
            rank: _rates(by_rank[rank])
            for rank in sorted(by_rank, key=int)
        },
        "by_context": {
            context: _rates(by_context[context])
            for context in sorted(by_context, key=int)
        },
        "top_semantic_mismatches": mismatch_pairs.most_common(100),
        "semantic_definition": (
            "same option type/card/attack and same target card state; "
            "interchangeable copies and multi-select order are collapsed"
        ),
        "intent_definition": (
            "same option type/card/attack; target instance is ignored"
        ),
        "rank_weight": "1/sqrt(leaderboard_rank)",
    }
    ranker_decisions = int(aggregate["ranker_decisions"])
    if ranker_decisions:
        gates = []
        for key, correct in aggregate.items():
            if not key.startswith("ranker_gate_") or not key.endswith("_correct"):
                continue
            stem = key.removesuffix("_correct")
            gates.append({
                "gate": stem.removeprefix("ranker_gate_"),
                "semantic": correct / ranker_decisions,
                "model_selection_rate": (
                    aggregate[f"{stem}_selected"] / ranker_decisions
                ),
            })
        gates.sort(
            key=lambda item: (item["semantic"], -item["model_selection_rate"]),
            reverse=True,
        )
        report["raw_ranker"] = {
            "decisions": ranker_decisions,
            "semantic_top1": (
                aggregate["ranker_semantic_top1"] / ranker_decisions
            ),
            "semantic_top2": (
                aggregate["ranker_semantic_top2"] / ranker_decisions
            ),
            "semantic_top3": (
                aggregate["ranker_semantic_top3"] / ranker_decisions
            ),
            "fallback_or_ranker_oracle": (
                aggregate["ranker_fallback_oracle"] / ranker_decisions
            ),
            "errors": int(aggregate["ranker_errors"]),
            "best_confidence_gates": gates[:15],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "games": report["games"],
        **report["metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
