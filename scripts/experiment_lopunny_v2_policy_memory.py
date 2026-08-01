"""Leakage-free canonical policy-memory coverage study for Lopunny v2.

Only train decisions populate memory.  A validation key is answered when all
train examples with that key agree strongly enough on the same semantic
action.  Several progressively coarser public-state schemas are compared;
episode ids, split labels, rewards, and future actions never enter a key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _semantic_action(
    semantics: np.ndarray,
    labels: np.ndarray,
    start: int,
    end: int,
) -> tuple[tuple[int | float, ...], ...]:
    chosen = np.flatnonzero(labels[start:end] == 1)
    return tuple(sorted(
        tuple(value.item() for value in semantics[start + local])
        for local in chosen
    ))


def _legal_signature(
    semantics: np.ndarray,
    start: int,
    end: int,
) -> tuple[tuple[int | float, ...], ...]:
    return tuple(sorted(set(
        tuple(value.item() for value in row)
        for row in semantics[start:end]
    )))


def _columns(names: list[str], schema: str) -> np.ndarray:
    if schema == "full":
        return np.arange(len(names), dtype=np.int64)

    always = {
        "turn", "turn_action_count", "first_player_is_self",
        "energy_attached", "retreated", "stadium_played",
        "supporter_played", "self_hand_count", "self_deck_count",
        "self_prize_count", "self_bench_count", "opp_hand_count",
        "opp_prize_count", "opp_bench_count", "self_active_id",
        "self_active_hp", "self_active_energy", "self_active_special_energy",
        "self_active_tool_count", "opp_active_id", "opp_active_hp",
        "opp_active_energy", "opp_active_special_energy",
        "select_type", "select_context", "select_min_count",
        "select_max_count", "legal_option_count",
    }
    prefixes = (
        "hand_", "field_", "discard_", "offered_option_type_",
        "offered_card_", "turn_log_card_",
    )
    board_tokens = (
        "self_bench_slot_", "opp_bench_slot_",
    )
    selected = []
    for index, name in enumerate(names):
        keep = name in always or name.startswith(prefixes)
        if schema in {"board", "sequence"}:
            keep = keep or name.startswith(board_tokens)
        if schema == "sequence":
            keep = keep or name.startswith("recent_log_")
            keep = keep or name.startswith("turn_self_log_type_")
            keep = keep or name.startswith("turn_opp_log_type_")
        if schema == "abstract":
            keep = (
                name in always
                or name.startswith((
                    "hand_", "field_", "offered_option_type_",
                    "offered_card_", "turn_log_card_",
                ))
            )
        if keep:
            selected.append(index)
    return np.asarray(selected, dtype=np.int64)


def _quantized_row(
    row: np.ndarray,
    names: list[str],
    columns: np.ndarray,
    turn_cap: int,
    hp_bucket: int,
) -> tuple[int, ...]:
    result = []
    for index in columns:
        value = float(row[index])
        name = names[index]
        if name == "turn":
            value = min(value, turn_cap)
        elif hp_bucket > 1 and (
            name.endswith("_hp") or name.endswith("_damage")
            or "_hp_" in name or "_damage_" in name
        ):
            value = round(value / hp_bucket)
        result.append(int(round(value)))
    return tuple(result)


def _digest(parts: Iterable[object]) -> str:
    return hashlib.blake2b(repr(tuple(parts)).encode(), digest_size=12).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files}
    names = arrays["count_feature_names"].astype(str).tolist()
    groups = arrays["groups"]
    starts, ends = _ranges(groups)
    splits = arrays["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    # Memory is safest and most useful for the single-choice MAIN sequence.
    eligible = (
        (arrays["select_contexts"] == 0)
        & (arrays["forced"] == 0)
        & (arrays["chosen_counts"] == 1)
    )
    train = train[eligible[train]]
    validation = validation[eligible[validation]]

    actions = {
        int(decision): _semantic_action(
            arrays["semantics"], arrays["labels"],
            int(starts[decision]), int(ends[decision]),
        )
        for decision in np.concatenate([train, validation])
    }
    legal = {
        int(decision): _legal_signature(
            arrays["semantics"], int(starts[decision]), int(ends[decision])
        )
        for decision in np.concatenate([train, validation])
    }

    experiments = []
    for schema in ("full", "sequence", "board", "abstract"):
        columns = _columns(names, schema)
        for include_legal in (False, True):
            for hp_bucket in (1, 20, 40):
                groups_by_key: dict[str, Counter[tuple]] = defaultdict(Counter)
                for decision in train:
                    row = _quantized_row(
                        arrays["count_features"][decision], names, columns,
                        turn_cap=20, hp_bucket=hp_bucket,
                    )
                    key = _digest((row, legal[int(decision)] if include_legal else ()))
                    groups_by_key[key][actions[int(decision)]] += 1
                for minimum_support in (1, 2, 3, 5):
                    for minimum_purity in (0.80, 0.90, 1.0):
                        memory = {}
                        for key, counts in groups_by_key.items():
                            support = sum(counts.values())
                            action, frequency = counts.most_common(1)[0]
                            if (
                                support >= minimum_support
                                and frequency / support >= minimum_purity
                            ):
                                memory[key] = action
                        answered = correct = resolved = 0
                        for decision in validation:
                            row = _quantized_row(
                                arrays["count_features"][decision], names, columns,
                                turn_cap=20, hp_bucket=hp_bucket,
                            )
                            key = _digest((
                                row, legal[int(decision)] if include_legal else ()
                            ))
                            predicted = memory.get(key)
                            if predicted is None:
                                continue
                            answered += 1
                            available = Counter(legal[int(decision)])
                            can_resolve = all(
                                available[item] >= count
                                for item, count in Counter(predicted).items()
                            )
                            resolved += int(can_resolve)
                            correct += int(
                                can_resolve and predicted == actions[int(decision)]
                            )
                        experiments.append({
                            "schema": schema,
                            "columns": int(len(columns)),
                            "include_legal_signature": include_legal,
                            "hp_bucket": hp_bucket,
                            "minimum_support": minimum_support,
                            "minimum_purity": minimum_purity,
                            "memory_keys": len(memory),
                            "validation_decisions": int(len(validation)),
                            "answered": answered,
                            "coverage": answered / max(1, len(validation)),
                            "resolved_rate": resolved / max(1, answered),
                            "accuracy_when_answered": correct / max(1, answered),
                            "correct_decisions": correct,
                        })
    best = sorted(
        experiments,
        key=lambda row: (
            row["correct_decisions"],
            row["accuracy_when_answered"],
            row["coverage"],
        ),
        reverse=True,
    )[:20]
    report = {
        "cache": str(args.cache.resolve()),
        "train_decisions": int(len(train)),
        "validation_decisions": int(len(validation)),
        "test_read": False,
        "selection_note": (
            "Rank by correct validation decisions; deployment still requires "
            "an override comparison against base errors."
        ),
        "best": best,
        "experiments": experiments,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(best[:10], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
