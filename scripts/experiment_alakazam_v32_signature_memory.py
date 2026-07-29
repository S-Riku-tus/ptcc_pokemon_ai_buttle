"""Evaluate leakage-free strategy memory over recurring legal-option signatures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _columns(names: list[str], wanted: tuple[str, ...]) -> np.ndarray:
    return np.asarray([names.index(name) for name in wanted], dtype=np.int64)


def _tuple(row: np.ndarray, columns: np.ndarray) -> tuple[int, ...]:
    return tuple(int(round(float(row[index]))) for index in columns)


def _bucket(value: float, width: int) -> int:
    return int(round(float(value))) // width


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("blend_scores", type=Path)
    parser.add_argument("blend_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as saved:
        features = saved["features"]
        labels = saved["labels"]
        groups = saved["groups"]
        splits = saved["splits"].astype(str)
        names = saved["feature_names"].astype(str).tolist()
    starts, ends = _ranges(groups)
    action_columns = _columns(
        names,
        (
            "option_type",
            "candidate_card_id",
            "candidate_attack_id",
            "candidate_target_id",
            "candidate_area",
            "candidate_inplay_area",
            "action_type",
        ),
    )
    card_columns = _columns(
        names,
        (
            "option_type",
            "candidate_card_id",
            "candidate_attack_id",
            "action_type",
        ),
    )
    family_column = names.index("action_type")
    state_columns = {
        name: names.index(name)
        for name in (
            "turn",
            "self_hand_count",
            "self_deck_count",
            "self_prize_count",
            "self_bench_count",
            "opp_hand_count",
            "opp_deck_count",
            "opp_prize_count",
            "opp_bench_count",
            "self_active_id",
            "opp_active_id",
            "supporter_played",
            "energy_attached",
            "current_powerful_hand_damage",
        )
    }

    def action_key(row: np.ndarray) -> tuple[int, ...]:
        return _tuple(row, action_columns)

    def option_keys(
        start: int,
        end: int,
        columns: np.ndarray,
    ) -> tuple[tuple[int, ...], ...]:
        return tuple(sorted(_tuple(features[row], columns) for row in range(start, end)))

    def family_signature(start: int, end: int) -> tuple[object, ...]:
        return (
            "family",
            tuple(sorted(
                int(round(float(features[row, family_column])))
                for row in range(start, end)
            )),
        )

    def card_signature(start: int, end: int) -> tuple[object, ...]:
        return ("cards", option_keys(start, end, card_columns))

    def option_signature(start: int, end: int) -> tuple[object, ...]:
        return ("options", option_keys(start, end, action_columns))

    def context(start: int, *, detailed: bool) -> tuple[int, ...]:
        row = features[start]
        core = (
            _bucket(row[state_columns["turn"]], 2),
            int(round(float(row[state_columns["self_active_id"]]))),
            int(round(float(row[state_columns["opp_active_id"]]))),
            int(round(float(row[state_columns["self_prize_count"]]))),
            int(round(float(row[state_columns["opp_prize_count"]]))),
            int(round(float(row[state_columns["supporter_played"]]))),
            int(round(float(row[state_columns["energy_attached"]]))),
        )
        if not detailed:
            return core
        return core + (
            _bucket(row[state_columns["self_hand_count"]], 2),
            _bucket(row[state_columns["self_deck_count"]], 5),
            int(round(float(row[state_columns["self_bench_count"]]))),
            _bucket(row[state_columns["opp_hand_count"]], 2),
            _bucket(row[state_columns["opp_deck_count"]], 5),
            int(round(float(row[state_columns["opp_bench_count"]]))),
            _bucket(
                row[state_columns["current_powerful_hand_damage"]],
                20,
            ),
        )

    signature_functions: dict[
        str,
        Callable[[int, int], tuple[object, ...]],
    ] = {
        "family": family_signature,
        "cards": card_signature,
        "options": option_signature,
        "cards_core_context": lambda start, end: (
            *card_signature(start, end),
            context(start, detailed=False),
        ),
        "options_core_context": lambda start, end: (
            *option_signature(start, end),
            context(start, detailed=False),
        ),
        "cards_detailed_context": lambda start, end: (
            *card_signature(start, end),
            context(start, detailed=True),
        ),
    }

    blend = json.loads(args.blend_report.read_text(encoding="utf-8"))
    model_order = list(blend["model_order"])
    weights = np.asarray(blend["selected_weights"], dtype=np.float32)
    with np.load(args.blend_scores, allow_pickle=False) as saved:
        validation_scores = sum(
            weight * saved[f"validation_{name}"]
            for name, weight in zip(model_order, weights)
        )
        test_scores = sum(
            weight * saved[f"test_{name}"]
            for name, weight in zip(model_order, weights)
        )
        validation_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]
        validation_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]

    split_decisions = {
        split: np.flatnonzero(splits == split)
        for split in ("train", "validation", "test")
    }
    score_by_split = {
        "validation": validation_scores,
        "test": test_scores,
    }
    label_by_split = {
        "validation": validation_labels,
        "test": test_labels,
    }
    group_by_split = {
        "validation": validation_groups,
        "test": test_groups,
    }

    def evaluate(
        split: str,
        memory: dict[tuple[object, ...], Counter[tuple[int, ...]]],
        signature: Callable[[int, int], tuple[object, ...]],
        *,
        minimum: int,
        confidence: float,
    ) -> dict[str, float | int]:
        scores = score_by_split[split]
        split_labels = label_by_split[split]
        split_groups = group_by_split[split]
        local_starts, local_ends = _ranges(split_groups)
        decisions = split_decisions[split]
        correct = memory_used = memory_correct = 0
        for local, decision in enumerate(decisions):
            source_start, source_end = starts[decision], ends[decision]
            local_start, local_end = local_starts[local], local_ends[local]
            base_top = int(
                np.argmax(scores[local_start:local_end])
            )
            selected_local = base_top
            counts = memory.get(signature(source_start, source_end))
            if counts:
                action, votes = counts.most_common(1)[0]
                total = sum(counts.values())
                matches = [
                    row - source_start
                    for row in range(source_start, source_end)
                    if action_key(features[row]) == action
                ]
                if (
                    total >= minimum
                    and votes / total >= confidence
                    and matches
                ):
                    selected_local = max(
                        matches,
                        key=lambda index: scores[local_start + index],
                    )
                    memory_used += 1
                    memory_correct += int(
                        split_labels[local_start + selected_local] == 1
                    )
            correct += int(
                split_labels[local_start + selected_local] == 1
            )
        count = len(decisions)
        return {
            "top1": correct / count,
            "memory_coverage": memory_used / count,
            "memory_covered_top1": (
                memory_correct / memory_used if memory_used else 0.0
            ),
            "memory_used": memory_used,
        }

    experiments = []
    memories = {}
    for name, signature in signature_functions.items():
        memory: dict[
            tuple[object, ...],
            Counter[tuple[int, ...]],
        ] = defaultdict(Counter)
        for decision in split_decisions["train"]:
            start, end = starts[decision], ends[decision]
            positive = int(np.flatnonzero(labels[start:end] == 1)[0])
            memory[signature(start, end)][
                action_key(features[start + positive])
            ] += 1
        memories[name] = memory
        for minimum in (1, 2, 3, 5, 10, 20):
            for confidence in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0):
                metrics = evaluate(
                    "validation",
                    memory,
                    signature,
                    minimum=minimum,
                    confidence=confidence,
                )
                experiments.append({
                    "signature": name,
                    "minimum": minimum,
                    "confidence": confidence,
                    **metrics,
                })
    selected = max(
        experiments,
        key=lambda row: (
            row["top1"],
            -row["memory_coverage"],
            row["minimum"],
            row["confidence"],
        ),
    )
    test = evaluate(
        "test",
        memories[str(selected["signature"])],
        signature_functions[str(selected["signature"])],
        minimum=int(selected["minimum"]),
        confidence=float(selected["confidence"]),
    )
    report = {
        "base_validation_top1": ensemble._accuracy(
            validation_scores,
            validation_labels,
            validation_groups.tolist(),
        ),
        "base_test_top1": ensemble._accuracy(
            test_scores,
            test_labels,
            test_groups.tolist(),
        ),
        "selected_on_validation": selected,
        "strict_test": test,
        "top_validation_experiments": sorted(
            experiments,
            key=lambda row: (
                -row["top1"],
                row["memory_coverage"],
            ),
        )[:20],
        "target_top1": 0.9,
        "target_met": test["top1"] >= 0.9,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
