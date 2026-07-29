"""Measure exact held-out state recurrence without using held-out labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        labels = cached["labels"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        names = cached["feature_names"].astype(str).tolist()
    starts, ends = _ranges(groups)
    candidate_start = names.index("option_type")
    train = np.flatnonzero(splits == "train")

    for mode in ("state", "full", "rounded_full"):
        memory: dict[bytes, Counter[int]] = defaultdict(Counter)
        for decision in train:
            start, end = starts[decision], ends[decision]
            positive = int(np.flatnonzero(labels[start:end] == 1)[0])
            if mode == "state":
                key = features[start, :candidate_start].tobytes()
            elif mode == "full":
                key = features[start:end].tobytes()
            else:
                key = np.round(features[start:end], 2).tobytes()
            memory[key][positive] += 1
        for split in ("validation", "test"):
            indices = np.flatnonzero(splits == split)
            hits = correct = 0
            for decision in indices:
                start, end = starts[decision], ends[decision]
                positive = int(np.flatnonzero(labels[start:end] == 1)[0])
                if mode == "state":
                    key = features[start, :candidate_start].tobytes()
                elif mode == "full":
                    key = features[start:end].tobytes()
                else:
                    key = np.round(features[start:end], 2).tobytes()
                if key in memory:
                    hits += 1
                    correct += int(memory[key].most_common(1)[0][0] == positive)
            print({
                "mode": mode,
                "split": split,
                "decisions": len(indices),
                "coverage": hits / len(indices),
                "covered_accuracy": correct / max(hits, 1),
                "total_accuracy": correct / len(indices),
                "unique_train_keys": len(memory),
            })

    semantic_names = (
        "option_type",
        "candidate_card_id",
        "candidate_attack_id",
        "candidate_area",
        "candidate_inplay_area",
        "candidate_target_id",
        "candidate_target_hp",
        "candidate_target_energy",
        "action_type",
    )
    semantic_columns = [names.index(name) for name in semantic_names]
    noisy_prefixes = (
        "history_",
        "turn_self_log_",
        "turn_opp_log_",
        "recent_log_",
    )
    noisy_names = {
        "public_log_count",
        "current_turn_log_count",
        "psychic_hit_probability_draw2",
        "psychic_hit_probability_draw3",
    }
    canonical_state_columns = [
        index
        for index, name in enumerate(names[:candidate_start])
        if not name.startswith(noisy_prefixes) and name not in noisy_names
    ]

    def semantic(decision: int, local: int) -> bytes:
        start = starts[decision]
        return features[start + local, semantic_columns].tobytes()

    def canonical_key(decision: int) -> bytes:
        start, end = starts[decision], ends[decision]
        state = features[start, canonical_state_columns].tobytes()
        options = sorted(
            features[row, semantic_columns].tobytes()
            for row in range(start, end)
        )
        return state + b"".join(options)

    memory: dict[bytes, Counter[bytes]] = defaultdict(Counter)
    for decision in train:
        start, end = starts[decision], ends[decision]
        positive = int(np.flatnonzero(labels[start:end] == 1)[0])
        memory[canonical_key(decision)][semantic(decision, positive)] += 1
    repeated = sum(sum(counts.values()) for counts in memory.values() if sum(counts.values()) > 1)
    consistent = sum(
        counts.most_common(1)[0][1]
        for counts in memory.values()
        if sum(counts.values()) > 1
    )
    print({
        "mode": "canonical_strategy",
        "train_repeated_decisions": repeated,
        "train_repeated_consistency": consistent / max(repeated, 1),
        "unique_train_keys": len(memory),
    })
    for split in ("validation", "test"):
        indices = np.flatnonzero(splits == split)
        hits = correct = 0
        for decision in indices:
            start, end = starts[decision], ends[decision]
            positive = int(np.flatnonzero(labels[start:end] == 1)[0])
            key = canonical_key(decision)
            if key in memory:
                hits += 1
                correct += int(
                    memory[key].most_common(1)[0][0]
                    == semantic(decision, positive)
                )
        print({
            "mode": "canonical_strategy",
            "split": split,
            "coverage": hits / len(indices),
            "covered_accuracy": correct / max(hits, 1),
            "total_accuracy": correct / len(indices),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
