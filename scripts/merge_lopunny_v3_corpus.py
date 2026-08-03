"""Merge the frozen v2 corpus with newly collected Majkel trajectories.

The old corpus is treated as historical training data.  Only the genuinely
new episode IDs are eligible for the v3 chronological validation/test split,
which prevents the v1/v2 holdouts from being reused as evidence for v3.
Candidate-level arrays are sliced with their decision groups so the merged
NPZ remains directly consumable by the Lopunny trainers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CANDIDATE_KEYS = {"features", "labels", "semantics"}
NAME_KEYS = {"feature_names", "count_feature_names"}


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cached:
        return {key: cached[key] for key in cached.files}


def _candidate_ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _take_decisions(
    arrays: dict[str, np.ndarray], decisions: np.ndarray
) -> dict[str, np.ndarray]:
    starts, ends = _candidate_ranges(arrays["groups"])
    candidate_rows = np.concatenate([
        np.arange(starts[index], ends[index], dtype=np.int64)
        for index in decisions
    ]) if len(decisions) else np.empty(0, dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for key, values in arrays.items():
        if key in NAME_KEYS:
            result[key] = values
        elif key in CANDIDATE_KEYS:
            result[key] = values[candidate_rows]
        else:
            result[key] = values[decisions]
    return result


def _check_schema(old: dict[str, np.ndarray], fresh: dict[str, np.ndarray]) -> None:
    if set(old) != set(fresh):
        raise ValueError(
            f"NPZ key mismatch: old-only={sorted(set(old) - set(fresh))}, "
            f"fresh-only={sorted(set(fresh) - set(old))}"
        )
    for key in NAME_KEYS:
        if not np.array_equal(old[key], fresh[key]):
            raise ValueError(f"Feature schema mismatch for {key}")
    for key in old:
        if key in NAME_KEYS:
            continue
        if old[key].ndim != fresh[key].ndim or old[key].shape[1:] != fresh[key].shape[1:]:
            raise ValueError(
                f"Shape mismatch for {key}: {old[key].shape} vs {fresh[key].shape}"
            )


def _concat(
    old: dict[str, np.ndarray], fresh: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    for key in old:
        if key in NAME_KEYS:
            merged[key] = old[key]
        else:
            merged[key] = np.concatenate([old[key], fresh[key]], axis=0)
    return merged


def _episode_counts(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    splits = arrays["splits"].astype(str)
    return {
        split: int(len(np.unique(arrays["episode_ids"][splits == split])))
        for split in ("train", "validation", "test")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation-games", type=int, default=80)
    parser.add_argument("--test-games", type=int, default=100)
    args = parser.parse_args()

    old = _load(args.old)
    fresh = _load(args.fresh)
    _check_schema(old, fresh)

    old_ids = set(map(int, np.unique(old["episode_ids"])))
    fresh_ids = sorted(map(int, np.unique(fresh["episode_ids"])))
    overlap = old_ids & set(fresh_ids)
    fresh_ids = [episode_id for episode_id in fresh_ids if episode_id not in overlap]
    required = args.validation_games + args.test_games
    if len(fresh_ids) <= required:
        raise ValueError(
            f"Need more than {required} genuinely new episodes, got {len(fresh_ids)}"
        )

    fresh_keep = np.flatnonzero(np.isin(fresh["episode_ids"], fresh_ids))
    fresh = _take_decisions(fresh, fresh_keep)
    validation_ids = set(fresh_ids[-required:-args.test_games])
    test_ids = set(fresh_ids[-args.test_games:])

    old["splits"] = np.full(len(old["groups"]), "train", dtype="U10")
    fresh["splits"] = np.asarray([
        "test" if int(episode_id) in test_ids
        else "validation" if int(episode_id) in validation_ids
        else "train"
        for episode_id in fresh["episode_ids"]
    ], dtype="U10")

    merged = _concat(old, fresh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)

    split_values = merged["splits"].astype(str)
    report: dict[str, Any] = {
        "old_corpus": str(args.old.resolve()),
        "fresh_corpus": str(args.fresh.resolve()),
        "output": str(args.output.resolve()),
        "overlap_episode_count_removed": len(overlap),
        "old_episode_count": len(old_ids),
        "genuinely_new_episode_count": len(fresh_ids),
        "total_episode_count": int(len(np.unique(merged["episode_ids"]))),
        "split_episodes": _episode_counts(merged),
        "split_decisions": {
            split: int(np.sum(split_values == split))
            for split in ("train", "validation", "test")
        },
        "split_policy": {
            "old": "all historical episodes assigned to train",
            "fresh": "ascending episode_id chronological split",
            "validation_games": args.validation_games,
            "test_games": args.test_games,
            "test_min_episode_id": min(test_ids),
            "test_max_episode_id": max(test_ids),
        },
        "candidate_rows": int(len(merged["labels"])),
        "decisions": int(len(merged["groups"])),
        "feature_count": int(len(merged["feature_names"])),
        "count_feature_count": int(len(merged["count_feature_names"])),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
