"""Rewrite only the episode-based split labels of an extracted NPZ corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation-min-episode", type=int, required=True)
    parser.add_argument("--test-min-episode", type=int, required=True)
    args = parser.parse_args()
    if args.validation_min_episode >= args.test_min_episode:
        raise ValueError("validation boundary must precede test boundary")

    with np.load(args.source, allow_pickle=False) as cached:
        arrays = {name: cached[name] for name in cached.files}
    episode_ids = arrays["episode_ids"].astype(np.int64)
    splits = np.where(
        episode_ids >= args.test_min_episode,
        "test",
        np.where(
            episode_ids >= args.validation_min_episode,
            "validation",
            "train",
        ),
    )
    arrays["splits"] = splits

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "validation_min_episode": args.validation_min_episode,
        "test_min_episode": args.test_min_episode,
        "split_decisions": {
            split: int(np.count_nonzero(splits == split))
            for split in ("train", "validation", "test")
        },
        "split_episodes": {
            split: int(len(np.unique(episode_ids[splits == split])))
            for split in ("train", "validation", "test")
        },
        "episode_overlap": {
            left + "_" + right: int(len(
                set(episode_ids[splits == left].tolist())
                & set(episode_ids[splits == right].tolist())
            ))
            for left, right in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        },
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
