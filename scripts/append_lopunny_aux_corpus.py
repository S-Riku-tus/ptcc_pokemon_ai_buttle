"""Append lower-trust exact-deck trajectories as training-only auxiliaries.

The base corpus's train/validation/test assignments are preserved byte-for-
byte.  Auxiliary episode IDs are remapped to unique negative values so the
existing recency weighting treats them as older/lower-weight evidence and so
same-episode opposite-seat trajectories remain distinct rather than leaking
into the Majkel validation/test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.merge_lopunny_v3_corpus import CANDIDATE_KEYS, NAME_KEYS, _check_schema


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cached:
        return {key: cached[key] for key in cached.files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--aux", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    base = _load(args.base)
    sources: list[dict[str, Any]] = []
    next_episode = -1
    for path in args.aux:
        aux = _load(path)
        _check_schema(base, aux)
        original_ids = np.unique(aux["episode_ids"])
        mapping = {
            int(episode_id): next_episode - index
            for index, episode_id in enumerate(original_ids)
        }
        next_episode -= len(original_ids)
        aux["episode_ids"] = np.asarray([
            mapping[int(episode_id)] for episode_id in aux["episode_ids"]
        ], dtype=np.int64)
        aux["splits"] = np.full(len(aux["groups"]), "train", dtype="U10")
        for key in base:
            if key in NAME_KEYS:
                continue
            base[key] = np.concatenate((base[key], aux[key]), axis=0)
        sources.append({
            "path": str(path.resolve()),
            "episodes": int(len(original_ids)),
            "decisions": int(len(aux["groups"])),
            "candidate_rows": int(len(aux["labels"])),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **base)
    split_values = base["splits"].astype(str)
    report = {
        "base": str(args.base.resolve()),
        "auxiliary_sources": sources,
        "output": str(args.output.resolve()),
        "auxiliary_policy": "train only; unique negative episode IDs",
        "split_decisions": {
            split: int(np.sum(split_values == split))
            for split in ("train", "validation", "test")
        },
        "decisions": int(len(base["groups"])),
        "candidate_rows": int(len(base["labels"])),
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
