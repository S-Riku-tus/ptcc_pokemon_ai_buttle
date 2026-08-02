"""Distil a trained Grimmsnarl ranker into the standard-library runtime model.

The runtime cannot import LightGBM, so the booster is dumped to the compact
tree JSON that ``ml_runtime.tree_score`` walks. For a teacher-conditioned
model the chosen pilot's dense category code is baked in, because the runtime
has no team id to read off the observation - it *is* the pilot now.

The exported feature order is the booster's own, and the runtime builds its
vectors from that list by name, so a feature reordering cannot silently
misalign the two.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.distill import compact_booster  # noqa: E402


def team_codes(corpus_path: Path) -> dict[int, int]:
    """Same dense mapping the trainer builds: sorted team ids to 0..N-1."""
    data = np.load(corpus_path, allow_pickle=False)
    teams = sorted({int(x) for x in data["team_ids"]})
    return {team: index for index, team in enumerate(teams)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-team", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    booster = lgb.Booster(model_file=str(args.model))
    model = compact_booster(booster, kind="grimmsnarl_ranker")

    if "teacher_team_id" in model["feature_names"]:
        if args.teacher_team is None:
            raise SystemExit(
                "model is teacher-conditioned; pass --teacher-team"
            )
        codes = team_codes(args.corpus)
        if args.teacher_team not in codes:
            raise SystemExit(
                f"team {args.teacher_team} is not in the corpus"
            )
        model["teacher_team_id"] = args.teacher_team
        model["teacher_team_code"] = codes[args.teacher_team]
    elif args.teacher_team is not None:
        raise SystemExit(
            "model has no teacher_team_id feature; --teacher-team is unused"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(model, separators=(",", ":")), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "trees": len(model["trees"]),
        "features": len(model["feature_names"]),
        "teacher_team_id": model.get("teacher_team_id"),
        "teacher_team_code": model.get("teacher_team_code"),
        "bytes": args.output.stat().st_size,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
