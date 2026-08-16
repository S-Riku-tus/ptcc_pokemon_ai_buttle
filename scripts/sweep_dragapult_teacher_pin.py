"""Which teacher identity should the deployed model be pinned to?

The model is teacher-conditioned: a categorical column carries whose policy to
reproduce, and the exported agent hard-codes one value.  v1 and v2 both pinned
the highest-rated pilot without testing the choice.  That pilot may be the one
the model reproduces worst, and its idiosyncrasies are then baked into every
game we play.

This sweeps every identity in the corpus.  For each candidate pin it overwrites
the team column on *all* held-out rows and measures Top-1 against what the
teachers actually did - pooled, and broken down by which pilot generated the
decision.  A pin that agrees with the whole cohort is a safer deployment than
one that agrees only with itself.

Usage:
  python scripts/sweep_dragapult_teacher_pin.py \
      --model data/ml/dragapult_v2/ranker_full.txt \
      --corpus data/ml/dragapult_v2/corpus_full.npz \
      --split test \
      --report experiments/dragapult_ml_v2/teacher_pin_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--split-report", type=Path,
        help=("Training report whose split_boundaries define the per-team "
              "split.  The corpus file stores the pre-resplit global cut, so "
              "without this the held-out block is a different one."),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    data = np.load(args.corpus, allow_pickle=False)
    names = [str(value) for value in data["feature_names"]]
    # The trainer appends teacher_team_id per split rather than widening the
    # base array, so the column is in the model but never in the corpus file.
    if "teacher_team_id" in names:
        raise SystemExit("unexpected: corpus already carries the team column")

    groups = data["groups"].astype(np.int64)
    offsets = np.r_[0, np.cumsum(groups)[:-1]].astype(np.int64)
    splits = data["splits"]
    team_ids = data["team_ids"].astype(np.int64)
    contexts = data["contexts"].astype(np.int64)
    labels = data["labels"]
    features = data["features"]

    if args.split_report:
        boundaries = json.loads(
            args.split_report.read_text(encoding="utf-8")
        )["split_boundaries"]
        episode_ids = data["episode_ids"].astype(np.int64)
        assigned = np.empty(len(episode_ids), dtype="<U10")
        for team, (validation_min, test_min) in boundaries.items():
            mask = team_ids == int(team)
            assigned[mask] = np.where(
                episode_ids[mask] >= int(test_min), "test",
                np.where(episode_ids[mask] >= int(validation_min),
                         "validation", "train"),
            )
        splits = assigned

    block = np.flatnonzero(splits == args.split)
    if not len(block):
        raise SystemExit(f"no decisions in split {args.split}")

    teams = sorted({int(value) for value in team_ids})
    codes = {team: index for index, team in enumerate(teams)}
    booster = lgb.Booster(model_file=str(args.model))
    best = booster.best_iteration or 0

    # One contiguous matrix of every candidate row in the split, plus an index
    # from decision to its slice, so each pin is a single predict() call.
    row_index: list[np.ndarray] = []
    truth: list[int] = []
    for decision in block:
        start, size = offsets[decision], groups[decision]
        rows = np.arange(start, start + size)
        row_index.append(rows)
        truth.append(int(np.argmax(labels[rows])))
    flat = np.concatenate(row_index)
    matrix = np.empty((len(flat), features.shape[1] + 1), dtype=np.float32)
    matrix[:, :features.shape[1]] = features[flat]
    team_column = features.shape[1]
    starts = np.r_[0, np.cumsum([len(rows) for rows in row_index])[:-1]]
    sizes = np.array([len(rows) for rows in row_index])
    truth_array = np.array(truth)
    decision_teams = team_ids[block]
    decision_contexts = contexts[block]

    results = []
    for team in teams:
        matrix[:, team_column] = float(codes[team])
        scores = booster.predict(matrix, num_iteration=best)
        picked = np.array([
            int(np.argmax(scores[start:start + size]))
            for start, size in zip(starts, sizes)
        ])
        correct = picked == truth_array
        by_team = {
            str(other): round(float(correct[decision_teams == other].mean()), 4)
            for other in teams
        }
        main = decision_contexts == 0
        results.append({
            "pin": team,
            "pooled_top1": round(float(correct.mean()), 4),
            "self_top1": by_team[str(team)],
            "others_top1": round(float(
                correct[decision_teams != team].mean()), 4),
            "main_top1": round(float(correct[main].mean()), 4),
            "by_team": by_team,
        })
        print(f"pin {team}  pooled {results[-1]['pooled_top1']:.4f}  "
              f"self {results[-1]['self_top1']:.4f}  "
              f"others {results[-1]['others_top1']:.4f}  "
              f"MAIN {results[-1]['main_top1']:.4f}")

    results.sort(key=lambda row: -row["pooled_top1"])
    print(f"\nbest pooled: {results[0]['pin']} at {results[0]['pooled_top1']}")
    print(f"worst pooled: {results[-1]['pin']} at {results[-1]['pooled_top1']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"split": args.split, "decisions": int(len(block)),
                        "results": results}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
