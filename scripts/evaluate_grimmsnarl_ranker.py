"""Evaluate a trained Grimmsnarl ranker on a frozen corpus split.

This intentionally reuses the corpus, split and metric implementation from
``train_grimmsnarl_v2_teacher.py``.  It makes old/new model comparisons honest:
both models see the exact same decisions without retraining either model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from train_grimmsnarl_v2_teacher import (
    Corpus,
    error_taxonomy,
    select_decisions,
    top1,
    topk,
    wilson,
)


def grouped_rates(
    values: np.ndarray, correct: np.ndarray
) -> dict[str, dict[str, float | int]]:
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for value, hit in zip(values, correct):
        counts[int(value)][0] += 1
        counts[int(value)][1] += int(hit)
    return {
        str(value): {
            "decisions": total,
            "top1": round(hits / total, 4),
        }
        for value, (total, hits) in sorted(counts.items())
    }


def episode_bootstrap_delta(
    episode_ids: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    """Cluster-bootstrap the Top-1 delta, keeping each game intact."""
    episodes, inverse = np.unique(episode_ids, return_inverse=True)
    candidate_hits = np.bincount(
        inverse, weights=candidate.astype(np.int8), minlength=len(episodes)
    )
    baseline_hits = np.bincount(
        inverse, weights=baseline.astype(np.int8), minlength=len(episodes)
    )
    decisions = np.bincount(inverse, minlength=len(episodes))
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(episodes), size=len(episodes))
        denominator = decisions[draw].sum()
        deltas[index] = (
            candidate_hits[draw].sum() - baseline_hits[draw].sum()
        ) / denominator
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "unit": "episode",
        "samples": samples,
        "seed": seed,
        "delta_top1_95": [round(float(low), 4), round(float(high), 4)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--baseline-model", type=Path,
        help="Optional old model for a paired comparison on identical rows.",
    )
    parser.add_argument(
        "--num-iteration", type=int,
        help="Evaluate only the first N trees of --model.",
    )
    parser.add_argument(
        "--baseline-num-iteration", type=int,
        help="Evaluate only the first N trees of --baseline-model.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split", default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--teams", default="",
                        help="Comma-separated label team ids to evaluate.")
    parser.add_argument(
        "--pin-team", type=int,
        help="Use this team feature for every row instead of the label team.",
    )
    parser.add_argument(
        "--pin-team-code", type=int,
        help=(
            "Override the dense categorical code used for --pin-team. This "
            "is required when evaluating an old model on a corpus whose "
            "teacher set (and therefore automatic dense mapping) changed."
        ),
    )
    parser.add_argument(
        "--baseline-pin-team-code", type=int,
        help=(
            "Categorical pilot code used only by --baseline-model. This "
            "allows a paired comparison when candidate and baseline were "
            "trained with different dense teacher mappings."
        ),
    )
    parser.add_argument("--team-feature", action="store_true")
    parser.add_argument("--split-mode", default="per-team",
                        choices=["global", "per-team"])
    parser.add_argument("--validation-fraction", type=float, default=0.12)
    parser.add_argument("--test-fraction", type=float, default=0.12)
    args = parser.parse_args()
    if args.num_iteration is not None and args.num_iteration <= 0:
        parser.error("--num-iteration must be positive")
    if (
        args.baseline_num_iteration is not None
        and args.baseline_num_iteration <= 0
    ):
        parser.error("--baseline-num-iteration must be positive")
    if args.baseline_num_iteration is not None and args.baseline_model is None:
        parser.error("--baseline-num-iteration requires --baseline-model")

    corpus = Corpus(args.corpus)
    boundaries = None
    if args.split_mode == "per-team":
        boundaries = corpus.resplit_per_team(
            args.validation_fraction, args.test_fraction
        )
    if args.team_feature:
        corpus.add_team_feature()

    teams = {
        int(value) for value in args.teams.split(",") if value.strip()
    } or None
    decisions = select_decisions(corpus, args.split, teams, None)
    if not len(decisions):
        raise SystemExit("selected split has no decisions")
    if args.pin_team is not None:
        if not args.team_feature:
            parser.error("--pin-team requires --team-feature")
        if args.pin_team not in corpus.team_codes and args.pin_team_code is None:
            parser.error(f"--pin-team {args.pin_team} is absent from corpus")
        if args.pin_team_code is not None:
            if args.pin_team_code < 0:
                parser.error("--pin-team-code cannot be negative")
            corpus.team_codes[args.pin_team] = args.pin_team_code
    elif args.pin_team_code is not None:
        parser.error("--pin-team-code requires --pin-team")
    if args.baseline_pin_team_code is not None:
        if args.baseline_model is None:
            parser.error("--baseline-pin-team-code requires --baseline-model")
        if args.pin_team is None or not args.team_feature:
            parser.error(
                "--baseline-pin-team-code requires --pin-team and "
                "--team-feature"
            )
        if args.baseline_pin_team_code < 0:
            parser.error("--baseline-pin-team-code cannot be negative")

    def load_model(path: Path) -> lgb.Booster:
        model = lgb.Booster(model_file=str(path))
        if model.num_feature() != len(corpus.names):
            raise SystemExit(
                "feature width mismatch: "
                f"model={model.num_feature()} corpus={len(corpus.names)} "
                f"path={path}"
            )
        model_names = model.feature_name()
        if model_names != corpus.names:
            mismatch = next(
                (
                    (index, expected, actual)
                    for index, (expected, actual) in enumerate(
                        zip(model_names, corpus.names)
                    )
                    if expected != actual
                ),
                None,
            )
            raise SystemExit(f"feature names mismatch: {mismatch} path={path}")
        return model

    booster = load_model(args.model)
    baseline = load_model(args.baseline_model) if args.baseline_model else None

    matrix = corpus.matrix(decisions, pin_team=args.pin_team)
    scores = booster.predict(matrix, num_iteration=args.num_iteration)
    if baseline is not None:
        if args.baseline_pin_team_code is None:
            baseline_scores = baseline.predict(
                matrix, num_iteration=args.baseline_num_iteration
            )
        else:
            baseline_matrix = matrix.copy()
            baseline_matrix[:, -1] = args.baseline_pin_team_code
            baseline_scores = baseline.predict(
                baseline_matrix, num_iteration=args.baseline_num_iteration
            )
            del baseline_matrix
    else:
        baseline_scores = None
    del matrix
    sizes = corpus.groups[decisions]
    offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
    correct = top1(scores, corpus, decisions, offsets)
    hits = int(correct.sum())
    low, high = wilson(hits, len(decisions))

    result: dict[str, Any] = {
        "corpus": str(args.corpus.resolve()),
        "model": str(args.model.resolve()),
        "model_iterations": int(min(
            booster.num_trees(), args.num_iteration or booster.num_trees()
        )),
        "split": args.split,
        "split_mode": args.split_mode,
        "split_boundaries": boundaries,
        "label_teams": sorted(teams) if teams else "all",
        "pin_team": args.pin_team,
        "pin_team_code": (
            corpus.team_codes[args.pin_team]
            if args.pin_team is not None else None
        ),
        "baseline_pin_team_code": args.baseline_pin_team_code,
        "team_feature": args.team_feature,
        "episodes": int(len(np.unique(corpus.episode_ids[decisions]))),
        "submissions": int(len(np.unique(corpus.submission_ids[decisions]))),
        "decisions": int(len(decisions)),
        "top1": round(hits / len(decisions), 4),
        "top1_hits": hits,
        "top1_wilson95": [low, high],
        "top2": topk(corpus, decisions, offsets, scores, 2),
        "top3": topk(corpus, decisions, offsets, scores, 3),
        "taxonomy": error_taxonomy(
            corpus, decisions, offsets, scores, corpus.names
        ),
        "top1_by_team": grouped_rates(corpus.team_ids[decisions], correct),
        "top1_by_context": grouped_rates(corpus.contexts[decisions], correct),
        "top1_by_teacher_action": grouped_rates(
            corpus.teacher_action_types[decisions], correct
        ),
    }
    if baseline is not None and baseline_scores is not None:
        baseline_correct = top1(
            baseline_scores, corpus, decisions, offsets
        )
        candidate_only = int(np.sum(correct & ~baseline_correct))
        baseline_only = int(np.sum(~correct & baseline_correct))
        both = int(np.sum(correct & baseline_correct))
        neither = int(np.sum(~correct & ~baseline_correct))
        result["baseline"] = {
            "model": str(args.baseline_model.resolve()),
            "model_iterations": int(min(
                baseline.num_trees(),
                args.baseline_num_iteration or baseline.num_trees(),
            )),
            "top1": round(float(baseline_correct.mean()), 4),
            "top1_hits": int(baseline_correct.sum()),
        }
        result["paired_comparison"] = {
            "candidate_minus_baseline_top1": round(
                float(correct.mean() - baseline_correct.mean()), 4
            ),
            "both_correct": both,
            "candidate_only_correct": candidate_only,
            "baseline_only_correct": baseline_only,
            "neither_correct": neither,
            "net_correct_decisions": candidate_only - baseline_only,
            "cluster_bootstrap": episode_bootstrap_delta(
                corpus.episode_ids[decisions], correct, baseline_correct
            ),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "decisions": result["decisions"],
        "top1": result["top1"],
        "top1_wilson95": result["top1_wilson95"],
        "top3": result["top3"],
        "order_insensitive_top1":
            result["taxonomy"]["order_insensitive_top1"],
        "baseline_top1": (
            result.get("baseline", {}).get("top1")
        ),
        "candidate_minus_baseline_top1": (
            result.get("paired_comparison", {}).get(
                "candidate_minus_baseline_top1"
            )
        ),
        "delta_top1_95": (
            result.get("paired_comparison", {})
            .get("cluster_bootstrap", {})
            .get("delta_top1_95")
        ),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
