"""Train an elite-cohort Grimmsnarl imitation ranker.

The v2 ranker conditions on an exact pilot id.  That is excellent for the
most deterministic pilots, but the highest-rated pilots are individually
noisy and have little data each.  v3 replaces the exact id with a binary
strength tier: rank 4/5/9/11/13 share one value, while the remaining pilots
still provide board-mechanics examples under the other value.

Early stopping is computed only on the elite cohort.  The chronological test
block is scored once after the iteration count has been selected.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_grimmsnarl_v2_teacher import (  # noqa: E402
    Corpus,
    Top1Metric,
    error_taxonomy,
    make_dataset,
    select_decisions,
    top1,
    topk,
    wilson,
)


DEFAULT_ELITE = (16371703, 16422241, 16463316, 16561259, 16531269)


class EliteCorpus(Corpus):
    """Corpus whose conditioning column identifies a strength cohort."""

    def add_elite_feature(self, elite: set[int]) -> None:
        if self.team_feature:
            return
        self.team_feature = True
        self.team_codes = {
            int(team): int(int(team) in elite)
            for team in np.unique(self.team_ids)
        }
        self.names.append("teacher_elite_tier")
        self.categorical.append("teacher_elite_tier")


def _evaluate(
    booster: lgb.Booster,
    corpus: Corpus,
    decisions: np.ndarray,
) -> dict[str, Any]:
    matrix = corpus.matrix(decisions)
    scores = booster.predict(matrix, num_iteration=booster.best_iteration)
    del matrix
    sizes = corpus.groups[decisions]
    offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
    correct = top1(scores, corpus, decisions, offsets)
    hits = int(correct.sum())
    low, high = wilson(hits, len(decisions))

    def breakdown(values: np.ndarray) -> dict[str, dict[str, float | int]]:
        buckets: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for slot, value in enumerate(values):
            bucket = buckets[int(value)]
            bucket[0] += 1
            bucket[1] += int(correct[slot])
        return {
            str(key): {
                "decisions": total,
                "top1": round(agreed / total, 4),
            }
            for key, (total, agreed) in sorted(buckets.items())
        }

    return {
        "decisions": int(len(decisions)),
        "top1": round(float(correct.mean()), 4),
        "top1_wilson95": [low, high],
        "top2": topk(corpus, decisions, offsets, scores, 2),
        "top3": topk(corpus, decisions, offsets, scores, 3),
        "taxonomy": error_taxonomy(
            corpus, decisions, offsets, scores, corpus.names
        ),
        "top1_by_context": breakdown(corpus.contexts[decisions]),
        "top1_by_team": breakdown(corpus.team_ids[decisions]),
        "top1_by_teacher_action": breakdown(
            corpus.teacher_action_types[decisions]
        ),
    }


def _baseline(
    model_path: Path,
    corpus_path: Path,
    elite: set[int],
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, Any]:
    corpus = Corpus(corpus_path)
    corpus.resplit_per_team(validation_fraction, test_fraction)
    corpus.add_team_feature()
    booster = lgb.Booster(model_file=str(model_path))
    output: dict[str, Any] = {}
    for split in ("validation", "test"):
        block = select_decisions(corpus, split, elite, None)
        output[split] = _evaluate(booster, corpus, block)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path)
    parser.add_argument(
        "--elite-teams",
        default=",".join(map(str, DEFAULT_ELITE)),
        help="Comma-separated team ids forming the shared elite tier.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.12)
    parser.add_argument("--test-fraction", type=float, default=0.12)
    parser.add_argument("--num-leaves", type=int, default=255)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-data-in-leaf", type=int, default=40)
    parser.add_argument("--feature-fraction", type=float, default=0.5)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--num-boost-round", type=int, default=4000)
    parser.add_argument("--early-stopping", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    elite = {
        int(value) for value in args.elite_teams.split(",") if value.strip()
    }
    if not elite:
        parser.error("--elite-teams cannot be empty")

    corpus = EliteCorpus(args.corpus)
    boundaries = corpus.resplit_per_team(
        args.validation_fraction, args.test_fraction
    )
    unknown = elite - set(map(int, np.unique(corpus.team_ids)))
    if unknown:
        parser.error(f"elite teams absent from corpus: {sorted(unknown)}")
    corpus.add_elite_feature(elite)

    train = select_decisions(corpus, "train", None, None)
    validation_all = select_decisions(corpus, "validation", None, None)
    test_all = select_decisions(corpus, "test", None, None)
    validation = validation_all[
        np.isin(corpus.team_ids[validation_all], list(elite))
    ]
    test = test_all[np.isin(corpus.team_ids[test_all], list(elite))]
    print(
        f"decisions train={len(train)} elite-validation={len(validation)} "
        f"elite-test={len(test)} features={len(corpus.names)}",
        flush=True,
    )

    train_set = make_dataset(corpus, train)
    validation_set = make_dataset(corpus, validation, reference=train_set)
    params: dict[str, Any] = {
        "objective": "lambdarank",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.lambda_l2,
        "seed": args.seed,
        "verbosity": -1,
        "num_threads": args.threads,
        "metric": "None",
        "lambdarank_truncation_level": 12,
        "label_gain": [0, 1],
    }
    evals: dict[str, dict[str, list[float]]] = {}
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[validation_set],
        valid_names=["elite_validation"],
        feval=Top1Metric(corpus, validation),
        callbacks=[
            lgb.early_stopping(args.early_stopping),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals),
        ],
    )

    result: dict[str, Any] = {
        "method": "shared elite-tier conditioning",
        "corpus": str(args.corpus.resolve()),
        "elite_teams": sorted(elite),
        "elite_feature_value": 1,
        "split_mode": "chronological per-team",
        "split_boundaries": boundaries,
        "split_decisions": {
            "train_all": int(len(train)),
            "validation_elite": int(len(validation)),
            "test_elite": int(len(test)),
        },
        "params": params,
        "best_iteration": int(booster.best_iteration),
        "validation": _evaluate(booster, corpus, validation),
        "test": _evaluate(booster, corpus, test),
        "validation_curve_tail": {
            key: [round(float(value), 4) for value in values[-5:]]
            for key, values in evals.get("elite_validation", {}).items()
        },
        "test_read_once_after_validation_selection": True,
    }
    if args.baseline_model:
        result["baseline"] = _baseline(
            args.baseline_model,
            args.corpus,
            elite,
            args.validation_fraction,
            args.test_fraction,
        )
        result["delta"] = {
            split: round(
                result[split]["top1"]
                - result["baseline"][split]["top1"],
                4,
            )
            for split in ("validation", "test")
        }

    importance = sorted(
        zip(corpus.names, booster.feature_importance("gain")),
        key=lambda pair: -pair[1],
    )
    result["top_features"] = [
        {"name": name, "gain": round(float(gain), 1)}
        for name, gain in importance[:30]
    ]

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(
        str(args.output_model), num_iteration=booster.best_iteration
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "best_iteration": result["best_iteration"],
        "validation_top1": result["validation"]["top1"],
        "test_top1": result["test"]["top1"],
        "baseline_validation": (
            result.get("baseline", {}).get("validation", {}).get("top1")
        ),
        "baseline_test": (
            result.get("baseline", {}).get("test", {}).get("top1")
        ),
        "delta": result.get("delta"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
