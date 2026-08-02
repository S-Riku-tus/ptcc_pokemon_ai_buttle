"""Train and evaluate the Grimmsnarl imitation ranker.

Three defects from the Alakazam line are fixed by construction:

1. Early stopping is on strict Top-1, the metric the agent is actually judged
   on. v33 stopped on NDCG and shipped roughly half the trees it wanted.
2. Reported agreement is per-decision Top-1 on a chronological block that is
   never touched during fitting or configuration selection.
3. Teacher-cohort choice is an experiment, not an assumption. ``--teams`` and
   ``--min-agreement`` allow field-pooled, subset and single-pilot corpora to
   be compared on the same held-out block, and ``--leave-out-team`` measures
   whether the policy transfers to a pilot the model never saw.

The error taxonomy separates recoverable same-turn ordering errors (the model
picked something the teacher also played that turn, just in another order)
from genuine divergence. That distinction is what the Alakazam v36 report
identified as the real ceiling, so it is measured from the start.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


class Corpus:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.features = data["features"]
        self.labels = data["labels"]
        self.groups = data["groups"]
        self.splits = data["splits"]
        self.episode_ids = data["episode_ids"]
        self.team_ids = data["team_ids"]
        self.turns = data["turns"]
        self.won = data["won"]
        self.teacher_action_types = data["teacher_action_types"]
        self.names = [str(x) for x in data["feature_names"]]
        self.categorical = [str(x) for x in data["categorical"]]
        self.starts, self.ends = _ranges(self.groups)
        self.team_feature = False

    def rows_for(self, decisions: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.arange(self.starts[i], self.ends[i]) for i in decisions
        ]) if len(decisions) else np.zeros(0, dtype=np.int64)

    def add_team_feature(self) -> None:
        """Condition the ranker on which pilot is acting.

        Pooling 21 pilots buys data but forces one averaged policy. With the
        pilot exposed as a categorical the model can keep the shared mechanics
        and still express per-pilot habits; inference pins it to the pilot we
        want to copy. The column is materialised per split in ``matrix`` -
        widening the 2.3 GB base array in place costs a second copy and gets
        the process killed.
        """
        if not self.team_feature:
            self.team_feature = True
            # Dense 0..N-1 codes. Raw Kaggle team ids are ~1.6e7, and
            # LightGBM allocates categorical bins over the value range, so
            # feeding them raw makes construction pathologically slow.
            self.team_codes = {
                int(team): index
                for index, team in enumerate(sorted(set(
                    int(x) for x in self.team_ids
                )))
            }
            self.names.append("teacher_team_id")
            self.categorical.append("teacher_team_id")

    def resplit_per_team(self, validation: float, test: float) -> dict:
        """Hold out each pilot's own newest games instead of the field's.

        A single global chronological cut puts almost all of one pilot's games
        on one side of the boundary, so per-pilot test blocks come out tiny and
        wildly uneven. Cutting inside each pilot keeps the split honest - test
        games are still strictly later than that pilot's training games - and
        gives every pilot a test block worth quoting a confidence interval on.
        """
        splits = np.empty(len(self.groups), dtype=self.splits.dtype)
        boundaries: dict[str, list[int]] = {}
        for team in np.unique(self.team_ids):
            mask = self.team_ids == team
            episodes = np.sort(np.unique(self.episode_ids[mask]))
            total = len(episodes)
            test_size = max(1, int(round(total * test)))
            validation_size = max(1, int(round(total * validation)))
            train_end = max(1, total - test_size - validation_size)
            validation_min = int(episodes[train_end])
            test_min = int(
                episodes[min(total - 1, train_end + validation_size)]
            )
            block = self.episode_ids[mask]
            splits[mask] = np.where(
                block >= test_min, "test",
                np.where(block >= validation_min, "validation", "train"),
            )
            boundaries[str(int(team))] = [validation_min, test_min]
        self.splits = splits
        return boundaries

    def matrix(self, decisions: np.ndarray, pin_team: int | None = None):
        """Feature block for these decisions, built in bounded chunks."""
        rows = self.rows_for(decisions)
        width = self.features.shape[1] + int(self.team_feature)
        block = np.empty((len(rows), width), dtype=np.float32)
        step = 200_000
        for start in range(0, len(rows), step):
            window = rows[start:start + step]
            block[start:start + len(window), :self.features.shape[1]] = (
                self.features[window]
            )
        if self.team_feature:
            codes = (
                np.full(
                    len(decisions),
                    self.team_codes[int(pin_team)],
                    dtype=np.float32,
                )
                if pin_team is not None
                else np.asarray(
                    [self.team_codes[int(x)] for x in self.team_ids[decisions]],
                    dtype=np.float32,
                )
            )
            block[:, -1] = np.repeat(codes, self.groups[decisions])
        return block


def top1(scores: np.ndarray, corpus: Corpus, decisions: np.ndarray,
         row_offset: np.ndarray) -> np.ndarray:
    """Per-decision correctness of the argmax candidate."""
    correct = np.zeros(len(decisions), dtype=bool)
    for slot, decision in enumerate(decisions):
        start = row_offset[slot]
        size = int(corpus.groups[decision])
        window = scores[start:start + size]
        best = int(np.argmax(window))
        correct[slot] = bool(
            corpus.labels[corpus.starts[decision] + best] == 1
        )
    return correct


class Top1Metric:
    """LightGBM feval for strict Top-1 on the validation set."""

    def __init__(self, corpus: Corpus, decisions: np.ndarray):
        self.corpus = corpus
        self.decisions = decisions
        sizes = corpus.groups[decisions]
        self.offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)

    def __call__(self, preds: np.ndarray, dataset: lgb.Dataset):
        correct = top1(preds, self.corpus, self.decisions, self.offsets)
        return "top1", float(correct.mean()), True


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963985
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * ((phat * (1 - phat) + z * z / (4 * total)) / total) ** 0.5
    return (
        round((centre - margin) / denominator, 4),
        round((centre + margin) / denominator, 4),
    )


def error_taxonomy(
    corpus: Corpus,
    decisions: np.ndarray,
    offsets: np.ndarray,
    scores: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    """Split misses into same-turn ordering errors and genuine divergence."""
    action_col = names.index("action_type_id")
    card_col = names.index("candidate_card_id")
    attack_col = names.index("candidate_attack_id")
    target_col = names.index("candidate_target_id")

    def identity(row: int) -> tuple:
        return (
            int(corpus.features[row, action_col]),
            int(corpus.features[row, card_col]),
            int(corpus.features[row, attack_col]),
            int(corpus.features[row, target_col]),
        )

    # What the teacher actually played across each whole turn.
    turn_plays: dict[tuple, set[tuple]] = defaultdict(set)
    for slot, decision in enumerate(decisions):
        start = int(corpus.starts[decision])
        chosen = start + int(np.flatnonzero(
            corpus.labels[start:int(corpus.ends[decision])] == 1
        )[0])
        key = (int(corpus.episode_ids[decision]), int(corpus.turns[decision]))
        turn_plays[key].add(identity(chosen))

    counts: Counter[str] = Counter()
    confusion: Counter[tuple] = Counter()
    for slot, decision in enumerate(decisions):
        start = int(corpus.starts[decision])
        size = int(corpus.groups[decision])
        window = scores[offsets[slot]:offsets[slot] + size]
        predicted = start + int(np.argmax(window))
        if corpus.labels[predicted] == 1:
            counts["correct"] += 1
            continue
        chosen = start + int(np.flatnonzero(
            corpus.labels[start:start + size] == 1
        )[0])
        key = (int(corpus.episode_ids[decision]), int(corpus.turns[decision]))
        predicted_identity = identity(predicted)
        if predicted_identity in turn_plays[key]:
            counts["same_turn_ordering"] += 1
        elif predicted_identity[0] == identity(chosen)[0]:
            counts["same_action_type_divergence"] += 1
        else:
            counts["divergence"] += 1
        confusion[(identity(chosen)[0], predicted_identity[0])] += 1

    total = sum(counts.values())
    return {
        "counts": dict(counts),
        "rates": {
            key: round(value / total, 4) for key, value in counts.items()
        },
        "order_insensitive_top1": round(
            (counts["correct"] + counts["same_turn_ordering"]) / max(1, total),
            4,
        ),
        "top_action_confusions": [
            {"teacher": int(a), "predicted": int(b), "count": int(n)}
            for (a, b), n in confusion.most_common(12)
        ],
    }


def topk(corpus: Corpus, decisions: np.ndarray, offsets: np.ndarray,
         scores: np.ndarray, k: int) -> float:
    hits = 0
    for slot, decision in enumerate(decisions):
        start = int(corpus.starts[decision])
        size = int(corpus.groups[decision])
        window = scores[offsets[slot]:offsets[slot] + size]
        order = np.argsort(-window)[:k]
        hits += int(any(
            corpus.labels[start + int(index)] == 1 for index in order
        ))
    return round(hits / max(1, len(decisions)), 4)


def select_decisions(corpus: Corpus, split: str,
                     teams: set[int] | None,
                     leave_out: int | None) -> np.ndarray:
    mask = corpus.splits == split
    if teams is not None:
        mask &= np.isin(corpus.team_ids, list(teams))
    if leave_out is not None:
        if split == "train":
            mask &= corpus.team_ids != leave_out
        else:
            mask &= corpus.team_ids == leave_out
    return np.flatnonzero(mask)


def make_dataset(corpus: Corpus, decisions: np.ndarray,
                 reference: lgb.Dataset | None = None) -> lgb.Dataset:
    rows = corpus.rows_for(decisions)
    categorical = [
        name for name in corpus.categorical if name in corpus.names
    ]
    # free_raw_data lets LightGBM drop the dense copy once it is binned;
    # evaluation rebuilds the block it needs one split at a time.
    return lgb.Dataset(
        corpus.matrix(decisions),
        label=corpus.labels[rows],
        group=corpus.groups[decisions],
        feature_name=corpus.names,
        categorical_feature=categorical,
        reference=reference,
        free_raw_data=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--teams", default="",
                        help="Comma separated team ids; empty uses all.")
    parser.add_argument("--leave-out-team", type=int,
                        help="Train without this team, evaluate only on it.")
    parser.add_argument("--objective", default="lambdarank",
                        choices=["lambdarank", "binary"])
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
    parser.add_argument(
        "--team-feature", action="store_true",
        help="Expose the acting pilot as a categorical feature.",
    )
    parser.add_argument(
        "--split-mode", default="global", choices=["global", "per-team"],
    )
    parser.add_argument("--validation-fraction", type=float, default=0.12)
    parser.add_argument("--test-fraction", type=float, default=0.12)
    args = parser.parse_args()

    corpus = Corpus(args.corpus)
    split_boundaries = None
    if args.split_mode == "per-team":
        split_boundaries = corpus.resplit_per_team(
            args.validation_fraction, args.test_fraction
        )
    if args.team_feature:
        corpus.add_team_feature()
    teams = {
        int(value) for value in args.teams.split(",") if value.strip()
    } or None

    train = select_decisions(corpus, "train", teams, args.leave_out_team)
    validation = select_decisions(corpus, "validation", teams,
                                  args.leave_out_team)
    test = select_decisions(corpus, "test", teams, args.leave_out_team)
    if not len(train) or not len(validation) or not len(test):
        raise SystemExit(
            f"empty split: train={len(train)} validation={len(validation)} "
            f"test={len(test)}"
        )
    print(
        f"decisions train={len(train)} validation={len(validation)} "
        f"test={len(test)} features={len(corpus.names)}",
        flush=True,
    )

    train_set = make_dataset(corpus, train)
    validation_set = make_dataset(corpus, validation, reference=train_set)

    params: dict[str, Any] = {
        "objective": args.objective,
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
        # No built-in metric. lgb.early_stopping stops on whichever tracked
        # metric stalls first, so leaving ndcg@k enabled would let NDCG pick
        # the iteration count for a model deployed on strict Top-1. That is
        # the v33 defect; Top1Metric below is the only metric.
        "metric": "None",
    }
    if args.objective == "lambdarank":
        params["lambdarank_truncation_level"] = 12
        params["label_gain"] = [0, 1]

    metric = Top1Metric(corpus, validation)
    evals: dict[str, dict[str, list[float]]] = {}
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[validation_set],
        valid_names=["validation"],
        feval=metric,
        callbacks=[
            lgb.early_stopping(args.early_stopping, first_metric_only=False),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals),
        ],
    )

    results: dict[str, Any] = {
        "corpus": str(args.corpus.resolve()),
        "teams": sorted(teams) if teams else "all",
        "leave_out_team": args.leave_out_team,
        "params": params,
        "best_iteration": int(booster.best_iteration),
        "num_boost_round": args.num_boost_round,
        "team_feature": bool(args.team_feature),
        "split_mode": args.split_mode,
        "split_boundaries": split_boundaries,
        "split_decisions": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "split_episodes": {
            name: int(len(np.unique(corpus.episode_ids[block])))
            for name, block in
            (("train", train), ("validation", validation), ("test", test))
        },
    }

    for name, block in (("validation", validation), ("test", test)):
        matrix = corpus.matrix(block)
        scores = booster.predict(
            matrix, num_iteration=booster.best_iteration
        )
        del matrix
        sizes = corpus.groups[block]
        offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
        correct = top1(scores, corpus, block, offsets)
        hits = int(correct.sum())
        low, high = wilson(hits, len(block))
        results[name] = {
            "decisions": int(len(block)),
            "top1": round(float(correct.mean()), 4),
            "top1_wilson95": [low, high],
            "top2": topk(corpus, block, offsets, scores, 2),
            "top3": topk(corpus, block, offsets, scores, 3),
            "taxonomy": error_taxonomy(
                corpus, block, offsets, scores, corpus.names
            ),
            "top1_by_teacher_action": {},
        }
        by_action: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for slot, decision in enumerate(block):
            action = int(corpus.teacher_action_types[decision])
            by_action[action][0] += 1
            by_action[action][1] += int(correct[slot])
        results[name]["top1_by_teacher_action"] = {
            str(action): {
                "decisions": total,
                "top1": round(agree / total, 4),
            }
            for action, (total, agree) in sorted(by_action.items())
        }
        by_team: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for slot, decision in enumerate(block):
            team = int(corpus.team_ids[decision])
            by_team[team][0] += 1
            by_team[team][1] += int(correct[slot])
        results[name]["top1_by_team"] = {
            str(team): {
                "decisions": total,
                "top1": round(agree / total, 4),
            }
            for team, (total, agree) in sorted(
                by_team.items(), key=lambda kv: -kv[1][1] / max(1, kv[1][0])
            )
        }

    importance = sorted(
        zip(corpus.names, booster.feature_importance("gain")),
        key=lambda item: -item[1],
    )
    results["top_features"] = [
        {"name": name, "gain": round(float(gain), 1)}
        for name, gain in importance[:30]
    ]
    results["validation_curve_tail"] = {
        key: [round(float(v), 4) for v in values[-5:]]
        for key, values in evals.get("validation", {}).items()
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.output_model:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(
            str(args.output_model), num_iteration=booster.best_iteration
        )

    print(json.dumps({
        "best_iteration": results["best_iteration"],
        "validation_top1": results["validation"]["top1"],
        "test_top1": results["test"]["top1"],
        "test_top1_wilson95": results["test"]["top1_wilson95"],
        "test_top3": results["test"]["top3"],
        "test_order_insensitive_top1":
            results["test"]["taxonomy"]["order_insensitive_top1"],
        "test_taxonomy": results["test"]["taxonomy"]["rates"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
