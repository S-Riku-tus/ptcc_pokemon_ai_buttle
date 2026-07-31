"""v33 ablation: intra-turn state features and turn-order graded relevance.

Diagnosis that motivates this script: on the frozen Yushin holdout the v32
ranker is 78.71% Top-1, but 59.98% of its errors pick an action the teacher
also plays later in the same turn. Order-insensitive agreement is already
91.48%. The missing signal is therefore intra-turn ordering, not action choice.

Two independent fixes are measured here.

``turnstate``
    Eight columns describing what the acting player has already been offered
    and passed over during the current turn. They are recoverable at inference
    time from the agent's own decision stream, so they are legal runtime
    features.

``graded``
    LambdaRank relevance is raised for candidates the teacher plays later in
    the same turn instead of labelling them exactly like actions the teacher
    never plays. Evaluation stays strictly binary Top-1/2/3 on the recorded
    action.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

BASE_CATEGORICAL = {
    "action_type", "option_type", "candidate_card_id", "candidate_attack_id",
    "candidate_target_id", "self_active_id", "opp_active_id", "stadium_id",
    "fallback_action_type", "fallback_card_id",
}
SEMANTIC = (
    "option_type", "candidate_card_id", "candidate_attack_id",
    "candidate_target_id", "candidate_inplay_area",
)
TURN_FEATURES = (
    "turn_decision_index",
    "turn_candidate_offer_count",
    "turn_candidate_passed_over",
    "turn_candidate_offered_previous",
    "turn_candidate_first_offer_index",
    "turn_class_passed_over",
    "turn_class_offer_count",
    "turn_new_candidate",
)


def ranges(groups):
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def rows_for(groups, decisions):
    starts, ends = ranges(groups)
    return np.concatenate([
        np.arange(starts[d], ends[d], dtype=np.int64) for d in decisions
    ])


def topk(scores, labels, groups):
    starts, ends = ranges(groups)
    t1 = t2 = t3 = 0
    for a, b in zip(starts, ends):
        order = np.argsort(-scores[a:b], kind="stable")
        lab = labels[a:b]
        t1 += int(lab[order[0]] == 1)
        t2 += int(bool(np.any(lab[order[:2]] == 1)))
        t3 += int(bool(np.any(lab[order[:3]] == 1)))
    n = len(groups)
    return {"top1": t1 / n, "top2": t2 / n, "top3": t3 / n, "decisions": int(n)}


def turn_blocks(features, groups, episode_ids, names):
    """Yield the list of decision indices belonging to each acting turn."""
    starts, _ = ranges(groups)
    i_turn = names.index("turn")
    blocks = []
    current = None
    for decision in range(len(groups)):
        key = (
            int(episode_ids[decision]),
            int(features[starts[decision], i_turn]),
        )
        if key != current:
            current = key
            blocks.append([])
        blocks[-1].append(decision)
    return blocks


def build_turn_state(features, labels, groups, blocks, names):
    starts, ends = ranges(groups)
    sem_cols = [names.index(n) for n in SEMANTIC]
    cls_cols = [names.index(n) for n in ("action_type", "candidate_card_id")]
    extra = np.zeros((len(labels), len(TURN_FEATURES)), dtype=np.float32)

    for block in blocks:
        seen_cand: dict[tuple, tuple[int, int, int]] = {}
        seen_cls: dict[tuple, tuple[int, int]] = {}
        prev_offered: set[tuple] = set()
        for position, decision in enumerate(block):
            a, b = starts[decision], ends[decision]
            pos = a + int(np.flatnonzero(labels[a:b] == 1)[0])
            chosen_sem = tuple(features[pos, sem_cols].tolist())
            chosen_cls = tuple(features[pos, cls_cols].tolist())
            offered_now = set()
            for row in range(a, b):
                sem = tuple(features[row, sem_cols].tolist())
                cls = tuple(features[row, cls_cols].tolist())
                offered_now.add(sem)
                offers, passed, first = seen_cand.get(sem, (0, 0, -1))
                c_offers, c_passed = seen_cls.get(cls, (0, 0))
                extra[row] = (
                    position,
                    offers,
                    passed,
                    int(sem in prev_offered),
                    first if first >= 0 else position,
                    c_passed,
                    c_offers,
                    int(offers == 0),
                )
            for row in range(a, b):
                sem = tuple(features[row, sem_cols].tolist())
                cls = tuple(features[row, cls_cols].tolist())
                offers, passed, first = seen_cand.get(sem, (0, 0, -1))
                seen_cand[sem] = (
                    offers + 1,
                    passed + int(sem != chosen_sem),
                    first if first >= 0 else position,
                )
                c_offers, c_passed = seen_cls.get(cls, (0, 0))
                seen_cls[cls] = (
                    c_offers + 1,
                    c_passed + int(cls != chosen_cls),
                )
            prev_offered = offered_now
    return extra


def build_graded_labels(features, labels, groups, blocks, names, high, mid, low):
    """Chosen -> high, played later this turn -> mid/low, otherwise 0."""
    starts, ends = ranges(groups)
    sem_cols = [names.index(n) for n in SEMANTIC]
    graded = np.zeros(len(labels), dtype=np.int32)
    stats = Counter()
    for block in blocks:
        picks = []
        for decision in block:
            a, b = starts[decision], ends[decision]
            pos = a + int(np.flatnonzero(labels[a:b] == 1)[0])
            picks.append(tuple(features[pos, sem_cols].tolist()))
        for position, decision in enumerate(block):
            a, b = starts[decision], ends[decision]
            future = picks[position + 1:]
            rank = {}
            for offset, key in enumerate(future):
                rank.setdefault(key, offset)
            for row in range(a, b):
                if labels[row] == 1:
                    graded[row] = high
                    stats["chosen"] += 1
                    continue
                sem = tuple(features[row, sem_cols].tolist())
                if sem in rank:
                    graded[row] = mid if rank[sem] == 0 else low
                    stats["later_next" if rank[sem] == 0 else "later"] += 1
                else:
                    stats["never"] += 1
    return graded, dict(stats)


def fit_and_score(matrix, cols, train_labels, eval_labels_by_split, wtr,
                  rows, groups_by_split, seed, num_leaves, n_estimators,
                  min_child_samples, learning_rate, label_gain):
    categorical = [
        i for i, n in enumerate(cols)
        if n in BASE_CATEGORICAL or n.endswith("_id")
    ]
    params = dict(
        objective="lambdarank", metric="ndcg",
        num_leaves=num_leaves, n_estimators=n_estimators,
        learning_rate=learning_rate, min_child_samples=min_child_samples,
        max_depth=-1, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.88, reg_alpha=0.2, reg_lambda=1.0,
        random_state=seed, n_jobs=20, verbosity=-1,
    )
    if label_gain is not None:
        params["label_gain"] = label_gain
    model = lgb.LGBMRanker(**params)
    model.fit(
        matrix[rows["train"]], train_labels, group=groups_by_split["train"],
        sample_weight=wtr, feature_name=cols, categorical_feature=categorical,
        eval_set=[(matrix[rows["validation"]],
                   eval_labels_by_split["validation"])],
        eval_group=[groups_by_split["validation"]],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--recency-floor", type=float, default=0.25)
    parser.add_argument("--recency-power", type=float, default=2.0)
    parser.add_argument(
        "--variant", action="append", default=[],
        help="baseline | turnstate | graded | turnstate_graded",
    )
    args = parser.parse_args()
    variants = args.variant or [
        "baseline", "turnstate", "graded", "turnstate_graded",
    ]

    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        labels = cached["labels"]
        weights = cached["weights"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
        names = cached["feature_names"].astype(str).tolist()

    blocks = turn_blocks(features, groups, episode_ids, names)
    print(f"acting turns: {len(blocks)}  decisions: {len(groups)}", flush=True)
    extra = build_turn_state(features, labels, groups, blocks, names)
    graded, graded_stats = build_graded_labels(
        features, labels, groups, blocks, names, high=7, mid=3, low=1,
    )
    print("graded label counts:", graded_stats, flush=True)

    idx = TURN_FEATURES.index("turn_candidate_passed_over")
    hist: dict[int, Counter] = defaultdict(Counter)
    starts, ends = ranges(groups)
    train = np.flatnonzero(splits == "train")
    for decision in train:
        a, b = starts[decision], ends[decision]
        pos = a + int(np.flatnonzero(labels[a:b] == 1)[0])
        for row in range(a, b):
            bucket = min(int(extra[row, idx]), 4)
            hist[bucket]["rows"] += 1
            hist[bucket]["chosen"] += int(row == pos)
    prior = {
        str(k): {
            "rows": int(v["rows"]),
            "p_chosen": v["chosen"] / v["rows"],
        }
        for k, v in sorted(hist.items())
    }
    print("P(chosen | passed over n times this turn):", flush=True)
    for key, value in prior.items():
        print(f"  {key}: rows={value['rows']:8d} p={value['p_chosen']:.4f}")

    valid = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    rows = {
        "train": rows_for(groups, train),
        "validation": rows_for(groups, valid),
        "test": rows_for(groups, test),
    }
    groups_by_split = {
        "train": groups[train].astype(int),
        "validation": groups[valid].astype(int),
        "test": groups[test].astype(int),
    }
    eval_labels = {k: labels[v] for k, v in rows.items()}

    episodes = episode_ids[train]
    ordered = np.unique(episodes)
    pos_map = {
        int(e): i / max(len(ordered) - 1, 1) for i, e in enumerate(ordered)
    }
    floor, power = args.recency_floor, args.recency_power
    dec_mult = np.asarray(
        [floor + (1.0 - floor) * pos_map[int(e)] ** power for e in episodes],
        dtype=np.float32,
    )
    wtr = weights[rows["train"]] * np.repeat(
        dec_mult, groups_by_split["train"]
    )

    augmented = np.concatenate([features, extra], axis=1)
    all_names = names + list(TURN_FEATURES)
    label_gain = [0, 1, 3, 7, 15, 31, 63, 127]

    results = {}
    for variant in variants:
        use_turn = "turnstate" in variant
        use_graded = "graded" in variant
        matrix = augmented if use_turn else features
        cols = all_names if use_turn else names
        train_labels = (
            graded[rows["train"]] if use_graded else labels[rows["train"]]
        )
        model = fit_and_score(
            matrix, cols, train_labels, eval_labels, wtr, rows,
            groups_by_split, args.seed, args.num_leaves, args.n_estimators,
            args.min_child_samples, args.learning_rate,
            label_gain if use_graded else None,
        )
        best = int(model.best_iteration_ or args.n_estimators)
        entry = {"best_iteration": best}
        for split in ("train", "validation", "test"):
            entry[split] = topk(
                model.predict(matrix[rows[split]]).astype(np.float32),
                labels[rows[split]], groups_by_split[split],
            )
        results[variant] = entry
        print(json.dumps({variant: {
            "best_iteration": best,
            "train_top1": round(entry["train"]["top1"], 4),
            "val_top1": round(entry["validation"]["top1"], 4),
            "test_top1": round(entry["test"]["top1"], 4),
            "test_top2": round(entry["test"]["top2"], 4),
            "test_top3": round(entry["test"]["top3"], 4),
        }}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({
            "cache": str(args.cache),
            "turn_features": list(TURN_FEATURES),
            "graded_label_counts": graded_stats,
            "passed_over_prior": prior,
            "hyperparameters": {
                "num_leaves": args.num_leaves,
                "min_child_samples": args.min_child_samples,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "recency_floor": floor,
                "recency_power": power,
            },
            "results": results,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
