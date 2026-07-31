"""v33 ablation: conditional multi-teacher training.

The v32 ranker fits its own training decisions to 96.9% Top-1 but only reaches
78.1% on the frozen chronological holdout, so it is variance limited rather
than capacity limited. v31 already showed that pooling teachers without telling
the model whose policy it is imitating hurts, because different teachers answer
the same board differently.

This script pools every cohort that shares the deck and the 657-feature schema
and appends an explicit ``teacher_id`` column, turning the conflicting labels
into a conditional policy. Evaluation stays on the primary teacher's frozen
chronological holdout, and inference always conditions on the primary teacher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

BASE_CATEGORICAL = {
    "action_type", "option_type", "candidate_card_id", "candidate_attack_id",
    "candidate_target_id", "self_active_id", "opp_active_id", "stadium_id",
    "fallback_action_type", "fallback_card_id",
}


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
    return {
        "top1": t1 / n, "top2": t2 / n, "top3": t3 / n, "decisions": int(n),
    }


def load(path: Path):
    with np.load(path, allow_pickle=False) as cached:
        return {
            "features": cached["features"],
            "labels": cached["labels"],
            "weights": cached["weights"],
            "groups": cached["groups"],
            "splits": cached["splits"].astype(str),
            "episode_ids": cached["episode_ids"],
            "names": cached["feature_names"].astype(str).tolist(),
        }


def recency_multiplier(episode_ids, decisions, floor, power):
    episodes = episode_ids[decisions]
    ordered = np.unique(episodes)
    pos = {int(e): i / max(len(ordered) - 1, 1) for i, e in enumerate(ordered)}
    return np.asarray(
        [floor + (1.0 - floor) * pos[int(e)] ** power for e in episodes],
        dtype=np.float32,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument(
        "--auxiliary", action="append", default=[], metavar="ID:PATH",
        help="Extra cohort as teacher_id:path-to-npz.",
    )
    parser.add_argument("--primary-id", type=int, default=0)
    parser.add_argument("--aux-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1091)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--recency-floor", type=float, default=0.25)
    parser.add_argument("--recency-power", type=float, default=2.0)
    args = parser.parse_args()

    primary = load(args.primary)
    names = primary["names"] + ["teacher_id"]
    splits = primary["splits"]
    train = np.flatnonzero(splits == "train")
    valid = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")

    p_rows = {
        "train": rows_for(primary["groups"], train),
        "validation": rows_for(primary["groups"], valid),
        "test": rows_for(primary["groups"], test),
    }
    p_groups = {
        "train": primary["groups"][train].astype(int),
        "validation": primary["groups"][valid].astype(int),
        "test": primary["groups"][test].astype(int),
    }
    mult = recency_multiplier(
        primary["episode_ids"], train, args.recency_floor, args.recency_power,
    )
    primary_weight = primary["weights"][p_rows["train"]] * np.repeat(
        mult, p_groups["train"]
    )

    def with_id(matrix, teacher_id):
        column = np.full((len(matrix), 1), teacher_id, dtype=np.float32)
        return np.concatenate([matrix, column], axis=1)

    eval_sets = {
        split: (
            with_id(primary["features"][p_rows[split]], args.primary_id),
            primary["labels"][p_rows[split]],
            p_groups[split],
        )
        for split in ("train", "validation", "test")
    }

    train_blocks = [with_id(
        primary["features"][p_rows["train"]], args.primary_id
    )]
    label_blocks = [primary["labels"][p_rows["train"]]]
    weight_blocks = [primary_weight]
    group_blocks = [p_groups["train"]]
    cohort_report = [{
        "path": str(args.primary),
        "teacher_id": args.primary_id,
        "role": "primary",
        "decisions": int(len(train)),
    }]

    # A held-out episode must not re-enter training through another cohort,
    # which can happen when two teachers met each other in the same game.
    protected = set(
        primary["episode_ids"][np.r_[valid, test]].astype(np.int64).tolist()
    )
    for spec in args.auxiliary:
        teacher_id, _, path = spec.partition(":")
        cohort = load(Path(path))
        if cohort["names"] != primary["names"]:
            raise RuntimeError(f"schema mismatch for {path}")
        # Auxiliary cohorts contribute every decision; they are never used for
        # model selection or for the reported holdout.
        decisions = np.flatnonzero(np.asarray([
            int(episode) not in protected
            for episode in cohort["episode_ids"]
        ]))
        dropped = len(cohort["groups"]) - len(decisions)
        if dropped:
            print(f"  dropped {dropped} decisions from {path} that replay a "
                  f"held-out primary episode", flush=True)
        rows = rows_for(cohort["groups"], decisions)
        groups = cohort["groups"][decisions].astype(int)
        aux_mult = recency_multiplier(
            cohort["episode_ids"], decisions,
            args.recency_floor, args.recency_power,
        )
        train_blocks.append(with_id(cohort["features"][rows], int(teacher_id)))
        label_blocks.append(cohort["labels"][rows])
        weight_blocks.append(
            cohort["weights"][rows]
            * np.repeat(aux_mult, groups)
            * args.aux_weight
        )
        group_blocks.append(groups)
        cohort_report.append({
            "path": path,
            "teacher_id": int(teacher_id),
            "role": "auxiliary",
            "decisions": int(len(decisions)),
        })
        print(f"loaded aux {path} id={teacher_id} decisions={len(decisions)}",
              flush=True)

    x_train = np.concatenate(train_blocks, axis=0)
    y_train = np.concatenate(label_blocks, axis=0)
    w_train = np.concatenate(weight_blocks, axis=0)
    g_train = np.concatenate(group_blocks, axis=0)
    print(f"pooled train rows={len(y_train)} decisions={len(g_train)}",
          flush=True)

    categorical = [
        i for i, n in enumerate(names)
        if n in BASE_CATEGORICAL or n.endswith("_id")
    ]
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        num_leaves=args.num_leaves, n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        min_child_samples=args.min_child_samples, max_depth=-1,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.88,
        reg_alpha=0.2, reg_lambda=1.0, random_state=args.seed,
        n_jobs=20, verbosity=-1,
    )
    model.fit(
        x_train, y_train, group=g_train, sample_weight=w_train,
        feature_name=names, categorical_feature=categorical,
        eval_set=[eval_sets["validation"][:2]],
        eval_group=[eval_sets["validation"][2]],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    best = int(model.best_iteration_ or args.n_estimators)
    report = {"best_iteration": best, "cohorts": cohort_report}
    for split in ("train", "validation", "test"):
        matrix, labels, groups = eval_sets[split]
        report[split] = topk(
            model.predict(matrix).astype(np.float32), labels, groups
        )
    print(json.dumps({
        "best_iteration": best,
        "train_top1": round(report["train"]["top1"], 4),
        "val_top1": round(report["validation"]["top1"], 4),
        "test_top1": round(report["test"]["top1"], 4),
        "test_top2": round(report["test"]["top2"], 4),
        "test_top3": round(report["test"]["top3"], 4),
    }), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
