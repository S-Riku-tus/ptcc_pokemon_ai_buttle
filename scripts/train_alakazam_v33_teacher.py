"""Train and export the v33 Alakazam teacher ranker.

Everything here is chosen on the frozen validation block. The test block is
scored once, after the configuration and the blend weights are final.

Reported metrics
    top1/top2/top3   Strict agreement with the exact action the teacher played.
    turn_set         Order-insensitive agreement: the pick is counted correct
                     when the teacher plays that same semantic action anywhere
                     in the same turn. v33 measures it because 60% of the v32
                     Top-1 errors are intra-turn reorderings of actions the
                     teacher takes regardless.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.distill import compact_booster  # noqa: E402

ACTION_TYPES = (
    "ability", "attack", "bench", "boss", "end", "energy", "evolve",
    "hammer", "other", "retreat", "trainer", "xerosic",
)
ACTION_TYPE_MAP = {name: index for index, name in enumerate(ACTION_TYPES)}
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
    "turn_decision_index", "turn_candidate_offer_count",
    "turn_candidate_passed_over", "turn_candidate_offered_previous",
    "turn_candidate_first_offer_index", "turn_class_passed_over",
    "turn_class_offer_count", "turn_new_candidate",
)
LABEL_GAIN = [0, 1, 3, 7, 15, 31, 63, 127]

CONFIGS = {
    "large_leaf": dict(
        objective="lambdarank", num_leaves=127, min_child_samples=40,
        learning_rate=0.03, colsample_bytree=0.88, categorical_ids=True,
    ),
    "numeric_ids": dict(
        objective="lambdarank", num_leaves=127, min_child_samples=40,
        learning_rate=0.03, colsample_bytree=0.88, categorical_ids=False,
    ),
    "small_leaf": dict(
        objective="lambdarank", num_leaves=63, min_child_samples=20,
        learning_rate=0.04, colsample_bytree=0.7, categorical_ids=True,
    ),
    "xendcg": dict(
        objective="rank_xendcg", num_leaves=127, min_child_samples=40,
        learning_rate=0.03, colsample_bytree=0.88, categorical_ids=True,
    ),
}


def ranges(groups):
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def rows_for(groups, decisions):
    starts, ends = ranges(groups)
    return np.concatenate([
        np.arange(starts[d], ends[d], dtype=np.int64) for d in decisions
    ])


def turn_blocks(features, groups, episode_ids, names):
    starts, _ = ranges(groups)
    i_turn = names.index("turn")
    blocks: list[list[int]] = []
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


def graded_labels(features, labels, groups, blocks, names):
    """Chosen -> 7, played next -> 3, played later this turn -> 1, else 0."""
    starts, ends = ranges(groups)
    sem_cols = [names.index(n) for n in SEMANTIC]
    graded = np.zeros(len(labels), dtype=np.int32)
    counts: Counter[str] = Counter()
    for block in blocks:
        picks = []
        for decision in block:
            a, b = starts[decision], ends[decision]
            pos = a + int(np.flatnonzero(labels[a:b] == 1)[0])
            picks.append(tuple(features[pos, sem_cols].tolist()))
        for position, decision in enumerate(block):
            a, b = starts[decision], ends[decision]
            rank: dict[tuple, int] = {}
            for offset, key in enumerate(picks[position + 1:]):
                rank.setdefault(key, offset)
            for row in range(a, b):
                if labels[row] == 1:
                    graded[row] = 7
                    counts["chosen"] += 1
                    continue
                sem = tuple(features[row, sem_cols].tolist())
                if sem in rank:
                    graded[row] = 3 if rank[sem] == 0 else 1
                    counts[
                        "played_next" if rank[sem] == 0 else "played_later"
                    ] += 1
                else:
                    counts["not_played"] += 1
    return graded, dict(counts)


def turn_pick_sets(features, labels, groups, blocks, names):
    """Semantic actions the teacher plays somewhere in each acting turn."""
    starts, ends = ranges(groups)
    sem_cols = [names.index(n) for n in SEMANTIC]
    per_decision: dict[int, set[tuple]] = {}
    for block in blocks:
        picks = set()
        for decision in block:
            a, b = starts[decision], ends[decision]
            pos = a + int(np.flatnonzero(labels[a:b] == 1)[0])
            picks.add(tuple(features[pos, sem_cols].tolist()))
        for decision in block:
            per_decision[decision] = picks
    return per_decision, sem_cols


def evaluate(scores, labels, groups, decisions, features, row_index,
             sem_cols, pick_sets, action_types):
    starts, ends = ranges(np.asarray(groups))
    t1 = t2 = t3 = turn_set = 0
    by_action: dict[int, Counter] = {}
    for local, (a, b) in enumerate(zip(starts, ends)):
        block = scores[a:b]
        lab = labels[a:b]
        order = np.argsort(-block, kind="stable")
        correct = bool(lab[order[0]] == 1)
        t1 += int(correct)
        t2 += int(bool(np.any(lab[order[:2]] == 1)))
        t3 += int(bool(np.any(lab[order[:3]] == 1)))
        picked_row = row_index[a + int(order[0])]
        sem = tuple(features[picked_row, sem_cols].tolist())
        turn_set += int(correct or sem in pick_sets[int(decisions[local])])
        action = int(action_types[decisions[local]])
        bucket = by_action.setdefault(action, Counter())
        bucket["count"] += 1
        bucket["correct"] += int(correct)
    n = len(groups)
    return {
        "decisions": int(n),
        "top1": t1 / n,
        "top2": t2 / n,
        "top3": t3 / n,
        "turn_set": turn_set / n,
        "by_teacher_action": {
            ACTION_TYPES[action]: {
                "count": int(stats["count"]),
                "top1": stats["correct"] / stats["count"],
            }
            for action, stats in sorted(by_action.items())
        },
    }


def normalise(scores, groups):
    """Within-candidate-set z-scores, matching the runtime blend."""
    out = np.empty_like(scores, dtype=np.float64)
    starts, ends = ranges(np.asarray(groups))
    for a, b in zip(starts, ends):
        block = scores[a:b].astype(np.float64)
        mean = block.mean()
        scale = max(block.std(), 1e-5)
        out[a:b] = (block - mean) / scale
    return out


def fit(name, matrix, cols, x_rows, y, group, weight, eval_pack,
        seed, n_estimators, graded):
    config = CONFIGS[name]
    categorical = (
        [
            i for i, n in enumerate(cols)
            if n in BASE_CATEGORICAL or n.endswith("_id")
        ]
        if config["categorical_ids"] else "auto"
    )
    params: dict[str, Any] = dict(
        objective=config["objective"], metric="ndcg",
        num_leaves=config["num_leaves"], n_estimators=n_estimators,
        learning_rate=config["learning_rate"],
        min_child_samples=config["min_child_samples"], max_depth=-1,
        subsample=0.9, subsample_freq=1,
        colsample_bytree=config["colsample_bytree"],
        reg_alpha=0.2, reg_lambda=1.0, random_state=seed,
        n_jobs=20, verbosity=-1,
    )
    if graded:
        params["label_gain"] = LABEL_GAIN
    model = lgb.LGBMRanker(**params)
    kwargs: dict[str, Any] = {
        "X": matrix[x_rows], "y": y, "group": group, "sample_weight": weight,
        "feature_name": cols,
    }
    if config["categorical_ids"]:
        kwargs["categorical_feature"] = categorical
    kwargs.update({
        "eval_set": [(matrix[eval_pack[0]], eval_pack[1])],
        "eval_group": [eval_pack[2]],
        "callbacks": [lgb.early_stopping(80, verbose=False)],
    })
    model.fit(**kwargs)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1091])
    parser.add_argument("--n-estimators", type=int, default=2500)
    parser.add_argument("--recency-floor", type=float, default=0.25)
    parser.add_argument("--recency-power", type=float, default=2.0)
    parser.add_argument("--max-blend", type=int, default=3)
    parser.add_argument(
        "--min-blend-gain", type=float, default=0.002,
        help=(
            "Validation Top-1 a further ensemble member must add before it is "
            "deployed. Extra members multiply the pure-Python inference cost, "
            "so a member that only ties is not worth shipping."
        ),
    )
    parser.add_argument(
        "--models", nargs="+", default=["large_leaf", "numeric_ids"],
    )
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--corpus-report", type=Path)
    parser.add_argument("--teacher-team", default="Yushin Ito")
    parser.add_argument("--teacher-submission-id", type=int, default=54773249)
    args = parser.parse_args()

    corpus_report = (
        json.loads(args.corpus_report.read_text(encoding="utf-8"))
        if args.corpus_report is not None else {}
    )

    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        labels = cached["labels"]
        weights = cached["weights"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
        action_types = cached["teacher_action_types"]
        names = cached["feature_names"].astype(str).tolist()

    base_names = [n for n in names if n not in TURN_FEATURES]
    base_columns = [names.index(n) for n in base_names]
    base_features = np.ascontiguousarray(features[:, base_columns])
    blocks = turn_blocks(features, groups, episode_ids, names)
    graded, graded_counts = graded_labels(
        features, labels, groups, blocks, names
    )
    pick_sets, sem_cols = turn_pick_sets(
        features, labels, groups, blocks, names
    )

    decisions = {
        split: np.flatnonzero(splits == split)
        for split in ("train", "validation", "test")
    }
    rows = {k: rows_for(groups, v) for k, v in decisions.items()}
    group_sizes = {
        k: groups[v].astype(int) for k, v in decisions.items()
    }

    episodes = episode_ids[decisions["train"]]
    ordered = np.unique(episodes)
    position = {
        int(e): i / max(len(ordered) - 1, 1) for i, e in enumerate(ordered)
    }
    floor, power = args.recency_floor, args.recency_power
    multiplier = np.asarray(
        [floor + (1.0 - floor) * position[int(e)] ** power for e in episodes],
        dtype=np.float32,
    )
    train_weight = weights[rows["train"]] * np.repeat(
        multiplier, group_sizes["train"]
    )

    variants: dict[str, dict[str, Any]] = {}
    catalogue = []
    for model_name in args.models:
        for use_turn in (False, True):
            for use_graded in (False, True):
                catalogue.append((model_name, use_turn, use_graded))

    for model_name, use_turn, use_graded in catalogue:
        cols = names if use_turn else base_names
        matrix = features if use_turn else base_features
        key = (
            f"{model_name}"
            f"{'_turn' if use_turn else ''}"
            f"{'_graded' if use_graded else ''}"
        )
        for seed in args.seeds:
            tag = key if len(args.seeds) == 1 else f"{key}_s{seed}"
            y = (
                graded[rows["train"]] if use_graded
                else labels[rows["train"]]
            )
            model = fit(
                model_name, matrix, cols, rows["train"], y,
                group_sizes["train"], train_weight,
                (rows["validation"], labels[rows["validation"]],
                 group_sizes["validation"]),
                seed, args.n_estimators, use_graded,
            )
            best = int(model.best_iteration_ or args.n_estimators)
            scores = {
                split: model.predict(
                    matrix[rows[split]], num_iteration=best
                ).astype(np.float32)
                for split in ("validation", "test")
            }
            metrics = {
                split: evaluate(
                    scores[split], labels[rows[split]], group_sizes[split],
                    decisions[split], features, rows[split], sem_cols,
                    pick_sets, action_types,
                )
                for split in ("validation", "test")
            }
            variants[tag] = {
                "model_name": model_name,
                "use_turn_features": use_turn,
                "use_graded_labels": use_graded,
                "seed": seed,
                "best_iteration": best,
                "validation": metrics["validation"],
                "test": metrics["test"],
                "_model": model,
                "_matrix": matrix,
                "_cols": cols,
                "_scores": scores,
            }
            print(json.dumps({tag: {
                "best_iteration": best,
                "val_top1": round(metrics["validation"]["top1"], 4),
                "test_top1": round(metrics["test"]["top1"], 4),
                "test_top3": round(metrics["test"]["top3"], 4),
                "test_turn_set": round(metrics["test"]["turn_set"], 4),
            }}), flush=True)

    # Greedy forward blend selected on validation only.
    normalised = {
        tag: {
            split: normalise(entry["_scores"][split], group_sizes[split])
            for split in ("validation", "test")
        }
        for tag, entry in variants.items()
    }

    def blend_top1(members, split):
        total = np.zeros(len(rows[split]), dtype=np.float64)
        for tag, weight in members:
            total += weight * normalised[tag][split]
        starts, ends = ranges(np.asarray(group_sizes[split]))
        correct = sum(
            int(labels[rows[split]][a + int(np.argmax(total[a:b]))] == 1)
            for a, b in zip(starts, ends)
        )
        return correct / len(group_sizes[split])

    best_single = max(
        variants, key=lambda t: variants[t]["validation"]["top1"]
    )
    members = [(best_single, 1.0)]
    current = blend_top1(members, "validation")
    trace = [{
        "members": [[t, w] for t, w in members],
        "validation_top1": current,
    }]
    while len(members) < args.max_blend:
        candidate = None
        for tag in variants:
            if any(tag == existing for existing, _ in members):
                continue
            for weight in (0.3, 0.5, 0.7, 1.0, 1.3):
                score = blend_top1(members + [(tag, weight)], "validation")
                if candidate is None or score > candidate[0]:
                    candidate = (score, tag, weight)
        if candidate is None:
            break
        if candidate[0] < current + args.min_blend_gain:
            trace.append({
                "rejected_member": candidate[1],
                "rejected_weight": candidate[2],
                "validation_top1_with_member": candidate[0],
                "gain": candidate[0] - current,
                "required_gain": args.min_blend_gain,
            })
            print(f"blend rejected {candidate[1]} w={candidate[2]} "
                  f"gain={candidate[0] - current:+.4f} "
                  f"< {args.min_blend_gain}", flush=True)
            break
        current = candidate[0]
        members.append((candidate[1], candidate[2]))
        trace.append({
            "members": [[t, w] for t, w in members],
            "validation_top1": current,
        })
        print(f"blend += {candidate[1]} w={candidate[2]} "
              f"val_top1={current:.4f}", flush=True)

    blend_scores = {
        split: sum(
            weight * normalised[tag][split] for tag, weight in members
        )
        for split in ("validation", "test")
    }
    blend_metrics = {
        split: evaluate(
            blend_scores[split].astype(np.float32), labels[rows[split]],
            group_sizes[split], decisions[split], features, rows[split],
            sem_cols, pick_sets, action_types,
        )
        for split in ("validation", "test")
    }
    print(json.dumps({"blend": {
        "members": [[t, w] for t, w in members],
        "val_top1": round(blend_metrics["validation"]["top1"], 4),
        "test_top1": round(blend_metrics["test"]["top1"], 4),
        "test_top2": round(blend_metrics["test"]["top2"], 4),
        "test_top3": round(blend_metrics["test"]["top3"], 4),
        "test_turn_set": round(blend_metrics["test"]["turn_set"], 4),
    }}, ensure_ascii=False), flush=True)

    exported = []
    if not args.no_export:
        # Refit every deployed member on train+validation+test, matching the
        # v31/v32 convention of shipping a model fitted on the frozen corpus
        # after the held-out estimate has been recorded.
        every = np.arange(len(groups), dtype=np.int64)
        all_rows = rows_for(groups, every)
        all_groups = groups.astype(int)
        all_episodes = episode_ids
        ordered_all = np.unique(all_episodes)
        position_all = {
            int(e): i / max(len(ordered_all) - 1, 1)
            for i, e in enumerate(ordered_all)
        }
        multiplier_all = np.asarray(
            [
                floor + (1.0 - floor) * position_all[int(e)] ** power
                for e in all_episodes
            ],
            dtype=np.float32,
        )
        full_weight = weights[all_rows] * np.repeat(
            multiplier_all, all_groups
        )
        for index, (tag, weight) in enumerate(members):
            entry = variants[tag]
            matrix, cols = entry["_matrix"], entry["_cols"]
            y = graded[all_rows] if entry["use_graded_labels"] else labels
            config = CONFIGS[entry["model_name"]]
            params: dict[str, Any] = dict(
                objective=config["objective"], metric="ndcg",
                num_leaves=config["num_leaves"],
                n_estimators=entry["best_iteration"],
                learning_rate=config["learning_rate"],
                min_child_samples=config["min_child_samples"], max_depth=-1,
                subsample=0.9, subsample_freq=1,
                colsample_bytree=config["colsample_bytree"],
                reg_alpha=0.2, reg_lambda=1.0, random_state=entry["seed"],
                n_jobs=20, verbosity=-1,
            )
            if entry["use_graded_labels"]:
                params["label_gain"] = LABEL_GAIN
            final = lgb.LGBMRanker(**params)
            kwargs: dict[str, Any] = {
                "X": matrix[all_rows], "y": y, "group": all_groups,
                "sample_weight": full_weight, "feature_name": cols,
            }
            if config["categorical_ids"]:
                kwargs["categorical_feature"] = [
                    i for i, n in enumerate(cols)
                    if n in BASE_CATEGORICAL or n.endswith("_id")
                ]
            final.fit(**kwargs)
            compact = compact_booster(final.booster_, "ranker")
            compact.update({
                "ensemble_weight": float(weight),
                "ensemble_role": tag,
                "temperature": 1.0,
                "fallback_probability": 0.0,
                "fallback_margin": 0.0,
                "action_type_map": ACTION_TYPE_MAP,
                "legal_option_only": True,
                "runtime_scope": "v33_yushin_turn_order_ranker",
                "uses_turn_features": bool(entry["use_turn_features"]),
                "training_decisions": int(len(groups)),
                "training_candidate_rows": int(len(labels)),
                "teacher_team": args.teacher_team,
                "teacher_submission_id": args.teacher_submission_id,
                "teacher_trajectories": int(len(np.unique(episode_ids))),
                "teacher_cohorts": corpus_report.get("cohorts", {}),
                "training_recency_weight": {
                    "floor": floor,
                    "power": power,
                    "episode_order": "ascending_episode_id",
                },
                "label_definition": (
                    "turn_order_graded_relevance"
                    if entry["use_graded_labels"] else "binary_chosen_action"
                ),
                "baseline": "v29_runtime_choice_and_raw_ranker_score",
            })
            filename = (
                "ranker_model.json" if index == 0
                else f"ranker_model_{index}.json"
            )
            path = args.agent_dir / filename
            path.write_text(
                json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            exported.append({
                "file": filename,
                "role": tag,
                "weight": float(weight),
                "iterations": entry["best_iteration"],
                "bytes": path.stat().st_size,
                "uses_turn_features": bool(entry["use_turn_features"]),
            })
            print(f"exported {filename} ({path.stat().st_size} bytes)",
                  flush=True)

    report = {
        "cache": str(args.cache.resolve()),
        "agent_dir": str(args.agent_dir.resolve()),
        "split_decisions": {
            k: int(len(v)) for k, v in decisions.items()
        },
        "train_episodes": int(len(np.unique(episode_ids[decisions["train"]]))),
        "graded_label_counts": graded_counts,
        "recency": {"floor": floor, "power": power},
        "variants": {
            tag: {
                key: value for key, value in entry.items()
                if not key.startswith("_")
            }
            for tag, entry in variants.items()
        },
        "blend_selection_trace": trace,
        "blend": {
            "members": [[t, w] for t, w in members],
            "validation": blend_metrics["validation"],
            "test": blend_metrics["test"],
        },
        "exported": exported,
        "target_top1": 0.90,
        "target_met": bool(blend_metrics["test"]["top1"] >= 0.90),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
