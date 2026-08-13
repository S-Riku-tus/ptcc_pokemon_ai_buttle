"""Like-for-like v32/v33 prediction comparison on their shared holdout.

v31 and v32 were measured against different teachers, so their published Top-1
numbers were never like-for-like. The v33 corpus keeps the v32 validation and
test episodes bit-identical and only enlarges training, which makes a real
comparison possible.

Two different things are reported, and they must not be mixed up.

``held_out``
    The honest generalisation numbers. Both sides come from models fitted on
    training episodes only: v32 from its stored holdout scores, v33 from its
    training report. This is the comparison to quote.

``shipped_artifact_integrity``
    v31, v32 and v33 all ship a model refitted on the frozen corpus after the
    held-out estimate is recorded, so the shipped file has seen these
    decisions. Walking it with the submission runtime's pure-Python scorer
    must therefore reproduce them almost perfectly. This only proves the
    exported JSON and the walker agree with the trained model; it is not a
    generalisation measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.distill import tree_score  # noqa: E402

SEMANTIC = (
    "option_type", "candidate_card_id", "candidate_attack_id",
    "candidate_target_id", "candidate_inplay_area",
)


def ranges(groups):
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def turn_pick_sets(features, labels, groups, episode_ids, decisions, names):
    starts, ends = ranges(groups)
    sem = [names.index(name) for name in SEMANTIC]
    i_turn = names.index("turn")
    blocks: dict[tuple[int, int], list[int]] = {}
    for decision in decisions:
        key = (
            int(episode_ids[decision]),
            int(features[starts[decision], i_turn]),
        )
        blocks.setdefault(key, []).append(decision)
    per_decision = {}
    for members in blocks.values():
        picks = set()
        for decision in members:
            a, b = starts[decision], ends[decision]
            pos = a + int(np.flatnonzero(labels[a:b] == 1)[0])
            picks.add(tuple(features[pos, sem].tolist()))
        for decision in members:
            per_decision[decision] = picks
    return per_decision, sem


def score(scores_by_decision, features, labels, groups, decisions,
          pick_sets, sem):
    starts, ends = ranges(groups)
    t1 = t2 = t3 = turn_set = 0
    for decision in decisions:
        a, b = starts[decision], ends[decision]
        values = np.asarray(scores_by_decision[decision])
        order = np.argsort(-values, kind="stable")
        lab = labels[a:b]
        correct = bool(lab[order[0]] == 1)
        t1 += int(correct)
        t2 += int(bool(np.any(lab[order[:2]] == 1)))
        t3 += int(bool(np.any(lab[order[:3]] == 1)))
        key = tuple(features[a + int(order[0]), sem].tolist())
        turn_set += int(correct or key in pick_sets[decision])
    n = len(decisions)
    return {
        "decisions": int(n), "top1": t1 / n, "top2": t2 / n,
        "top3": t3 / n, "turn_set": turn_set / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v33-cache", type=Path, required=True)
    parser.add_argument("--v33-agent", type=Path, required=True)
    parser.add_argument("--v33-report", type=Path, required=True)
    parser.add_argument("--v32-cache", type=Path, required=True)
    parser.add_argument("--v32-scores", type=Path, required=True)
    parser.add_argument(
        "--v32-score-row", type=int, default=5,
        help="Row of test_scores holding v32's deployed recency tree.",
    )
    parser.add_argument(
        "--integrity-episodes", type=int, default=40,
        help="Holdout episodes to re-score with the pure-Python walker.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.v33_cache, allow_pickle=False) as cached:
        features = cached["features"]
        labels = cached["labels"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
        names = cached["feature_names"].astype(str).tolist()
    starts, ends = ranges(groups)
    test = np.flatnonzero(splits == "test")
    pick_sets, sem = turn_pick_sets(
        features, labels, groups, episode_ids, test, names
    )

    with np.load(args.v32_cache, allow_pickle=False) as cached:
        v32_splits = cached["splits"].astype(str)
        v32_groups = cached["groups"]
        v32_episode_ids = cached["episode_ids"]
        v32_features = cached["features"]
        v32_labels = cached["labels"]
        v32_names = cached["feature_names"].astype(str).tolist()
    v32_test = np.flatnonzero(v32_splits == "test")
    v32_picks, v32_sem = turn_pick_sets(
        v32_features, v32_labels, v32_groups, v32_episode_ids,
        v32_test, v32_names,
    )
    with np.load(args.v32_scores, allow_pickle=False) as cached:
        stored = cached["test_scores"][args.v32_score_row]
        stored_groups = cached["test_groups"]
    offsets = np.r_[0, np.cumsum(stored_groups, dtype=np.int64)[:-1]]
    v32_by_decision = {
        int(decision): stored[
            offsets[local]:offsets[local] + stored_groups[local]
        ]
        for local, decision in enumerate(v32_test)
    }
    v32_held_out = score(
        v32_by_decision, v32_features, v32_labels, v32_groups,
        v32_test, v32_picks, v32_sem,
    )

    report = json.loads(args.v33_report.read_text(encoding="utf-8"))
    v33_held_out = {
        key: report["blend"]["test"][key]
        for key in ("decisions", "top1", "top2", "top3", "turn_set")
    }

    models = []
    for index in (0, 1, 2):
        name = (
            "ranker_model.json" if index == 0
            else f"ranker_model_{index}.json"
        )
        path = args.v33_agent / name
        if not path.exists():
            continue
        model = json.loads(path.read_text(encoding="utf-8"))
        models.append((model, float(model.get("ensemble_weight", 1.0)), name))

    sampled = set(
        np.unique(episode_ids[test])[:args.integrity_episodes].tolist()
    )
    subset = test[[int(episode_ids[d]) in sampled for d in test]]
    walked = {}
    for decision in subset:
        a, b = starts[decision], ends[decision]
        total = np.zeros(b - a, dtype=np.float64)
        for model, weight, _ in models:
            columns = [names.index(n) for n in model["feature_names"]]
            raw = np.asarray([
                tree_score(features[row, columns].tolist(), model)
                for row in range(a, b)
            ])
            total += weight * (raw - raw.mean()) / max(raw.std(), 1e-5)
        walked[decision] = total
    integrity = score(
        walked, features, labels, groups, subset, pick_sets, sem
    )

    output = {
        "holdout_episodes": int(len(np.unique(episode_ids[test]))),
        "holdout_decisions": int(len(test)),
        "held_out": {
            "note": (
                "Both rows come from models fitted on training episodes only."
            ),
            "v32_deployed": v32_held_out,
            "v33_deployed": v33_held_out,
            "delta_points": {
                key: (v33_held_out[key] - v32_held_out[key]) * 100
                for key in ("top1", "top2", "top3", "turn_set")
            },
        },
        "shipped_artifact_integrity": {
            "note": (
                "In-sample by construction; confirms the exported JSON and "
                "the pure-Python walker agree with the trained model."
            ),
            "episodes": args.integrity_episodes,
            **integrity,
        },
        "v33_members": [
            {"file": name, "weight": weight,
             "role": model.get("ensemble_role"),
             "trees": len(model["trees"])}
            for model, weight, name in models
        ],
        "target_top1": 0.90,
        "target_met": bool(v33_held_out["top1"] >= 0.90),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
