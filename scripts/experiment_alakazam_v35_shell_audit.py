"""Score the deployed safety shell, not just the ranker underneath it.

Every reported number from v31 to v34 is the ranker's agreement with the
teacher over all scoped decisions. That is not what the agent plays. The v31
shell discards the ranker's pick and replays the v29 baseline whenever one of
six guards fires, and no version has measured whether those substitutions move
the played action towards the teacher or away from it.

Each guard reads features the corpus already stores, so the shell can be
replayed exactly from the cached candidate matrix plus a set of ranker scores.
That makes the audit deterministic, free of replay I/O, and able to score
alternative shells without rebuilding an agent.

Guard order mirrors ``ml_runtime.HybridRanker.choose``. Two details are easy to
get wrong and change which decisions a guard claims. The ``fallback_context``
the guards receive is the *v29 baseline* row (``v29_selected``), not the pure
deterministic policy row (``fallback_selected``). And the lethal estimate
means a Powerful Hand that KOs the opposing *active*, not one that wins the
game, so the lethal guard fires several times per game rather than once.

The reported ``played_*`` metrics are the ones the ladder sees: agreement of
the action the agent actually returns, with the same ordering/divergence split
the residual analysis uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_alakazam_v33_teacher import (  # noqa: E402
    ACTION_TYPE_MAP,
    ACTION_TYPES,
    ranges,
    rows_for,

)

DUDUNSPARCE_CARD_ID = 66
# Action classes that can spend the turn's single Supporter and so make a
# deferred Boss route unplayable. ``trainer`` covers both Items and Supporters,
# and the corpus has no column separating them, so this over-blocks slightly.
SUPPORTER_SLOT = frozenset(
    ACTION_TYPE_MAP[name] for name in ("trainer", "xerosic", "hammer")
)

# ``lethal`` and ``boss`` name how far each guard reaches.
#   always          v31-v34 behaviour: veto the ranker outright
#   end_only        veto only when the ranker wants to end the turn, which is
#                   the case the guard's stated rationale actually covers
#   end_or_weak     also veto a ranker attack that is not itself lethal
#   end_or_trainer  (boss only) also veto a ranker pick that could consume the
#                   turn's single Supporter slot and strand the Boss route
#   off             no guard
SHELLS: dict[str, dict[str, Any]] = {
    "v34": {"lethal": "always", "boss": "always", "ko": True,
            "note": "deployed v31-v34 shell"},
    "lethal_end_only": {"lethal": "end_only", "boss": "always", "ko": True},
    "lethal_end_or_weak": {"lethal": "end_or_weak", "boss": "always",
                           "ko": True},
    "boss_end_only": {"lethal": "always", "boss": "end_only", "ko": True},
    "boss_off": {"lethal": "always", "boss": "off", "ko": True},
    "boss_end_or_trainer": {"lethal": "always", "boss": "end_or_trainer",
                            "ko": True},
    "lethal_end_or_weak_boss_end_only": {
        "lethal": "end_or_weak", "boss": "end_only", "ko": True},
    "lethal_end_or_weak_boss_end_or_trainer": {
        "lethal": "end_or_weak", "boss": "end_or_trainer", "ko": True},
    "lethal_end_or_weak_boss_off": {
        "lethal": "end_or_weak", "boss": "off", "ko": True},
    "lethal_end_only_boss_end_only": {
        "lethal": "end_only", "boss": "end_only", "ko": True},
    "ko_reachable_only": {"lethal": "end_or_weak", "boss": "end_only",
                          "ko": "reachable"},
    "no_ko_guard": {"lethal": "end_or_weak", "boss": "end_only", "ko": False},
    "unguarded": {"lethal": "off", "boss": "off", "ko": False,
                  "note": "upper bound: the ranker plays every decision"},
}


def guard_reason(top, baseline, features, columns, attack_available, shell):
    """The first guard that fires, or None when the ranker's pick stands."""
    action = int(features[top, columns["action_type"]])
    baseline_action = int(features[baseline, columns["action_type"]])
    lethal_available = bool(
        features[baseline, columns["attack_lethal_estimate"]]
    )

    # Every guard falls back to the same baseline action, so guard order only
    # changes attribution, never the played action. It is kept faithful to the
    # runtime anyway: the deployed lethal guard is checked before the ranker
    # is consulted at all, while the narrowed forms need the ranker's pick.
    if shell["lethal"] == "always" and lethal_available:
        return "lethal_guard"
    if action == ACTION_TYPE_MAP["other"]:
        return "unmodelled_other"
    if features[top, columns["breaks_current_ko_estimate"]] and (
        shell["ko"] is True
        or (shell["ko"] == "reachable" and attack_available)
    ):
        return "breaks_current_ko"

    if baseline_action == ACTION_TYPE_MAP["boss"] and (
        action != ACTION_TYPE_MAP["boss"]
    ):
        if shell["boss"] == "always" or (
            shell["boss"] in ("end_only", "end_or_trainer")
            and action == ACTION_TYPE_MAP["end"]
        ) or (
            shell["boss"] == "end_or_trainer" and action in SUPPORTER_SLOT
        ):
            return "preserve_fallback_boss_route"

    if (
        action == ACTION_TYPE_MAP["end"]
        and attack_available
        and int(features[top, columns["has_ready_active_alakazam"]]) == 1
    ):
        return "end_with_ready_attack"
    if (
        action == ACTION_TYPE_MAP["ability"]
        and int(features[top, columns["candidate_card_id"]])
        == DUDUNSPARCE_CARD_ID
        and int(features[top, columns["self_board_count"]]) <= 2
    ):
        return "dudunsparce_body_floor"

    if lethal_available:
        if shell["lethal"] in ("end_only", "end_or_weak") and (
            action == ACTION_TYPE_MAP["end"]
        ):
            return "lethal_declined_by_end"
        if (
            shell["lethal"] == "end_or_weak"
            and action == ACTION_TYPE_MAP["attack"]
            and not features[top, columns["attack_lethal_estimate"]]
        ):
            return "lethal_declined_by_weak_attack"
    return None


def audit(scores, labels, features, columns, row_index, group_sizes,
          decisions, action_types, sem_cols, pick_sets, shell):
    starts, ends = ranges(np.asarray(group_sizes))
    totals: Counter[str] = Counter()
    by_reason: dict[str, Counter] = {}
    by_class: dict[int, Counter] = {}

    for local, (a, b) in enumerate(zip(starts, ends)):
        block_rows = row_index[a:b]
        order = np.argsort(-scores[a:b], kind="stable")
        top = int(block_rows[order[0]])
        lab = labels[a:b]
        teacher = int(block_rows[int(np.flatnonzero(lab == 1)[0])])

        hits = np.flatnonzero(
            features[block_rows, columns["v29_selected"]] == 1
        )
        if len(hits) == 0:
            totals["baseline_missing"] += 1
            baseline = top
        else:
            baseline = int(block_rows[int(hits[0])])

        attack_available = bool(np.any(
            features[block_rows, columns["action_type"]]
            == ACTION_TYPE_MAP["attack"]
        ))
        reason = guard_reason(
            top, baseline, features, columns, attack_available, shell
        )
        played = baseline if reason is not None else top

        model_ok = top == teacher
        played_ok = played == teacher
        picks = pick_sets[int(decisions[local])]
        played_in_turn = played_ok or tuple(
            features[played, sem_cols].tolist()
        ) in picks

        action = int(action_types[decisions[local]])
        bucket = by_class.setdefault(action, Counter())
        for target in (totals, bucket):
            target["count"] += 1
            target["model_correct"] += int(model_ok)
            target["played_correct"] += int(played_ok)
            target["baseline_correct"] += int(baseline == teacher)
            target["played_turn_set"] += int(played_in_turn)
            target["blocked"] += int(reason is not None)
            target["regret"] += int(
                reason is not None and model_ok and not played_ok
            )
            target["save"] += int(
                reason is not None and played_ok and not model_ok
            )

        if reason is None:
            continue
        stats = by_reason.setdefault(reason, Counter())
        stats["count"] += 1
        stats["model_correct"] += int(model_ok)
        stats["played_correct"] += int(played_ok)
        stats["model_equals_baseline"] += int(top == baseline)

    def summarise(stats):
        n = max(1, stats["count"])
        return {
            "count": int(stats["count"]),
            "model_top1": stats["model_correct"] / n,
            "played_top1": stats["played_correct"] / n,
            "played_turn_set": stats["played_turn_set"] / n,
            "baseline_top1": stats["baseline_correct"] / n,
            "blocked_rate": stats["blocked"] / n,
            "regret": int(stats["regret"]),
            "save": int(stats["save"]),
            "net": int(stats["save"] - stats["regret"]),
        }

    return {
        "overall": summarise(totals),
        "by_guard": {
            reason: {
                "count": int(stats["count"]),
                "model_would_have_matched": (
                    stats["model_correct"] / max(1, stats["count"])
                ),
                "played_matched": (
                    stats["played_correct"] / max(1, stats["count"])
                ),
                "no_op_share": (
                    stats["model_equals_baseline"] / max(1, stats["count"])
                ),
                "net_decisions": int(
                    stats["played_correct"] - stats["model_correct"]
                ),
            }
            for reason, stats in sorted(
                by_reason.items(), key=lambda kv: kv[1]["count"], reverse=True
            )
        },
        "by_teacher_action": {
            ACTION_TYPES[action]: summarise(stats)
            for action, stats in sorted(by_class.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--shells", nargs="+", default=list(SHELLS))
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        labels = cached["labels"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
        action_types = cached["teacher_action_types"]
        names = cached["feature_names"].astype(str).tolist()
    with np.load(args.scores, allow_pickle=False) as cached:
        score_blocks = {
            split: cached[split] for split in args.splits if split in cached
        }

    from scripts.train_alakazam_v33_teacher import (
        turn_blocks, turn_pick_sets,
    )
    blocks = turn_blocks(features, groups, episode_ids, names)
    pick_sets, sem_cols = turn_pick_sets(
        features, labels, groups, blocks, names
    )

    columns = {
        name: names.index(name)
        for name in (
            "action_type", "breaks_current_ko_estimate",
            "attack_lethal_estimate", "has_ready_active_alakazam",
            "self_board_count", "candidate_card_id", "v29_selected",
        )
    }

    report: dict[str, Any] = {}
    for shell_name in args.shells:
        shell = SHELLS[shell_name]
        report[shell_name] = {"config": shell}
        for split, scores in score_blocks.items():
            decisions = np.flatnonzero(splits == split)
            row_index = rows_for(groups, decisions)
            report[shell_name][split] = audit(
                scores, labels[row_index], features, columns, row_index,
                groups[decisions].astype(int), decisions, action_types,
                sem_cols, pick_sets, shell,
            )
        summary = {
            split: {
                key: round(report[shell_name][split]["overall"][key], 4)
                for key in ("played_top1", "played_turn_set", "blocked_rate")
            }
            for split in score_blocks
        }
        print(json.dumps({shell_name: summary}), flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
