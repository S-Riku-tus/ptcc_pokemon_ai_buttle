"""Extract the v33 candidate corpus from the enlarged same-teacher index.

Two things differ from the v32 extraction.

Split boundaries are pinned to explicit episode IDs instead of percentiles, so
the 289 newly recovered games can only enter training and the frozen
validation/test episodes stay bit-identical to the ones v32 reported on. Every
new episode is chronologically older than the boundary, so this is a pure
training-set enlargement.

Eight intra-turn columns are appended. They describe what the acting player has
already been offered and passed over during the current turn, which the v32
feature set could not express because it scores every candidate as a pure
function of the current observation. They are derived from the agent's own
decision stream and are therefore reproducible at inference time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from train_alakazam_v31_teacher import _extract_chunk  # noqa: E402

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


def build_turn_state(features, labels, groups, episode_ids, names):
    """Intra-turn offer/pass history, in the order the teacher acted."""
    starts, ends = ranges(groups)
    sem_cols = [names.index(n) for n in SEMANTIC]
    cls_cols = [names.index(n) for n in ("action_type", "candidate_card_id")]
    i_turn = names.index("turn")

    extra = np.zeros((len(labels), len(TURN_FEATURES)), dtype=np.float32)
    seen_cand: dict[tuple, tuple[int, int, int]] = {}
    seen_cls: dict[tuple, tuple[int, int]] = {}
    prev_offered: set[tuple] = set()
    current_key = None
    position = 0

    for decision in range(len(groups)):
        a, b = starts[decision], ends[decision]
        key = (int(episode_ids[decision]), int(features[a, i_turn]))
        if key != current_key:
            current_key = key
            seen_cand = {}
            seen_cls = {}
            prev_offered = set()
            position = 0
        pos = a + int(np.flatnonzero(labels[a:b] == 1)[0])
        chosen_sem = tuple(features[pos, sem_cols].tolist())
        chosen_cls = tuple(features[pos, cls_cols].tolist())
        # Count each distinct semantic candidate once per decision. The
        # runtime can only cheaply deduplicate raw options by this same key,
        # so both sides must agree or the columns drift apart in play.
        offered_now: dict[tuple, tuple] = {}
        for row in range(a, b):
            sem = tuple(features[row, sem_cols].tolist())
            cls = tuple(features[row, cls_cols].tolist())
            offered_now.setdefault(sem, cls)
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
        for sem, cls in offered_now.items():
            offers, passed, first = seen_cand.get(sem, (0, 0, -1))
            seen_cand[sem] = (
                offers + 1,
                passed + int(sem != chosen_sem),
                first if first >= 0 else position,
            )
            c_offers, c_passed = seen_cls.get(cls, (0, 0))
            seen_cls[cls] = (c_offers + 1, c_passed + int(cls != chosen_cls))
        prev_offered = set(offered_now)
        position += 1
    return extra


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher_index", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--validation-min-episode", type=int, default=88002730,
        help="First episode ID of the frozen v32 validation block.",
    )
    parser.add_argument(
        "--test-min-episode", type=int, default=88050020,
        help="First episode ID of the frozen v32 test block.",
    )
    args = parser.parse_args()

    with args.teacher_index.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    # One acting seat per episode keeps each decision stream contiguous, which
    # the intra-turn history depends on.
    unique: dict[int, dict[str, str]] = {}
    for row in rows:
        unique.setdefault(int(row["episode_id"]), row)
    rows = [unique[key] for key in sorted(unique)]

    for row in rows:
        episode_id = int(row["episode_id"])
        row["split"] = (
            "test" if episode_id >= args.test_min_episode
            else "validation" if episode_id >= args.validation_min_episode
            else "train"
        )
    split_counts = Counter(row["split"] for row in rows)
    print(f"trajectories={len(rows)} splits={dict(split_counts)}", flush=True)

    workers = min(max(1, args.workers), len(rows))
    chunks = [rows[index::workers] for index in range(workers)]
    agent_dir = str(args.agent_dir.resolve())
    if workers == 1:
        parts = [_extract_chunk(agent_dir, chunks[0])]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            parts = list(
                executor.map(_extract_chunk, [agent_dir] * workers, chunks)
            )

    names = parts[0]["feature_names"]
    if any(part["feature_names"] != names for part in parts):
        raise RuntimeError("worker feature schemas differ")
    arrays = {
        key: np.concatenate([part[key] for part in parts])
        for key in (
            "features", "labels", "weights", "groups", "fallback_correct",
            "teacher_action_types", "episode_ids", "ranks",
        )
    }
    arrays["splits"] = np.asarray(
        sum((part["splits"] for part in parts), [])
    )

    extra = build_turn_state(
        arrays["features"], arrays["labels"], arrays["groups"],
        arrays["episode_ids"], names,
    )
    features = np.concatenate([arrays["features"], extra], axis=1)
    all_names = names + list(TURN_FEATURES)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        labels=arrays["labels"],
        weights=arrays["weights"],
        groups=arrays["groups"],
        splits=arrays["splits"],
        fallback_correct=arrays["fallback_correct"],
        teacher_action_types=arrays["teacher_action_types"],
        episode_ids=arrays["episode_ids"],
        ranks=arrays["ranks"],
        feature_names=np.asarray(all_names),
    )

    stats: Counter[str] = Counter()
    for part in parts:
        stats.update(part["stats"])
    split_values = arrays["splits"]
    report = {
        "teacher_index": str(args.teacher_index.resolve()),
        "cache": str(args.output.resolve()),
        "trajectories": len(rows),
        "trajectory_splits": dict(split_counts),
        "features": len(all_names),
        "base_features": len(names),
        "turn_features": list(TURN_FEATURES),
        "decisions": int(len(arrays["groups"])),
        "candidate_rows": int(len(arrays["labels"])),
        "split_decisions": {
            split: int(np.count_nonzero(split_values == split))
            for split in ("train", "validation", "test")
        },
        "validation_min_episode": args.validation_min_episode,
        "test_min_episode": args.test_min_episode,
        "extraction_stats": dict(stats),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
