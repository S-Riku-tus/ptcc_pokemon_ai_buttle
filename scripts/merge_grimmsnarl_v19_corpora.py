"""Merge Grimmsnarl corpora with newest episode/seat relations taking priority.

The v19 fit keeps the broad 21-pilot mechanics corpus, then replaces stale
copies with the current-top-four corpus and finally with the refreshed
high-rated trajectories.  Deduplication is at the trajectory boundary, never
at a decision row, so intra-turn history remains intact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    loaded = [np.load(path, allow_pickle=False) for path in args.input]
    try:
        names = [str(value) for value in loaded[0]["feature_names"]]
        categorical = [str(value) for value in loaded[0]["categorical"]]
        for path, data in zip(args.input[1:], loaded[1:]):
            if [str(value) for value in data["feature_names"]] != names:
                raise ValueError(f"feature schema mismatch: {path}")
            if [str(value) for value in data["categorical"]] != categorical:
                raise ValueError(f"categorical schema mismatch: {path}")

        # Later inputs replace older copies of the same trajectory.
        owner: dict[tuple[int, int], int] = {}
        relation_counts: list[dict[tuple[int, int], int]] = []
        for source, data in enumerate(loaded):
            keys = [
                (int(episode), int(seat))
                for episode, seat in zip(data["episode_ids"], data["seats"])
            ]
            counts: dict[tuple[int, int], int] = {}
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
                owner[key] = source
            relation_counts.append(counts)

        decision_parts: dict[str, list[np.ndarray]] = {
            key: [] for key in (
                "groups", "episode_ids", "team_ids", "submission_ids",
                "seats", "turns", "contexts", "won",
                "teacher_action_types",
            )
        }
        feature_parts: list[np.ndarray] = []
        label_parts: list[np.ndarray] = []
        source_report: list[dict[str, object]] = []
        for source, (path, data) in enumerate(zip(args.input, loaded)):
            starts, ends = ranges(data["groups"])
            keep = np.asarray([
                owner[(int(episode), int(seat))] == source
                for episode, seat in zip(data["episode_ids"], data["seats"])
            ])
            decisions = np.flatnonzero(keep)
            rows = np.concatenate([
                np.arange(starts[index], ends[index]) for index in decisions
            ]) if len(decisions) else np.zeros(0, dtype=np.int64)
            feature_parts.append(data["features"][rows])
            label_parts.append(data["labels"][rows])
            for key in decision_parts:
                if key == "submission_ids" and key not in data.files:
                    decision_parts[key].append(
                        np.full(len(decisions), -1, dtype=np.int64)
                    )
                else:
                    decision_parts[key].append(data[key][decisions])
            all_relations = set(relation_counts[source])
            kept_relations = {
                key for key in all_relations if owner[key] == source
            }
            source_report.append({
                "path": str(path.resolve()),
                "input_relations": len(all_relations),
                "kept_relations": len(kept_relations),
                "replaced_relations": len(all_relations - kept_relations),
                "kept_decisions": int(len(decisions)),
                "kept_candidate_rows": int(len(rows)),
            })

        features = np.concatenate(feature_parts)
        labels = np.concatenate(label_parts)
        arrays = {
            key: np.concatenate(parts) for key, parts in decision_parts.items()
        }
        order = np.lexsort((arrays["seats"], arrays["episode_ids"]))
        if not np.array_equal(order, np.arange(len(order))):
            starts, ends = ranges(arrays["groups"])
            row_order = np.concatenate([
                np.arange(starts[index], ends[index]) for index in order
            ])
            features = features[row_order]
            labels = labels[row_order]
            arrays = {key: value[order] for key, value in arrays.items()}

        # The trainer's per-team split is authoritative.  A valid global
        # chronological split is retained for tools that inspect the cache
        # directly.
        unique_episodes = np.unique(arrays["episode_ids"])
        validation_min = unique_episodes[int(len(unique_episodes) * 0.76)]
        test_min = unique_episodes[int(len(unique_episodes) * 0.88)]
        splits = np.where(
            arrays["episode_ids"] >= test_min,
            "test",
            np.where(
                arrays["episode_ids"] >= validation_min,
                "validation",
                "train",
            ),
        )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            features=features,
            labels=labels,
            splits=splits,
            feature_names=np.asarray(names),
            categorical=np.asarray(categorical),
            **arrays,
        )
        report = {
            "inputs_oldest_to_newest": source_report,
            "deduplication_key": ["episode_id", "seat_index"],
            "relations": len(owner),
            "episodes": int(len(np.unique(arrays["episode_ids"]))),
            "teams": sorted(int(value) for value in np.unique(arrays["team_ids"])),
            "decisions": int(len(arrays["groups"])),
            "candidate_rows": int(len(labels)),
            "features": len(names),
            "validation_min_episode": int(validation_min),
            "test_min_episode": int(test_min),
            "output": str(args.output.resolve()),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        for data in loaded:
            data.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
