"""Prove the v33 runtime reproduces the corpus intra-turn columns exactly.

The eight intra-turn columns are the only v33 features that depend on the
agent's own decision history rather than on the current observation alone, so
they are the only place a train/serve skew can hide. This replays held-out
teacher episodes through the shipped ``_TurnHistory``, feeding it the recorded
actions, and compares every produced cell against the corpus cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent  # noqa: E402


def ranges(groups):
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def read_replay(row: dict[str, str], archives: dict[str, Any]) -> dict:
    path = row["storage_path"]
    if row["storage_type"] == "zip":
        archive = archives.get(path)
        if archive is None:
            archive = zipfile.ZipFile(path)
            archives[path] = archive
        return json.loads(archive.read(row["replay_path"]))
    return json.loads(
        (Path(path) / row["replay_path"]).read_text(encoding="utf-8")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("teacher_index", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _, _, main_module = load_dir_agent(args.agent_dir.resolve())
    runtime_module = sys.modules["ml_runtime"]
    turn_names = list(runtime_module.TURN_FEATURE_NAMES)
    runtime = main_module._RUNTIME

    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        groups = cached["groups"]
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"]
        names = cached["feature_names"].astype(str).tolist()
    starts, ends = ranges(groups)
    turn_columns = [names.index(name) for name in turn_names]
    semantic_columns = [
        names.index(name) for name in (
            "option_type", "candidate_card_id", "candidate_attack_id",
            "candidate_target_id", "candidate_inplay_area",
        )
    ]

    import csv
    with args.teacher_index.open(encoding="utf-8-sig", newline="") as handle:
        rows = {int(r["episode_id"]): r for r in csv.DictReader(handle)}

    held_out = [
        episode for episode in np.unique(episode_ids[splits == "test"])
    ][:args.episodes]
    by_episode: dict[int, list[int]] = {}
    for decision in range(len(groups)):
        by_episode.setdefault(int(episode_ids[decision]), []).append(decision)

    archives: dict[str, Any] = {}
    checked = mismatched = 0
    examples = []
    for episode in held_out:
        row = rows[int(episode)]
        replay = read_replay(row, archives)
        seat = int(row["seat_index"])
        steps = replay.get("steps") or []
        decisions = iter(by_episode[int(episode)])
        runtime.turn_history.reset()
        for index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[index + 1]):
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            action = (steps[index + 1][seat] or {}).get("action")
            if (
                record.get("status") != "ACTIVE"
                or int(select.get("context", -1)) != 0
                or int(select.get("type", -1)) != 0
                or int(select.get("minCount") or 0) != 1
                or int(select.get("maxCount") or 0) != 1
                or len(options) < 2
                or not isinstance(action, list)
                or len(action) != 1
                or not isinstance(action[0], int)
                or not 0 <= action[0] < len(options)
            ):
                continue

            current = observation.get("current") or {}
            keys = runtime._turn_keys(current, select, options)
            produced = runtime.turn_history.columns(current, keys)

            decision = next(decisions)
            a, b = starts[decision], ends[decision]
            expected = features[a:b][:, turn_columns]
            corpus_keys = features[a:b][:, semantic_columns]
            # Corpus rows are deduplicated representatives, which can be finer
            # grained than the intra-turn semantic key, so align by key rather
            # than by position.
            runtime_by_key = {}
            for position, (semantic, _) in enumerate(keys):
                runtime_by_key.setdefault(semantic, produced[position])
            for slot in range(len(expected)):
                key = tuple(corpus_keys[slot].tolist())
                checked += 1
                row = runtime_by_key.get(key)
                if row is None:
                    mismatched += 1
                    if len(examples) < 5:
                        examples.append({
                            "episode": int(episode),
                            "decision": int(decision),
                            "reason": "corpus candidate absent at runtime",
                            "key": [float(v) for v in key],
                        })
                    continue
                got = [float(row[name]) for name in turn_names]
                want = [float(value) for value in expected[slot]]
                if got != want:
                    mismatched += 1
                    if len(examples) < 5:
                        examples.append({
                            "episode": int(episode),
                            "decision": int(decision),
                            "candidate": slot,
                            "runtime": got,
                            "corpus": want,
                        })
            runtime.turn_history.record(current, keys, action[0])

    for archive in archives.values():
        archive.close()
    report = {
        "cache": str(args.cache),
        "agent_dir": str(args.agent_dir),
        "episodes": len(held_out),
        "candidate_rows_checked": checked,
        "mismatched": mismatched,
        "examples": examples,
        "parity": mismatched == 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if mismatched == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
