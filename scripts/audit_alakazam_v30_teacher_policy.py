"""Audit teacher consistency and replay-memory upper bounds for v30.

The report separates three different claims that must not be conflated:

* exact replay recall: can a policy answer a scene copied from its memory?
* canonical consistency: do serial-independent identical scenes agree?
* episode-held-out retrieval: does that memory transfer to unseen games?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
V30_DIR = ROOT / "agents" / "alakazam" / "alakazam_ml_v30"
sys.path.insert(0, str(V30_DIR))

from evaluate_alakazam_teacher_imitation import _display, _label
from teacher_memory import (
    semantic_action_key,
    semantic_option_key,
    teacher_memory_keys,
)


UNORDERED_ZONES = {"hand", "discard", "prize"}


def _normalise(
    value: Any,
    *,
    drop_serial: bool,
    parent_key: str = "",
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalise(
                child,
                drop_serial=drop_serial,
                parent_key=str(key),
            )
            for key, child in sorted(value.items())
            if not (drop_serial and key == "serial")
        }
    if isinstance(value, list):
        children = [
            _normalise(child, drop_serial=drop_serial)
            for child in value
        ]
        if parent_key in UNORDERED_ZONES:
            children.sort(
                key=lambda child: json.dumps(
                    child,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return children
    return value


def _hash(value: Any) -> str:
    return hashlib.blake2b(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8"),
        digest_size=12,
    ).hexdigest()


def _option_labels(observation: dict[str, Any]) -> list[str]:
    options = (observation.get("select") or {}).get("option") or []
    return [
        _display(_label(observation, [index], level="semantic"))
        for index in range(len(options))
    ]


def _keys(observation: dict[str, Any]) -> tuple[str, str, str]:
    exact_key, canonical_key = teacher_memory_keys(observation)

    select = observation.get("select") or {}
    canonical_select = {
        "type": int(select.get("type", -1)),
        "context": int(select.get("context", -1)),
        "minCount": int(select.get("minCount") or 0),
        "maxCount": int(select.get("maxCount") or 0),
        "remainDamageCounter": int(
            select.get("remainDamageCounter") or 0
        ),
        "remainEnergyCost": _normalise(
            select.get("remainEnergyCost"),
            drop_serial=True,
        ),
        "contextCard": _normalise(
            select.get("contextCard"),
            drop_serial=True,
        ),
        "effect": _normalise(
            select.get("effect"),
            drop_serial=True,
        ),
        "options": sorted(
            semantic_option_key(observation, index)
            for index in range(
                len((select.get("option") or []))
            )
        ),
    }
    public_state = {
        "current": _normalise(
            observation.get("current") or {},
            drop_serial=True,
        ),
        "select": canonical_select,
    }
    state_key = _hash(public_state)

    return exact_key, canonical_key, state_key


def _read_replay(
    row: dict[str, str],
    archives: dict[str, zipfile.ZipFile],
) -> dict[str, Any]:
    storage_path = row["storage_path"]
    if row["storage_type"] == "zip":
        archive = archives.get(storage_path)
        if archive is None:
            archive = zipfile.ZipFile(storage_path)
            archives[storage_path] = archive
        return json.loads(archive.read(row["replay_path"]))
    return json.loads(
        (Path(storage_path) / row["replay_path"]).read_text(
            encoding="utf-8",
        )
    )


def _iter_decisions(
    rows: Iterable[dict[str, str]],
) -> Iterable[dict[str, Any]]:
    archives: dict[str, zipfile.ZipFile] = {}
    try:
        for row in rows:
            replay = _read_replay(row, archives)
            seat = int(row["seat_index"])
            steps = replay.get("steps") or []
            for step_index, step in enumerate(steps[:-1]):
                if seat >= len(step) or seat >= len(steps[step_index + 1]):
                    continue
                record = step[seat] or {}
                observation = record.get("observation") or {}
                select = observation.get("select")
                if select is None or record.get("status") != "ACTIVE":
                    continue
                recorded = (
                    steps[step_index + 1][seat] or {}
                ).get("action")
                if (
                    not isinstance(recorded, list)
                    or len(recorded) == 60
                    or any(not isinstance(index, int) for index in recorded)
                ):
                    continue
                options = select.get("option") or []
                if any(not 0 <= index < len(options) for index in recorded):
                    continue
                exact_key, canonical_key, state_key = _keys(observation)
                semantic = semantic_action_key(observation, recorded)
                intent = _display(
                    _label(observation, recorded, level="intent")
                )
                yield {
                    "episode_id": int(row["episode_id"]),
                    "cohort": row["source_cohort"],
                    "rank": int(row["leaderboard_rank"]),
                    "context": int(select.get("context", -1)),
                    "select_type": int(select.get("type", -1)),
                    "option_count": len(options),
                    "semantic": semantic,
                    "intent": intent,
                    "exact_key": exact_key,
                    "canonical_key": canonical_key,
                    "state_key": state_key,
                }
    finally:
        for archive in archives.values():
            archive.close()


def _majority_metrics(
    groups: dict[str, Counter[str]],
) -> dict[str, Any]:
    decisions = sum(sum(labels.values()) for labels in groups.values())
    correct = sum(max(labels.values()) for labels in groups.values())
    repeated = {
        key: labels
        for key, labels in groups.items()
        if sum(labels.values()) > 1
    }
    repeated_decisions = sum(sum(labels.values()) for labels in repeated.values())
    repeated_correct = sum(max(labels.values()) for labels in repeated.values())
    conflicting = {
        key: labels
        for key, labels in repeated.items()
        if len(labels) > 1
    }
    conflict_decisions = sum(
        sum(labels.values()) for labels in conflicting.values()
    )
    return {
        "keys": len(groups),
        "decisions": decisions,
        "majority_accuracy": correct / decisions if decisions else None,
        "repeated_keys": len(repeated),
        "repeated_decisions": repeated_decisions,
        "repeated_majority_accuracy": (
            repeated_correct / repeated_decisions
            if repeated_decisions
            else None
        ),
        "conflicting_keys": len(conflicting),
        "conflict_decisions": conflict_decisions,
        "top_conflicts": [
            {
                "key": key,
                "decisions": sum(labels.values()),
                "labels": labels.most_common(),
            }
            for key, labels in sorted(
                conflicting.items(),
                key=lambda item: sum(item[1].values()),
                reverse=True,
            )[:30]
        ],
    }


def _split_episodes(
    rows: list[dict[str, str]],
) -> tuple[set[int], set[int]]:
    by_cohort: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_cohort[row["source_cohort"]].add(int(row["episode_id"]))
    train: set[int] = set()
    test: set[int] = set()
    for episodes in by_cohort.values():
        ordered = sorted(episodes)
        boundary = max(1, int(len(ordered) * 0.8))
        train.update(ordered[:boundary])
        test.update(ordered[boundary:])
    return train, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher_index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.teacher_index.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    train_episodes, test_episodes = _split_episodes(rows)

    counts: Counter[str] = Counter()
    by_context: dict[str, Counter[str]] = defaultdict(Counter)
    groups: dict[str, dict[str, Counter[str]]] = {
        name: defaultdict(Counter)
        for name in ("exact", "canonical", "state")
    }
    train_memory: dict[str, dict[str, Counter[str]]] = {
        name: defaultdict(Counter)
        for name in ("exact", "canonical", "state")
    }
    heldout: dict[str, Counter[str]] = {
        name: Counter()
        for name in ("exact", "canonical", "state")
    }

    decisions = list(_iter_decisions(rows))
    for decision in decisions:
        counts["decisions"] += 1
        counts[f"context_{decision['context']}"] += 1
        counts[f"intent_{decision['intent']}"] += 1
        by_context[str(decision["context"])][decision["semantic"]] += 1
        for name in ("exact", "canonical", "state"):
            key = decision[f"{name}_key"]
            groups[name][key][decision["semantic"]] += 1
            if decision["episode_id"] in train_episodes:
                train_memory[name][key][decision["semantic"]] += 1

    for decision in decisions:
        if decision["episode_id"] not in test_episodes:
            continue
        for name in ("exact", "canonical", "state"):
            memory = train_memory[name].get(decision[f"{name}_key"])
            heldout[name]["decisions"] += 1
            if not memory:
                continue
            heldout[name]["hits"] += 1
            prediction = memory.most_common(1)[0][0]
            if prediction == decision["semantic"]:
                heldout[name]["correct"] += 1

    report = {
        "teacher_index": str(args.teacher_index.resolve()),
        "trajectories": len(rows),
        "decisions": counts["decisions"],
        "train_episodes": len(train_episodes),
        "test_episodes": len(test_episodes),
        "contexts": {
            key.removeprefix("context_"): value
            for key, value in counts.items()
            if key.startswith("context_")
        },
        "teacher_intents": {
            key.removeprefix("intent_"): value
            for key, value in counts.most_common()
            if key.startswith("intent_")
        },
        "memory_upper_bounds": {
            name: _majority_metrics(value)
            for name, value in groups.items()
        },
        "chronological_episode_holdout_retrieval": {
            name: {
                "decisions": value["decisions"],
                "hits": value["hits"],
                "coverage": (
                    value["hits"] / value["decisions"]
                    if value["decisions"]
                    else None
                ),
                "correct": value["correct"],
                "accuracy_on_hits": (
                    value["correct"] / value["hits"]
                    if value["hits"]
                    else None
                ),
                "overall_accuracy": (
                    value["correct"] / value["decisions"]
                    if value["decisions"]
                    else None
                ),
            }
            for name, value in heldout.items()
        },
        "top_actions_by_context": {
            context: labels.most_common(30)
            for context, labels in sorted(
                by_context.items(),
                key=lambda item: int(item[0]),
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "trajectories": report["trajectories"],
        "decisions": report["decisions"],
        "memory": {
            name: {
                "majority_accuracy": value["majority_accuracy"],
                "repeated_decisions": value["repeated_decisions"],
                "conflict_decisions": value["conflict_decisions"],
            }
            for name, value in report["memory_upper_bounds"].items()
        },
        "holdout": report["chronological_episode_holdout_retrieval"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
