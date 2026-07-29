"""Diagnose whether the imitation ceiling is temporal or episode-contextual."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_alakazam_v31_teacher as teacher  # noqa: E402


def _metrics(
    model: Any,
    arrays: dict[str, Any],
    decisions: np.ndarray,
) -> dict[str, Any]:
    scores, labels, groups = teacher._predict_for_decisions(
        model, arrays, decisions
    )
    result, _ = teacher._evaluate(
        scores,
        labels,
        groups,
        arrays["fallback_correct"][decisions],
        arrays["teacher_action_types"][decisions],
        1.0,
    )
    return result


def _random_episode_split(
    episode_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes = rng.permutation(np.unique(episode_ids))
    train_end = int(len(episodes) * 0.70)
    validation_end = int(len(episodes) * 0.80)
    train_set = set(episodes[:train_end].tolist())
    validation_set = set(episodes[train_end:validation_end].tolist())
    train = np.flatnonzero(np.asarray([
        episode in train_set for episode in episode_ids
    ]))
    validation = np.flatnonzero(np.asarray([
        episode in validation_set for episode in episode_ids
    ]))
    test = np.flatnonzero(np.asarray([
        episode not in train_set and episode not in validation_set
        for episode in episode_ids
    ]))
    return train, validation, test


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("schema_cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.schema_cache, allow_pickle=False) as schema:
        desired = schema["feature_names"].astype(str).tolist()
    with np.load(args.cache, allow_pickle=False) as cached:
        names = cached["feature_names"].astype(str).tolist()
        columns = [names.index(name) for name in desired]
        arrays: dict[str, Any] = {
            "features": cached["features"][:, columns],
        }
        for key in (
            "labels",
            "weights",
            "groups",
            "fallback_correct",
            "teacher_action_types",
            "episode_ids",
        ):
            arrays[key] = cached[key]
        chronological = cached["splits"].astype(str)
    rng = np.random.default_rng(741)
    random_episode = _random_episode_split(arrays["episode_ids"], rng)
    shuffled = rng.permutation(len(arrays["groups"]))
    train_end = int(len(shuffled) * 0.70)
    validation_end = int(len(shuffled) * 0.80)
    random_decision = (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )
    split_sets = {
        "chronological_episode": (
            np.flatnonzero(chronological == "train"),
            np.flatnonzero(chronological == "validation"),
            np.flatnonzero(chronological == "test"),
        ),
        "random_episode": random_episode,
        "random_decision_same_episode_leakage": random_decision,
    }
    report = {}
    for name, (train, validation, test) in split_sets.items():
        model = teacher._fit(
            arrays,
            desired,
            train,
            n_estimators=900,
            validation_indices=validation,
        )
        report[name] = {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "best_iteration": int(model.best_iteration_ or 900),
            "validation_metrics": _metrics(model, arrays, validation),
            "test_metrics": _metrics(model, arrays, test),
        }
        print({
            "split": name,
            "validation_top1": report[name]["validation_metrics"][
                "semantic_top1"
            ],
            "test_top1": report[name]["test_metrics"]["semantic_top1"],
        }, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
