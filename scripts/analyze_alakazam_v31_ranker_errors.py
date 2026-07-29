"""Summarize held-out semantic mistakes for one cached v31 ranker corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_alakazam_v31_teacher as teacher  # noqa: E402


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    return starts, ends


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument(
        "--schema-cache",
        type=Path,
        help="Optionally restrict features to this cache's feature schema.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    desired_names = None
    if args.schema_cache is not None:
        with np.load(args.schema_cache, allow_pickle=False) as schema:
            desired_names = schema["feature_names"].astype(str).tolist()
    with np.load(args.cache, allow_pickle=False) as cached:
        cached_names = cached["feature_names"].astype(str).tolist()
        names = cached_names if desired_names is None else desired_names
        columns = [cached_names.index(name) for name in names]
        arrays: dict[str, Any] = {
            "features": cached["features"][:, columns],
        }
        arrays.update({
            key: cached[key]
            for key in (
                "labels",
                "weights",
                "groups",
                "fallback_correct",
                "teacher_action_types",
                "episode_ids",
                "ranks",
            )
        })
        splits = cached["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    model = teacher._fit(
        arrays,
        names,
        train,
        n_estimators=900,
        validation_indices=validation,
    )
    scores = model.predict(arrays["features"]).astype(np.float32)
    starts, ends = _ranges(arrays["groups"])
    column = {name: index for index, name in enumerate(names)}
    pairs: Counter[str] = Counter()
    action_confusion: Counter[str] = Counter()
    by_turn: Counter[str] = Counter()
    by_options: Counter[str] = Counter()
    examples = []
    correct = 0
    for decision in test:
        start, end = starts[decision], ends[decision]
        predicted = int(np.argmax(scores[start:end]))
        teacher_local = int(
            np.flatnonzero(arrays["labels"][start:end] == 1)[0]
        )
        correct += int(predicted == teacher_local)
        if predicted == teacher_local:
            continue
        teacher_row = arrays["features"][start + teacher_local]
        predicted_row = arrays["features"][start + predicted]
        teacher_action = teacher.ACTION_TYPES[
            int(teacher_row[column["action_type"]])
        ]
        predicted_action = teacher.ACTION_TYPES[
            int(predicted_row[column["action_type"]])
        ]
        teacher_card = int(teacher_row[column["candidate_card_id"]])
        predicted_card = int(predicted_row[column["candidate_card_id"]])
        teacher_target = int(teacher_row[column["candidate_target_id"]])
        predicted_target = int(predicted_row[column["candidate_target_id"]])
        pair = (
            f"{teacher_action}:{teacher_card}:{teacher_target}"
            f" <- {predicted_action}:{predicted_card}:{predicted_target}"
        )
        pairs[pair] += 1
        action_confusion[f"{teacher_action} <- {predicted_action}"] += 1
        turn = int(teacher_row[column["turn"]])
        turn_bin = "early" if turn <= 4 else "mid" if turn <= 10 else "late"
        by_turn[turn_bin] += 1
        option_count = end - start
        option_bin = (
            "2-4"
            if option_count <= 4
            else "5-8"
            if option_count <= 8
            else "9-15"
            if option_count <= 15
            else "16+"
        )
        by_options[option_bin] += 1
        if len(examples) < 100:
            order = np.argsort(-scores[start:end], kind="stable")[:5]
            examples.append({
                "episode_id": int(arrays["episode_ids"][decision]),
                "turn": turn,
                "self_active_id": int(
                    teacher_row[column["self_active_id"]]
                ),
                "opp_active_id": int(
                    teacher_row[column["opp_active_id"]]
                ),
                "hand_count": int(
                    teacher_row[column["self_hand_count"]]
                ),
                "deck_count": int(
                    teacher_row[column["self_deck_count"]]
                ),
                "options": [
                    {
                        "rank": rank + 1,
                        "score": float(scores[start + local]),
                        "teacher": bool(
                            arrays["labels"][start + local] == 1
                        ),
                        "action": teacher.ACTION_TYPES[
                            int(arrays["features"][
                                start + local, column["action_type"]
                            ])
                        ],
                        "card_id": int(arrays["features"][
                            start + local, column["candidate_card_id"]
                        ]),
                        "target_id": int(arrays["features"][
                            start + local, column["candidate_target_id"]
                        ]),
                        "target_hp": int(arrays["features"][
                            start + local, column["candidate_target_hp"]
                        ]),
                    }
                    for rank, local in enumerate(order)
                ],
            })
    errors = len(test) - correct
    report = {
        "cache": str(args.cache.resolve()),
        "test_decisions": int(len(test)),
        "top1": correct / len(test),
        "errors": errors,
        "top_action_confusions": action_confusion.most_common(50),
        "top_semantic_pairs": pairs.most_common(100),
        "errors_by_turn": dict(by_turn),
        "errors_by_option_count": dict(by_options),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "top1": report["top1"],
        "errors": errors,
        "top_action_confusions": report["top_action_confusions"][:15],
        "top_semantic_pairs": report["top_semantic_pairs"][:20],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
