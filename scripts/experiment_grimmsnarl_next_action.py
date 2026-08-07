"""Rerank Grimmsnarl candidates with a decision-level next-action model.

The production ranker scores candidates independently.  Most of its misses on
high-rated pilots are actions that the teacher performs later in the same turn,
so this experiment exposes the *whole offered set* to a second-stage model.
Model and blend selection use validation only; the chronological test block is
read once after the winning configuration is fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_grimmsnarl_v2_teacher import (  # noqa: E402
    Corpus,
    select_decisions,
)


def _key(row: np.ndarray, columns: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(row[column]) for column in columns)


def _mode_columns(names: list[str], mode: str) -> tuple[int, ...]:
    fields = {
        "action": ("action_type_id",),
        "action_card": ("action_type_id", "candidate_card_id"),
        "action_card_attack": (
            "action_type_id", "candidate_card_id", "candidate_attack_id",
        ),
    }[mode]
    return tuple(names.index(name) for name in fields)


def _blocks(corpus: Corpus, decisions: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(corpus.starts[decision]), int(corpus.ends[decision]))
        for decision in decisions
    ]


def _chosen_rows(corpus: Corpus, decisions: np.ndarray) -> np.ndarray:
    rows = []
    for start, end in _blocks(corpus, decisions):
        chosen = np.flatnonzero(corpus.labels[start:end] == 1)
        if len(chosen) != 1:
            raise RuntimeError(f"decision has {len(chosen)} chosen rows")
        rows.append(start + int(chosen[0]))
    return np.asarray(rows, dtype=np.int64)


def _state_columns(corpus: Corpus, train: np.ndarray) -> np.ndarray:
    """Keep columns invariant inside every observed training decision."""
    # ``Corpus.add_team_feature`` appends the runtime-only teacher column to
    # names; the stored feature matrix deliberately remains at its base width.
    base_width = int(corpus.features.shape[1])
    varying = np.zeros(base_width, dtype=bool)
    for start, end in _blocks(corpus, train):
        if end - start > 1:
            varying |= np.any(
                corpus.features[start + 1:end] != corpus.features[start],
                axis=0,
            )
    # These fields are candidate-valued even when a small teacher slice never
    # happened to offer two different values.  Excluding them prevents the raw
    # first option from becoming an accidental set representation.
    candidate_exact = {
        "option_type", "action_type_id", "boss_opp_bench_value",
        "boss_opp_bench_low_hp_value", "retreat_active_damage_value",
    }
    candidate_prefixes = (
        "candidate_", "target_", "evolve_", "energy_target_",
    )
    for index, name in enumerate(corpus.names[:base_width]):
        if name in candidate_exact or name.startswith(candidate_prefixes):
            varying[index] = True
    return np.flatnonzero(~varying)


def _base_scores(
    corpus: Corpus,
    booster: lgb.Booster,
    decisions: np.ndarray,
    pin_team: int,
) -> np.ndarray:
    matrix = corpus.matrix(decisions, pin_team=pin_team)
    try:
        return booster.predict(matrix).astype(np.float32)
    finally:
        del matrix


def _normalise(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = scores.copy().astype(np.float32)
    offset = 0
    for size in groups:
        end = offset + int(size)
        block = out[offset:end]
        block -= float(block.mean())
        block /= max(float(block.std()), 1e-5)
        offset = end
    return out


def _class_map(
    corpus: Corpus,
    train: np.ndarray,
    columns: tuple[int, ...],
) -> dict[tuple[int, ...], int]:
    keys = {
        _key(corpus.features[row], columns)
        for row in _chosen_rows(corpus, train)
    }
    return {key: index for index, key in enumerate(sorted(keys))}


def _decision_matrix(
    corpus: Corpus,
    decisions: np.ndarray,
    scores: np.ndarray,
    state_columns: np.ndarray,
    class_map: dict[tuple[int, ...], int],
    key_columns: tuple[int, ...],
    team_code: int,
) -> np.ndarray:
    """State plus count/max/mass/gap summaries for each offered class."""
    classes = len(class_map)
    blocks = _blocks(corpus, decisions)
    first_rows = np.asarray([start for start, _ in blocks], dtype=np.int64)
    state = corpus.features[first_rows][:, state_columns]
    out = np.zeros((len(decisions), classes * 4 + 5), dtype=np.float32)
    score_offset = 0
    for slot, (start, end) in enumerate(blocks):
        size = end - start
        block = scores[score_offset:score_offset + size].astype(np.float64)
        top = float(block.max())
        exp = np.exp(np.clip(block - top, -50.0, 0.0))
        probability = exp / max(float(exp.sum()), 1e-12)
        unknown = 0
        for local, row in enumerate(range(start, end)):
            cls = class_map.get(_key(corpus.features[row], key_columns))
            if cls is None:
                unknown += 1
                continue
            offset = cls * 4
            out[slot, offset] += 1
            out[slot, offset + 1] = max(
                out[slot, offset + 1], float(block[local])
            )
            out[slot, offset + 2] += float(probability[local])
        for cls in range(classes):
            offset = cls * 4
            out[slot, offset + 3] = out[slot, offset + 1] - top
        order = np.sort(block)[::-1]
        margin = float(order[0] - order[1]) if len(order) > 1 else 50.0
        entropy = float(-(
            probability * np.log(np.maximum(probability, 1e-12))
        ).sum())
        out[slot, -5:] = (top, margin, entropy, size, unknown)
        score_offset += size
    team = np.full((len(decisions), 1), team_code, dtype=np.float32)
    return np.ascontiguousarray(np.hstack((state, team, out)))


def _targets(
    corpus: Corpus,
    decisions: np.ndarray,
    class_map: dict[tuple[int, ...], int],
    columns: tuple[int, ...],
) -> np.ndarray:
    return np.asarray([
        class_map.get(_key(corpus.features[row], columns), -1)
        for row in _chosen_rows(corpus, decisions)
    ], dtype=np.int32)


def _probabilities(model: lgb.LGBMClassifier, matrix: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(matrix)
    out = np.full((len(matrix), int(max(model.classes_)) + 1), 1e-8)
    for column, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, column]
    return out


def _score(
    corpus: Corpus,
    decisions: np.ndarray,
    base: np.ndarray,
    probabilities: np.ndarray,
    class_map: dict[tuple[int, ...], int],
    columns: tuple[int, ...],
    alpha: float,
) -> dict[str, float | int]:
    base = _normalise(base, corpus.groups[decisions])
    correct = base_correct = class_correct = unknown_targets = 0
    score_offset = 0
    for slot, (start, end) in enumerate(_blocks(corpus, decisions)):
        size = end - start
        labels = corpus.labels[start:end]
        target = int(np.flatnonzero(labels == 1)[0])
        target_class = class_map.get(
            _key(corpus.features[start + target], columns)
        )
        if target_class is None:
            unknown_targets += 1
        block = base[score_offset:score_offset + size].astype(np.float64)
        prior = np.empty(size, dtype=np.float64)
        candidate_classes: list[int | None] = []
        for local, row in enumerate(range(start, end)):
            cls = class_map.get(_key(corpus.features[row], columns))
            candidate_classes.append(cls)
            prior[local] = (
                probabilities[slot, cls] if cls is not None else 1e-8
            )
        base_pick = int(np.argmax(block))
        pick = int(np.argmax(block + alpha * np.log(np.maximum(prior, 1e-8))))
        base_correct += int(base_pick == target)
        correct += int(pick == target)
        class_correct += int(
            target_class is not None and candidate_classes[pick] == target_class
        )
        score_offset += size
    total = max(len(decisions), 1)
    return {
        "decisions": int(len(decisions)),
        "top1": correct / total,
        "base_top1": base_correct / total,
        "picked_class_accuracy": class_correct / total,
        "unknown_targets": unknown_targets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--team", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-model", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    corpus = Corpus(args.corpus)
    corpus.resplit_per_team(0.12, 0.12)
    corpus.add_team_feature()
    if args.team not in corpus.team_codes:
        raise SystemExit(f"team {args.team} is absent from corpus")
    decisions = {
        split: select_decisions(corpus, split, {args.team}, None)
        for split in ("train", "validation", "test")
    }
    booster = lgb.Booster(model_file=str(args.base_model))
    if booster.feature_name() != corpus.names:
        raise SystemExit("base model and corpus feature schemas differ")
    scores = {
        split: _base_scores(corpus, booster, block, args.team)
        for split, block in decisions.items()
    }
    state_columns = _state_columns(corpus, decisions["train"])
    team_code = corpus.team_codes[args.team]
    validation_runs: list[dict[str, Any]] = []
    fitted: dict[tuple[str, int], tuple[lgb.LGBMClassifier, dict, tuple]] = {}
    configurations = (
        {"num_leaves": 31, "min_child_samples": 35, "max_depth": 9},
        {"num_leaves": 63, "min_child_samples": 25, "max_depth": 12},
    )
    alphas = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)
    for mode in ("action", "action_card", "action_card_attack"):
        columns = _mode_columns(corpus.names, mode)
        class_map = _class_map(corpus, decisions["train"], columns)
        train_y = _targets(corpus, decisions["train"], class_map, columns)
        validation_y = _targets(
            corpus, decisions["validation"], class_map, columns
        )
        train_x = _decision_matrix(
            corpus, decisions["train"], scores["train"], state_columns,
            class_map, columns, team_code,
        )
        validation_x = _decision_matrix(
            corpus, decisions["validation"], scores["validation"],
            state_columns, class_map, columns, team_code,
        )
        known_validation = validation_y >= 0
        for config_index, config in enumerate(configurations):
            model = lgb.LGBMClassifier(
                objective="multiclass", n_estimators=700,
                learning_rate=0.04, subsample=0.9, subsample_freq=1,
                colsample_bytree=0.8, reg_alpha=0.2, reg_lambda=1.0,
                random_state=8100 + config_index, n_jobs=16, verbosity=-1,
                **config,
            )
            model.fit(
                train_x, train_y,
                eval_set=[(
                    validation_x[known_validation],
                    validation_y[known_validation],
                )],
                callbacks=[lgb.early_stopping(60, verbose=False)],
            )
            probability = _probabilities(model, validation_x)
            for alpha in alphas:
                metrics = _score(
                    corpus, decisions["validation"], scores["validation"],
                    probability, class_map, columns, alpha,
                )
                validation_runs.append({
                    "mode": mode, "config": config_index,
                    "classes": len(class_map), "alpha": alpha,
                    "best_iteration": int(model.best_iteration_ or 700),
                    **metrics,
                })
            fitted[(mode, config_index)] = (model, class_map, columns)
        print(json.dumps({
            "mode": mode,
            "best": max(
                (row for row in validation_runs if row["mode"] == mode),
                key=lambda row: (row["top1"], -row["alpha"]),
            ),
        }), flush=True)
        del train_x, validation_x

    selected = max(
        validation_runs,
        key=lambda row: (
            row["top1"], row["picked_class_accuracy"],
            -row["alpha"], -row["config"],
        ),
    )
    model, class_map, columns = fitted[
        (selected["mode"], selected["config"])
    ]
    test_x = _decision_matrix(
        corpus, decisions["test"], scores["test"], state_columns,
        class_map, columns, team_code,
    )
    test_probability = _probabilities(model, test_x)
    test = _score(
        corpus, decisions["test"], scores["test"], test_probability,
        class_map, columns, selected["alpha"],
    )
    report = {
        "method": "decision-level offered-set next-action reranker",
        "corpus": str(args.corpus.resolve()),
        "base_model": str(args.base_model.resolve()),
        "team": args.team,
        "split": "per-team chronological 76/12/12",
        "selection_rule": "maximum validation strict Top-1; test read once",
        "state_features": int(len(state_columns)),
        "validation_runs": validation_runs,
        "selected": selected,
        "test": test,
        "target_top1": 0.90,
        "target_met": bool(test["top1"] >= 0.90),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.output_model:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(
            str(args.output_model),
            num_iteration=int(model.best_iteration_ or 700),
        )
    print(json.dumps({"selected": selected, "test": test}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
