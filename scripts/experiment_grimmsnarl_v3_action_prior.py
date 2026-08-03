"""Validate an elite next-action prior over the frozen v2.1 ranker.

The deployed v2.1 model is pinned to pilot 16494330.  This experiment keeps
its concrete candidate scores, then adds a decision-level prior learned only
from rank 4/5/9/11/13.  The prior predicts the next action family from the
public state and the complete legal menu.  A validation-only alpha controls
how strongly it may reorder v2's candidates; the chronological test block is
read once after alpha selection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_grimmsnarl_v2_teacher import (  # noqa: E402
    Corpus,
    error_taxonomy,
    top1,
    wilson,
)


ELITE = (16371703, 16422241, 16463316, 16561259, 16531269)
PINNED_V2_TEACHER = 16494330
ACTION_COUNT = 17


def _blocks(corpus: Corpus, split: str, teams: set[int]) -> np.ndarray:
    return np.flatnonzero(
        (corpus.splits == split) & np.isin(corpus.team_ids, list(teams))
    )


def _invariant_columns(corpus: Corpus, decisions: np.ndarray) -> np.ndarray:
    """Columns that describe the decision, not a particular candidate."""
    varying = np.zeros(corpus.features.shape[1], dtype=bool)
    # A deterministic evenly-spaced audit covers rare late-game contexts while
    # bounding startup time. Candidate columns vary often and are found early.
    if len(decisions) > 20_000:
        audit = decisions[np.linspace(
            0, len(decisions) - 1, 20_000, dtype=np.int64
        )]
    else:
        audit = decisions
    for decision in audit:
        start, end = corpus.starts[decision], corpus.ends[decision]
        block = corpus.features[start:end]
        if len(block) > 1:
            varying |= np.any(block != block[0], axis=0)
    # Identity of the arbitrary first option is never a state feature even if
    # a thin sample happened not to vary it. Exclude candidate/option/menu
    # fields by construction and add the menu back as symmetric summaries.
    forbidden_prefixes = (
        "candidate_", "ctx_", "offered_", "option_", "retreat_to_",
        "energy_target_", "damage_target_", "boss_",
    )
    keep = [
        index
        for index, name in enumerate(
            corpus.names[:corpus.features.shape[1]]
        )
        if not varying[index]
        and name not in {
            "action_type_id", "option_type", "select_context",
            "select_type",
        }
        and not name.startswith(forbidden_prefixes)
    ]
    return np.asarray(keep, dtype=np.int64)


def _base_scores(
    booster: lgb.Booster,
    corpus: Corpus,
    decisions: np.ndarray,
    pin_team: int,
) -> np.ndarray:
    matrix = corpus.matrix(decisions, pin_team=pin_team)
    scores = booster.predict(matrix, num_iteration=booster.best_iteration)
    del matrix
    return np.asarray(scores, dtype=np.float32)


def _decision_matrix(
    corpus: Corpus,
    decisions: np.ndarray,
    scores: np.ndarray,
    invariant: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    width = len(invariant) + ACTION_COUNT * 3 + 4
    output = np.empty((len(decisions), width), dtype=np.float32)
    cursor = 0
    for slot, decision in enumerate(decisions):
        start, end = corpus.starts[decision], corpus.ends[decision]
        size = int(end - start)
        local_scores = scores[cursor:cursor + size]
        actions = np.rint(
            corpus.features[start:end, corpus.names.index("action_type_id")]
        ).astype(np.int16)
        counts = np.bincount(actions, minlength=ACTION_COUNT).astype(np.float32)
        maxima = np.full(ACTION_COUNT, -20.0, dtype=np.float32)
        means = np.full(ACTION_COUNT, -20.0, dtype=np.float32)
        for action in np.unique(actions):
            values = local_scores[actions == action]
            maxima[int(action)] = float(values.max())
            means[int(action)] = float(values.mean())
        ordered = np.sort(local_scores)
        top = float(ordered[-1])
        margin = top - float(ordered[-2]) if size > 1 else 0.0
        centred = local_scores - float(local_scores.max())
        probability = np.exp(centred)
        probability /= max(float(probability.sum()), 1e-8)
        entropy = float(-np.sum(probability * np.log(probability + 1e-8)))
        output[slot] = np.concatenate((
            corpus.features[start, invariant],
            counts,
            maxima,
            means,
            np.asarray([top, margin, entropy, size], dtype=np.float32),
        ))
        cursor += size
    names = [corpus.names[index] for index in invariant]
    names += [f"menu_action_{action}_count" for action in range(ACTION_COUNT)]
    names += [f"menu_action_{action}_max_v2" for action in range(ACTION_COUNT)]
    names += [f"menu_action_{action}_mean_v2" for action in range(ACTION_COUNT)]
    names += ["v2_top_score", "v2_margin", "v2_entropy", "menu_size"]
    return output, names


def _blend(
    corpus: Corpus,
    decisions: np.ndarray,
    base_scores: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    alpha: float,
) -> np.ndarray:
    class_column = {int(value): index for index, value in enumerate(classes)}
    action_column = corpus.names.index("action_type_id")
    output = np.empty_like(base_scores)
    cursor = 0
    for slot, decision in enumerate(decisions):
        start, end = corpus.starts[decision], corpus.ends[decision]
        size = int(end - start)
        base = base_scores[cursor:cursor + size]
        z = (base - float(base.mean())) / max(float(base.std()), 1e-5)
        actions = np.rint(
            corpus.features[start:end, action_column]
        ).astype(np.int16)
        prior = np.asarray([
            np.log(max(
                float(probabilities[slot, class_column[int(action)]]), 1e-8
            )) if int(action) in class_column else -20.0
            for action in actions
        ], dtype=np.float32)
        output[cursor:cursor + size] = z + alpha * prior
        cursor += size
    return output


def _metrics(
    corpus: Corpus,
    decisions: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    offsets = np.r_[
        0, np.cumsum(corpus.groups[decisions])[:-1]
    ].astype(np.int64)
    correct = top1(scores, corpus, decisions, offsets)
    hits = int(correct.sum())
    low, high = wilson(hits, len(decisions))
    by_team: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    by_context: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for slot, decision in enumerate(decisions):
        for key, store in (
            (int(corpus.team_ids[decision]), by_team),
            (int(corpus.contexts[decision]), by_context),
        ):
            store[key][0] += 1
            store[key][1] += int(correct[slot])

    def serialise(values: dict[int, list[int]]) -> dict[str, Any]:
        return {
            str(key): {
                "decisions": total,
                "top1": round(agreed / total, 4),
            }
            for key, (total, agreed) in sorted(values.items())
        }

    return {
        "decisions": int(len(decisions)),
        "top1": round(float(correct.mean()), 4),
        "top1_wilson95": [low, high],
        "top1_by_team": serialise(by_team),
        "top1_by_context": serialise(by_context),
        "taxonomy": error_taxonomy(
            corpus, decisions, offsets, scores, corpus.names
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=1200)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()

    corpus = Corpus(args.corpus)
    corpus.resplit_per_team(0.12, 0.12)
    corpus.add_team_feature()
    elite = set(ELITE)
    blocks = {
        split: _blocks(corpus, split, elite)
        for split in ("train", "validation", "test")
    }
    guard = {
        split: _blocks(corpus, split, {PINNED_V2_TEACHER})
        for split in ("validation", "test")
    }
    invariant = _invariant_columns(corpus, blocks["train"])
    booster = lgb.Booster(model_file=str(args.base_model))

    base: dict[str, np.ndarray] = {}
    matrices: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    for split, decisions in blocks.items():
        base[split] = _base_scores(
            booster, corpus, decisions, PINNED_V2_TEACHER
        )
        matrices[split], names = _decision_matrix(
            corpus, decisions, base[split], invariant
        )
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("decision feature schema drift")

    targets = corpus.teacher_action_types[blocks["train"]].astype(np.int32)
    validation_targets = corpus.teacher_action_types[
        blocks["validation"]
    ].astype(np.int32)
    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=args.trees,
        learning_rate=0.025,
        num_leaves=127,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=20260803,
        n_jobs=args.threads,
        verbosity=-1,
    )
    model.fit(
        matrices["train"], targets,
        eval_set=[(matrices["validation"], validation_targets)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        feature_name=feature_names,
    )
    best_iteration = int(model.best_iteration_ or args.trees)
    validation_probability = model.predict_proba(
        matrices["validation"], num_iteration=best_iteration
    )
    grid = (0.0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
    validation_runs: list[dict[str, Any]] = []
    for alpha in grid:
        scores = _blend(
            corpus, blocks["validation"], base["validation"],
            validation_probability, model.classes_, alpha,
        )
        row = {"alpha": alpha}
        row.update(_metrics(corpus, blocks["validation"], scores))
        validation_runs.append(row)
    selected = max(
        validation_runs,
        key=lambda row: (row["top1"], -row["alpha"]),
    )

    test_probability = model.predict_proba(
        matrices["test"], num_iteration=best_iteration
    )
    test_scores = _blend(
        corpus, blocks["test"], base["test"], test_probability,
        model.classes_, float(selected["alpha"]),
    )
    test = _metrics(corpus, blocks["test"], test_scores)

    # Behavioural guard: the prior is not trained on the currently pinned
    # teacher, so quantify how much it changes agreement with that teacher.
    guard_report: dict[str, Any] = {}
    for split, decisions in guard.items():
        guard_base = _base_scores(
            booster, corpus, decisions, PINNED_V2_TEACHER
        )
        guard_matrix, _ = _decision_matrix(
            corpus, decisions, guard_base, invariant
        )
        guard_probability = model.predict_proba(
            guard_matrix, num_iteration=best_iteration
        )
        guard_blend = _blend(
            corpus, decisions, guard_base, guard_probability,
            model.classes_, float(selected["alpha"]),
        )
        guard_report[split] = {
            "base": _metrics(corpus, decisions, guard_base),
            "action_prior": _metrics(corpus, decisions, guard_blend),
        }

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(
        str(args.output_model), num_iteration=best_iteration
    )
    result = {
        "method": "elite action-family prior over pinned v2.1",
        "elite_teams": list(ELITE),
        "pinned_v2_teacher": PINNED_V2_TEACHER,
        "split_mode": "chronological per-team 76/12/12",
        "fit_decisions": int(len(blocks["train"])),
        "invariant_state_features": int(len(invariant)),
        "decision_features": int(matrices["train"].shape[1]),
        "feature_names": feature_names,
        "classes": [int(value) for value in model.classes_],
        "best_iteration": best_iteration,
        "selection_rule": "maximum elite validation strict Top-1",
        "validation_runs": validation_runs,
        "selected": selected,
        "test": test,
        "guard_pinned_teacher": guard_report,
        "test_read_once_after_validation_selection": True,
        "fit_seconds": round(time.perf_counter() - started, 3),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "best_iteration": best_iteration,
        "base_validation": validation_runs[0]["top1"],
        "selected_validation": selected["top1"],
        "alpha": selected["alpha"],
        "base_test": _metrics(
            corpus, blocks["test"], base["test"]
        )["top1"],
        "selected_test": test["top1"],
        "guard": {
            split: {
                "base": values["base"]["top1"],
                "action_prior": values["action_prior"]["top1"],
            }
            for split, values in guard_report.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
