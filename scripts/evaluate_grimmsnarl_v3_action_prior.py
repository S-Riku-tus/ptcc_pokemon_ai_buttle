"""Select a conservative elite-action-prior blend on validation only.

The unconstrained action prior improves agreement with the five strongest
teachers but deliberately moves away from the v2.1 pilot.  This evaluator
chooses the strongest blend whose v2-pilot validation loss is at most the
configured guard, then scores both chronological test blocks once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_grimmsnarl_v3_action_prior import (  # noqa: E402
    ELITE,
    PINNED_V2_TEACHER,
    _base_scores,
    _blend,
    _blocks,
    _decision_matrix,
    _metrics,
)
from scripts.train_grimmsnarl_v2_teacher import Corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--action-model", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--max-pinned-validation-loss", type=float, default=0.005,
    )
    args = parser.parse_args()

    source = json.loads(args.training_report.read_text(encoding="utf-8"))
    state_count = int(source["invariant_state_features"])
    state_names = source["feature_names"][:state_count]
    classes = np.asarray(source["classes"], dtype=np.int32)
    best_iteration = int(source["best_iteration"])

    corpus = Corpus(args.corpus)
    corpus.resplit_per_team(0.12, 0.12)
    corpus.add_team_feature()
    invariant = np.asarray(
        [corpus.names.index(name) for name in state_names], dtype=np.int64
    )
    base_model = lgb.Booster(model_file=str(args.base_model))
    action_model = lgb.Booster(model_file=str(args.action_model))

    teams = {
        "elite": set(ELITE),
        "pinned": {PINNED_V2_TEACHER},
    }
    blocks = {
        cohort: {
            split: _blocks(corpus, split, ids)
            for split in ("validation", "test")
        }
        for cohort, ids in teams.items()
    }
    scores: dict[str, dict[str, np.ndarray]] = {
        cohort: {} for cohort in teams
    }
    probabilities: dict[str, dict[str, np.ndarray]] = {
        cohort: {} for cohort in teams
    }
    for cohort in teams:
        for split in ("validation", "test"):
            decisions = blocks[cohort][split]
            base = _base_scores(
                base_model, corpus, decisions, PINNED_V2_TEACHER
            )
            matrix, _ = _decision_matrix(
                corpus, decisions, base, invariant
            )
            scores[cohort][split] = base
            probabilities[cohort][split] = np.asarray(
                action_model.predict(
                    matrix, num_iteration=best_iteration
                ),
                dtype=np.float32,
            )

    alpha_grid = (
        0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12,
        0.15, 0.20, 0.30, 0.50, 0.75, 1.0,
    )
    baseline = {
        cohort: _metrics(
            corpus, blocks[cohort]["validation"],
            scores[cohort]["validation"],
        )["top1"]
        for cohort in teams
    }
    validation_runs: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        row: dict[str, Any] = {"alpha": alpha}
        for cohort in teams:
            decisions = blocks[cohort]["validation"]
            blended = _blend(
                corpus, decisions, scores[cohort]["validation"],
                probabilities[cohort]["validation"], classes, alpha,
            )
            top1 = _metrics(corpus, decisions, blended)["top1"]
            row[f"{cohort}_top1"] = top1
            row[f"{cohort}_delta"] = round(
                top1 - baseline[cohort], 4
            )
        row["guard_pass"] = bool(
            row["pinned_delta"] >= -args.max_pinned_validation_loss
        )
        validation_runs.append(row)

    eligible = [
        row for row in validation_runs
        if row["guard_pass"] and row["elite_delta"] > 0
    ]
    if not eligible:
        raise SystemExit("no blend improves elite validation within guard")
    selected = max(
        eligible,
        key=lambda row: (row["elite_top1"], row["pinned_top1"], -row["alpha"]),
    )
    alpha = float(selected["alpha"])

    test: dict[str, Any] = {}
    for cohort in teams:
        decisions = blocks[cohort]["test"]
        base = _metrics(corpus, decisions, scores[cohort]["test"])
        blended_scores = _blend(
            corpus, decisions, scores[cohort]["test"],
            probabilities[cohort]["test"], classes, alpha,
        )
        blended = _metrics(corpus, decisions, blended_scores)
        test[cohort] = {
            "base": base,
            "conservative_prior": blended,
            "delta": round(blended["top1"] - base["top1"], 4),
        }

    result = {
        "method": "guarded elite action-prior blend",
        "source_training_report": str(args.training_report.resolve()),
        "elite_teams": list(ELITE),
        "pinned_v2_teacher": PINNED_V2_TEACHER,
        "selection_rule": (
            "maximum elite validation Top-1 subject to pinned-pilot "
            f"loss <= {args.max_pinned_validation_loss:.4f}"
        ),
        "max_pinned_validation_loss": args.max_pinned_validation_loss,
        "validation_baseline": baseline,
        "validation_runs": validation_runs,
        "selected": selected,
        "test": test,
        "test_read_once_after_guarded_validation_selection": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected": selected,
        "test": {
            cohort: {
                "base": values["base"]["top1"],
                "prior": values["conservative_prior"]["top1"],
                "delta": values["delta"],
            }
            for cohort, values in test.items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
