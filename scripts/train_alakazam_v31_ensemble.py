"""Fit the validation-selected v31 two-ranker ensemble on all Majkel games."""

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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

import train_alakazam_v31_teacher as teacher  # noqa: E402
from ml.core.distill import compact_booster  # noqa: E402


def _fit(
    arrays: dict[str, Any],
    names: list[str],
    *,
    iterations: int,
    categorical_ids: bool,
    num_leaves: int,
    min_child_samples: int,
    colsample_bytree: float,
    seed: int,
) -> lgb.LGBMRanker:
    decisions = np.arange(len(arrays["groups"]), dtype=np.int64)
    x, y, weights, groups = teacher._select_decisions(arrays, decisions)
    categorical = [
        index
        for index, name in enumerate(names)
        if name in teacher.BASE_CATEGORICAL or name.endswith("_id")
    ] if categorical_ids else []
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=iterations,
        learning_rate=0.025 if num_leaves == 255 else 0.03,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )
    model.fit(
        x,
        y,
        group=groups,
        sample_weight=weights,
        feature_name=names,
        categorical_feature=categorical,
    )
    return model


def _write(
    model: lgb.LGBMRanker,
    path: Path,
    *,
    role: str,
    weight: float,
    trajectories: int,
    decisions: int,
) -> int:
    compact = compact_booster(model.booster_, "ranker")
    compact.update({
        "temperature": 1.0,
        "fallback_probability": 0.0,
        "fallback_margin": 0.0,
        "action_type_map": teacher.ACTION_TYPE_MAP,
        "legal_option_only": True,
        "runtime_scope": "v31_majkel_two_ranker_ensemble",
        "ensemble_role": role,
        "ensemble_weight": weight,
        "training_decisions": decisions,
        "teacher_trajectories": trajectories,
        "teacher_cohorts": {"majkel_full": trajectories},
        "baseline": "v29_runtime_choice_and_raw_ranker_score",
    })
    path.write_text(
        json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("schema_cache", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.schema_cache, allow_pickle=False) as schema:
        names = schema["feature_names"].astype(str).tolist()
    with np.load(args.cache, allow_pickle=False) as cached:
        original_names = cached["feature_names"].astype(str).tolist()
        columns = [original_names.index(name) for name in names]
        arrays: dict[str, Any] = {
            "features": cached["features"][:, columns],
            "labels": cached["labels"],
            "weights": cached["weights"],
            "groups": cached["groups"],
        }
        trajectories = len(np.unique(cached["episode_ids"]))
    large = _fit(
        arrays,
        names,
        iterations=410,
        categorical_ids=True,
        num_leaves=255,
        min_child_samples=55,
        colsample_bytree=0.80,
        seed=1086,
    )
    numeric = _fit(
        arrays,
        names,
        iterations=524,
        categorical_ids=False,
        num_leaves=127,
        min_child_samples=35,
        colsample_bytree=0.88,
        seed=305,
    )
    args.agent_dir.mkdir(parents=True, exist_ok=True)
    large_path = args.agent_dir / "ranker_model.json"
    numeric_path = args.agent_dir / "ranker_numeric_model.json"
    report = {
        "features": len(names),
        "teacher_trajectories": trajectories,
        "teacher_decisions": len(arrays["groups"]),
        "ensemble": [
            {
                "path": str(large_path.resolve()),
                "role": "categorical_large_leaf",
                "weight": 1.0,
                "iterations": 410,
                "bytes": _write(
                    large,
                    large_path,
                    role="categorical_large_leaf",
                    weight=1.0,
                    trajectories=trajectories,
                    decisions=len(arrays["groups"]),
                ),
            },
            {
                "path": str(numeric_path.resolve()),
                "role": "numeric_id_diversifier",
                "weight": 1.3,
                "iterations": 524,
                "bytes": _write(
                    numeric,
                    numeric_path,
                    role="numeric_id_diversifier",
                    weight=1.3,
                    trajectories=trajectories,
                    decisions=len(arrays["groups"]),
                ),
            },
        ],
        "selection_evidence": str(
            (
                ROOT
                / "experiments"
                / "alakazam_ml_v31"
                / "majkel_ranker_ensemble.json"
            ).resolve()
        ),
        "heldout_semantic_top1": 0.7630988023952096,
        "heldout_semantic_top2_reference": 0.895,
        "target_top1": 0.90,
        "target_met": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
