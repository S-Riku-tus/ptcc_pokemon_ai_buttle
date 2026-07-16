from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb

from .distill import compact_booster
from .matrix import build_matrix_store
from .matrix_train import run_matrix_training


FOCUS_FALLBACK = {"boss", "retreat", "xerosic", "hammer"}


def _harden_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema = dict(schema)
    thresholds = dict(schema.get("action_type_thresholds") or {})
    for action in FOCUS_FALLBACK:
        thresholds[action] = 1.01
    thresholds["energy"] = max(0.85, float(thresholds.get("energy", 0.0)))
    schema["action_type_thresholds"] = thresholds
    schema["fallback_probability"] = float(schema.get("fallback_probability", 0.55))
    schema["fallback_margin"] = float(schema.get("fallback_margin", 0.12))
    schema["focus_action_policy"] = {
        "boss": "always_fallback",
        "retreat": "always_fallback",
        "xerosic": "always_fallback",
        "hammer": "always_fallback_due_submission_holdout_failure",
        "energy": "ml_only_if_probability>=0.85_and_margin>=0.12",
    }
    return schema


def train_all(
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    seed: int = 741,
    confidence_threshold: float = 0.55,
) -> dict[str, Any]:
    del seed, confidence_threshold  # deterministic settings are defined in matrix_train
    if not (processed_dir / "matrix" / "matrix_schema.json").exists():
        build_matrix_store(processed_dir, progress=True)
    summary = run_matrix_training(processed_dir, models_dir, reports_dir, progress=True)

    schema_path = models_dir / "model_schema.json"
    schema = _harden_schema(json.loads(schema_path.read_text(encoding="utf-8")))
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    booster = lgb.Booster(model_file=str(models_dir / "ranker.txt"))
    compact = compact_booster(booster, "ranker")
    compact.update({
        "temperature": schema["temperature"],
        "fallback_probability": schema["fallback_probability"],
        "fallback_margin": schema["fallback_margin"],
        "action_type_thresholds": schema["action_type_thresholds"],
        "action_type_map": schema["action_type_map"],
        "focus_action_policy": schema["focus_action_policy"],
        "legal_option_only": True,
    })
    (models_dir / "ranker_model.json").write_text(
        json.dumps(compact, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    # Keep the historical filename as an alias used by existing tooling.
    (models_dir / "ranker_model.txt").write_text(
        (models_dir / "ranker.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )

    time_metrics = summary["splits"].get("time_holdout", {})
    (reports_dir / "action_type_metrics.json").write_text(
        json.dumps(time_metrics.get("action_type_metrics", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "ablation_results.json").write_text(
        json.dumps(summary.get("ablations", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # The old value head used policy state and one teacher variant only. It is
    # intentionally retired rather than silently mixing stale predictions.
    (models_dir / "value_model.json").write_text(
        json.dumps({
            "status": "disabled",
            "reason": "not used by the hybrid policy; retraining deferred until cross-team value targets are validated",
        }, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[2]
    parser.add_argument("--processed", default=str(base / "data" / "ml" / "alakazam" / "processed"))
    parser.add_argument("--models", default=str(base / "data" / "ml" / "alakazam" / "models"))
    parser.add_argument("--reports", default=str(base / "data" / "ml" / "alakazam" / "reports"))
    args = parser.parse_args()
    summary = train_all(Path(args.processed), Path(args.models), Path(args.reports))
    print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
