"""Create the v22 submission candidate from the frozen v8 champion.

The only policy change is the model's global teacher condition: team 16494330
(code 16) becomes the 1220.2-rated same-deck pilot 16371703 (code 0).  The
ranker's trees, feature schema, deck, fallback, and planner remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8"
TARGET = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v22"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {TARGET}")
    shutil.copytree(
        SOURCE,
        TARGET,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

    model_path = TARGET / "ranker_model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    assert model["teacher_team_id"] == 16494330
    assert model["teacher_team_code"] == 16
    assert "teacher_team_id" in model["feature_names"]
    model["teacher_team_id"] = 16371703
    model["teacher_team_code"] = 0
    model_path.write_text(
        json.dumps(model, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    main_path = TARGET / "main.py"
    source = main_path.read_text(encoding="utf-8")
    marker = "from __future__ import annotations"
    _old_doc, body = source.split(marker, 1)
    main_path.write_text(
        '"""Grimmsnarl ML v22: v8 mechanics pinned to the 1220 pilot.\n\n'
        "This candidate changes only the categorical teacher condition used by\n"
        "the frozen v8 ranker.  The deck, trees, features, fallback policy and\n"
        "one-ply planner are byte-identical to v8.\n"
        '"""\n\n'
        + marker + body,
        encoding="utf-8",
    )

    metadata = {
        "name": "grimmsnarl_ml_v22",
        "version": "22.0.0",
        "role": "experimental_submission_candidate",
        "archetype": "marnies_grimmsnarl_ex",
        "parent_agent": "grimmsnarl_ml_v8",
        "created_at": "2026-08-13T17:00:00+09:00",
        "deck_hash": "9714ab5c3996f6cc",
        "deck_changed": False,
        "policy": (
            "Frozen v8 policy with the entire ranker conditioned on same-deck "
            "team 16371703 (stored code 0, observed rating 1220.2) instead of "
            "the incumbent team 16494330 (stored code 16)."
        ),
        "ranker": {
            "source_agent": "grimmsnarl_ml_v8",
            "trees": len(model["trees"]),
            "features": len(model["feature_names"]),
            "teacher_team_id": 16371703,
            "teacher_team_code": 0,
            "source_teacher_team_id": 16494330,
            "source_teacher_team_code": 16,
            "trees_changed": False,
            "sha256": sha256(model_path),
        },
        "change_scope": {
            "policy_files_changed_from_v8": ["ranker_model.json"],
            "documentation_files_changed_from_v8": ["main.py", "metadata.json"],
            "identical_runtime_files": [
                "deck.csv",
                "fallback_policy.py",
                "ml_features.py",
                "ml_planner.py",
                "ml_runtime.py",
                "policy_base.py",
            ],
            "full_policy_pin_supersedes_froslass_escalation": True,
        },
        "policy_escalation": {
            "mode": "class",
            "teacher_team_id": 16371703,
            "teacher_code": 0,
            "trigger_context": 0,
            "trigger_feature": "evolve_froslass",
            "effective_runtime": (
                "no separate escalation: the global teacher pin is already code 0"
            ),
        },
        "impact_evidence": {
            "stored_games": 133,
            "decisions": 11658,
            "changed_decisions": 1647,
            "games_touched": 133,
            "changed_actions_per_game": 12.38345864661654,
            "changed_decision_rate": 0.14127637673700462,
            "gate": "LARGE_ENOUGH_TO_IMPLEMENT",
            "report": (
                "experiments/grimmsnarl_1100_diagnosis/"
                "policy_impact_v22_full.json"
            ),
        },
        "submission_package": {
            "path": "artifacts/grimmsnarl_ml_v22_submission.tar.gz",
            "bytes": 10926438,
            "sha256": (
                "c2b1097ead1c68ce689d07e5d60810470449cb4630be7e6b2d9b6ecf671fb33b"
            ),
            "archive_entries": 18,
            "extracted_import_smoke": "PASS",
            "deck_size": 60,
        },
        "known_limits": [
            "Strength is unverified until real ladder episodes are collected.",
        ],
        "description": (
            "Experimental v22 submission candidate using the same-deck "
            "1220.2-rated pilot condition."
        ),
    }
    (TARGET / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"created: {TARGET}")
    print(f"ranker_sha256: {metadata['ranker']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
