"""End-to-end smoke test for Champion-Challenger with real agents + cg engine.

This verifies the whole pipeline runs to completion and writes artifacts. It is
NOT a performance judgement. Marked ``smoke`` so it can be deselected:

    pytest -m "not smoke"       # fast unit tests only
    pytest -m smoke             # this end-to-end check only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

CHAMPION = "alakazam_ml_v3"
CHALLENGER = "alakazam_ml_v2_expanded"

pytestmark = pytest.mark.smoke


def _cg_available() -> bool:
    try:
        import run_champion_challenger  # noqa: F401  (adds vendor to sys.path)
        from cg.game import battle_start  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _cg_available(), reason="cg engine not importable")
def test_end_to_end_writes_artifacts(tmp_path):
    import run_champion_challenger

    rc = run_champion_challenger.main(
        [
            "--champion", CHAMPION,
            "--challenger", CHALLENGER,
            "--games", "2",
            "--no-baseline",
            "--output-root", str(tmp_path),
        ]
    )
    assert rc == 0

    runs = list(tmp_path.glob("*_" + CHAMPION + "_vs_" + CHALLENGER))
    assert runs, "no run directory produced"
    run_dir = runs[0]

    for name in [
        "config_resolved.json",
        "environment.json",
        "game_results.csv",
        "seed_pair_results.csv",
        "agent_metrics.csv",
        "matchup_metrics.csv",
        "promotion_report.json",
        "promotion_report.md",
        "run.log",
    ]:
        assert (run_dir / name).exists(), f"missing artifact {name}"

    report = json.loads((run_dir / "promotion_report.json").read_text(encoding="utf-8"))
    assert report["head_to_head"]["games"] == 2
    assert report["judgement"]["verdict"] in {
        "PROMOTE_RECOMMENDED", "HOLD", "REJECT", "INVALID_EVALUATION"
    }
    # champion metadata / model must not have been altered by the evaluation
    assert report["meta"]["champion"] == CHAMPION


@pytest.mark.skipif(not _cg_available(), reason="cg engine not importable")
def test_champion_files_unchanged_after_run(tmp_path):
    """The evaluation must not mutate the Champion directory."""
    import cc_core
    import run_champion_challenger

    champ_dir = cc_core.resolve_agent_dir(CHAMPION)
    before = {
        p.name: p.stat().st_mtime
        for p in champ_dir.iterdir()
        if p.is_file()
    }
    model_hash_before = cc_core.model_hash(champ_dir)

    run_champion_challenger.main(
        [
            "--champion", CHAMPION,
            "--challenger", CHALLENGER,
            "--games", "2",
            "--no-baseline",
            "--output-root", str(tmp_path),
        ]
    )

    after = {
        p.name: p.stat().st_mtime
        for p in champ_dir.iterdir()
        if p.is_file()
    }
    assert before == after, "champion files changed during evaluation"
    assert cc_core.model_hash(champ_dir) == model_hash_before
