"""vfinal must be v22's policy, exactly, and provably.

The search layer measured worse than v22 in a calibrated 320-game paired arena,
so it ships disabled.  That is only worth anything if "disabled" means the
agent is byte-for-byte the champion and not a champion with a quiet behaviour
change in it, which is what these assertions pin.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AGENT = Path(__file__).resolve().parents[1]
V22 = AGENT.parent / "grimmsnarl_ml_v22"
if str(AGENT) not in sys.path:
    sys.path.insert(0, str(AGENT))

import turn_search  # noqa: E402

POLICY_FILES = (
    "deck.csv",
    "fallback_policy.py",
    "ml_features.py",
    "ml_planner.py",
    "ml_runtime.py",
    "policy_base.py",
    "ranker_model.json",
)


def test_every_policy_file_is_byte_identical_to_v22() -> None:
    for name in POLICY_FILES:
        assert (AGENT / name).read_bytes() == (V22 / name).read_bytes(), name


def test_the_search_layer_is_off() -> None:
    assert turn_search.ENABLED is False
    assert turn_search.build("deck.csv") is None


def test_the_agent_loads_with_no_search_component() -> None:
    sys.path.insert(0, str(AGENT.parents[2] / "scripts"))
    from agent_loader import load_dir_agent_module  # noqa: PLC0415

    module = load_dir_agent_module(AGENT)
    snapshot = module.diag_snapshot()
    assert snapshot["search_load_error"] is None
    assert snapshot["search"] == {}
    assert module._SEARCH is None


def test_metadata_records_the_measurement_that_disabled_it() -> None:
    meta = json.loads((AGENT / "metadata.json").read_text(encoding="utf-8"))
    evidence = meta["turn_search"]
    assert evidence["enabled"] is False
    assert evidence["arena"]["null_control_win_rate"] == 0.4917
    assert evidence["arena"]["opening_only_win_rate"] == 0.478
    assert evidence["arena"]["committed_line_win_rate"] == 0.3438
