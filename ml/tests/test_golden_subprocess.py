from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_golden_states_in_isolated_process():
    suite = Path(__file__).with_name("golden_states_suite.py")
    legacy_agent = suite.parents[2] / "agents" / "alakazam_ml_v2_expanded"
    if not legacy_agent.is_dir():
        pytest.skip("historical alakazam_ml_v2_expanded agent is not present")
    result = subprocess.run(
        [sys.executable, str(suite)],
        cwd=suite.parents[2],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
