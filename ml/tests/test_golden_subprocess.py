from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_golden_states_in_isolated_process():
    suite = Path(__file__).with_name("golden_states_suite.py")
    result = subprocess.run(
        [sys.executable, str(suite)],
        cwd=suite.parents[2],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
