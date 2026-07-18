from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


AGENT = Path(__file__).resolve().parent


def _runtime_module():
    sys.path.insert(0, str(AGENT))
    try:
        spec = importlib.util.spec_from_file_location("alakazam_v6_shadow_runtime_test", AGENT / "ml_runtime.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(AGENT))


def test_live_override_is_opt_in(monkeypatch):
    monkeypatch.delenv("ALAKAZAM_ML_ENABLE_OVERRIDE", raising=False)
    runtime = _runtime_module().HybridRanker()
    assert runtime.model is not None
    assert runtime.enable_override is False
    assert runtime.snapshot()["runtime_scope"] == "shadow_guarded_v3_base"


def test_experiment_can_explicitly_enable_override(monkeypatch):
    monkeypatch.setenv("ALAKAZAM_ML_ENABLE_OVERRIDE", "1")
    runtime = _runtime_module().HybridRanker()
    assert runtime.enable_override is True
