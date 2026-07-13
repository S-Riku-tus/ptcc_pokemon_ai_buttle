from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from contextlib import contextmanager
from itertools import count
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "vendor"
AGENTS_DIR = ROOT / "agents"
SHARED_BASE_DIR = AGENTS_DIR / "_base"
SHARED_POLICY_BASE_PATH = SHARED_BASE_DIR / "policy_base.py"

_LOAD_COUNTER = count()
_SHARED_POLICY_MODULE_NAME = "_shared_policy_base"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _prepend_sys_path(path: Path):
    text = str(path)
    inserted = False
    if text not in sys.path:
        sys.path.insert(0, text)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


@contextmanager
def _temporary_module_alias(name: str, module: Any):
    had_previous = name in sys.modules
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if had_previous:
            sys.modules[name] = previous
        else:
            sys.modules.pop(name, None)


def load_shared_policy_base():
    module = sys.modules.get(_SHARED_POLICY_MODULE_NAME)
    if module is not None:
        return module
    with _prepend_sys_path(VENDOR_DIR):
        return _load_module(_SHARED_POLICY_MODULE_NAME, SHARED_POLICY_BASE_PATH)


def load_dir_agent_module(agent_dir: Path):
    agent_dir = agent_dir.resolve()
    main_path = agent_dir / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(main_path)

    load_id = next(_LOAD_COUNTER)
    main_module_name = f"agent_{agent_dir.name}_{load_id}"
    local_policy_path = agent_dir / "policy_base.py"
    policy_module = (
        _load_module(f"policy_base_{agent_dir.name}_{load_id}", local_policy_path)
        if local_policy_path.exists()
        else load_shared_policy_base()
    )

    with _prepend_sys_path(VENDOR_DIR), _prepend_sys_path(agent_dir):
        with _temporary_module_alias("policy_base", policy_module):
            return _load_module(main_module_name, main_path)


def load_dir_agent(agent_dir: Path):
    module = load_dir_agent_module(agent_dir)
    return module.agent, get_agent_diag(module), module


def get_agent_diag(module) -> dict[str, Any] | None:
    for attr in ("_DIAG", "DIAG"):
        value = getattr(module, attr, None)
        if isinstance(value, dict):
            return value
    return None


def diag_snapshot(diag: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(diag, dict):
        return None
    errors = diag.get("errors")
    decisions = int(diag.get("decisions", 0) or 0)
    policy_fallback = int(diag.get("policy_fallback", 0) or 0)
    obs_fallback = int(diag.get("obs_fallback", 0) or 0)
    snapshot = {
        "decisions": decisions,
        "policy_ok": int(diag.get("policy_ok", 0) or 0),
        "policy_fallback": policy_fallback,
        "obs_fallback": obs_fallback,
        "deck_returns": int(diag.get("deck_returns", 0) or 0),
        "errors": dict(errors) if isinstance(errors, dict) else {},
    }
    snapshot["fallback_rate"] = (
        (policy_fallback + obs_fallback) / decisions if decisions else 0.0
    )
    return snapshot


def diag_delta(
    before: dict[str, Any] | None, after: dict[str, Any] | None
) -> dict[str, Any] | None:
    if after is None:
        return None
    before = before or {}
    after_errors = Counter((after.get("errors") or {}) if isinstance(after, dict) else {})
    before_errors = Counter((before.get("errors") or {}) if isinstance(before, dict) else {})
    delta = {
        "decisions": int(after.get("decisions", 0) or 0) - int(before.get("decisions", 0) or 0),
        "policy_ok": int(after.get("policy_ok", 0) or 0) - int(before.get("policy_ok", 0) or 0),
        "policy_fallback": int(after.get("policy_fallback", 0) or 0)
        - int(before.get("policy_fallback", 0) or 0),
        "obs_fallback": int(after.get("obs_fallback", 0) or 0)
        - int(before.get("obs_fallback", 0) or 0),
        "deck_returns": int(after.get("deck_returns", 0) or 0)
        - int(before.get("deck_returns", 0) or 0),
        "errors": dict(after_errors - before_errors),
    }
    delta["fallback_rate"] = (
        (delta["policy_fallback"] + delta["obs_fallback"]) / delta["decisions"]
        if delta["decisions"]
        else 0.0
    )
    return delta
