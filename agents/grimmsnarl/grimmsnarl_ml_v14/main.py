"""Marnie's Grimmsnarl ex ML v14: champion plus guarded residuals.

v13 proved that matchup routing can be useful, but its two new specialists
were not safe replacements for the v8 champion: the wall state machine went
1-10 and the generic v9 model was not actually trained as an Alakazam expert.
v14 therefore keeps the exact v8 ranker on every matchup and permits only two
narrow, auditable corrections:

* the measured Petrel/dead-Unfair-Stamp residual from v10; and
* a wall guard that replaces a zero-prize, zero-active-damage Shadow Bullet
  only when the fallback has a non-closing development action available.

The public matchup router remains as telemetry.  It never selects a different
ranker, so late detection cannot stitch incompatible policies together.
"""

from __future__ import annotations

import os
from typing import Any

import fallback_policy
from fallback_policy import agent as _fallback_agent
from ml_runtime import Ranker
from policy_router import PolicyRouter

_ROUTER = PolicyRouter()

_RANKER: Ranker | None = None
_LOAD_ERROR: str | None = None
if os.environ.get("GRIMMSNARL_ML_DISABLE") != "1":
    try:
        _RANKER = Ranker()
    except Exception as error:  # missing/corrupt model must not crash a game
        _LOAD_ERROR = f"{type(error).__name__}: {error}"

try:
    from ml_planner import Planner

    _PLANNER = Planner()
    _PLANNER_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    _PLANNER = None
    _PLANNER_ERROR = f"{type(error).__name__}: {error}"
_PLANNER_DISABLED = (
    _PLANNER is None or os.environ.get("GRIMMSNARL_PLANNER_DISABLE") == "1"
)

try:
    from ml_residual import Residual

    _RESIDUAL = Residual()
    _RESIDUAL_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    _RESIDUAL = None
    _RESIDUAL_ERROR = f"{type(error).__name__}: {error}"
_RESIDUAL_DISABLED = (
    _RESIDUAL is None or os.environ.get("GRIMMSNARL_RESIDUAL_DISABLE") == "1"
)

try:
    from matchup_guard import WallSafetyGuard

    _WALL_GUARD = WallSafetyGuard()
    _WALL_GUARD_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    _WALL_GUARD = None
    _WALL_GUARD_ERROR = f"{type(error).__name__}: {error}"
_WALL_GUARD_DISABLED = (
    _WALL_GUARD is None
    or os.environ.get("GRIMMSNARL_WALL_GUARD_DISABLE") == "1"
)


def _rule_index(rule_choice: Any) -> int | None:
    if (
        isinstance(rule_choice, list)
        and len(rule_choice) == 1
        and isinstance(rule_choice[0], int)
    ):
        return rule_choice[0]
    return None


def _note_external(
    observation: dict[str, Any],
    select: dict[str, Any],
    chosen: int,
) -> None:
    if _RANKER is not None and not _RANKER.teacher_forced:
        _RANKER.observe_external(observation, chosen)
    if _PLANNER is not None:
        _PLANNER.note(observation, select, chosen)


def _choose(observation: Any):
    if not isinstance(observation, dict) or observation.get("select") is None:
        return _fallback_agent(observation)

    # Track public matchup evidence and prize history on every observation,
    # before any failure/defer path. Neither tracker chooses the base policy.
    _ROUTER.choose(observation)
    if _RESIDUAL is not None:
        _RESIDUAL.note(observation)

    # The fallback's trackers require the complete trajectory. Its answer is
    # normally advisory and is used by the wall guard only on its narrow gate.
    rule_choice = _fallback_agent(observation)
    if _RANKER is None:
        return rule_choice

    select = observation.get("select") or {}
    if not _RANKER.is_scorable(select):
        chosen = _rule_index(rule_choice)
        if chosen is not None:
            _note_external(observation, select, chosen)
        return rule_choice

    index = _RANKER.choose(observation)
    if index is None:
        chosen = _rule_index(rule_choice)
        if chosen is None:
            chosen = 0
        _note_external(observation, select, chosen)
        return rule_choice

    if not _PLANNER_DISABLED:
        index = _PLANNER.adjust(observation, select, index, _RANKER.last_scores)
    if not _RESIDUAL_DISABLED:
        index = _RESIDUAL.adjust(
            observation, select, index, _RANKER, _RANKER.last_scores
        )
    if not _WALL_GUARD_DISABLED:
        index = _WALL_GUARD.adjust(
            observation, select, index, rule_choice
        )

    _RANKER.commit(index)
    if _PLANNER is not None and not _RANKER.teacher_forced:
        _PLANNER.note(observation, select, index)
    return [index]


def observe_external(observation: Any, chosen: int) -> None:
    """Advance every stateful layer with a replay's actual action."""
    if not isinstance(observation, dict):
        return
    _ROUTER.choose(observation)
    if _RANKER is not None:
        _RANKER.observe_external(observation, chosen)
    if _PLANNER is not None:
        _PLANNER.note(observation, observation.get("select") or {}, chosen)
    if _RESIDUAL is not None:
        _RESIDUAL.note(observation)


def diag_reset() -> None:
    fallback_policy.DIAG.clear()
    fallback_policy.DIAG.update(fallback_policy._fresh_diag())
    _ROUTER.reset()
    if _RANKER is not None:
        _RANKER.reset()
    if _PLANNER is not None:
        _PLANNER.reset()
    if _RESIDUAL is not None:
        _RESIDUAL.reset()
    if _WALL_GUARD is not None:
        _WALL_GUARD.reset()


def diag_snapshot() -> dict[str, Any]:
    return {
        "fallback": dict(fallback_policy.DIAG),
        "router": _ROUTER.snapshot(),
        "ml": _RANKER.snapshot() if _RANKER is not None else {},
        "planner": _PLANNER.snapshot() if _PLANNER is not None else {},
        "residual": _RESIDUAL.snapshot() if _RESIDUAL is not None else {},
        "wall_guard": (
            _WALL_GUARD.snapshot() if _WALL_GUARD is not None else {}
        ),
        "load_error": _LOAD_ERROR,
        "planner_load_error": _PLANNER_ERROR,
        "residual_load_error": _RESIDUAL_ERROR,
        "wall_guard_load_error": _WALL_GUARD_ERROR,
    }


# IMPORTANT: Kaggle's loader selects the last callable defined in main.py.
def agent(observation: Any):
    return _choose(observation)
