"""Marnie's Grimmsnarl ex ML v13: sticky matchup experts.

v13 is a conservative recovery from v11/v12.  It keeps the exact v8 policy in
the matchups where v8 was strongest, uses the proven v9 consensus model only
against the Alakazam line, and delegates publicly identified wall decks to the
deterministic wall-aware fallback policy.  The route is chosen from public
cards and then held for the game.  v12's broad whole-turn search is absent.

The arithmetic planner inherited from v8 remains: it can only replace a choice
when the observation proves an immediate prize/survival domination or a safe
wall unlock.  It is not a general value search.
"""

from __future__ import annotations

import os
from typing import Any

import fallback_policy
from fallback_policy import agent as _fallback_agent
from ml_runtime import Ranker
from policy_router import ALAKAZAM, PENDING, WALL, PolicyRouter

_ROUTER = PolicyRouter()

# v8 is the eager control and remains available if the specialist cannot load.
_RANKER_V8: Ranker | None = None
_RANKER_V9: Ranker | None = None
_RANKER: Ranker | None = None  # compatibility for the existing replay tools
_V9_ATTEMPTED = False
_LOAD_ERRORS: dict[str, str] = {}

if os.environ.get("GRIMMSNARL_ML_DISABLE") != "1":
    try:
        _RANKER_V8 = Ranker("ranker_model.json")
        _RANKER = _RANKER_V8
    except Exception as error:  # missing/corrupt model must not crash a game
        _LOAD_ERRORS["v8"] = f"{type(error).__name__}: {error}"


def _load_v9() -> Ranker | None:
    global _RANKER_V9, _V9_ATTEMPTED
    if _V9_ATTEMPTED:
        return _RANKER_V9
    _V9_ATTEMPTED = True
    if os.environ.get("GRIMMSNARL_ML_DISABLE") == "1":
        return None
    try:
        _RANKER_V9 = Ranker("ranker_model_v9.json")
        # Replay evaluators set the historical `_RANKER` flag before the lazy
        # specialist exists.  Carry that mode across when it is created.
        if _RANKER_V8 is not None and _RANKER_V8.teacher_forced:
            _RANKER_V9.teacher_forced = True
    except Exception as error:  # noqa: BLE001
        _LOAD_ERRORS["v9"] = f"{type(error).__name__}: {error}"
        _RANKER_V9 = None
    return _RANKER_V9


def _ranker_for(route: str) -> Ranker | None:
    if route == WALL:
        return None
    if route == ALAKAZAM:
        return _load_v9() or _RANKER_V8
    return _RANKER_V8


# The narrow planner is optional.  Failure falls back to the selected model,
# never to a crash and never to the removed v12 search.
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


def _rule_index(rule_choice: Any) -> int | None:
    if (
        isinstance(rule_choice, list)
        and len(rule_choice) == 1
        and isinstance(rule_choice[0], int)
    ):
        return rule_choice[0]
    return None


def _note_external(
    ranker: Ranker,
    observation: dict[str, Any],
    select: dict[str, Any],
    chosen: int,
) -> None:
    if not ranker.teacher_forced:
        ranker.observe_external(observation, chosen)
        if _PLANNER is not None:
            _PLANNER.note(observation, select, chosen)


def _choose(observation: Any):
    if not isinstance(observation, dict) or observation.get("select") is None:
        return _fallback_agent(observation)

    # Always run the fallback once: its prize and temporary-immunity trackers
    # must see the complete trajectory even when an ML expert supplies action.
    rule_choice = _fallback_agent(observation)
    route = _ROUTER.choose(observation)

    # Wall is an explicit state-machine expert.  It contains the deck's
    # zero-damage, Boss-unlock and alternative-attacker rules and cannot be
    # overwritten by an "attacked" reward like v12's broad search was.
    if route == WALL:
        return rule_choice

    ranker = _ranker_for(route)
    if ranker is None:
        return rule_choice
    if ranker is not _RANKER_V8 and _RANKER_V8 is not None:
        if _RANKER_V8.teacher_forced:
            ranker.teacher_forced = True

    select = observation.get("select") or {}
    if not ranker.is_scorable(select):
        chosen = _rule_index(rule_choice)
        if chosen is not None:
            _note_external(ranker, observation, select, chosen)
        return rule_choice

    index = ranker.choose(observation)
    if index is None:
        chosen = _rule_index(rule_choice)
        if chosen is None:
            chosen = 0
        _note_external(ranker, observation, select, chosen)
        return rule_choice

    if not _PLANNER_DISABLED:
        index = _PLANNER.adjust(observation, select, index, ranker.last_scores)
    ranker.commit(index)
    if _PLANNER is not None and not ranker.teacher_forced:
        _PLANNER.note(observation, select, index)
    return [index]


def observe_external(observation: Any, chosen: int) -> None:
    """Advance the active expert with a replay's action."""
    if not isinstance(observation, dict):
        return
    route = _ROUTER.route
    if route == PENDING:
        route = _ROUTER.choose(observation)
    if route == WALL:
        return
    ranker = _ranker_for(route)
    if ranker is not None:
        ranker.observe_external(observation, chosen)
    if _PLANNER is not None:
        _PLANNER.note(observation, observation.get("select") or {}, chosen)


def diag_reset() -> None:
    fallback_policy.DIAG.clear()
    fallback_policy.DIAG.update(fallback_policy._fresh_diag())
    _ROUTER.reset()
    if _RANKER_V8 is not None:
        _RANKER_V8.reset()
    if _RANKER_V9 is not None:
        _RANKER_V9.reset()
    if _PLANNER is not None:
        _PLANNER.reset()


def diag_snapshot() -> dict[str, Any]:
    return {
        "fallback": dict(fallback_policy.DIAG),
        "router": _ROUTER.snapshot(),
        "ml": {
            "v8": _RANKER_V8.snapshot() if _RANKER_V8 is not None else {},
            "v9": _RANKER_V9.snapshot() if _RANKER_V9 is not None else {},
            "v9_loaded": _RANKER_V9 is not None,
        },
        "planner": _PLANNER.snapshot() if _PLANNER is not None else {},
        "load_errors": dict(_LOAD_ERRORS),
        "planner_load_error": _PLANNER_ERROR,
    }


# IMPORTANT: Kaggle's loader selects the last callable defined in main.py.
def agent(observation: Any):
    return _choose(observation)
