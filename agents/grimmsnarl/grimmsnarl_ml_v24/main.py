"""Grimmsnarl ML v24: v22 plus mirror-only Froslass suppression.

The 60 cards, ranker, features, fallback policy and one-ply planner are v22.
The only new decision is a public-information mirror veto: if v22 selects a
Froslass evolution, use its best-scored non-Froslass alternative instead.
"""

from __future__ import annotations

import os

import fallback_policy
from fallback_policy import agent as _fallback_agent
from ml_runtime import Ranker

try:
    from mirror_froslass import MirrorFroslassGuard

    _MIRROR_FROSLASS = MirrorFroslassGuard()
    _MIRROR_FROSLASS_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    _MIRROR_FROSLASS = None
    _MIRROR_FROSLASS_ERROR = f"{type(error).__name__}: {error}"
_MIRROR_FROSLASS_DISABLED = (
    _MIRROR_FROSLASS is None
    or os.environ.get("GRIMMSNARL_V24_FROSLASS_DISABLE") == "1"
)

_RANKER: Ranker | None = None
_LOAD_ERROR: str | None = None
if os.environ.get("GRIMMSNARL_ML_DISABLE") != "1":
    try:
        _RANKER = Ranker()
    except Exception as error:  # missing/corrupt model must not crash the game
        _LOAD_ERROR = f"{type(error).__name__}: {error}"

# The planner is an override layer, not a dependency: if it cannot be imported
# the agent must still play the ranker's answer rather than fail to load.
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


def _choose(observation):
    if not isinstance(observation, dict) or observation.get("select") is None:
        return _fallback_agent(observation)

    if _MIRROR_FROSLASS is not None:
        _MIRROR_FROSLASS.observe(observation)
    rule_choice = _fallback_agent(observation)
    if _RANKER is None:
        return _guard_rule_choice(observation, rule_choice)

    select = observation.get("select") or {}
    if not _RANKER.is_scorable(select):
        rule_choice = _guard_rule_choice(observation, rule_choice)
        chosen = (
            rule_choice[0]
            if isinstance(rule_choice, list) and len(rule_choice) == 1
            else None
        )
        if chosen is not None and not _RANKER.teacher_forced:
            _RANKER.observe_external(observation, chosen)
            if _PLANNER is not None:
                _PLANNER.note(observation, select, chosen)
        return rule_choice

    index = _RANKER.choose(observation)
    if index is None:
        # Feature or scoring failure: keep the rule answer, and keep the
        # intra-turn history aligned with what was actually played.
        rule_choice = _guard_rule_choice(observation, rule_choice)
        chosen = (
            rule_choice[0]
            if isinstance(rule_choice, list) and rule_choice
            else 0
        )
        if not _RANKER.teacher_forced:
            _RANKER.observe_external(observation, chosen)
            if _PLANNER is not None:
                _PLANNER.note(observation, select, chosen)
        return rule_choice
    if not _PLANNER_DISABLED:
        index = _PLANNER.adjust(
            observation, select, index, _RANKER.last_scores
        )
    if not _MIRROR_FROSLASS_DISABLED:
        index = _MIRROR_FROSLASS.adjust(
            observation, select, index, _RANKER.last_scores
        )
    _RANKER.commit(index)
    if _PLANNER is not None and not _RANKER.teacher_forced:
        _PLANNER.note(observation, select, index)
    return [index]


def _guard_rule_choice(observation, rule_choice):
    """Apply the same veto if the ranker is unavailable or defers."""
    if _MIRROR_FROSLASS_DISABLED:
        return rule_choice
    if not (
        isinstance(rule_choice, list)
        and len(rule_choice) == 1
        and isinstance(rule_choice[0], int)
    ):
        return rule_choice
    select = observation.get("select") or {}
    moved = _MIRROR_FROSLASS.adjust(
        observation, select, rule_choice[0], None
    )
    return rule_choice if moved == rule_choice[0] else [moved]


def observe_external(observation, chosen):
    """Advance both histories with an action we did not choose.

    Teacher-forced evaluation replays a stored game, so the ranker's intra-turn
    columns and the planner's per-turn heal budget both have to follow the
    teacher rather than our own suggestion. Evaluators that only called
    ``Ranker.observe_external`` would leave the planner counting nothing.
    """
    if _RANKER is not None:
        _RANKER.observe_external(observation, chosen)
    if _PLANNER is not None and isinstance(observation, dict):
        _PLANNER.note(observation, observation.get("select") or {}, chosen)


def diag_reset():
    fallback_policy.DIAG.clear()
    fallback_policy.DIAG.update(fallback_policy._fresh_diag())
    if _RANKER is not None:
        _RANKER.reset()
    if _PLANNER is not None:
        _PLANNER.reset()
    if _MIRROR_FROSLASS is not None:
        _MIRROR_FROSLASS.reset()


def diag_snapshot():
    return {
        "fallback": dict(fallback_policy.DIAG),
        "ml": _RANKER.snapshot() if _RANKER is not None else {},
        "planner": _PLANNER.snapshot() if _PLANNER is not None else {},
        "mirror_froslass": (
            _MIRROR_FROSLASS.snapshot()
            if _MIRROR_FROSLASS is not None else {}
        ),
        "load_error": _LOAD_ERROR,
        "planner_load_error": _PLANNER_ERROR,
        "mirror_froslass_load_error": _MIRROR_FROSLASS_ERROR,
    }


# IMPORTANT: Kaggle's loader selects the last callable defined in main.py.
def agent(observation):
    return _choose(observation)
