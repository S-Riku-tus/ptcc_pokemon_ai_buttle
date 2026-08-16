"""Exact-list Dragapult v1: multi-teacher imitation with safe fallback."""

from __future__ import annotations

import os

import fallback_policy
from fallback_policy import MY_DECK, agent as _fallback_agent


_RANKER = None
_LOAD_ERROR = None
if os.environ.get("DRAGAPULT_ML_DISABLE") != "1":
    try:
        from ml_runtime import Ranker

        _RANKER = Ranker()
    except Exception as error:  # model absence/corruption must never lose a game
        _LOAD_ERROR = f"{type(error).__name__}: {error}"


def _single_choice(value):
    return value[0] if isinstance(value, list) and len(value) == 1 else None


def _choose(observation):
    if not isinstance(observation, dict):
        return []
    if observation.get("select") is None:
        if _RANKER is not None:
            _RANKER.reset()
        _fallback_agent(observation)
        return list(MY_DECK)

    fallback = list(_fallback_agent(observation))
    if _RANKER is None:
        return fallback

    index = _RANKER.choose(observation)
    if index is None:
        external = _single_choice(fallback)
        if external is not None and not _RANKER.teacher_forced:
            _RANKER.observe_external(observation, external)
        return fallback

    _RANKER.commit(index)
    return [index]


def observe_external(observation, chosen):
    """Advance history with the teacher action during offline replay."""
    if _RANKER is not None:
        _RANKER.observe_external(observation, chosen)


def diag_reset():
    fallback_policy.diag_reset()
    if _RANKER is not None:
        _RANKER.reset()


def diag_snapshot():
    return {
        "ml": _RANKER.snapshot() if _RANKER is not None else {},
        "fallback": fallback_policy.diag_snapshot(),
        "load_error": _LOAD_ERROR,
    }


# The competition loader chooses the last callable defined in this module.
def agent(observation):
    return _choose(observation)

