"""Marnie's Grimmsnarl ex ML v3: guarded elite action-order prior.

Decision split:

* Every measured single-pick select is scored by the unchanged v2.1 ranker.
  A small (alpha 0.10) decision-level prior learned from rank 4/5/9/11/13 may
  reorder action families; the v2.1 ranker still chooses the concrete card or
  target. The coefficient was selected under a validation guard that limits
  loss against v2's pinned pilot.
* Multi-pick selects - Punk Up's five-energy attachment, skill ordering,
  two-card discards - and the face-down prize picks stay with the
  ``marnies_grimmsnarl_ex_v7`` rule policy. The prize zone exposes no card
  ids, so nothing can be learned there.

The rule policy is invoked on every observation regardless, so its prize
tracker and diagnostics stay consistent; on ranker-owned choices its answer
is discarded in favour of the two-stage model.
"""

from __future__ import annotations

import os

import fallback_policy
from fallback_policy import agent as _fallback_agent
from ml_runtime import Ranker

_RANKER: Ranker | None = None
_LOAD_ERROR: str | None = None
if os.environ.get("GRIMMSNARL_ML_DISABLE") != "1":
    try:
        _RANKER = Ranker()
    except Exception as error:  # missing/corrupt model must not crash the game
        _LOAD_ERROR = f"{type(error).__name__}: {error}"


def _choose(observation):
    if not isinstance(observation, dict) or observation.get("select") is None:
        return _fallback_agent(observation)

    rule_choice = _fallback_agent(observation)
    if _RANKER is None:
        return rule_choice

    select = observation.get("select") or {}
    if not _RANKER.is_scorable(select):
        chosen = (
            rule_choice[0]
            if isinstance(rule_choice, list) and len(rule_choice) == 1
            else None
        )
        if chosen is not None and not _RANKER.teacher_forced:
            _RANKER.observe_external(observation, chosen)
        return rule_choice

    index = _RANKER.choose(observation)
    if index is None:
        # Feature or scoring failure: keep the rule answer, and keep the
        # intra-turn history aligned with what was actually played.
        chosen = (
            rule_choice[0]
            if isinstance(rule_choice, list) and rule_choice
            else 0
        )
        if not _RANKER.teacher_forced:
            _RANKER.observe_external(observation, chosen)
        return rule_choice
    _RANKER.commit(index)
    return [index]


def diag_reset():
    fallback_policy.DIAG.clear()
    fallback_policy.DIAG.update(fallback_policy._fresh_diag())
    if _RANKER is not None:
        _RANKER.reset()


def diag_snapshot():
    return {
        "fallback": dict(fallback_policy.DIAG),
        "ml": _RANKER.snapshot() if _RANKER is not None else {},
        "load_error": _LOAD_ERROR,
    }


# IMPORTANT: Kaggle's loader selects the last callable defined in main.py.
def agent(observation):
    return _choose(observation)
