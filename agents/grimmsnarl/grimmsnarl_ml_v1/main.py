"""Marnie's Grimmsnarl ex v1 ML: imitation ranker over the top-50 corpus.

Decision split:

* MAIN (``context == 0``, single pick, two or more options) is decided by the
  imitation ranker's argmax, with no rule veto. That is the decision class the
  ranker was fitted on, and the Alakazam line showed a safety shell over these
  decisions silently eats several points of the agreement it was trained for.
* Every other context - Punk Up attachment, searches, discards, coin flips,
  initial Active, forced switches - stays with the ``marnies_grimmsnarl_ex_v7``
  rule policy, which is already validated on those paths.

The rule policy is invoked on every observation regardless, so its prize
tracker and diagnostics stay consistent; on MAIN its answer is discarded in
favour of the ranker's.
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
    if not Ranker.is_main(select):
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
