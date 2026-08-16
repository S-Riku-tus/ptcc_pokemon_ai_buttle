"""Grimmsnarl vfinal: the v22 champion policy with prize-level search authority.

v22 is the best-evidenced agent this line has produced - 190 stored ladder
games, implied strength 1008.8 - and nothing in v23 through v29 separated from
it by more than the ~130 Elo single-run noise floor.  So v22 is kept intact,
byte for byte, as the policy: the same 60, the same 2,000-tree ranker pinned to
the 1220-rated pilot, the same fallback and one-ply planner.

The one thing added is the lever the endgame diagnosis identified as never
having been tested: a search layer with actual authority.  ``turn_search``
enumerates the rest of our own turn from the real engine and may overrule the
ranker only when a different first action leads to a line that takes strictly
more prizes, on every determinization it samples.  Every other decision is
v22's, unchanged, and any failure inside the layer returns v22's answer.
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

# Search is the only new component.  It must never be able to stop the agent
# from answering, so a load failure is recorded and ignored.


def _search_multi_pick(observation, select):
    """What the rule policy would pick for a multi-select, inside the search.

    Punk Up's five Energy and Poffin's basics are multi-picks, and v22 trims
    both below the offered maximum.  A search that always took the maximum
    would be modelling an agent we do not ship, and would then propose lines
    the real turn cannot reproduce.
    """
    del select
    return _fallback_agent(observation)


def _snapshot_rule_state():
    return dict(fallback_policy.TEMP_IMMUNITY), dict(fallback_policy.DIAG)


def _restore_rule_state(saved):
    immunity, diag = saved
    fallback_policy.TEMP_IMMUNITY.clear()
    fallback_policy.TEMP_IMMUNITY.update(immunity)
    fallback_policy.DIAG.clear()
    fallback_policy.DIAG.update(diag)


try:
    import turn_search

    _SEARCH = turn_search.build(
        "deck.csv",
        multi_pick=_search_multi_pick,
        state_guard=(_snapshot_rule_state, _restore_rule_state),
    )
    _SEARCH_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    _SEARCH = None
    _SEARCH_ERROR = f"{type(error).__name__}: {error}"


def _choose(observation):
    if not isinstance(observation, dict) or observation.get("select") is None:
        return _fallback_agent(observation)

    # A committed line runs to the end of the turn before the ranker is
    # consulted again.  Playing only a line's opening and then handing the turn
    # back was measured to collect 19 of its 36 prizes and to cost 73 damage a
    # turn against not overriding at all, so an opening is never played alone.
    if _SEARCH is not None and not (
        _RANKER is not None and _RANKER.teacher_forced
    ):
        planned = _SEARCH.planned(observation)
        if planned is not None:
            # Advance the histories exactly the way v22 does for an action it
            # did not score itself: once, and only for a single-pick.
            if len(planned) == 1:
                observe_external(observation, planned[0])
            return planned

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
            if _PLANNER is not None:
                _PLANNER.note(observation, select, chosen)
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
            if _PLANNER is not None:
                _PLANNER.note(observation, select, chosen)
        return rule_choice
    if not _PLANNER_DISABLED:
        index = _PLANNER.adjust(
            observation, select, index, _RANKER.last_scores
        )
    if _SEARCH is not None and not _RANKER.teacher_forced:
        _SEARCH.budget.note(observation)
        try:
            improved = _SEARCH.suggest(observation, index)
        except Exception:  # noqa: BLE001
            improved = None
        if improved is not None:
            index = improved
    _RANKER.commit(index)
    if _PLANNER is not None and not _RANKER.teacher_forced:
        _PLANNER.note(observation, select, index)
    return [index]


def observe_external(observation, chosen):
    """Advance every history with an action we did not choose.

    Teacher-forced evaluation replays a stored game, so the ranker's intra-turn
    columns and the planner's per-turn heal budget both have to follow the
    teacher rather than our own suggestion.
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
    if _SEARCH is not None:
        _SEARCH.reset()


def diag_snapshot():
    return {
        "fallback": dict(fallback_policy.DIAG),
        "ml": _RANKER.snapshot() if _RANKER is not None else {},
        "planner": _PLANNER.snapshot() if _PLANNER is not None else {},
        "search": _SEARCH.snapshot() if _SEARCH is not None else {},
        "load_error": _LOAD_ERROR,
        "planner_load_error": _PLANNER_ERROR,
        "search_load_error": _SEARCH_ERROR,
    }


# IMPORTANT: Kaggle's loader selects the last callable defined in main.py.
def agent(observation):
    return _choose(observation)
