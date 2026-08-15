"""Grimmsnarl ML v29: measured mixture of race and elite specialists.

v25's current-pilot ranker is the default because its two ladder samples were
better than v22 on ordinary prize races (25-9, 73.5%, versus 94-50, 65.3%).
That same policy collapsed against damage-immune wall decks (3-15), where v22
was 27-23.  v28 therefore synchronized both rankers and switched to v22 on
public wall evidence.

v29 keeps that validated base and gives v22 two additional cells where the
stored ladder record favours its 1220-rated teacher: Mega Lopunny / Mega
Froslass (10-9 versus v25's 1-3) and Hydrapple ex (2-1 versus 2-3), despite
v22 facing the stronger opponents in both. Pure Teal Mask Ogerpon remains on
v25 because neither policy solves the deck's Grass weakness. The inherited
deterministic wall and deck-clock guards still run after the selected ranker
and one-ply planner.
"""

from __future__ import annotations

import os
from typing import Any

import fallback_policy
from fallback_policy import agent as _fallback_agent
from ml_runtime import Ranker


_RACE: Ranker | None = None
_WALL: Ranker | None = None
_RACE_LOAD_ERROR: str | None = None
_WALL_LOAD_ERROR: str | None = None
if os.environ.get("GRIMMSNARL_ML_DISABLE") != "1":
    try:
        _RACE = Ranker("ranker_model.json")       # v25 / AlphaTCG
    except Exception as error:  # missing/corrupt model must not crash play
        _RACE_LOAD_ERROR = f"{type(error).__name__}: {error}"
    try:
        _WALL = Ranker("ranker_v22_model.json")   # v22 / 1220 pilot
    except Exception as error:
        _WALL_LOAD_ERROR = f"{type(error).__name__}: {error}"


def _optional(component: str, constructor: str):
    try:
        module = __import__(component, fromlist=[constructor])
        return getattr(module, constructor)(), None
    except Exception as error:  # each layer independently fails closed
        return None, f"{type(error).__name__}: {error}"


_PLANNER, _PLANNER_ERROR = _optional("ml_planner", "Planner")
_ROUTER, _ROUTER_ERROR = _optional("policy_router", "PolicyRouter")
_TRAJECTORY, _TRAJECTORY_ERROR = _optional(
    "wall_trajectory", "WallTrajectoryGuard"
)
_WALL_BREAK, _WALL_BREAK_ERROR = _optional("wall_break", "WallBreakGuard")
_DECK_CLOCK, _DECK_CLOCK_ERROR = _optional("deck_clock", "DeckClockGuard")

_PLANNER_DISABLED = (
    _PLANNER is None or os.environ.get("GRIMMSNARL_PLANNER_DISABLE") == "1"
)
_LAST_TRACE: dict[str, Any] = {}


def _single(choice: Any) -> int | None:
    if isinstance(choice, list) and len(choice) == 1:
        value = choice[0]
        return int(value) if isinstance(value, int) else None
    return None


def _advance_ranker(ranker: Ranker | None, observation: dict[str, Any],
                    chosen: int, had_pending: bool) -> None:
    if ranker is None:
        return
    if had_pending:
        ranker.commit(chosen)
    elif not ranker.teacher_forced:
        ranker.observe_external(observation, chosen)


def _observe_route(observation: dict[str, Any]) -> str:
    route = "v8_default"
    if _ROUTER is not None:
        route = _ROUTER.choose(observation)
    wall_known = route == "v8_wall_guarded"
    if _TRAJECTORY is not None:
        _TRAJECTORY.note(observation, wall_known=wall_known)
    if _WALL_BREAK is not None:
        _WALL_BREAK.note(observation)
    return route


def _policy_name(route: str) -> str:
    """Return the owning policy; the override exists for paired probes."""
    forced = os.environ.get(
        "GRIMMSNARL_V29_POLICY",
        os.environ.get("GRIMMSNARL_V28_POLICY", "auto"),
    ).lower()
    if forced in {"v22", "wall"}:
        return "v22"
    if forced in {"v25", "race"}:
        return "v25"
    elite_routes = {
        "v8_wall_guarded",
        "v29_lopunny_elite",
        "v29_hydrapple_elite",
    }
    return "v22" if route in elite_routes else "v25"


def _choose(observation):
    if not isinstance(observation, dict) or observation.get("select") is None:
        return _fallback_agent(observation)

    select = observation.get("select") or {}
    route = _observe_route(observation)
    policy = _policy_name(route)
    rule_choice = _fallback_agent(observation)
    rule_index = _single(rule_choice)

    # Both histories always follow the final action.  This makes a late public
    # wall reveal safe: the v22 specialist has seen the exact same turn prefix.
    if _RACE is not None and _WALL is not None:
        _WALL.teacher_forced = _RACE.teacher_forced

    race_pending = False
    wall_pending = False
    race_index: int | None = None
    wall_index: int | None = None
    if _RACE is not None and _RACE.is_scorable(select):
        race_index = _RACE.choose(observation)
        race_pending = race_index is not None
    if _WALL is not None and _WALL.is_scorable(select):
        wall_index = _WALL.choose(observation)
        wall_pending = wall_index is not None

    selected_ranker = _WALL if policy == "v22" else _RACE
    selected_index = wall_index if policy == "v22" else race_index
    selected_scores = (
        selected_ranker.last_scores if selected_ranker is not None else {}
    )
    if selected_index is None:
        if rule_index is None:
            return rule_choice
        index = rule_index
    else:
        index = selected_index
        if not _PLANNER_DISABLED:
            index = _PLANNER.adjust(
                observation, select, index, selected_scores
            )
    planner_index = index

    if _DECK_CLOCK is not None:
        index = _DECK_CLOCK.adjust(
            observation, select, index, selected_scores
        )
    deck_index = index
    if _TRAJECTORY is not None:
        index = _TRAJECTORY.adjust(
            observation, select, index, rule_choice
        )
    trajectory_index = index
    if _WALL_BREAK is not None:
        index = _WALL_BREAK.adjust(
            observation, select, index, rule_choice
        )
    wall_break_index = index

    options = list(select.get("option") or [])
    if not 0 <= index < len(options):
        if rule_index is None:
            return rule_choice
        index = rule_index

    _LAST_TRACE.clear()
    _LAST_TRACE.update({
        "route": route,
        "policy": policy,
        "first_player_is_self": int(
            (observation.get("current") or {}).get("firstPlayer", -1)
            == (observation.get("current") or {}).get("yourIndex", 0)
        ),
        "rule": rule_index,
        "v25_race": race_index,
        "v22_wall": wall_index,
        "selected": selected_index,
        "planner": planner_index,
        "deck_clock": deck_index,
        "wall_trajectory": trajectory_index,
        "wall_break": wall_break_index,
        "final": index,
    })

    _advance_ranker(_RACE, observation, index, race_pending)
    _advance_ranker(_WALL, observation, index, wall_pending)
    teacher_forced = bool(
        (_RACE is not None and _RACE.teacher_forced)
        or (_WALL is not None and _WALL.teacher_forced)
    )
    if _PLANNER is not None and not teacher_forced:
        _PLANNER.note(observation, select, index)
    return [index]


def observe_external(observation, chosen):
    """Advance both policies and the planner with a stored teacher action."""
    if _RACE is not None:
        _RACE.observe_external(observation, chosen)
    if _WALL is not None:
        _WALL.observe_external(observation, chosen)
    if _PLANNER is not None and isinstance(observation, dict):
        _PLANNER.note(observation, observation.get("select") or {}, chosen)


def diag_reset():
    fallback_policy.DIAG.clear()
    fallback_policy.DIAG.update(fallback_policy._fresh_diag())
    _LAST_TRACE.clear()
    for component in (
        _RACE, _WALL, _PLANNER, _ROUTER, _TRAJECTORY,
        _WALL_BREAK, _DECK_CLOCK,
    ):
        if component is not None and hasattr(component, "reset"):
            component.reset()


def _snapshot(component) -> dict[str, Any]:
    if component is None or not hasattr(component, "snapshot"):
        return {}
    return component.snapshot()


def diag_snapshot():
    return {
        "fallback": dict(fallback_policy.DIAG),
        "race_v25": _snapshot(_RACE),
        "wall_v22": _snapshot(_WALL),
        "planner": _snapshot(_PLANNER),
        "router": _snapshot(_ROUTER),
        "wall_trajectory": _snapshot(_TRAJECTORY),
        "wall_break": _snapshot(_WALL_BREAK),
        "deck_clock": _snapshot(_DECK_CLOCK),
        "last_trace": dict(_LAST_TRACE),
        "load_errors": {
            "race_v25": _RACE_LOAD_ERROR,
            "wall_v22": _WALL_LOAD_ERROR,
            "planner": _PLANNER_ERROR,
            "router": _ROUTER_ERROR,
            "wall_trajectory": _TRAJECTORY_ERROR,
            "wall_break": _WALL_BREAK_ERROR,
            "deck_clock": _DECK_CLOCK_ERROR,
        },
    }


# IMPORTANT: Kaggle's loader selects the last callable defined in main.py.
def agent(observation):
    return _choose(observation)
