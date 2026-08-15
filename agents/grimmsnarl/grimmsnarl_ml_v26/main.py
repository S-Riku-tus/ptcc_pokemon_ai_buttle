"""Grimmsnarl ML v26: v22 champion plus guarded H2 planning.

v22 remains the default policy.  v25's compact AlphaTCG model is loaded only
as a second opinion at the root of a publicly confirmed mirror; it never owns
the final action.  The Search API can then compare the candidates through the
opponent reply and our next turn.  Deterministic wall-route and deck-clock
guards cover the two directly observed v25 failure mechanisms.

Every added layer is fail-closed: no model, no search API, no budget,
disagreement between hidden-state samples, or any exception returns the v22
answer (apart from the deliberately narrow deterministic safety guards).
"""

from __future__ import annotations

import os
from typing import Any

import fallback_policy
from fallback_policy import agent as _fallback_agent
from ml_runtime import Ranker


_RANKER: Ranker | None = None
_PEER: Ranker | None = None
_LOAD_ERROR: str | None = None
_PEER_LOAD_ERROR: str | None = None
if os.environ.get("GRIMMSNARL_ML_DISABLE") != "1":
    try:
        _RANKER = Ranker("ranker_model.json")       # v22 / 1220 pilot
    except Exception as error:  # missing/corrupt model must never crash play
        _LOAD_ERROR = f"{type(error).__name__}: {error}"
    try:
        _PEER = Ranker("ranker_v25_model.json")     # candidate generator only
    except Exception as error:
        _PEER_LOAD_ERROR = f"{type(error).__name__}: {error}"


def _optional(component: str, constructor: str):
    try:
        module = __import__(component, fromlist=[constructor])
        return getattr(module, constructor)(), None
    except Exception as error:  # each layer independently degrades to v22
        return None, f"{type(error).__name__}: {error}"


_PLANNER, _PLANNER_ERROR = _optional("ml_planner", "Planner")
_ROUTER, _ROUTER_ERROR = _optional("policy_router", "PolicyRouter")
_TRAJECTORY, _TRAJECTORY_ERROR = _optional(
    "wall_trajectory", "WallTrajectoryGuard"
)
_WALL_BREAK, _WALL_BREAK_ERROR = _optional("wall_break", "WallBreakGuard")
_DECK_CLOCK, _DECK_CLOCK_ERROR = _optional("deck_clock", "DeckClockGuard")
_H2, _H2_ERROR = _optional("h2_search", "H2SearchPlanner")

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


def _search_mirror(observation: dict[str, Any], route: str) -> bool:
    """Require both the Grimmsnarl line and this list's engine in public.

    Seeing a lone Grimmsnarl is enough for matchup telemetry, but not enough
    to assume the opponent's hidden 60 for search.  Froslass/Snorunt or
    Munkidori alongside the line is the public signature used by v26.
    """
    if route != "v8_mirror":
        return False
    current = observation.get("current") or {}
    players = current.get("players") or []
    your = int(current.get("yourIndex", 0) or 0)
    if len(players) < 2 or your not in (0, 1):
        return False
    opponent = players[1 - your]
    public_ids: set[int] = set()
    for area in ("active", "bench", "discard"):
        for card in opponent.get(area) or []:
            if not isinstance(card, dict):
                continue
            public_ids.add(int(card.get("id", -1)))
            public_ids.update(
                int(previous.get("id", -1))
                for previous in card.get("preEvolution") or []
                if isinstance(previous, dict)
            )
    return bool(
        public_ids & {646, 647, 648}
        and public_ids & {104, 112, 860}
    )


def _choose(observation):
    if not isinstance(observation, dict) or observation.get("select") is None:
        return _fallback_agent(observation)

    select = observation.get("select") or {}
    route = _observe_route(observation)
    rule_choice = _fallback_agent(observation)
    rule_index = _single(rule_choice)
    if _RANKER is None:
        return rule_choice

    # Teacher-forced probes set the champion flag.  The peer must follow the
    # same contract or it would commit our proposal and then the teacher too.
    if _PEER is not None:
        _PEER.teacher_forced = _RANKER.teacher_forced

    base_pending = False
    peer_pending = False
    base_index: int | None = None
    peer_index: int | None = None
    if _RANKER.is_scorable(select):
        base_index = _RANKER.choose(observation)
        base_pending = base_index is not None
    if _PEER is not None and _PEER.is_scorable(select):
        peer_index = _PEER.choose(observation)
        peer_pending = peer_index is not None

    if base_index is None:
        if rule_index is None:
            return rule_choice
        index = rule_index
    else:
        index = base_index
        if not _PLANNER_DISABLED:
            index = _PLANNER.adjust(
                observation, select, index, _RANKER.last_scores
            )
    planner_index = index

    # The statistical planner is allowed only in exact public mirrors.  It
    # receives the planner-adjusted index, so it cannot undo a one-ply proof.
    if _H2 is not None and base_pending:
        index = _H2.adjust(
            observation,
            index,
            _RANKER.last_scores,
            _RANKER,
            peer_index,
            is_mirror=_search_mirror(observation, route),
        )
    h2_index = index
    if _DECK_CLOCK is not None:
        index = _DECK_CLOCK.adjust(
            observation, select, index, _RANKER.last_scores
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
    wall_index = index

    options = list(select.get("option") or [])
    if not 0 <= index < len(options):
        # A malformed optional layer is not allowed to turn a legal rule answer
        # into an invalid action.
        if rule_index is None:
            return rule_choice
        index = rule_index

    _LAST_TRACE.clear()
    _LAST_TRACE.update({
        "route": route,
        "rule": rule_index,
        "ranker": base_index,
        "peer": peer_index,
        "planner": planner_index,
        "h2": h2_index,
        "deck_clock": deck_index,
        "wall_trajectory": trajectory_index,
        "wall_break": wall_index,
        "final": index,
    })

    _advance_ranker(_RANKER, observation, index, base_pending)
    _advance_ranker(_PEER, observation, index, peer_pending)
    if _PLANNER is not None and not _RANKER.teacher_forced:
        _PLANNER.note(observation, select, index)
    return [index]


def observe_external(observation, chosen):
    """Advance every stateful layer with a stored teacher action."""
    if _RANKER is not None:
        _RANKER.observe_external(observation, chosen)
    if _PEER is not None:
        _PEER.observe_external(observation, chosen)
    if _PLANNER is not None and isinstance(observation, dict):
        _PLANNER.note(observation, observation.get("select") or {}, chosen)


def diag_reset():
    fallback_policy.DIAG.clear()
    fallback_policy.DIAG.update(fallback_policy._fresh_diag())
    _LAST_TRACE.clear()
    for component in (
        _RANKER, _PEER, _PLANNER, _ROUTER, _TRAJECTORY,
        _WALL_BREAK, _DECK_CLOCK, _H2,
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
        "ml": _snapshot(_RANKER),
        "peer": _snapshot(_PEER),
        "planner": _snapshot(_PLANNER),
        "router": _snapshot(_ROUTER),
        "wall_trajectory": _snapshot(_TRAJECTORY),
        "wall_break": _snapshot(_WALL_BREAK),
        "deck_clock": _snapshot(_DECK_CLOCK),
        "h2_search": _snapshot(_H2),
        "last_trace": dict(_LAST_TRACE),
        "load_errors": {
            "ranker": _LOAD_ERROR,
            "peer": _PEER_LOAD_ERROR,
            "planner": _PLANNER_ERROR,
            "router": _ROUTER_ERROR,
            "wall_trajectory": _TRAJECTORY_ERROR,
            "wall_break": _WALL_BREAK_ERROR,
            "deck_clock": _DECK_CLOCK_ERROR,
            "h2_search": _H2_ERROR,
        },
    }


# IMPORTANT: Kaggle's loader selects the last callable defined in main.py.
def agent(observation):
    return _choose(observation)
