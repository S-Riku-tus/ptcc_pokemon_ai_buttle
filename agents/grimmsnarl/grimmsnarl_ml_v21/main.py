"""Marnie's Grimmsnarl ex ML v20.

v20 keeps v19's deck and the verified v15-v18 safety layers.  Its single
teacher-unconditioned ranker is retrained without outcome weighting: sparse,
publicly observable attack-delay, backup-chain, Punk Up, promotion, wall and
mirror-endgame states receive twice the fit weight.  Twenty new features make
the Active and backup Grimmsnarl-line ETA, candidate ETA gain, over-funding and
ready-promotion opportunity visible to the model.

``horizon_prize.py`` is the only new policy shell.  After the first Shadow
Bullet in a mirror endgame it can break a raw-score near-tie when one legal
target has a strictly larger two-turn Prize ceiling.  Immediate KO, wall,
attack-access and preservation invariants retain the final word.  No hidden
deck identity, teacher identity, replay result or opponent hand is read.
"""

from __future__ import annotations

import os
from typing import Any

import fallback_policy
import policy_router
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

try:
    from attack_access import AttackAccessGuard

    _ROUTE = AttackAccessGuard()
    _ROUTE_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    _ROUTE = None
    _ROUTE_ERROR = f"{type(error).__name__}: {error}"
_ROUTE_DISABLED = (
    _ROUTE is None or os.environ.get("GRIMMSNARL_ROUTE_DISABLE") == "1"
)

try:
    from wall_break import WallBreakGuard

    _WALL_BREAK = WallBreakGuard()
    _WALL_BREAK_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    _WALL_BREAK = None
    _WALL_BREAK_ERROR = f"{type(error).__name__}: {error}"
_WALL_BREAK_DISABLED = (
    _WALL_BREAK is None
    or os.environ.get("GRIMMSNARL_WALL_BREAK_DISABLE") == "1"
)

try:
    from mirror_prize import MirrorPrizeGuard

    _MIRROR_PRIZE = MirrorPrizeGuard()
    _MIRROR_PRIZE_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    _MIRROR_PRIZE = None
    _MIRROR_PRIZE_ERROR = f"{type(error).__name__}: {error}"
_MIRROR_PRIZE_DISABLED = (
    _MIRROR_PRIZE is None
    or os.environ.get("GRIMMSNARL_MIRROR_PRIZE_DISABLE") == "1"
)

try:
    from horizon_prize import HorizonPrizePlanner

    _HORIZON_PRIZE = HorizonPrizePlanner()
    _HORIZON_PRIZE_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    _HORIZON_PRIZE = None
    _HORIZON_PRIZE_ERROR = f"{type(error).__name__}: {error}"
_HORIZON_PRIZE_DISABLED = (
    _HORIZON_PRIZE is None
    or os.environ.get("GRIMMSNARL_HORIZON_PRIZE_DISABLE") == "1"
)


def _note_matchup(route: str) -> None:
    """Send public mirror evidence to safety/tie-break layers, not the model."""
    is_mirror = route == policy_router.MIRROR
    if _RANKER is not None:
        _RANKER.suspend_escalation = is_mirror
    if _MIRROR_PRIZE is not None:
        _MIRROR_PRIZE.set_mirror(is_mirror)
    if _HORIZON_PRIZE is not None:
        _HORIZON_PRIZE.set_mirror(is_mirror)


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


def _route(
    observation: dict[str, Any],
    select: dict[str, Any],
    index: int,
    rule_choice: Any,
) -> int:
    """Both routes, applied to whichever layer decided.

    The two never contend for the same board: ``attack_access`` requires the
    Shadow Bullet to be worth something, and ``wall_break`` requires it to be
    worth nothing.  Ordering them anyway means the wall route has the last
    word, which is the direction that cannot lose a prize.
    """
    if not _ROUTE_DISABLED:
        index = _ROUTE.adjust(observation, select, index, rule_choice)
    if not _WALL_BREAK_DISABLED:
        index = _WALL_BREAK.adjust(observation, select, index, rule_choice)
    if not _HORIZON_PRIZE_DISABLED:
        index = _HORIZON_PRIZE.adjust(
            observation, select, index,
            _RANKER.last_scores if _RANKER is not None else None,
        )
    if not _MIRROR_PRIZE_DISABLED:
        index = _MIRROR_PRIZE.adjust(
            observation, select, index,
            _RANKER.last_scores if _RANKER is not None else None,
        )
    return index


def _rule_path(
    observation: dict[str, Any],
    select: dict[str, Any],
    rule_choice: Any,
    *,
    note: bool,
) -> Any:
    """Return the rule policy's selection, with the route invariant applied.

    Single-pick selects are the only ones the guard can speak about; a
    multi-pick search keeps the rule policy's whole answer untouched.
    """
    chosen = _rule_index(rule_choice)
    if chosen is None:
        return rule_choice
    forced = _route(observation, select, chosen, rule_choice)
    if note:
        _note_external(observation, select, forced)
    return rule_choice if forced == chosen else [forced]


def _choose(observation: Any):
    if not isinstance(observation, dict) or observation.get("select") is None:
        return _fallback_agent(observation)

    # Track public matchup evidence, prize history and the turn counter on
    # every observation, before any failure/defer path. None of them chooses
    # the base policy.
    route = _ROUTER.choose(observation)
    _note_matchup(route)
    if _RESIDUAL is not None:
        _RESIDUAL.note(observation)
    if _ROUTE is not None:
        _ROUTE.note(observation)
    if _WALL_BREAK is not None:
        _WALL_BREAK.note(observation)
    if _MIRROR_PRIZE is not None:
        _MIRROR_PRIZE.note(observation)
    if _HORIZON_PRIZE is not None:
        _HORIZON_PRIZE.note(observation)

    # The fallback's trackers require the complete trajectory. Its answer is
    # normally advisory and is used by the wall guard and the bridge attack
    # only on their narrow gates.
    rule_choice = _fallback_agent(observation)
    select = observation.get("select") or {}
    if _RANKER is None:
        return _rule_path(observation, select, rule_choice, note=False)

    if not _RANKER.is_scorable(select):
        return _rule_path(observation, select, rule_choice, note=True)

    index = _RANKER.choose(observation)
    if index is None:
        if _rule_index(rule_choice) is None:
            _note_external(observation, select, 0)
            return rule_choice
        return _rule_path(observation, select, rule_choice, note=True)

    if not _PLANNER_DISABLED:
        index = _PLANNER.adjust(
            observation, select, index, _RANKER.last_scores
        )
    if not _RESIDUAL_DISABLED:
        index = _RESIDUAL.adjust(
            observation, select, index, _RANKER, _RANKER.last_scores
        )
    if not _WALL_GUARD_DISABLED:
        index = _WALL_GUARD.adjust(
            observation, select, index, rule_choice
        )
    # Last, so the invariant has the final word over advisory layers.
    index = _route(observation, select, index, rule_choice)

    _RANKER.commit(index)
    if _PLANNER is not None and not _RANKER.teacher_forced:
        _PLANNER.note(observation, select, index)
    return [index]


def observe_external(observation: Any, chosen: int) -> None:
    """Advance every stateful layer with a replay's actual action."""
    if not isinstance(observation, dict):
        return
    _note_matchup(_ROUTER.choose(observation))
    if _RANKER is not None:
        _RANKER.observe_external(observation, chosen)
    if _PLANNER is not None:
        _PLANNER.note(observation, observation.get("select") or {}, chosen)
    if _RESIDUAL is not None:
        _RESIDUAL.note(observation)
    if _ROUTE is not None:
        _ROUTE.note(observation)
    if _WALL_BREAK is not None:
        _WALL_BREAK.note(observation)
    if _MIRROR_PRIZE is not None:
        _MIRROR_PRIZE.note(observation)
        _MIRROR_PRIZE.record(
            observation, observation.get("select") or {}, chosen
        )
    if _HORIZON_PRIZE is not None:
        _HORIZON_PRIZE.note(observation)
        _HORIZON_PRIZE.record(
            observation, observation.get("select") or {}, chosen
        )


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
    if _ROUTE is not None:
        _ROUTE.reset()
    if _WALL_BREAK is not None:
        _WALL_BREAK.reset()
    if _MIRROR_PRIZE is not None:
        _MIRROR_PRIZE.reset()
    if _HORIZON_PRIZE is not None:
        _HORIZON_PRIZE.reset()


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
        "attack_access": _ROUTE.snapshot() if _ROUTE is not None else {},
        "wall_break": (
            _WALL_BREAK.snapshot() if _WALL_BREAK is not None else {}
        ),
        "mirror_prize": (
            _MIRROR_PRIZE.snapshot() if _MIRROR_PRIZE is not None else {}
        ),
        "horizon_prize": (
            _HORIZON_PRIZE.snapshot()
            if _HORIZON_PRIZE is not None else {}
        ),
        "load_error": _LOAD_ERROR,
        "planner_load_error": _PLANNER_ERROR,
        "residual_load_error": _RESIDUAL_ERROR,
        "wall_guard_load_error": _WALL_GUARD_ERROR,
        "attack_access_load_error": _ROUTE_ERROR,
        "wall_break_load_error": _WALL_BREAK_ERROR,
        "mirror_prize_load_error": _MIRROR_PRIZE_ERROR,
        "horizon_prize_load_error": _HORIZON_PRIZE_ERROR,
    }


# IMPORTANT: Kaggle's loader selects the last callable defined in main.py.
def agent(observation: Any):
    return _choose(observation)
