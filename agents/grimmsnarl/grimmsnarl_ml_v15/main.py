"""Marnie's Grimmsnarl ex ML v15: v14 plus the attack-access invariant.

v14 is v8's ranker on every matchup plus two narrow residuals, and its ladder
run says the deficit is not in building the board.  It evolves more Grimmsnarl
ex a game than the rank-3 pilot (2.21 against 1.95), converts Boss's Orders to
an attack on 79% of plays against their 82%, and once it starts attacking it
does not stop (0.27 non-Shadow turns after the first Shadow Bullet in losses,
against their 1.20).  What it does is start late: first Shadow Bullet on turn
3.60 against their 2.96, and 19.0% of games with a Shadow Bullet by our second
turn against their 38.0%.

The whole tail sits in one state.  A Grimmsnarl ex that *can pay* Shadow Bullet
and a Grimmsnarl ex that can *be in the Active spot this turn* are different
facts, and nothing in v8 computes the second one: ``_active_can_retreat`` reads
only the Energy already attached, so a Snorunt Active with no Energy, a
Darkness in hand and an unused manual attachment reads as "no attack this
turn" when the truth is attach -> retreat -> promote -> Shadow Bullet.  Every
non-Grimmsnarl body this deck plays retreats for exactly one Energy, so that
route is always one attachment long, and the resource it needs is the one every
other play also wants.

v15 therefore adds exactly one module and changes nothing else:

* ``attack_access.py`` forces the next step of an ETA-0 attack route while a
  ready Grimmsnarl is unreachable, refuses to end a turn with a worthwhile
  Shadow Bullet unspent, and takes v8's own bridge attack (Filch / Corkscrew
  Punch) instead of v8's END.  Both forced steps are already v8's own
  top-scored rules (990,000 for the escape attachment, 995,000 for the
  retreat); the guard only makes the ranker respect them, on a strict subset of
  the boards where v8 would apply them.

Everything else - the 60 cards, the ranker, the fallback policy, the planner,
the Petrel/dead-Stamp residual, the wall safety gate and the telemetry router -
is byte-identical to v14.  ``GRIMMSNARL_ROUTE_DISABLE=1`` restores v14 exactly,
which is how the same-board A/B is measured.
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
    """The attack-access invariant, applied to whichever layer decided."""
    if _ROUTE_DISABLED:
        return index
    return _ROUTE.adjust(observation, select, index, rule_choice)


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
    _ROUTER.choose(observation)
    if _RESIDUAL is not None:
        _RESIDUAL.note(observation)
    if _ROUTE is not None:
        _ROUTE.note(observation)

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
    # Last, so the invariant has the final word over every advisory layer.
    index = _route(observation, select, index, rule_choice)

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
    if _ROUTE is not None:
        _ROUTE.note(observation)


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
        "load_error": _LOAD_ERROR,
        "planner_load_error": _PLANNER_ERROR,
        "residual_load_error": _RESIDUAL_ERROR,
        "wall_guard_load_error": _WALL_GUARD_ERROR,
        "attack_access_load_error": _ROUTE_ERROR,
    }


# IMPORTANT: Kaggle's loader selects the last callable defined in main.py.
def agent(observation: Any):
    return _choose(observation)
