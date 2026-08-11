"""Marnie's Grimmsnarl ex ML v18: seal the post-Shadow Prize invariant.

v18 adds one mirror-only safety layer over v17.  After our first Shadow
Bullet, damage placement may choose among several opposing Benched Pokemon.
When the damage being placed now proves a knockout, the legal target worth the
most Prize cards dominates every lower-Prize target.  ``mirror_prize.py``
seals that arithmetic invariant and leaves every non-lethal target plan to the
ranker.  The additional exact-60 mirror audit found no broader safe override:
v17 already matched all five high-rated two-Prize Adrena-Brain finishes.

v17 is v16 plus one deliberately narrow correction.  v16 identified eight
stored decisions where the last Impidimp/Morgrem capable of crossing a damage
wall was evolved into another Grimmsnarl ex even though a fuelled Grimmsnarl ex
already existed.  Its PRESERVE guard stopped only two: BREAK's eight-turn ETA
gate also rejected the other six, usually because an Impidimp's current
10-damage attack was too slow.  That conflated "attack with this body now"
with "keep this body so it can become Morgrem next turn".

All eight decisions offered a provably worthless Shadow Bullet and END.  v17
uses that free closing action instead of consuming the last future breaker.
The active BREAK route, model, deck, mirror escalation gate and every non-wall
decision remain v16.  No retraining is required or performed: this is a
decision-invariant repair over the same 2,000-tree model.

The remainder records the v16 evidence base retained by this version.

v15 solved getting to the attack, and its 110 rated games say so without
ambiguity.  First Shadow Bullet on own turn 2.84 against v14's 3.60; 82.7% of
games have one by turn 3; only 4 of 187 turn-ends leave a fuelled Grimmsnarl ex
unused; and in the losses the turn of the *first Grimmsnarl ex* and the turn of
the *first Shadow Bullet* are the same number, 3.048.  There is no access gap
left to close.

Re-measuring the same 110 games decided v16 by elimination.  Four candidate
levers were tested at decision level and three of them are already saturated:

* a playable Boss's Orders would have added a prize on **1** of 446 swings;
* the Bench-30 passed over an offered lethal on **2** of 397 shots;
* Adrena-Brain was taken on **98.6%** of the turns it was offered;
* a Grimmsnarl-line play was passed over at END **once** in 176 turns.

Every one of those is offer-side, not decision-side, so no preference change
can move them.  Two things are not saturated, and v16 is exactly those two.

**1. A swing that is provably worth nothing** (``wall_break.py``).  84 Shadow
Bullets over the two runs were thrown at an Active that prevents all damage
from us while the Bench-30 could take no prize either.  The wall matchup is
7-8, 8.25 swings a game in the losses, 86.4% of them taking nothing, and two
games ended in deck-out.  Marnie's Morgrem is neither a Pokemon ex nor a
Pokemon with an Ability, so its Corkscrew Punch lands through Crustle, Sylveon,
Cornerstone Mask Ogerpon ex and Neutralization Zone alike.  v16 routes to it
instead of throwing the dead swing, and stops spending the last one on a
Grimmsnarl ex evolution while that wall is up.

**2. A teacher escalation that generalised too far** (``ml_runtime.py``).  v6
handed the Froslass evolve to pilot 16371703 because the pinned teacher takes
it 95.7% of the time against that pilot's 80.5%.  On mirror boards the shipped
combination produces neither: replaying all 104 stored mirror decisions that
offered the evolve, v15 takes it **4** times with the escalation on and **33**
with it off, while the mirror opponents - on the identical 60 cards - take it
**12 of 12** (Fisher p = 0.0001).  A rate no teacher exhibits is not imitation,
so v16 suspends that one escalation on mirror boards and leaves it untouched
everywhere else, where our 85.6% uptake already matches the pilot it copies.

Everything else - the 60 cards, the ranker weights, the fallback policy, the
planner, the Petrel/dead-Stamp residual, the wall safety gate, the
attack-access invariant and the telemetry router - is byte-identical to v15.
``GRIMMSNARL_WALL_BREAK_DISABLE=1`` and ``GRIMMSNARL_ESCALATION_MIRROR=on``
restore v15 one change at a time, and the two fire on disjoint sets of games.
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


def _note_matchup(route: str) -> None:
    """Tell the ranker whether this is a mirror, from public evidence only.

    The router already classifies the matchup off the opponent's Active,
    Bench, discard and pre-evolutions - no hidden hand or deck - and v15
    already computes it on every observation for telemetry.  v16 is the first
    version to read it, and it reads it for one thing: the Froslass evolve
    escalation stands down in the mirror.
    """
    is_mirror = route == policy_router.MIRROR
    if _RANKER is not None:
        _RANKER.suspend_escalation = is_mirror
    if _MIRROR_PRIZE is not None:
        _MIRROR_PRIZE.set_mirror(is_mirror)


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
        "load_error": _LOAD_ERROR,
        "planner_load_error": _PLANNER_ERROR,
        "residual_load_error": _RESIDUAL_ERROR,
        "wall_guard_load_error": _WALL_GUARD_ERROR,
        "attack_access_load_error": _ROUTE_ERROR,
        "wall_break_load_error": _WALL_BREAK_ERROR,
        "mirror_prize_load_error": _MIRROR_PRIZE_ERROR,
    }


# IMPORTANT: Kaggle's loader selects the last callable defined in main.py.
def agent(observation: Any):
    return _choose(observation)
