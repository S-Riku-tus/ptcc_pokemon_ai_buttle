"""Exact-list Dragapult v2: typed-route imitation with one mechanical guard.

v1.0 shipped an untyped model and lost tempo to duplicate-colour attachments.
v1.1 answered that with a broad deterministic override of energy, evolution,
retreat, Boss and search.  Measured against the teachers on the frozen test
split, those overrides agreed with the pilots on 37.9% of the 1,509 decisions
they seized, against the model's own 73.5%: the guard was replacing imitation
with a worse policy everywhere except the one case it was written for.

v2 fixes the cause instead.  The ranker now sees typed Energy per body, the
route ETA before and after each candidate, and whether a candidate completes
the Fire+Psychic pair, so it can express the distinction v1 was blind to.  The
only override left is mechanical rather than strategic: never spend the turn's
attachment on a colour a body already holds when the same decision offers an
attachment that completes a route pair.  Phantom Dive costs exactly one Fire
and one Psychic and every body in this deck retreats for one, so a second copy
of a colour it already has cannot be the better action.

``DRAGAPULT_GUARD_DISABLE=1`` turns even that off, so the guard's contribution
can be measured rather than assumed.
"""

from __future__ import annotations

import os

import fallback_policy
from fallback_policy import MY_DECK, agent as _fallback_agent


FIRE_ENERGY = 2
PSYCHIC_ENERGY = 5
DRAGAPULT_LINE = (119, 120, 121)
OPT_ATTACH = 8
MAIN_CONTEXT = 0

_RANKER = None
_LOAD_ERROR = None
_GUARD_ENABLED = os.environ.get("DRAGAPULT_GUARD_DISABLE") != "1"
# "overrides" is the key the offline runtime evaluator reads to attribute a
# decision to the guard rather than to the model.
_GUARD_STATS = {"seen": 0, "overrides": 0, "duplicate_color": 0}

if os.environ.get("DRAGAPULT_ML_DISABLE") != "1":
    try:
        from ml_runtime import Ranker

        _RANKER = Ranker()
    except Exception as error:  # model absence/corruption must never lose a game
        _LOAD_ERROR = f"{type(error).__name__}: {error}"


def _single_choice(value):
    return value[0] if isinstance(value, list) and len(value) == 1 else None


def _route_attach(observation, option):
    """(source energy id, target energies) for a route attachment, else None."""
    if int(option.get("type", -1)) != OPT_ATTACH:
        return None
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    mine = players[your] if your in (0, 1) else {}
    hand = mine.get("hand") or []
    index = int(option.get("index", -1))
    if not 0 <= index < len(hand) or not isinstance(hand[index], dict):
        return None
    source = int(hand[index].get("id", -1))
    if source not in (FIRE_ENERGY, PSYCHIC_ENERGY):
        return None
    area = int(option.get("inPlayArea", -1))
    target_index = int(option.get("inPlayIndex", -1))
    zone = mine.get("active") if area == 4 else mine.get("bench") if area == 5 else []
    if not isinstance(zone, list) or not 0 <= target_index < len(zone):
        return None
    target = zone[target_index]
    if not isinstance(target, dict) or int(target.get("id", -1)) not in DRAGAPULT_LINE:
        return None
    return source, [int(value) for value in target.get("energies") or []]


def _completing_attach(observation, options):
    """Index of the best attachment that completes a Fire+Psychic pair.

    Prefers the latest evolution stage, because that body attacks soonest.
    """
    best = None
    best_stage = -1
    stages = {119: 0, 120: 1, 121: 2}
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    mine = players[your] if your in (0, 1) else {}
    for position, option in enumerate(options):
        facts = _route_attach(observation, option)
        if facts is None:
            continue
        source, energies = facts
        if source in energies:
            continue
        other = PSYCHIC_ENERGY if source == FIRE_ENERGY else FIRE_ENERGY
        if other not in energies:
            continue
        area = int(option.get("inPlayArea", -1))
        target_index = int(option.get("inPlayIndex", -1))
        zone = (mine.get("active") if area == 4 else mine.get("bench")) or []
        target = (
            zone[target_index]
            if isinstance(zone, list) and 0 <= target_index < len(zone) else {}
        )
        stage = stages.get(int((target or {}).get("id", -1)), -1)
        if stage > best_stage:
            best, best_stage = position, stage
    return best


def _guarded_index(observation, ml_index):
    """Replace a strictly dominated duplicate-colour attachment, else None."""
    if not _GUARD_ENABLED:
        return None
    select = observation.get("select") or {}
    if int(select.get("context", -1)) != MAIN_CONTEXT:
        return None
    options = select.get("option") or []
    if not 0 <= ml_index < len(options):
        return None
    facts = _route_attach(observation, options[ml_index])
    if facts is None:
        return None
    source, energies = facts
    if source not in energies:
        return None
    _GUARD_STATS["seen"] += 1
    replacement = _completing_attach(observation, options)
    if replacement is None or replacement == ml_index:
        return None
    _GUARD_STATS["duplicate_color"] += 1
    _GUARD_STATS["overrides"] += 1
    return replacement


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

    replacement = _guarded_index(observation, index)
    if replacement is not None:
        index = replacement
    _RANKER.commit(index)
    return [index]


def observe_external(observation, chosen):
    """Advance history with the teacher action during offline replay."""
    if _RANKER is not None:
        _RANKER.observe_external(observation, chosen)


def diag_reset():
    fallback_policy.diag_reset()
    for key in _GUARD_STATS:
        _GUARD_STATS[key] = 0
    if _RANKER is not None:
        _RANKER.reset()


def diag_snapshot():
    return {
        "ml": _RANKER.snapshot() if _RANKER is not None else {},
        "fallback": fallback_policy.diag_snapshot(),
        "guard": dict(_GUARD_STATS),
        "load_error": _LOAD_ERROR,
    }


# The competition loader chooses the last callable defined in this module.
def agent(observation):
    return _choose(observation)
