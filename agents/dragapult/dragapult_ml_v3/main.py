"""Exact-list Dragapult v3: typed route, board width, bounded loss guards.

v1.0 shipped an untyped model and lost tempo to duplicate-colour attachments.
v1.1 answered that with a broad deterministic override of energy, evolution,
retreat, Boss and search.  Measured against the teachers on the frozen test
split, those overrides agreed with the pilots on 37.9% of the 1,509 decisions
they seized, against the model's own 73.5%: the guard was replacing imitation
with a worse policy everywhere except the one case it was written for.

v2 fixes the Energy cause instead.  The ranker now sees typed Energy per body, the
route ETA before and after each candidate, and whether a candidate completes
the Fire+Psychic pair, so it can express the distinction v1 was blind to.  The
only override left is mechanical rather than strategic: never spend the turn's
attachment on a colour a body already holds when the same decision offers an
attachment that completes a route pair.  Phantom Dive costs exactly one Fire
and one Psychic and every body in this deck retreats for one, so a second copy
of a colour it already has cannot be the better action.

v3 adds observable board-width routes and one second bounded guard: do not turn
a zero-Energy Active Drakloak into a two-Prize Dragapult when it still cannot
attack and the evolution does not provide the measured survival exception.
``DRAGAPULT_GUARD_DISABLE=1`` turns both guards off, so their contribution can
be measured rather than assumed.
"""

from __future__ import annotations

import os

import fallback_policy
from fallback_policy import MY_DECK, agent as _fallback_agent


FIRE_ENERGY = 2
PSYCHIC_ENERGY = 5
DRAGAPULT_LINE = (119, 120, 121)
DRAKLOAK = 120
DRAGAPULT_EX = 121
OPT_ATTACH = 8
OPT_EVOLVE = 9
MAIN_CONTEXT = 0

_RANKER = None
_LOAD_ERROR = None
_GUARD_ENABLED = os.environ.get("DRAGAPULT_GUARD_DISABLE") != "1"
# "overrides" is the key the offline runtime evaluator reads to attribute a
# decision to the guard rather than to the model.
_GUARD_STATS = {
    "seen": 0,
    "overrides": 0,
    "duplicate_color": 0,
    "active_unpowered_evolution": 0,
    "no_safe_alternative": 0,
}

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


def _own_state(observation):
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    mine = players[your] if your in (0, 1) and len(players) == 2 else {}
    return current, mine


def _evolve_target(observation, option):
    """Return ``(area, index, body)`` even when the log omits target fields."""
    _, mine = _own_state(observation)
    active = mine.get("active") or []
    if isinstance(active, dict):
        active = [active]
    bench = mine.get("bench") or []
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("targetIndex"))
    if area is not None and index is not None:
        zone = active if int(area) == 4 else bench if int(area) == 5 else []
        slot = int(index)
        if isinstance(zone, list) and 0 <= slot < len(zone):
            return int(area), slot, zone[slot]
    candidates = []
    for candidate_area, zone in ((4, active), (5, bench)):
        for slot, body in enumerate(zone if isinstance(zone, list) else []):
            if isinstance(body, dict) and int(body.get("id", -1)) == DRAKLOAK:
                candidates.append((candidate_area, slot, body))
    return candidates[0] if len(candidates) == 1 else None


def _active_gets_an_energy_this_turn(observation, options):
    """Whether any currently legal attachment gives the Active a Jet attack."""
    current, mine = _own_state(observation)
    if bool(current.get("energyAttached")):
        return False
    hand = mine.get("hand") or []
    for option in options:
        if int(option.get("type", -1)) != OPT_ATTACH:
            continue
        if int(option.get("inPlayArea", -1)) != 4:
            continue
        if int(option.get("inPlayIndex", -1)) != 0:
            continue
        source_index = int(option.get("index", -1))
        if not 0 <= source_index < len(hand):
            continue
        source = hand[source_index]
        if isinstance(source, dict) and int(source.get("id", -1)) in (2, 5, 7):
            return True
    return False


def _survives_ready_phantom_dive(observation, target):
    """Known survival exception: evolving lets the Active live through 200.

    The held-out teacher counterexample is a Dragapult mirror: a damaged
    Drakloak is knocked out by Phantom Dive, while evolving increases its
    remaining HP enough to survive.  We deliberately keep this exact rather
    than estimate arbitrary incoming damage from printed attacks; effects such
    as Premium Power Pro and energy-scaled attacks made that estimate unsafe in
    the two live failures this guard is meant to fix.
    """
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    if your not in (0, 1) or len(players) != 2:
        return False
    opponent_active = players[1 - your].get("active") or []
    if not opponent_active or not isinstance(opponent_active[0], dict):
        return False
    threat = opponent_active[0]
    if int(threat.get("id", -1)) != DRAGAPULT_EX:
        return False
    colors = [int(value) for value in threat.get("energies") or []]
    if FIRE_ENERGY not in colors or PSYCHIC_ENERGY not in colors:
        return False
    remaining_hp = int(target.get("hp", 0) or 0)
    old_max_hp = int(target.get("maxHp", 90) or 90)
    projected_hp = remaining_hp + max(0, 320 - old_max_hp)
    return projected_hp > 200


def _is_dead_active_evolution(observation, index):
    """True only for a zero-Energy Active evolution with no attack this turn.

    Dragapult ex can use Jet Headbutt for one Colorless Energy, so lacking the
    full Fire+Psychic pair is not enough to justify an override.  This guard
    binds only when the Active Drakloak has zero Energy and no legal attachment
    can give the evolved body even that 70-damage attack.  It therefore covers
    the observed two-prize donation without suppressing normal one-colour or
    bench development.
    """
    if not _GUARD_ENABLED:
        return False
    select = observation.get("select") or {}
    if int(select.get("context", -1)) != MAIN_CONTEXT:
        return False
    options = select.get("option") or []
    if not 0 <= index < len(options):
        return False
    option = options[index]
    if int(option.get("type", -1)) != OPT_EVOLVE:
        return False
    _, mine = _own_state(observation)
    hand = mine.get("hand") or []
    source_index = int(option.get("index", -1))
    source = hand[source_index] if 0 <= source_index < len(hand) else {}
    if not isinstance(source, dict) or int(source.get("id", -1)) != DRAGAPULT_EX:
        return False
    resolved = _evolve_target(observation, option)
    if resolved is None:
        return False
    area, _, target = resolved
    if area != 4 or int(target.get("id", -1)) != DRAKLOAK:
        return False
    if target.get("energies"):
        return False
    if _survives_ready_phantom_dive(observation, target):
        return False
    return not _active_gets_an_energy_this_turn(observation, options)


def _best_safe_alternative(observation, rejected):
    """Second-best scored representative that is not the same dead evolve."""
    if _RANKER is None:
        return None
    for position, _ in sorted(
        _RANKER.last_scores.items(), key=lambda item: (-item[1], item[0])
    ):
        if position == rejected:
            continue
        if not _is_dead_active_evolution(observation, position):
            return position
    return None


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
    if facts is not None:
        source, energies = facts
        if source in energies:
            _GUARD_STATS["seen"] += 1
            replacement = _completing_attach(observation, options)
            if replacement is not None and replacement != ml_index:
                _GUARD_STATS["duplicate_color"] += 1
                _GUARD_STATS["overrides"] += 1
                return replacement
    if not _is_dead_active_evolution(observation, ml_index):
        return None
    _GUARD_STATS["seen"] += 1
    _GUARD_STATS["active_unpowered_evolution"] += 1
    replacement = _best_safe_alternative(observation, ml_index)
    if replacement is None:
        _GUARD_STATS["no_safe_alternative"] += 1
        return None
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
