"""Exact-list Dragapult v1: multi-teacher imitation with safe fallback."""

from __future__ import annotations

import os

import fallback_policy
from fallback_policy import MY_DECK, agent as _fallback_agent


_RANKER = None
_LOAD_ERROR = None
_GUARD_STATS = {
    "overrides": 0,
    "energy": 0,
    "evolution": 0,
    "route_selection": 0,
    "retreat": 0,
    "boss": 0,
    "premature_end": 0,
}
if os.environ.get("DRAGAPULT_ML_DISABLE") != "1":
    try:
        from ml_runtime import Ranker

        _RANKER = Ranker()
    except Exception as error:  # model absence/corruption must never lose a game
        _LOAD_ERROR = f"{type(error).__name__}: {error}"


def _single_choice(value):
    return value[0] if isinstance(value, list) and len(value) == 1 else None


def _played_card_id(observation, option):
    if int(option.get("type", -1)) != 7:  # PLAY
        return -1
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    hand = (players[your] if your in (0, 1) else {}).get("hand") or []
    index = int(option.get("index", -1))
    if not 0 <= index < len(hand) or not isinstance(hand[index], dict):
        return -1
    return int(hand[index].get("id", -1))


def _option_card(observation, option):
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    player = int(option.get("playerIndex", your))
    owner = players[player] if player in (0, 1) else {}
    area = int(option.get("area", -1))
    zones = {
        1: select.get("deck") or [],
        2: owner.get("hand") or [],
        3: owner.get("discard") or [],
        4: owner.get("active") or [],
        5: owner.get("bench") or [],
        12: current.get("looking") or [],
    }
    zone = zones.get(area, [])
    index = int(option.get("index", -1))
    if not isinstance(zone, list) or not 0 <= index < len(zone):
        return None
    return zone[index] if isinstance(zone[index], dict) else None


def _raw_phantom_ready(card):
    return (
        isinstance(card, dict)
        and int(card.get("id", -1)) == 121
        and 2 in (card.get("energies") or [])
        and 5 in (card.get("energies") or [])
    )


def _guard_reason(observation, ml_index, fallback_index):
    """Return why the deterministic route policy must own this decision."""
    select = observation.get("select") or {}
    options = select.get("option") or []
    if not (0 <= ml_index < len(options)) or fallback_index is None:
        return None
    option = options[ml_index]
    context = int(select.get("context", -1))
    option_type = int(option.get("type", -1))

    # Search/effect sub-selections are where typed Energy and the immediate
    # missing evolution stage matter.  v1 did not represent those facts.
    if context == 7:
        # TO_HAND is also used by Poké Pad and generic draw effects.  Only
        # seize searches that actually expose the evolution line; forcing all
        # 1,846 test searches made the deterministic policy too broad.
        candidate_ids = {
            int(card.get("id", -1))
            for card in (_option_card(observation, candidate) for candidate in options)
            if isinstance(card, dict)
        }
        effect_id = int((select.get("effect") or {}).get("id", -1))
        if candidate_ids.intersection({119, 120, 121}) and effect_id in (1097, 1121):
            return "route_selection"
        return None
    if context in (18, 19, 21, 22, 37):
        return "route_selection"
    if context in (3, 4):
        fallback_card = _option_card(observation, options[fallback_index])
        ml_card = _option_card(observation, option)
        return "retreat" if _raw_phantom_ready(fallback_card) and not _raw_phantom_ready(ml_card) else None
    if context != 0:  # MAIN
        return None
    if option_type == 8:  # ATTACH
        return "energy"
    if option_type == 9:  # EVOLVE
        return "evolution"
    if option_type == 12:  # RETREAT
        return "retreat"
    if option_type == 7 and _played_card_id(observation, option) == 1182:
        return "boss"
    if option_type == 14 and fallback_index != ml_index:  # END
        if any(int(candidate.get("type", -1)) == 13 for candidate in options):
            return "premature_end"
    return None


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

    external = _single_choice(fallback)
    reason = _guard_reason(observation, index, external)
    if reason is not None and external is not None:
        _GUARD_STATS["overrides"] += 1
        _GUARD_STATS[reason] += 1
        # Preserve the ranker's turn-history features using the action that is
        # actually returned, not the rejected ML action.
        _RANKER.commit(external)
        return fallback

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
