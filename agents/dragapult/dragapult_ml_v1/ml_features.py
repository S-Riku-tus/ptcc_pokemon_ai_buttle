"""Leakage-safe, deck-aware features for the Dragapult imitation ranker.

The same module is imported by the offline corpus builder and the submitted
runtime.  It deliberately reads only the acting player's observation: hidden
opponent cards, the eventual reward, and future replay state never enter a
feature.  The representation is otherwise mostly deck agnostic; the exact
Dragapult list is used only for compact count features and a few useful route
flags.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


# Engine enum values.  Keeping them local makes this module standard-library
# only and prevents a training/runtime dependency mismatch.
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_STADIUM = 7
AREA_LOOKING = 12

OPT_NUMBER = 0
OPT_YES = 1
OPT_NO = 2
OPT_CARD = 3
OPT_TOOL_CARD = 4
OPT_ENERGY_CARD = 5
OPT_ENERGY = 6
OPT_PLAY = 7
OPT_ATTACH = 8
OPT_EVOLVE = 9
OPT_ABILITY = 10
OPT_DISCARD = 11
OPT_RETREAT = 12
OPT_ATTACK = 13
OPT_END = 14
OPT_SKILL = 15

MAIN_CONTEXT = 0

# Every context whose alternatives can be represented as a single ranked
# candidate.  Export-time validation decides the smaller set actually owned
# by a trained model; thin contexts stay with the deterministic fallback.
SCORABLE_CONTEXTS = frozenset(range(0, 45))

# Exact list 202ee2cec6cbe8b4.
FIRE_ENERGY = 2
PSYCHIC_ENERGY = 5
DARK_ENERGY = 7
MUNKIDORI = 112
DREEPY = 119
DRAKLOAK = 120
DRAGAPULT_EX = 121
FEZANDIPITI_EX = 140
BUDEW = 235
MEOWTH_EX = 1071
UNFAIR_STAMP = 1080
BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
CRUSHING_HAMMER = 1120
ULTRA_BALL = 1121
POKE_PAD = 1152
BOSS = 1182
CRISPIN = 1198
JUDGE = 1213
LILLIE = 1227
DAWN = 1231
JAMMING_TOWER = 1246

KEY_CARD_IDS = (
    FIRE_ENERGY, PSYCHIC_ENERGY, DARK_ENERGY, MUNKIDORI, DREEPY, DRAKLOAK,
    DRAGAPULT_EX, FEZANDIPITI_EX, BUDEW, MEOWTH_EX, UNFAIR_STAMP,
    BUDDY_POFFIN, NIGHT_STRETCHER, CRUSHING_HAMMER, ULTRA_BALL, POKE_PAD,
    BOSS, CRISPIN, JUDGE, LILLIE, DAWN, JAMMING_TOWER,
)
ENERGY_IDS = {FIRE_ENERGY, PSYCHIC_ENERGY, DARK_ENERGY}
POKEMON_IDS = {
    MUNKIDORI, DREEPY, DRAKLOAK, DRAGAPULT_EX, FEZANDIPITI_EX, BUDEW,
    MEOWTH_EX,
}
BASIC_IDS = {MUNKIDORI, DREEPY, FEZANDIPITI_EX, BUDEW, MEOWTH_EX}
EVOLUTION_IDS = {DRAKLOAK, DRAGAPULT_EX}
ITEM_IDS = {
    UNFAIR_STAMP, BUDDY_POFFIN, NIGHT_STRETCHER, CRUSHING_HAMMER,
    ULTRA_BALL, POKE_PAD,
}
SUPPORTER_IDS = {BOSS, CRISPIN, JUDGE, LILLIE, DAWN}
STADIUM_IDS = {JAMMING_TOWER}
RULE_BOX_IDS = {DRAGAPULT_EX, FEZANDIPITI_EX, MEOWTH_EX}

PETTY_GRUDGE = 150
BITE = 151
DRAGON_HEADBUTT = 152
JET_HEADBUTT = 153
PHANTOM_DIVE = 154
MIND_BEND = 141
CRUEL_ARROW = 183
ITCHY_POLLEN = 323
TUCK_TAIL = 1546

ACTION_TYPES = (
    "ability", "attack", "bench", "boss", "end", "energy", "evolve",
    "flag", "item", "number", "other", "retreat", "select", "skill",
    "stadium", "stamp", "supporter",
)


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _cards(owner: dict[str, Any], area: str) -> list[dict[str, Any]]:
    value = owner.get(area) or []
    return [card for card in value if isinstance(card, dict)]


def _players(current: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    players = current.get("players") or [{}, {}]
    players = list(players) + [{}, {}]
    your = _as_int(current.get("yourIndex"), 0)
    your = your if your in (0, 1) else 0
    me = players[your] if isinstance(players[your], dict) else {}
    opp = players[1 - your] if isinstance(players[1 - your], dict) else {}
    return your, me, opp


def _in_play(owner: dict[str, Any]) -> list[dict[str, Any]]:
    return _cards(owner, "active") + _cards(owner, "bench")


def _energy_count(card: dict[str, Any] | None) -> int:
    if not isinstance(card, dict):
        return 0
    energies = card.get("energies")
    if isinstance(energies, list):
        return len(energies)
    cards = card.get("energyCards")
    return len(cards) if isinstance(cards, list) else 0


def _energy_type_count(card: dict[str, Any] | None, energy_type: int) -> int:
    if not isinstance(card, dict):
        return 0
    return sum(int(value) == energy_type for value in (card.get("energies") or []))


def _route_eta(card: dict[str, Any] | None, attached: int | None = None) -> int:
    if not isinstance(card, dict):
        return 99
    stage = {DREEPY: 0, DRAKLOAK: 1, DRAGAPULT_EX: 2}.get(
        _as_int(card.get("id")), -1
    )
    if stage < 0:
        return 99
    energies = [int(value) for value in (card.get("energies") or [])]
    if attached is not None:
        energies.append(int(attached))
    missing_colors = int(FIRE_ENERGY not in energies) + int(
        PSYCHIC_ENERGY not in energies
    )
    return (2 - stage) + missing_colors


def _damage(card: dict[str, Any] | None) -> int:
    if not isinstance(card, dict):
        return 0
    return max(0, _as_int(card.get("maxHp"), 0) - _as_int(card.get("hp"), 0))


def _card_at(
    current: dict[str, Any],
    select: dict[str, Any],
    area: int,
    index: int,
    player_index: int,
) -> dict[str, Any] | None:
    your, me, opp = _players(current)
    owner = me if player_index == your else opp
    if area == AREA_DECK:
        zone = select.get("deck") or []
    elif area == AREA_HAND:
        zone = owner.get("hand") or []
    elif area == AREA_DISCARD:
        zone = owner.get("discard") or []
    elif area == AREA_ACTIVE:
        zone = owner.get("active") or []
    elif area == AREA_BENCH:
        zone = owner.get("bench") or []
    elif area == AREA_PRIZE:
        zone = owner.get("prize") or []
    elif area == AREA_STADIUM:
        zone = current.get("stadium") or []
    elif area == AREA_LOOKING:
        zone = current.get("looking") or []
    else:
        return None
    if not isinstance(zone, list) or not 0 <= index < len(zone):
        return None
    card = zone[index]
    return card if isinstance(card, dict) else None


def _active(owner: dict[str, Any]) -> dict[str, Any]:
    cards = _cards(owner, "active")
    return cards[0] if cards else {}


def _board_features(prefix: str, owner: dict[str, Any], out: dict[str, Any]) -> None:
    active = _cards(owner, "active")
    bench = _cards(owner, "bench")
    ordered = active[:1] + bench[:5]
    for slot in range(6):
        card = ordered[slot] if slot < len(ordered) else {}
        out[f"{prefix}_slot{slot}_id"] = _as_int(card.get("id"))
        out[f"{prefix}_slot{slot}_hp"] = _as_int(card.get("hp"), 0)
        out[f"{prefix}_slot{slot}_max_hp"] = _as_int(card.get("maxHp"), 0)
        out[f"{prefix}_slot{slot}_damage"] = _damage(card)
        out[f"{prefix}_slot{slot}_energy"] = _energy_count(card)
        out[f"{prefix}_slot{slot}_fire"] = _energy_type_count(card, FIRE_ENERGY)
        out[f"{prefix}_slot{slot}_psychic"] = _energy_type_count(card, PSYCHIC_ENERGY)
        out[f"{prefix}_slot{slot}_dark"] = _energy_type_count(card, DARK_ENERGY)
        out[f"{prefix}_slot{slot}_tools"] = len(card.get("tools") or [])
        out[f"{prefix}_slot{slot}_evolution_depth"] = len(
            card.get("preEvolution") or []
        )
        out[f"{prefix}_slot{slot}_appeared"] = int(
            bool(card.get("appearThisTurn"))
        )


def state_features(current: dict[str, Any]) -> dict[str, float | int]:
    """Describe only the public information in the acting observation."""
    your, me, opp = _players(current)
    mine = _in_play(me)
    theirs = _in_play(opp)
    hand = _cards(me, "hand")
    discard = _cards(me, "discard")
    opp_discard = _cards(opp, "discard")
    turn = _as_int(current.get("turn"), 0)
    self_prizes = len(me.get("prize") or [])
    opp_prizes = len(opp.get("prize") or [])
    self_active = _active(me)
    opp_active = _active(opp)

    out: dict[str, float | int] = {
        "turn": turn,
        "turn_action_count": _as_int(current.get("turnActionCount"), 0),
        "first_player_is_self": int(_as_int(current.get("firstPlayer")) == your),
        "energy_attached": int(bool(current.get("energyAttached"))),
        "retreated": int(bool(current.get("retreated"))),
        "stadium_played": int(bool(current.get("stadiumPlayed"))),
        "supporter_played": int(bool(current.get("supporterPlayed"))),
        "self_hand_count": _as_int(me.get("handCount"), len(hand)),
        "self_deck_count": _as_int(me.get("deckCount"), 0),
        "self_prize_count": self_prizes,
        "self_bench_count": len(_cards(me, "bench")),
        "opp_hand_count": _as_int(opp.get("handCount"), 0),
        "opp_deck_count": _as_int(opp.get("deckCount"), 0),
        "opp_prize_count": opp_prizes,
        "opp_bench_count": len(_cards(opp, "bench")),
        "prize_lead": opp_prizes - self_prizes,
        "self_total_energy": sum(_energy_count(card) for card in mine),
        "opp_total_energy": sum(_energy_count(card) for card in theirs),
        "self_total_damage": sum(_damage(card) for card in mine),
        "opp_total_damage": sum(_damage(card) for card in theirs),
        "self_active_id": _as_int(self_active.get("id")),
        "self_active_hp": _as_int(self_active.get("hp"), 0),
        "self_active_energy": _energy_count(self_active),
        "opp_active_id": _as_int(opp_active.get("id")),
        "opp_active_hp": _as_int(opp_active.get("hp"), 0),
        "opp_active_energy": _energy_count(opp_active),
        "stadium_id": _as_int((current.get("stadium") or [{}])[0].get("id"))
        if current.get("stadium") else -1,
        "early_game": int(turn <= 4),
        "mid_game": int(5 <= turn <= 10),
        "late_game": int(turn >= 11),
        "phantom_ready_active": int(
            _as_int(self_active.get("id")) == DRAGAPULT_EX
            and FIRE_ENERGY in (self_active.get("energies") or [])
            and PSYCHIC_ENERGY in (self_active.get("energies") or [])
        ),
        "any_dragapult": int(any(_as_int(card.get("id")) == DRAGAPULT_EX for card in mine)),
        "any_drakloak": int(any(_as_int(card.get("id")) == DRAKLOAK for card in mine)),
        "damaged_self_bodies": sum(_damage(card) > 0 for card in mine),
        "munkidori_dark_ready": int(any(
            _as_int(card.get("id")) == MUNKIDORI
            and DARK_ENERGY in (card.get("energies") or [])
            for card in mine
        )),
        "opp_bench_60_or_less": sum(
            0 < _as_int(card.get("hp"), 0) <= 60 for card in _cards(opp, "bench")
        ),
        "opp_bench_30_or_less": sum(
            0 < _as_int(card.get("hp"), 0) <= 30 for card in _cards(opp, "bench")
        ),
    }
    for owner, prefix in ((me, "self"), (opp, "opp")):
        for status in ("asleep", "burned", "confused", "paralyzed", "poisoned"):
            out[f"{prefix}_{status}"] = int(bool(owner.get(status)))
    _board_features("self", me, out)
    _board_features("opp", opp, out)

    counts = {
        "hand": Counter(_as_int(card.get("id")) for card in hand),
        "field": Counter(_as_int(card.get("id")) for card in mine),
        "discard": Counter(_as_int(card.get("id")) for card in discard),
        "opp_field": Counter(_as_int(card.get("id")) for card in theirs),
        "opp_discard": Counter(_as_int(card.get("id")) for card in opp_discard),
    }
    for card_id in KEY_CARD_IDS:
        for prefix, counter in counts.items():
            out[f"{prefix}_{card_id}"] = counter[card_id]
    return out


def observation_features(observation: dict[str, Any]) -> dict[str, float | int]:
    """Decision-local public history; never inspect later replay steps."""
    logs = observation.get("logs") or []
    counts = Counter(
        _as_int(log.get("type"))
        for log in logs if isinstance(log, dict)
    )
    select = observation.get("select") or {}
    return {
        "observation_step": _as_int(observation.get("step"), 0),
        "remaining_time": float(observation.get("remainingOverageTime") or 0.0),
        "log_count": len(logs),
        "log_moves": counts[6],
        "log_switches": counts[8],
        "log_plays": counts[10],
        "log_attaches": counts[11],
        "log_evolves": counts[12],
        "log_attacks": counts[15],
        "select_type": _as_int(select.get("type")),
        "select_context": _as_int(select.get("context")),
        "select_min": _as_int(select.get("minCount"), 0),
        "select_max": _as_int(select.get("maxCount"), 0),
        "select_option_count": len(select.get("option") or []),
        "remain_damage_counter": _as_int(select.get("remainDamageCounter"), 0),
        "remain_energy_cost": _as_int(select.get("remainEnergyCost"), 0),
        "effect_id": _as_int((select.get("effect") or {}).get("id")),
    }


def _candidate_card(
    current: dict[str, Any], select: dict[str, Any], option: dict[str, Any]
) -> tuple[dict[str, Any] | None, int, int]:
    your, _, _ = _players(current)
    option_type = _as_int(option.get("type"))
    area = _as_int(option.get("area"))
    player = _as_int(option.get("playerIndex"), your)
    index = _as_int(option.get("index"))
    if option_type in (OPT_PLAY, OPT_ATTACH, OPT_EVOLVE, OPT_ENERGY):
        area, player = AREA_HAND, your
    card = _card_at(current, select, area, index, player)
    return card, area, player


def _action_type(option_type: int, card_id: int) -> str:
    if option_type == OPT_ABILITY:
        return "ability"
    if option_type == OPT_ATTACK:
        return "attack"
    if option_type == OPT_END:
        return "end"
    if option_type == OPT_RETREAT:
        return "retreat"
    if option_type == OPT_EVOLVE:
        return "evolve"
    if option_type in (OPT_ATTACH, OPT_ENERGY, OPT_ENERGY_CARD):
        return "energy"
    if option_type in (OPT_YES, OPT_NO):
        return "flag"
    if option_type == OPT_NUMBER:
        return "number"
    if option_type == OPT_SKILL:
        return "skill"
    if option_type in (OPT_CARD, OPT_TOOL_CARD, OPT_DISCARD):
        return "select"
    if option_type == OPT_PLAY:
        if card_id == BOSS:
            return "boss"
        if card_id == UNFAIR_STAMP:
            return "stamp"
        if card_id in ITEM_IDS:
            return "item"
        if card_id in SUPPORTER_IDS:
            return "supporter"
        if card_id in STADIUM_IDS:
            return "stadium"
        if card_id in POKEMON_IDS:
            return "bench"
    return "other"


def option_features(
    current: dict[str, Any],
    select: dict[str, Any],
    option: dict[str, Any],
    *,
    base_state: dict[str, Any] | None = None,
    option_position: int = 0,
) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = dict(base_state or state_features(current))
    your, me, _ = _players(current)
    option_type = _as_int(option.get("type"))
    card, area, player = _candidate_card(current, select, option)
    card_id = _as_int(card.get("id")) if card else -1
    inplay_area = _as_int(option.get("inPlayArea"))
    inplay_index = _as_int(option.get("inPlayIndex"))
    target = _card_at(current, select, inplay_area, inplay_index, player)
    target_id = _as_int(target.get("id")) if target else -1
    target_energies = [int(value) for value in (target.get("energies") or [])] if target else []
    is_route_attach = (
        option_type == OPT_ATTACH
        and card_id in (FIRE_ENERGY, PSYCHIC_ENERGY)
        and target_id in (DREEPY, DRAKLOAK, DRAGAPULT_EX)
    )
    context_card = select.get("contextCard") or {}
    ctx_owner = _as_int(context_card.get("playerIndex"))

    hand_counts = Counter(_as_int(c.get("id")) for c in _cards(me, "hand"))
    field_counts = Counter(_as_int(c.get("id")) for c in _in_play(me))
    discard_counts = Counter(_as_int(c.get("id")) for c in _cards(me, "discard"))
    out.update({
        "action_type": _action_type(option_type, card_id),
        "option_type": option_type,
        "option_position": option_position,
        "candidate_card_id": card_id,
        "candidate_attack_id": _as_int(option.get("attackId")),
        "candidate_area": area,
        "candidate_owner_is_self": int(player == your),
        "candidate_inplay_area": inplay_area,
        "candidate_inplay_index": inplay_index,
        "candidate_target_id": target_id,
        "candidate_target_hp": _as_int(target.get("hp"), 0) if target else 0,
        "candidate_target_max_hp": _as_int(target.get("maxHp"), 0) if target else 0,
        "candidate_target_damage": _damage(target),
        "candidate_target_energy": _energy_count(target),
        "candidate_target_fire": _energy_type_count(target, FIRE_ENERGY),
        "candidate_target_psychic": _energy_type_count(target, PSYCHIC_ENERGY),
        "candidate_target_dark": _energy_type_count(target, DARK_ENERGY),
        "candidate_target_route_eta": _route_eta(target),
        "candidate_route_eta_after": (
            _route_eta(target, card_id) if is_route_attach else _route_eta(target)
        ),
        "candidate_attach_duplicate_color": int(
            is_route_attach and card_id in target_energies
        ),
        "candidate_attach_completes_colors": int(
            is_route_attach
            and card_id not in target_energies
            and (
                (card_id == FIRE_ENERGY and PSYCHIC_ENERGY in target_energies)
                or (card_id == PSYCHIC_ENERGY and FIRE_ENERGY in target_energies)
            )
        ),
        "candidate_card_hp": _as_int(card.get("hp"), 0) if card else 0,
        "candidate_card_max_hp": _as_int(card.get("maxHp"), 0) if card else 0,
        "candidate_card_energy": _energy_count(card),
        "candidate_is_energy": int(card_id in ENERGY_IDS),
        "candidate_is_pokemon": int(card_id in POKEMON_IDS),
        "candidate_is_basic": int(card_id in BASIC_IDS),
        "candidate_is_evolution": int(card_id in EVOLUTION_IDS),
        "candidate_is_rule_box": int(card_id in RULE_BOX_IDS),
        "candidate_hand_count": hand_counts[card_id],
        "candidate_field_count": field_counts[card_id],
        "candidate_discard_count": discard_counts[card_id],
        "ctx_card_id": _as_int(context_card.get("id")),
        "ctx_area": _as_int(context_card.get("area")),
        "ctx_owner_is_self": int(ctx_owner == your) if ctx_owner >= 0 else -1,
        "ctx_number": _as_int(option.get("number")),
        "is_phantom_dive": int(_as_int(option.get("attackId")) == PHANTOM_DIVE),
        "is_itchy_pollen": int(_as_int(option.get("attackId")) == ITCHY_POLLEN),
        "is_recon_directive": int(option_type == OPT_ABILITY and target_id == DRAKLOAK),
        "is_adrena_brain": int(option_type == OPT_ABILITY and target_id == MUNKIDORI),
    })
    return out


def assert_no_leakage(feature_columns: list[str]) -> None:
    """Fail closed when an accidental label/future/private field is added."""
    forbidden_exact = {
        "action", "selected", "reward", "result", "winner", "won",
        "opponent_hand_cards", "opponent_deck_cards", "prize_card_id",
    }
    forbidden_fragments = (
        "future_", "next_state", "final_reward", "teacher_choice",
        "hidden_opponent", "opp_hand_card_", "opp_deck_card_",
    )
    bad = sorted(
        name for name in feature_columns
        if name.lower() in forbidden_exact
        or any(fragment in name.lower() for fragment in forbidden_fragments)
    )
    if bad:
        raise ValueError(f"leakage-prone feature columns: {bad}")
