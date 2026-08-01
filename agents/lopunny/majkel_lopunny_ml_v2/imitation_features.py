"""Pure-dict features shared by training and the submission runtime.

The original feature skeleton was proved on the Alakazam imitation series.
This version keeps its useful public-state representation, adds the complete
Majkel1337 Mega Lopunny deck vocabulary, and exposes every selection context
rather than only ``MAIN``.  It deliberately depends only on the standard
library so exactly the same code runs during offline extraction and on Kaggle.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

# Retain the old vocabulary because opponent Alakazam states are frequent, but
# add every card in the teacher's exact Lopunny list.  Counts are public at
# runtime (own hand/field/discard and both public boards); prize identities are
# never used.
KEY_CARD_IDS = [
    5, 11, 13, 14, 19, 66, 104, 112, 140, 174, 305, 343, 648, 741,
    742, 743, 848, 849, 1079, 1081, 1086, 1097, 1121, 1122, 1129,
    1152, 1174, 1182, 1197, 1225, 1227, 1229, 1231, 1266,
]
ENERGY_IDS = {3, 5, 11, 13, 14, 17, 19}
PSYCHIC_ENERGY_IDS = {5, 19}
BASIC_IDS = {66, 140, 174, 305, 343, 741, 848}
SPECIFIC_ACTIONS = {1182: "boss", 1197: "xerosic", 1081: "hammer"}
ROUTE_IDS = {741, 742, 743, 1079}
FROSLASS_ID = 104
GRIMMSNARL_EX_ID = 648
MUNKIDORI_ID = 112
FROSLASS_VULNERABLE_IDS = {66, 140, 343, 742}
ONE_ENERGY_PIVOT_IDS = {140, 305, 343, 741, 742}
MIST_ENERGY_ID = 11
MIST_HIGH_SIGNATURE_IDS = {344, 345, 878, 879, 304}
MIST_MEDIUM_SIGNATURE_IDS = {1030, 1031}

BUNEARY_ID = 848
MEGA_LOPUNNY_EX_ID = 849
FAN_ROTOM_ID = 174
ULTRA_BALL_ID = 1121
POKEGEAR_ID = 1122
AIR_BALLOON_ID = 1174
LILLIE_ID = 1227
WALLY_ID = 1229
GALE_THRUST_ID = 1225
SPIKY_HOPPER_ID = 1226
LOPUNNY_ENERGY_IDS = {11, 13, 14}
ACTION_NAMES = (
    "other", "attack", "retreat", "end", "evolve", "ability",
    "energy", "boss", "xerosic", "hammer", "bench", "trainer",
)


def encode_action_type(name: str) -> int:
    return ACTION_NAMES.index(name) if name in ACTION_NAMES else 0


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [x for x in (player.get(area) or []) if isinstance(x, dict)]


def _in_play(player: dict[str, Any]) -> list[dict[str, Any]]:
    return _cards(player, "active") + _cards(player, "bench")


def _energy_cards(card: dict[str, Any]) -> list[Any]:
    return list(card.get("energyCards") or card.get("energies") or [])


def _attached_id(value: Any) -> int:
    if isinstance(value, dict):
        return int(value.get("id", value.get("cardId", -1)))
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _energy_count(card: dict[str, Any]) -> int:
    return len(_energy_cards(card))


def _special_energy_count(card: dict[str, Any]) -> int:
    count = 0
    for energy in _energy_cards(card):
        if isinstance(energy, dict):
            card_id = int(energy.get("id", energy.get("cardId", -1)))
            if card_id not in {3, 5, 17}:
                count += 1
    return count


def _card_at_area(player: dict[str, Any], area: int | None, index: int | None) -> dict[str, Any] | None:
    if index is None:
        return None
    if area == 4:
        seq = _cards(player, "active")
    elif area == 5:
        seq = _cards(player, "bench")
    elif area == 2:
        seq = _cards(player, "hand")
    elif area == 3:
        seq = _cards(player, "discard")
    else:
        return None
    return seq[index] if 0 <= index < len(seq) else None


def _attached_card_at_option(
    current: dict[str, Any], option: dict[str, Any]
) -> dict[str, Any] | None:
    """Resolve TOOL_CARD / ENERGY_CARD options without serial-based leakage."""
    players = current.get("players", [{}, {}])
    your = int(current.get("yourIndex", 0))
    player_index = int(option.get("playerIndex", your))
    if player_index not in (0, 1):
        player_index = your
    target = _card_at_area(
        players[player_index],
        option.get("inPlayArea"),
        option.get("inPlayIndex"),
    )
    if not isinstance(target, dict):
        return None
    area = int(option.get("area", -1))
    values = target.get("energyCards") if area == 8 else target.get("tools")
    values = [value for value in (values or []) if isinstance(value, dict)]
    index = option.get("index")
    return values[index] if isinstance(index, int) and 0 <= index < len(values) else None


def candidate_card(
    current: dict[str, Any],
    option: dict[str, Any],
    select: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    your = int(current.get("yourIndex", 0))
    players = current.get("players", [{}, {}])
    player = players[your]
    option_type = int(option.get("type", -1))
    if option_type in (7, 8, 9):
        index = option.get("index")
        hand = _cards(player, "hand")
        return hand[index] if isinstance(index, int) and 0 <= index < len(hand) else None
    if option_type == 10:
        return _card_at_area(player, option.get("area"), option.get("index"))
    if option_type in (4, 5):
        return _attached_card_at_option(current, option)
    if option_type in (3, 6, 11):
        if int(option.get("area", -1)) == 1 and select is not None:
            index = option.get("index")
            deck = [
                card
                for card in (select.get("deck") or [])
                if isinstance(card, dict)
            ]
            return (
                deck[index]
                if isinstance(index, int) and 0 <= index < len(deck)
                else None
            )
        player_index = int(option.get("playerIndex", your))
        if player_index not in (0, 1):
            player_index = your
        return _card_at_area(
            players[player_index], option.get("area"), option.get("index")
        )
    return None


def _at_least_one_probability(population: int, successes: int, draws: int) -> float:
    population = max(0, int(population))
    successes = max(0, min(int(successes), population))
    draws = max(0, min(int(draws), population))
    if draws == 0 or successes == 0:
        return 0.0
    failures = population - successes
    if failures < draws:
        return 1.0
    return 1.0 - math.comb(failures, draws) / math.comb(population, draws)


def candidate_target(current: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    players = current.get("players", [{}, {}])
    your = int(current.get("yourIndex", 0))
    target_player = int(option.get("playerIndex", your))
    if target_player not in (0, 1):
        target_player = your
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    return _card_at_area(players[target_player], area, index)


def action_type(
    current: dict[str, Any],
    option: dict[str, Any],
    select: dict[str, Any] | None = None,
) -> str:
    option_type = int(option.get("type", -1))
    if option_type == 13:
        return "attack"
    if option_type == 12:
        return "retreat"
    if option_type == 14:
        return "end"
    if option_type == 9:
        return "evolve"
    if option_type == 10:
        return "ability"
    if option_type in (7, 8):
        card = candidate_card(current, option, select) or {}
        card_id = int(card.get("id", -1))
        if card_id in SPECIFIC_ACTIONS:
            return SPECIFIC_ACTIONS[card_id]
        if option_type == 8 or card_id in ENERGY_IDS:
            return "energy"
        if card_id in BASIC_IDS:
            return "bench"
        return "trainer"
    return "other"


def _active_features(prefix: str, player: dict[str, Any], out: dict[str, float | int]) -> None:
    active = (_cards(player, "active") or [{}])[0]
    out[f"{prefix}_active_id"] = int(active.get("id", -1))
    out[f"{prefix}_active_hp"] = float(active.get("hp", 0))
    out[f"{prefix}_active_max_hp"] = float(active.get("maxHp", 0))
    out[f"{prefix}_active_damage"] = max(0.0, float(active.get("maxHp", 0)) - float(active.get("hp", 0)))
    out[f"{prefix}_active_energy"] = _energy_count(active)
    out[f"{prefix}_active_special_energy"] = _special_energy_count(active)
    out[f"{prefix}_active_tool_count"] = len(active.get("tools") or [])
    out[f"{prefix}_active_appear_this_turn"] = int(bool(active.get("appearThisTurn")))
    _attachment_identity_features(f"{prefix}_active", active, out)


def _attachment_identity_features(
    prefix: str,
    card: dict[str, Any],
    out: dict[str, float | int],
) -> None:
    """Expose exact attached-card identities instead of count-only summaries."""
    energy_ids = sorted(_attached_id(value) for value in _energy_cards(card))
    tool_ids = sorted(_attached_id(value) for value in (card.get("tools") or []))
    evolution_ids = [
        _attached_id(value)
        for value in (card.get("preEvolution") or [])
    ]
    for index in range(4):
        out[f"{prefix}_energy_id_{index}"] = (
            energy_ids[index] if index < len(energy_ids) else -1
        )
    for index in range(2):
        out[f"{prefix}_tool_id_{index}"] = (
            tool_ids[index] if index < len(tool_ids) else -1
        )
    for index in range(2):
        out[f"{prefix}_pre_evolution_id_{index}"] = (
            evolution_ids[index] if index < len(evolution_ids) else -1
        )


def _bench_aggregates(prefix: str, player: dict[str, Any], out: dict[str, float | int]) -> None:
    bench = _cards(player, "bench")
    hps = [float(card.get("hp", 0)) for card in bench]
    max_hps = [float(card.get("maxHp", 0)) for card in bench]
    energies = [_energy_count(card) for card in bench]
    out[f"{prefix}_bench_open"] = max(0, int(player.get("benchMax", 5)) - len(bench))
    out[f"{prefix}_bench_min_hp"] = min(hps) if hps else 0.0
    out[f"{prefix}_bench_max_hp"] = max(hps) if hps else 0.0
    out[f"{prefix}_bench_max_energy"] = max(energies) if energies else 0
    out[f"{prefix}_bench_total_damage"] = sum(max(0.0, max_hp - hp) for hp, max_hp in zip(hps, max_hps))
    out[f"{prefix}_bench_damaged_count"] = sum(int(hp < max_hp) for hp, max_hp in zip(hps, max_hps))
    out[f"{prefix}_bench_low_hp_count"] = sum(int(hp <= 80) for hp in hps)


def _ordered_board_features(
    prefix: str,
    player: dict[str, Any],
    out: dict[str, float | int],
) -> None:
    """Preserve the identities and condition of individual Bench slots.

    v28 only exposed aggregate Bench HP/Energy.  Expert MAIN sequencing changes
    materially when one specific Abra, Dudunsparce, or support pivot is ready,
    damaged, or newly played, so v29 keeps a bounded five-slot description.
    """
    bench = _cards(player, "bench")
    for index in range(5):
        card = bench[index] if index < len(bench) else {}
        max_hp = float(card.get("maxHp", 0))
        hp = float(card.get("hp", 0))
        out[f"{prefix}_bench_slot_{index}_id"] = int(card.get("id", -1))
        out[f"{prefix}_bench_slot_{index}_hp"] = hp
        out[f"{prefix}_bench_slot_{index}_damage"] = max(0.0, max_hp - hp)
        out[f"{prefix}_bench_slot_{index}_energy"] = _energy_count(card)
        out[f"{prefix}_bench_slot_{index}_special_energy"] = _special_energy_count(card)
        out[f"{prefix}_bench_slot_{index}_appear_this_turn"] = int(
            bool(card.get("appearThisTurn"))
        )
        _attachment_identity_features(
            f"{prefix}_bench_slot_{index}",
            card,
            out,
        )


def state_features(current: dict[str, Any]) -> dict[str, float | int]:
    players = current.get("players", [{}, {}])
    your = int(current.get("yourIndex", 0))
    me, opp = players[your], players[1 - your]
    me_in_play, opp_in_play = _in_play(me), _in_play(opp)
    self_hand_count = int(me.get("handCount", len(me.get("hand") or [])))
    self_deck_count = int(me.get("deckCount", 0))
    turn = int(current.get("turn", 0))
    out: dict[str, float | int] = {
        "turn": turn,
        "turn_action_count": int(current.get("turnActionCount", 0)),
        "first_player_is_self": int(current.get("firstPlayer", -1) == your),
        "energy_attached": int(bool(current.get("energyAttached"))),
        "retreated": int(bool(current.get("retreated"))),
        "stadium_played": int(bool(current.get("stadiumPlayed"))),
        "supporter_played": int(bool(current.get("supporterPlayed"))),
        "self_hand_count": self_hand_count,
        "self_deck_count": self_deck_count,
        "self_prize_count": len(me.get("prize") or []),
        "self_bench_count": len(me.get("bench") or []),
        "opp_hand_count": int(opp.get("handCount", 0)),
        "opp_deck_count": int(opp.get("deckCount", 0)),
        "opp_prize_count": len(opp.get("prize") or []),
        "opp_bench_count": len(opp.get("bench") or []),
        "self_total_energy": sum(_energy_count(c) for c in me_in_play),
        "opp_total_energy": sum(_energy_count(c) for c in opp_in_play),
        "self_total_hp": sum(float(c.get("hp", 0)) for c in me_in_play),
        "opp_total_hp": sum(float(c.get("hp", 0)) for c in opp_in_play),
        "self_status_count": sum(int(bool(me.get(k))) for k in ("asleep", "burned", "confused", "paralyzed", "poisoned")),
        "opp_status_count": sum(int(bool(opp.get(k))) for k in ("asleep", "burned", "confused", "paralyzed", "poisoned")),
        "early_game": int(turn <= 4),
        "mid_game": int(5 <= turn <= 10),
        "late_game": int(turn >= 11),
        "deck_low_10": int(self_deck_count <= 10),
        "deck_low_5": int(self_deck_count <= 5),
        "current_powerful_hand_damage": 20 * self_hand_count,
    }
    _active_features("self", me, out)
    _active_features("opp", opp, out)
    _bench_aggregates("self", me, out)
    _bench_aggregates("opp", opp, out)
    _ordered_board_features("self", me, out)
    _ordered_board_features("opp", opp, out)

    hand_counts = Counter(int(c.get("id", -1)) for c in _cards(me, "hand"))
    field_counts = Counter(int(c.get("id", -1)) for c in me_in_play)
    discard_counts = Counter(int(c.get("id", -1)) for c in _cards(me, "discard"))
    opp_field_counts = Counter(int(c.get("id", -1)) for c in opp_in_play)
    for card_id in KEY_CARD_IDS:
        out[f"hand_{card_id}"] = hand_counts[card_id]
        out[f"field_{card_id}"] = field_counts[card_id]
        out[f"discard_{card_id}"] = discard_counts[card_id]
        out[f"opp_field_{card_id}"] = opp_field_counts[card_id]

    active_cards = _cards(me, "active")
    bench_cards = _cards(me, "bench")
    ready_active_alakazam = sum(
        int(int(card.get("id", -1)) == 743 and _energy_count(card) >= 1)
        for card in active_cards
    )
    ready_bench_alakazam = sum(
        int(int(card.get("id", -1)) == 743 and _energy_count(card) >= 1)
        for card in bench_cards
    )
    ready_alakazam = ready_active_alakazam + ready_bench_alakazam
    field_route_bodies = field_counts[741] + field_counts[742] + field_counts[743]
    self_board_count = len(me_in_play)
    self_active_id = int(out["self_active_id"])
    support_pivot_ready = int(
        self_active_id in ONE_ENERGY_PIVOT_IDS
        and int(out["self_active_energy"]) == 0
        and ready_bench_alakazam > 0
    )
    prize_turn_floor = len(me.get("prize") or [])
    deck_runway_margin = self_deck_count - prize_turn_floor - 3
    visible_psychic_energy = sum(hand_counts[x] + discard_counts[x]
                                 for x in PSYCHIC_ENERGY_IDS)
    visible_psychic_energy += sum(
        int(
            (energy.get("id", energy.get("cardId", -1)) if isinstance(energy, dict) else energy)
            in PSYCHIC_ENERGY_IDS
        )
        for pokemon in me_in_play for energy in _energy_cards(pokemon)
    )
    psychic_energy_remaining = max(0, 6 - visible_psychic_energy)
    unseen_count = self_deck_count + len(me.get("prize") or [])
    has_psychic_in_hand = int(sum(hand_counts[x] for x in PSYCHIC_ENERGY_IDS) > 0)
    active_alakazam_unpowered = int(
        int(out["self_active_id"]) == 743 and int(out["self_active_energy"]) == 0
    )
    opp_discard_ids = [int(c.get("id", -1)) for c in _cards(opp, "discard")]
    opp_visible_energy_ids = []
    for pokemon in opp_in_play:
        for energy in _energy_cards(pokemon):
            if isinstance(energy, dict):
                opp_visible_energy_ids.append(int(energy.get("id", energy.get("cardId", -1))))
            else:
                opp_visible_energy_ids.append(int(energy))
    opp_visible_mist_count = (
        sum(card_id == MIST_ENERGY_ID for card_id in opp_discard_ids)
        + sum(card_id == MIST_ENERGY_ID for card_id in opp_visible_energy_ids)
    )
    opp_signature_ids = set(opp_field_counts) | set(opp_discard_ids)
    enrich_cycle_ready = int(
        field_counts[66] > 0 or (field_counts[305] > 0 and hand_counts[66] > 0)
    )
    enrich_draw_safe = int(self_deck_count - 4 > max(8, len(me.get("prize") or []) + 3))
    out.update({
        "ready_alakazam_count": ready_alakazam,
        "has_ready_alakazam": int(ready_alakazam > 0),
        "ready_active_alakazam_count": ready_active_alakazam,
        "has_ready_active_alakazam": int(ready_active_alakazam > 0),
        "ready_bench_alakazam_count": ready_bench_alakazam,
        "has_ready_backup_alakazam": int(ready_bench_alakazam > 0),
        "active_is_fezandipiti": int(int(out["self_active_id"]) == 140),
        "active_is_setup_wall": int(int(out["self_active_id"]) in {66, 140, 305, 343}),
        "active_is_one_energy_pivot": int(self_active_id in ONE_ENERGY_PIVOT_IDS),
        "support_pivot_ready": support_pivot_ready,
        "self_board_count": self_board_count,
        "self_last_body_risk": int(self_board_count <= 1),
        "field_route_body_count": field_route_bodies,
        "has_backup_route_body": int(field_route_bodies >= 2),
        "has_abra_anywhere": int(field_counts[741] + hand_counts[741] > 0),
        "has_bridge_anywhere": int(field_counts[742] + hand_counts[742] + hand_counts[1079] > 0),
        "has_alakazam_anywhere": int(field_counts[743] + hand_counts[743] > 0),
        "has_psychic_energy_in_hand": has_psychic_in_hand,
        "visible_psychic_energy_count": visible_psychic_energy,
        "psychic_energy_remaining_estimate": psychic_energy_remaining,
        "psychic_hit_probability_draw2": _at_least_one_probability(
            unseen_count, psychic_energy_remaining, min(2, self_deck_count)
        ),
        "psychic_hit_probability_draw3": _at_least_one_probability(
            unseen_count, psychic_energy_remaining, min(3, self_deck_count)
        ),
        "active_alakazam_unpowered": active_alakazam_unpowered,
        "emergency_energy_draw_state": int(
            active_alakazam_unpowered and not has_psychic_in_hand and self_deck_count <= 10
        ),
        "complete_route_components": int(field_counts[741] + field_counts[742] + field_counts[743] > 0)
        + int(hand_counts[742] + hand_counts[1079] + field_counts[742] > 0)
        + int(hand_counts[743] + field_counts[743] > 0)
        + has_psychic_in_hand,
        "needs_first_abra": int(field_counts[741] + field_counts[742] + field_counts[743] == 0),
        "needs_stage2": int(field_counts[743] == 0),
        "needs_attacker_energy": int(ready_alakazam == 0 and field_counts[743] > 0),
        "dudunsparce_engine_count": field_counts[66],
        "dunsparce_engine_count": field_counts[305],
        "opp_froslass_count": opp_field_counts[FROSLASS_ID],
        "opp_has_froslass": int(opp_field_counts[FROSLASS_ID] > 0),
        "opp_has_grimmsnarl_ex": int(opp_field_counts[GRIMMSNARL_EX_ID] > 0),
        "opp_munkidori_count": opp_field_counts[MUNKIDORI_ID],
        "opp_has_munkidori": int(opp_field_counts[MUNKIDORI_ID] > 0),
        "opp_spread_package_count": (
            opp_field_counts[FROSLASS_ID]
            + opp_field_counts[MUNKIDORI_ID]
            + opp_field_counts[GRIMMSNARL_EX_ID]
        ),
        "self_has_shaymin": int(field_counts[343] > 0),
        "self_froslass_vulnerable_count": sum(
            field_counts[card_id] for card_id in FROSLASS_VULNERABLE_IDS
        ),
        "deck_runway_margin": deck_runway_margin,
        "deck_pressure_risk": int(deck_runway_margin <= 4),
        "self_has_enriching_in_hand": int(hand_counts[13] > 0),
        "enrich_cycle_ready": enrich_cycle_ready,
        "enrich_draw_safe": enrich_draw_safe,
        "opp_visible_mist_count": opp_visible_mist_count,
        "opp_mist_signature_high": int(bool(opp_signature_ids & MIST_HIGH_SIGNATURE_IDS)),
        "opp_mist_signature_medium": int(bool(opp_signature_ids & MIST_MEDIUM_SIGNATURE_IDS)),
        "opp_active_has_prevent_energy": int(
            MIST_ENERGY_ID in [
                int(e.get("id", e.get("cardId", -1))) if isinstance(e, dict) else int(e)
                for e in _energy_cards((_cards(opp, "active") or [{}])[0])
            ]
        ),
    })
    ready_active_lopunny = sum(
        int(int(card.get("id", -1)) == MEGA_LOPUNNY_EX_ID and _energy_count(card) >= 1)
        for card in active_cards
    )
    ready_bench_lopunny = sum(
        int(int(card.get("id", -1)) == MEGA_LOPUNNY_EX_ID and _energy_count(card) >= 1)
        for card in bench_cards
    )
    spiky_ready_lopunny = sum(
        int(int(card.get("id", -1)) == MEGA_LOPUNNY_EX_ID and _energy_count(card) >= 2)
        for card in active_cards + bench_cards
    )
    out.update({
        "ready_active_lopunny_count": ready_active_lopunny,
        "ready_bench_lopunny_count": ready_bench_lopunny,
        "ready_lopunny_count": ready_active_lopunny + ready_bench_lopunny,
        "spiky_ready_lopunny_count": spiky_ready_lopunny,
        "has_ready_active_lopunny": int(ready_active_lopunny > 0),
        "has_ready_backup_lopunny": int(ready_bench_lopunny > 0),
        "has_spiky_ready_lopunny": int(spiky_ready_lopunny > 0),
        "buneary_route_count": field_counts[BUNEARY_ID],
        "lopunny_route_count": field_counts[MEGA_LOPUNNY_EX_ID],
        "has_buneary_in_hand": int(hand_counts[BUNEARY_ID] > 0),
        "has_lopunny_in_hand": int(hand_counts[MEGA_LOPUNNY_EX_ID] > 0),
        "has_air_balloon_in_hand": int(hand_counts[AIR_BALLOON_ID] > 0),
        "has_enriching_energy_in_hand_lopunny": int(hand_counts[13] > 0),
        "lopunny_special_energy_in_hand": sum(
            hand_counts[card_id] for card_id in LOPUNNY_ENERGY_IDS
        ),
    })
    stadium = current.get("stadium") or []
    out["stadium_id"] = int(stadium[0].get("id", -1)) if stadium and isinstance(stadium[0], dict) else -1
    return out


TRACKED_LOG_TYPES = (2, 4, 6, 8, 10, 11, 12, 15)


def _first_nested_id(value: Any) -> int:
    if isinstance(value, dict):
        if "id" in value or "cardId" in value:
            return int(value.get("id", value.get("cardId", -1)))
        for child in value.values():
            result = _first_nested_id(child)
            if result >= 0:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _first_nested_id(child)
            if result >= 0:
                return result
    return -1


def observation_features(
    observation: dict[str, Any],
) -> dict[str, float | int]:
    """Features v29 omitted: public action history and selection context."""
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    your = int(current.get("yourIndex", 0))
    logs = [
        event
        for event in (observation.get("logs") or [])
        if isinstance(event, dict)
    ]
    last_turn_start = -1
    for index, event in enumerate(logs):
        if int(event.get("type", -1)) == 2:
            last_turn_start = index
    turn_logs = logs[last_turn_start:] if last_turn_start >= 0 else logs

    remain_energy_cost = select.get("remainEnergyCost")
    if isinstance(remain_energy_cost, list):
        remain_energy_cost_count = len(remain_energy_cost)
    else:
        remain_energy_cost_count = max(0, _as_int(remain_energy_cost, 0))
    out: dict[str, float | int] = {
        "public_log_count": len(logs),
        "current_turn_log_count": len(turn_logs),
        "select_context_card_id": _first_nested_id(
            select.get("contextCard")
        ),
        "select_effect_card_id": _first_nested_id(select.get("effect")),
        "select_deck_count": len(select.get("deck") or []),
        "select_remain_damage_counter": int(
            select.get("remainDamageCounter") or 0
        ),
        "select_remain_energy_cost_count": remain_energy_cost_count,
        "current_looking_count": len(current.get("looking") or []),
        "search_begin_input": int(
            bool(observation.get("search_begin_input"))
        ),
    }
    for scope_name, events in (("history", logs), ("turn", turn_logs)):
        for log_type in TRACKED_LOG_TYPES:
            out[f"{scope_name}_self_log_type_{log_type}"] = sum(
                int(
                    int(event.get("type", -1)) == log_type
                    and int(event.get("playerIndex", -1)) == your
                )
                for event in events
            )
            out[f"{scope_name}_opp_log_type_{log_type}"] = sum(
                int(
                    int(event.get("type", -1)) == log_type
                    and int(event.get("playerIndex", -1)) == 1 - your
                )
                for event in events
            )
    for slot in range(6):
        event = logs[-1 - slot] if slot < len(logs) else {}
        player = int(event.get("playerIndex", -1))
        out[f"recent_log_{slot}_type"] = int(event.get("type", -1))
        out[f"recent_log_{slot}_player"] = (
            0 if player == your else 1 if player == 1 - your else -1
        )
        out[f"recent_log_{slot}_card_id"] = int(
            event.get("cardId", -1)
        )
        out[f"recent_log_{slot}_from_area"] = int(
            event.get("fromArea", -1)
        )
        out[f"recent_log_{slot}_to_area"] = int(
            event.get("toArea", -1)
        )
    for card_id in KEY_CARD_IDS:
        out[f"turn_log_card_{card_id}_count"] = sum(
            int(event.get("cardId", -1) == card_id)
            for event in turn_logs
        )
        out[f"select_deck_card_{card_id}_count"] = sum(
            int(card.get("id", -1) == card_id)
            for card in (select.get("deck") or [])
            if isinstance(card, dict)
        )
    looking_ids = []
    for value in current.get("looking") or []:
        card_id = _first_nested_id(value)
        if card_id >= 0:
            looking_ids.append(card_id)
    looking_ids.sort()
    for index in range(5):
        out[f"current_looking_id_{index}"] = (
            looking_ids[index] if index < len(looking_ids) else -1
        )
    return out


def option_features(
    current: dict[str, Any],
    select: dict[str, Any],
    option: dict[str, Any],
    base_state: dict[str, float | int] | None = None,
    observation: dict[str, Any] | None = None,
    option_position: int = -1,
) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = (
        dict(base_state)
        if base_state is not None
        else state_features(current)
    )
    if observation is not None:
        out.update(observation_features(observation))
    card = candidate_card(current, option, select) or {}
    target = candidate_target(current, option) or {}
    your = int(current.get("yourIndex", 0))
    action = action_type(current, option, select)
    option_type = int(option.get("type", -1))
    card_id = int(card.get("id", -1))
    target_id = int(target.get("id", -1))
    target_energy = _energy_count(target)
    hand_cost = int(option_type in (7, 8, 9))
    evolution_draw = 0
    if option_type == 9 and card_id == 742:
        evolution_draw = 2
    elif option_type == 9 and card_id == 743:
        evolution_draw = 3
    rare_candy_route = int(
        action == "trainer" and card_id == 1079
        and int(out["field_741"]) > 0 and int(out["hand_743"]) > 0
    )
    if rare_candy_route:
        # Candy and Alakazam leave hand, then Alakazam draws three.
        hand_cost = 2
        evolution_draw = 3
    enriching_draw = int(action == "energy" and card_id == 13) * 4
    candidate_draw = evolution_draw + enriching_draw
    post_hand_count = max(0, int(out["self_hand_count"]) - hand_cost + candidate_draw)
    current_damage = int(out["current_powerful_hand_damage"])
    post_damage = 20 * post_hand_count
    opp_hp = float(out["opp_active_hp"])
    self_active_id = int(out["self_active_id"])
    is_powerful_hand_attack = int(option_type == 13 and self_active_id == 743)
    current_ko = int(self_active_id == 743 and current_damage >= opp_hp > 0)
    post_ko = int(self_active_id == 743 and post_damage >= opp_hp > 0)
    target_area = int(option.get("inPlayArea", option.get("area", -1)))
    legal_options = list(select.get("option") or [])
    offered_action_counts = Counter(
        action_type(current, offered, select) for offered in legal_options
    )
    offered_card_counts = Counter(
        int((candidate_card(current, offered, select) or {}).get("id", -1))
        for offered in legal_options
    )
    preceding_options = (
        legal_options[:option_position]
        if 0 <= option_position <= len(legal_options)
        else []
    )
    preceding_action_counts = Counter(
        action_type(current, offered, select)
        for offered in preceding_options
    )
    preceding_card_counts = Counter(
        int((candidate_card(current, offered, select) or {}).get("id", -1))
        for offered in preceding_options
    )

    out.update({
        "legal_option_count": len(legal_options),
        "select_type": int(select.get("type", -1)),
        "select_context": int(select.get("context", -1)),
        "select_min_count": int(select.get("minCount", 0)),
        "select_max_count": int(select.get("maxCount", 0)),
        "option_type": option_type,
        "candidate_option_position": option_position,
        "candidate_option_reverse_position": (
            len(legal_options) - 1 - option_position
            if option_position >= 0
            else -1
        ),
        "candidate_raw_index": int(
            option.get("index")
            if option.get("index") is not None
            else -1
        ),
        "candidate_raw_inplay_index": int(
            option.get("inPlayIndex")
            if option.get("inPlayIndex") is not None
            else -1
        ),
        "candidate_raw_player_relative": (
            0
            if int(
                option.get("playerIndex")
                if option.get("playerIndex") is not None
                else your
            )
            == your
            else 1
        ),
        "candidate_same_action_preceding": preceding_action_counts[action],
        "candidate_same_card_preceding": preceding_card_counts[card_id],
        "candidate_is_first_action_copy": int(
            preceding_action_counts[action] == 0
        ),
        "candidate_is_first_card_copy": int(
            preceding_card_counts[card_id] == 0
        ),
        "candidate_card_id": card_id,
        "candidate_attack_id": int(option.get("attackId", -1)),
        "candidate_number": _as_int(option.get("number"), -1),
        "candidate_skill_id": _as_int(option.get("skillId"), -1),
        "candidate_special_condition": _as_int(
            option.get("specialCondition"), -1
        ),
        "candidate_area": int(option.get("area", -1)),
        "candidate_inplay_area": int(option.get("inPlayArea", -1)),
        "candidate_target_id": target_id,
        "candidate_target_hp": float(target.get("hp", 0)),
        "candidate_target_max_hp": float(target.get("maxHp", 0)),
        "candidate_target_energy": target_energy,
        "candidate_target_special_energy": _special_energy_count(target),
        "candidate_target_appear_this_turn": int(bool(target.get("appearThisTurn"))),
        "candidate_target_is_active": int(target_area == 4),
        "candidate_target_is_bench": int(target_area == 5),
        "candidate_hand_cost": hand_cost,
        "candidate_evolution_draw_count": evolution_draw,
        "candidate_enriching_draw_count": enriching_draw,
        "candidate_total_draw_count": candidate_draw,
        "candidate_net_hand_delta": candidate_draw - hand_cost,
        "post_action_hand_count": post_hand_count,
        "post_action_powerful_hand_damage": post_damage,
        "current_ko_estimate": current_ko,
        "post_action_ko_estimate": post_ko,
        "breaks_current_ko_estimate": int(current_ko and not post_ko),
        "preserves_current_ko_estimate": int(not current_ko or post_ko),
        "powerful_hand_attack_option": is_powerful_hand_attack,
        "attack_lethal_estimate": int(is_powerful_hand_attack and current_damage >= opp_hp > 0),
        "attack_overkill_estimate": max(0.0, current_damage - opp_hp) if is_powerful_hand_attack else 0.0,
        "candidate_is_route_card": int(card_id in ROUTE_IDS),
        "candidate_is_abra": int(card_id == 741),
        "candidate_is_kadabra": int(card_id == 742),
        "candidate_is_alakazam": int(card_id == 743),
        "candidate_is_rare_candy": int(card_id == 1079),
        "candidate_is_psychic_energy": int(card_id in PSYCHIC_ENERGY_IDS),
        "candidate_is_enriching_energy": int(card_id == 13),
        "candidate_enrich_cycle_target": int(
            card_id == 13 and target_id in {66, 305}
            and int(out["enrich_cycle_ready"]) == 1
        ),
        "candidate_enrich_draw_safe": int(
            card_id == 13 and int(out["enrich_draw_safe"]) == 1
        ),
        "target_is_abra": int(target_id == 741),
        "target_is_kadabra": int(target_id == 742),
        "target_is_alakazam": int(target_id == 743),
        "target_is_dunsparce": int(target_id == 305),
        "target_is_dudunsparce": int(target_id == 66),
        "candidate_fills_missing_abra": int(card_id == 741 and int(out["needs_first_abra"]) == 1),
        "candidate_fills_stage2": int(card_id == 743 and int(out["needs_stage2"]) == 1),
        "candidate_fills_energy": int(action == "energy" and int(out["needs_attacker_energy"]) == 1),
        "rare_candy_route_available": rare_candy_route,
        "rare_candy_projected_damage": post_damage if rare_candy_route else 0,
        "rare_candy_immediate_ko_estimate": int(
            rare_candy_route and post_damage >= opp_hp > 0
            and (int(out["self_active_energy"]) > 0 or int(out["has_psychic_energy_in_hand"]) == 1)
        ),
        "kadabra_draws_toward_candy_for_active_abra": int(
            action == "evolve" and card_id == 742 and target_area == 5
            and self_active_id == 741 and int(out["hand_743"]) > 0
            and int(out["hand_1079"]) == 0
        ),
        "setup_dunsparce_over_abra": int(
            int(select.get("context", -1)) == 1 and card_id == 305
            and int(out["hand_741"]) > 0 and int(out["hand_305"]) > 0
        ),
        "same_action_option_count": offered_action_counts[action],
        "same_card_option_count": offered_card_counts[card_id],
        "action_type": action,
    })
    for name in ("attack", "retreat", "end", "evolve", "ability", "energy", "boss", "xerosic", "hammer", "bench", "trainer", "other"):
        out[f"is_{name}"] = int(action == name)
        out[f"offered_{name}_count"] = offered_action_counts[name]
        out[f"candidate_is_only_{name}"] = int(
            action == name and offered_action_counts[name] == 1
        )
    for offered_card_id in KEY_CARD_IDS:
        out[f"offered_card_{offered_card_id}_count"] = (
            offered_card_counts[offered_card_id]
        )

    # Explicit action/state interactions make context learnable without requiring
    # deep trees to rediscover every conjunction from scratch.
    out.update({
        "hammer_opp_active_energy_value": int(action == "hammer") * int(out["opp_active_energy"]),
        "hammer_opp_active_special_energy_value": int(action == "hammer") * int(out["opp_active_special_energy"]),
        "xerosic_opp_hand_value": int(action == "xerosic") * int(out["opp_hand_count"]),
        "boss_opp_bench_value": int(action == "boss") * int(out["opp_bench_count"]),
        "boss_opp_bench_low_hp_value": int(action == "boss") * int(out["opp_bench_low_hp_count"]),
        "retreat_active_damage_value": int(action == "retreat") * float(out["self_active_damage"]),
        "retreat_status_value": int(action == "retreat") * int(out["self_status_count"]),
        "energy_target_existing_energy_value": int(action == "energy") * target_energy,
        "energy_target_is_alakazam_value": int(action == "energy" and target_id == 743),
        "energy_target_is_fezandipiti_value": int(action == "energy" and target_id == 140),
        "energy_support_pivot_value": int(
            action == "energy"
            and target_area == 4
            and target_id in ONE_ENERGY_PIVOT_IDS
            and int(out["support_pivot_ready"]) == 1
        ),
        "evolve_target_is_abra_value": int(action == "evolve" and target_id == 741),
        "evolve_target_is_kadabra_value": int(action == "evolve" and target_id == 742),
        "evolve_kadabra_to_active": int(
            action == "evolve" and card_id == 742 and target_area == 4
        ),
        "evolve_kadabra_to_bench": int(
            action == "evolve" and card_id == 742 and target_area == 5
        ),
        "evolve_kadabra_target_damage": (
            int(action == "evolve" and card_id == 742)
            * max(0.0, float(target.get("maxHp", 0)) - float(target.get("hp", 0)))
        ),
        "evolve_kadabra_under_froslass": int(
            action == "evolve" and card_id == 742 and int(out["opp_has_froslass"]) == 1
        ),
        "bench_shaymin_into_grimmsnarl": int(
            action == "bench" and card_id == 343 and int(out["opp_has_grimmsnarl_ex"]) == 1
        ),
        "bench_shaymin_into_spread_package": int(
            action == "bench" and card_id == 343 and int(out["opp_spread_package_count"]) > 0
        ),
        "bench_backup_route_under_boardout_risk": int(
            action == "bench" and card_id == 741 and int(out["self_board_count"]) <= 2
        ),
        "bench_early_game_value": int(action == "bench") * int(out["early_game"]),
        "ability_low_deck_risk": int(action == "ability") * int(out["deck_low_5"]),
        "trainer_low_deck_risk": int(action == "trainer") * int(out["deck_low_5"]),
        "optional_draw_under_deck_pressure": int(
            action in {"ability", "trainer"} and int(out["deck_pressure_risk"]) == 1
        ),
        "emergency_energy_draw_value": int(
            action == "ability" and int(out["emergency_energy_draw_state"]) == 1
        ) * float(out["psychic_hit_probability_draw3"]),
        "end_has_attack_ready_penalty": int(action == "end") * int(out["has_ready_alakazam"]),
        "end_ready_active_alakazam_penalty": int(action == "end") * int(out["has_ready_active_alakazam"]),
        "attack_with_ready_active_alakazam": int(action == "attack") * int(out["has_ready_active_alakazam"]),
        "retreat_to_ready_backup_value": int(action == "retreat") * int(out["has_ready_backup_alakazam"]),
        "retreat_support_pivot_value": int(action == "retreat") * int(out["support_pivot_ready"]),
        "ability_repositions_to_ready_backup_value": (
            int(action == "ability")
            * int(out["self_active_id"] == 66)
            * int(out["has_ready_backup_alakazam"])
        ),
        "evolve_builds_active_alakazam": int(
            action == "evolve" and card_id == 743 and target_area == 4
        ),
        "evolve_builds_backup_alakazam": int(
            action == "evolve" and card_id == 743 and target_area == 5
        ),
    })
    attack_id = int(option.get("attackId", -1))
    # Gale Thrust's 170 bonus is active after a self Switch/Retreat in the
    # current turn.  The public log is the only runtime-observable source, so
    # keep both the conservative 60 and switched 230 estimates visible.
    self_switches = int(out.get("turn_self_log_type_8", 0))
    gale_damage = 230 if self_switches > 0 else 60
    attack_damage = (
        gale_damage if attack_id == GALE_THRUST_ID
        else 160 if attack_id == SPIKY_HOPPER_ID
        else 0
    )
    out.update({
        "candidate_is_buneary": int(card_id == BUNEARY_ID),
        "candidate_is_lopunny": int(card_id == MEGA_LOPUNNY_EX_ID),
        "candidate_is_wally": int(card_id == WALLY_ID),
        "candidate_is_lillie": int(card_id == LILLIE_ID),
        "candidate_is_air_balloon": int(card_id == AIR_BALLOON_ID),
        "candidate_is_ultra_ball": int(card_id == ULTRA_BALL_ID),
        "candidate_is_pokegear": int(card_id == POKEGEAR_ID),
        "candidate_is_lopunny_energy": int(card_id in LOPUNNY_ENERGY_IDS),
        "target_is_buneary": int(target_id == BUNEARY_ID),
        "target_is_lopunny": int(target_id == MEGA_LOPUNNY_EX_ID),
        "evolve_builds_active_lopunny": int(
            action == "evolve" and card_id == MEGA_LOPUNNY_EX_ID and target_area == 4
        ),
        "evolve_builds_backup_lopunny": int(
            action == "evolve" and card_id == MEGA_LOPUNNY_EX_ID and target_area == 5
        ),
        "energy_target_is_lopunny": int(
            action == "energy" and target_id == MEGA_LOPUNNY_EX_ID
        ),
        "energy_target_is_buneary": int(
            action == "energy" and target_id == BUNEARY_ID
        ),
        "gale_thrust_option": int(attack_id == GALE_THRUST_ID),
        "spiky_hopper_option": int(attack_id == SPIKY_HOPPER_ID),
        "gale_thrust_switch_observed": int(
            attack_id == GALE_THRUST_ID and self_switches > 0
        ),
        "lopunny_attack_damage_estimate": attack_damage,
        "lopunny_attack_lethal_estimate": int(
            attack_damage > 0 and 0 < float(out["opp_active_hp"]) <= attack_damage
        ),
        "end_has_lopunny_attack_penalty": int(action == "end")
        * int(out["has_ready_active_lopunny"]),
        "attack_with_ready_active_lopunny": int(action == "attack")
        * int(out["has_ready_active_lopunny"]),
        "retreat_to_ready_backup_lopunny": int(action == "retreat")
        * int(out["has_ready_backup_lopunny"]),
    })
    return out


def semantic_option_key(
    current: dict[str, Any],
    select: dict[str, Any],
    option: dict[str, Any],
    option_position: int = -1,
) -> tuple[int | float, ...]:
    """Canonical action identity used for duplicate-tolerant agreement.

    Hand/deck indices and card serials are intentionally omitted: choosing the
    first or second indistinguishable copy is the same game action.  Board
    targets retain their area/index and condition because two copies can have
    different HP, Energy, tools, or evolution history.
    """
    feature = option_features(
        current, select, option, option_position=option_position
    )
    return semantic_feature_key(feature)


def semantic_feature_key(
    feature: dict[str, float | int | str],
) -> tuple[int | float, ...]:
    """Fast semantic identity when candidate features already exist."""
    area = int(feature.get("candidate_area", -1))
    board_index = (
        int(feature.get("candidate_raw_index", -1))
        if area in (4, 5, 8, 9, 10)
        else -1
    )
    return (
        int(feature.get("option_type", -1)),
        int(feature.get("candidate_card_id", -1)),
        int(feature.get("candidate_attack_id", -1)),
        int(feature.get("candidate_number", -1)),
        int(feature.get("candidate_skill_id", -1)),
        int(feature.get("candidate_special_condition", -1)),
        int(feature.get("candidate_raw_player_relative", -1)),
        int(feature.get("candidate_area", -1)),
        int(feature.get("candidate_inplay_area", -1)),
        int(feature.get("candidate_raw_inplay_index", -1)),
        board_index,
        int(feature.get("candidate_target_id", -1)),
        float(feature.get("candidate_target_hp", 0)),
        int(feature.get("candidate_target_energy", 0)),
    )


def decision_features(
    observation: dict[str, Any],
) -> dict[str, float | int]:
    """Candidate-independent features for predicting a variable pick count."""
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    options = [
        option for option in (select.get("option") or [])
        if isinstance(option, dict)
    ]
    out: dict[str, float | int] = dict(state_features(current))
    out.update(observation_features(observation))
    out.update({
        "select_type": int(select.get("type", -1)),
        "select_context": int(select.get("context", -1)),
        "select_min_count": int(select.get("minCount", 0)),
        "select_max_count": int(select.get("maxCount", 0)),
        "legal_option_count": len(options),
    })
    option_types = Counter(int(option.get("type", -1)) for option in options)
    offered_cards = Counter(
        int((candidate_card(current, option, select) or {}).get("id", -1))
        for option in options
    )
    for option_type in range(17):
        out[f"offered_option_type_{option_type}_count"] = option_types[option_type]
    for card_id in KEY_CARD_IDS:
        out[f"offered_card_{card_id}_count"] = offered_cards[card_id]
    return out


LEAKAGE_DENYLIST = {
    "reward", "target_reward", "target_win", "target_loss", "result", "winner",
    "initial_deck_json", "deck_hash", "deck_type", "majkel_distance", "visualize",
}


def assert_no_leakage(feature_columns: list[str]) -> None:
    leaked = sorted(set(feature_columns) & LEAKAGE_DENYLIST)
    if leaked:
        raise ValueError(f"Leakage columns present in policy features: {leaked}")
