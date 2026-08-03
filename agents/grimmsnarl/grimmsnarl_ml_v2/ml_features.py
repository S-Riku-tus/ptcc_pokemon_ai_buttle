"""Feature extraction for the Marnie's Grimmsnarl ex imitation ranker.

Standard library only: this module ships inside the Kaggle runtime and is also
imported by the offline corpus builder, so training and inference cannot drift.

Structure follows the Alakazam v29-v35 module, which is the part of that line
that worked: generic board/hand/log description plus a bounded set of
archetype interactions. The archetype half is rewritten for this deck.

Deck (hash 9714ab5c3996f6cc, 21 of the top-50 teams run it card for card):

    10 Basic {D} Energy 7      4 Marnie's Impidimp 646
     3 Marnie's Morgrem 647    3 Marnie's Grimmsnarl ex 648
     2 Snorunt 860             2 Froslass 104
     4 Munkidori 112           3 Rare Candy 1079
     1 Unfair Stamp 1080       4 Buddy-Buddy Poffin 1086
     3 Night Stretcher 1097    1 Pokegear 3.0 1122
     1 Tool Scrapper 1137      4 Poke Pad 1152
     2 Boss's Orders 1182      4 Team Rocket's Petrel 1219
     4 Lillie's Determination 1227
     1 Dawn 1231               4 Spikemuth Gym 1259

The engine is Impidimp -> Morgrem -> Grimmsnarl ex, or Impidimp -> Rare Candy
-> Grimmsnarl ex. Evolving into Grimmsnarl ex triggers Punk Up, which searches
up to five Basic {D} Energy out of the deck and attaches them anywhere, so the
evolve step and the energy step are not independent decisions the way they are
in most decks. Shadow Bullet is 180 to the Active plus 30 to one Bench.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

# Engine area codes.
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6
AREA_NAMES = {
    AREA_HAND: "hand",
    AREA_DISCARD: "discard",
    AREA_ACTIVE: "active",
    AREA_BENCH: "bench",
}

# Select contexts v2 scores with the ranker. v1 scored MAIN only and left
# these to the rule policy, which agreed with the pinned teacher just 39.5%
# of the time on deck search and 50-65% on damage-counter placement.
MAIN_CONTEXT = 0
CTX_SWITCH = 3
CTX_TO_ACTIVE = 4
CTX_TO_HAND = 7
CTX_DAMAGE_COUNTER = 13
CTX_DAMAGE = 15
CTX_REMOVE_DAMAGE_COUNTER = 16
CTX_ATTACH_FROM = 21
CTX_ATTACH_TO = 22
CTX_REMOVE_COUNTER_COUNT = 40
CTX_ACTIVATE = 43

# Contexts whose options carry a resolvable card identity or a number, and
# where the teacher never declined across the corpus, so a plain argmax over
# candidates reproduces the decision.
SCORABLE_CONTEXTS = frozenset({
    MAIN_CONTEXT, 1, 2, CTX_SWITCH, CTX_TO_ACTIVE, 5, CTX_TO_HAND, 8,
    CTX_DAMAGE_COUNTER, 14, CTX_DAMAGE, CTX_REMOVE_DAMAGE_COUNTER, 17,
    CTX_ATTACH_FROM, CTX_ATTACH_TO, 26, 30, 37, 38, 39,
    CTX_REMOVE_COUNTER_COUNT, 41, CTX_ACTIVATE, 44,
})

DARK_ENERGY_ID = 7
IMPIDIMP_ID = 646
MORGREM_ID = 647
GRIMMSNARL_EX_ID = 648
SNORUNT_ID = 860
FROSLASS_ID = 104
MUNKIDORI_ID = 112
RARE_CANDY_ID = 1079
UNFAIR_STAMP_ID = 1080
POFFIN_ID = 1086
NIGHT_STRETCHER_ID = 1097
POKEGEAR_ID = 1122
TOOL_SCRAPPER_ID = 1137
POKE_PAD_ID = 1152
BOSS_ID = 1182
PETREL_ID = 1219
LILLIE_ID = 1227
DAWN_ID = 1231
SPIKEMUTH_GYM_ID = 1259

SHADOW_BULLET_ID = 937
SHADOW_BULLET_DAMAGE = 180.0
SHADOW_BULLET_BENCH_DAMAGE = 30.0
SHADOW_BULLET_COST = 2

KEY_CARD_IDS = [
    DARK_ENERGY_ID, FROSLASS_ID, MUNKIDORI_ID, IMPIDIMP_ID, MORGREM_ID,
    GRIMMSNARL_EX_ID, SNORUNT_ID, RARE_CANDY_ID, UNFAIR_STAMP_ID, POFFIN_ID,
    NIGHT_STRETCHER_ID, POKEGEAR_ID, TOOL_SCRAPPER_ID, POKE_PAD_ID, BOSS_ID,
    PETREL_ID, LILLIE_ID, DAWN_ID, SPIKEMUTH_GYM_ID,
]
ENERGY_IDS = {DARK_ENERGY_ID}
BASIC_POKEMON_IDS = {IMPIDIMP_ID, SNORUNT_ID, MUNKIDORI_ID}
MARNIE_LINE_IDS = {IMPIDIMP_ID, MORGREM_ID, GRIMMSNARL_EX_ID}
ROUTE_IDS = MARNIE_LINE_IDS | {RARE_CANDY_ID}
ABILITY_HOLDER_IDS = {FROSLASS_ID, MUNKIDORI_ID, GRIMMSNARL_EX_ID}
SUPPORTER_IDS = {PETREL_ID, LILLIE_ID, DAWN_ID, BOSS_ID}
ITEM_IDS = {
    RARE_CANDY_ID, POFFIN_ID, NIGHT_STRETCHER_ID, POKEGEAR_ID,
    TOOL_SCRAPPER_ID, POKE_PAD_ID, UNFAIR_STAMP_ID,
}
SPECIFIC_ACTIONS = {
    BOSS_ID: "boss",
    UNFAIR_STAMP_ID: "stamp",
    SPIKEMUTH_GYM_ID: "stadium",
}
# Froslass's Freezing Shroud hits every Pokemon that has an Ability, on both
# sides, so our own ability holders are a liability in the mirror.
FROSLASS_VULNERABLE_IDS = ABILITY_HOLDER_IDS

ACTION_TYPES = (
    "ability", "attack", "bench", "boss", "end", "energy", "evolve",
    "flag", "item", "number", "other", "retreat", "select", "skill",
    "stadium", "stamp", "supporter",
)


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


def _energy_count(card: dict[str, Any]) -> int:
    return len(_energy_cards(card))


def _dark_energy_count(card: dict[str, Any]) -> int:
    return sum(
        int(_attached_id(value) == DARK_ENERGY_ID)
        for value in _energy_cards(card)
    )


def _special_energy_count(card: dict[str, Any]) -> int:
    return sum(
        int(_attached_id(value) not in ENERGY_IDS)
        for value in _energy_cards(card)
    )


def _card_at_area(
    player: dict[str, Any],
    area: int | None,
    index: int | None,
) -> dict[str, Any] | None:
    if not isinstance(index, int):
        return None
    name = AREA_NAMES.get(int(area) if isinstance(area, int) else -1)
    if name is None:
        return None
    seq = _cards(player, name)
    return seq[index] if 0 <= index < len(seq) else None


def candidate_card(
    current: dict[str, Any],
    option: dict[str, Any],
    select: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """The card the option acts *with* (hand card, or in-play ability holder)."""
    your = int(current.get("yourIndex", 0))
    player = (current.get("players") or [{}, {}])[your]
    option_type = int(option.get("type", -1))
    if option_type in (7, 8, 9):
        return _card_at_area(player, AREA_HAND, option.get("index"))
    if option_type == 10:
        return _card_at_area(player, option.get("area"), option.get("index"))
    if option_type == 3:
        if _int(option.get("area")) == AREA_DECK and select is not None:
            deck = [c for c in (select.get("deck") or []) if isinstance(c, dict)]
            index = option.get("index")
            if isinstance(index, int) and 0 <= index < len(deck):
                return deck[index]
            return None
        return _card_at_area(player, option.get("area"), option.get("index"))
    return None


def candidate_target(
    current: dict[str, Any],
    option: dict[str, Any],
) -> dict[str, Any] | None:
    """The card the option acts *on* (evolve/attach destination, ability target)."""
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    owner = int(option.get("playerIndex", your))
    if owner not in (0, 1):
        owner = your
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    return _card_at_area(players[owner], area, index)


def _size(value: Any) -> int:
    """Length of a list-valued select field, or the scalar itself."""
    if isinstance(value, (list, tuple, dict)):
        return len(value)
    return int(value) if isinstance(value, (int, float)) else 0


def _scalar(value: Any) -> int:
    if isinstance(value, (list, tuple, dict)):
        return len(value)
    return int(value) if isinstance(value, (int, float)) else 0


def _int(value: Any, default: int = -1) -> int:
    """Options omit fields rather than null them, but never assume it."""
    return int(value) if isinstance(value, (int, float)) else default


def area_cards(
    current: dict[str, Any],
    select: dict[str, Any],
    player: dict[str, Any],
    area: Any,
) -> list[dict[str, Any]]:
    """Resolve an option area to a card list.

    Area 1 indexes the select's own revealed deck list rather than a board
    zone. Area 6 is the face-down prize zone: the observation exposes no ids,
    so it stays unresolved and those options are told apart by slot only.
    """
    code = int(area) if isinstance(area, int) else -1
    if code == AREA_DECK:
        return [c for c in (select.get("deck") or []) if isinstance(c, dict)]
    name = AREA_NAMES.get(code)
    return _cards(player, name) if name else []


def resolve_option(
    current: dict[str, Any],
    select: dict[str, Any],
    option: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool, int]:
    """(card, owner_is_self, area) for a generic ``type: 3`` select option."""
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    owner = option.get("playerIndex")
    owner = int(owner) if isinstance(owner, int) and owner in (0, 1) else your
    player = players[owner] if owner < len(players) else {}
    area = _int(option.get("area"))
    cards = area_cards(current, select, player, area)
    index = option.get("index")
    card = (
        cards[index]
        if isinstance(index, int) and 0 <= index < len(cards) else None
    )
    return card, owner == your, area


def action_type(
    current: dict[str, Any],
    option: dict[str, Any],
    select: dict[str, Any] | None = None,
) -> str:
    option_type = int(option.get("type", -1))
    if option_type == 3:
        return "select"
    if option_type == 0:
        return "number"
    if option_type == 15:
        return "skill"
    if option_type in (1, 2):
        return "flag"
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
    if option_type == 8:
        return "energy"
    if option_type == 7:
        card_id = int((candidate_card(current, option, select) or {}).get("id", -1))
        if card_id in SPECIFIC_ACTIONS:
            return SPECIFIC_ACTIONS[card_id]
        if card_id in ENERGY_IDS:
            return "energy"
        if card_id in BASIC_POKEMON_IDS:
            return "bench"
        if card_id in SUPPORTER_IDS:
            return "supporter"
        if card_id in ITEM_IDS:
            return "item"
        return "item"
    return "other"


def _at_least_one_probability(
    population: int,
    successes: int,
    draws: int,
) -> float:
    population = max(0, int(population))
    successes = max(0, min(int(successes), population))
    draws = max(0, min(int(draws), population))
    if draws == 0 or successes == 0:
        return 0.0
    failures = population - successes
    if failures < draws:
        return 1.0
    return 1.0 - math.comb(failures, draws) / math.comb(population, draws)


def _attachment_identity_features(
    prefix: str,
    card: dict[str, Any],
    out: dict[str, float | int],
) -> None:
    energy_ids = sorted(_attached_id(v) for v in _energy_cards(card))
    tool_ids = sorted(_attached_id(v) for v in (card.get("tools") or []))
    evolution_ids = [_attached_id(v) for v in (card.get("preEvolution") or [])]
    for index in range(5):
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


def _active_features(
    prefix: str,
    player: dict[str, Any],
    out: dict[str, float | int],
) -> None:
    active = (_cards(player, "active") or [{}])[0]
    max_hp = float(active.get("maxHp", 0))
    hp = float(active.get("hp", 0))
    out[f"{prefix}_active_id"] = int(active.get("id", -1))
    out[f"{prefix}_active_hp"] = hp
    out[f"{prefix}_active_max_hp"] = max_hp
    out[f"{prefix}_active_damage"] = max(0.0, max_hp - hp)
    out[f"{prefix}_active_energy"] = _energy_count(active)
    out[f"{prefix}_active_dark_energy"] = _dark_energy_count(active)
    out[f"{prefix}_active_special_energy"] = _special_energy_count(active)
    out[f"{prefix}_active_tool_count"] = len(active.get("tools") or [])
    out[f"{prefix}_active_appear_this_turn"] = int(
        bool(active.get("appearThisTurn"))
    )
    out[f"{prefix}_active_evolution_depth"] = len(
        active.get("preEvolution") or []
    )
    _attachment_identity_features(f"{prefix}_active", active, out)


def _bench_aggregates(
    prefix: str,
    player: dict[str, Any],
    out: dict[str, float | int],
) -> None:
    bench = _cards(player, "bench")
    hps = [float(c.get("hp", 0)) for c in bench]
    max_hps = [float(c.get("maxHp", 0)) for c in bench]
    energies = [_energy_count(c) for c in bench]
    out[f"{prefix}_bench_open"] = max(
        0, int(player.get("benchMax", 5)) - len(bench)
    )
    out[f"{prefix}_bench_min_hp"] = min(hps) if hps else 0.0
    out[f"{prefix}_bench_max_hp"] = max(hps) if hps else 0.0
    out[f"{prefix}_bench_max_energy"] = max(energies) if energies else 0
    out[f"{prefix}_bench_total_energy"] = sum(energies)
    out[f"{prefix}_bench_total_damage"] = sum(
        max(0.0, m - h) for h, m in zip(hps, max_hps)
    )
    out[f"{prefix}_bench_damaged_count"] = sum(
        int(h < m) for h, m in zip(hps, max_hps)
    )
    out[f"{prefix}_bench_low_hp_count"] = sum(int(h <= 70) for h in hps)
    # Shadow Bullet's 30 to the Bench and Froslass checkup counters make
    # "one more hit kills it" a real sequencing input on both sides.
    out[f"{prefix}_bench_snipe_range_count"] = sum(
        int(0 < h <= SHADOW_BULLET_BENCH_DAMAGE) for h in hps
    )


def _ordered_board_features(
    prefix: str,
    player: dict[str, Any],
    out: dict[str, float | int],
) -> None:
    """Keep each Bench slot's identity and condition, not just aggregates."""
    bench = _cards(player, "bench")
    for index in range(5):
        card = bench[index] if index < len(bench) else {}
        max_hp = float(card.get("maxHp", 0))
        hp = float(card.get("hp", 0))
        out[f"{prefix}_bench_slot_{index}_id"] = int(card.get("id", -1))
        out[f"{prefix}_bench_slot_{index}_hp"] = hp
        out[f"{prefix}_bench_slot_{index}_damage"] = max(0.0, max_hp - hp)
        out[f"{prefix}_bench_slot_{index}_energy"] = _energy_count(card)
        out[f"{prefix}_bench_slot_{index}_dark_energy"] = _dark_energy_count(card)
        out[f"{prefix}_bench_slot_{index}_appear_this_turn"] = int(
            bool(card.get("appearThisTurn"))
        )
        out[f"{prefix}_bench_slot_{index}_evolution_depth"] = len(
            card.get("preEvolution") or []
        )
        _attachment_identity_features(
            f"{prefix}_bench_slot_{index}", card, out
        )


def _attacker_state(cards: list[dict[str, Any]]) -> tuple[int, int]:
    """(Grimmsnarl ex bodies, Grimmsnarl ex bodies that can attack now)."""
    bodies = 0
    ready = 0
    for card in cards:
        if int(card.get("id", -1)) != GRIMMSNARL_EX_ID:
            continue
        bodies += 1
        if _energy_count(card) >= SHADOW_BULLET_COST:
            ready += 1
    return bodies, ready


def state_features(current: dict[str, Any]) -> dict[str, float | int]:
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    me, opp = players[your], players[1 - your]
    me_in_play, opp_in_play = _in_play(me), _in_play(opp)
    hand = _cards(me, "hand")
    self_hand_count = int(me.get("handCount", len(hand)))
    self_deck_count = int(me.get("deckCount", 0))
    self_prize_count = len(me.get("prize") or [])
    opp_prize_count = len(opp.get("prize") or [])
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
        "self_prize_count": self_prize_count,
        "self_bench_count": len(me.get("bench") or []),
        "opp_hand_count": int(opp.get("handCount", 0)),
        "opp_deck_count": int(opp.get("deckCount", 0)),
        "opp_prize_count": opp_prize_count,
        "opp_bench_count": len(opp.get("bench") or []),
        "prize_lead": opp_prize_count - self_prize_count,
        "self_total_energy": sum(_energy_count(c) for c in me_in_play),
        "self_total_dark_energy": sum(
            _dark_energy_count(c) for c in me_in_play
        ),
        "opp_total_energy": sum(_energy_count(c) for c in opp_in_play),
        "self_total_hp": sum(float(c.get("hp", 0)) for c in me_in_play),
        "opp_total_hp": sum(float(c.get("hp", 0)) for c in opp_in_play),
        "self_status_count": sum(
            int(bool(me.get(k)))
            for k in ("asleep", "burned", "confused", "paralyzed", "poisoned")
        ),
        "opp_status_count": sum(
            int(bool(opp.get(k)))
            for k in ("asleep", "burned", "confused", "paralyzed", "poisoned")
        ),
        "early_game": int(turn <= 4),
        "mid_game": int(5 <= turn <= 10),
        "late_game": int(turn >= 11),
        "deck_low_10": int(self_deck_count <= 10),
        "deck_low_5": int(self_deck_count <= 5),
    }
    _active_features("self", me, out)
    _active_features("opp", opp, out)
    _bench_aggregates("self", me, out)
    _bench_aggregates("opp", opp, out)
    _ordered_board_features("self", me, out)
    _ordered_board_features("opp", opp, out)

    hand_counts = Counter(int(c.get("id", -1)) for c in hand)
    field_counts = Counter(int(c.get("id", -1)) for c in me_in_play)
    discard_counts = Counter(
        int(c.get("id", -1)) for c in _cards(me, "discard")
    )
    opp_field_counts = Counter(int(c.get("id", -1)) for c in opp_in_play)
    for card_id in KEY_CARD_IDS:
        out[f"hand_{card_id}"] = hand_counts[card_id]
        out[f"field_{card_id}"] = field_counts[card_id]
        out[f"discard_{card_id}"] = discard_counts[card_id]
        out[f"opp_field_{card_id}"] = opp_field_counts[card_id]

    active_cards = _cards(me, "active")
    bench_cards = _cards(me, "bench")
    active_bodies, active_ready = _attacker_state(active_cards)
    bench_bodies, bench_ready = _attacker_state(bench_cards)
    total_ready = active_ready + bench_ready
    self_active_id = int(out["self_active_id"])

    # Punk Up fires on the evolve into Grimmsnarl ex, so an evolve that is
    # available right now is also an energy engine, not only a body upgrade.
    candy_route = int(
        hand_counts[RARE_CANDY_ID] > 0
        and hand_counts[GRIMMSNARL_EX_ID] > 0
        and field_counts[IMPIDIMP_ID] > 0
    )
    morgrem_route = int(
        hand_counts[GRIMMSNARL_EX_ID] > 0 and field_counts[MORGREM_ID] > 0
    )
    punk_up_available = int(bool(candy_route or morgrem_route))

    marnie_bodies = (
        field_counts[IMPIDIMP_ID]
        + field_counts[MORGREM_ID]
        + field_counts[GRIMMSNARL_EX_ID]
    )
    unseen = self_deck_count + self_prize_count
    dark_seen = (
        hand_counts[DARK_ENERGY_ID]
        + discard_counts[DARK_ENERGY_ID]
        + sum(_dark_energy_count(c) for c in me_in_play)
    )
    dark_remaining = max(0, 10 - dark_seen)

    munkidori_ready = sum(
        int(
            int(c.get("id", -1)) == MUNKIDORI_ID
            and _dark_energy_count(c) > 0
        )
        for c in me_in_play
    )
    opp_damaged_bench = sum(
        int(float(c.get("hp", 0)) < float(c.get("maxHp", 0)))
        for c in _cards(opp, "bench")
    )
    stadium_cards = [
        c for c in (current.get("stadium") or []) if isinstance(c, dict)
    ]
    stadium_id = int(stadium_cards[0].get("id", -1)) if stadium_cards else -1

    out.update({
        "stadium_id": stadium_id,
        "spikemuth_gym_active": int(stadium_id == SPIKEMUTH_GYM_ID),
        "attacker_body_count": active_bodies + bench_bodies,
        "active_is_attacker": int(self_active_id == GRIMMSNARL_EX_ID),
        "active_attacker_ready": active_ready,
        "backup_attacker_ready": bench_ready,
        "ready_attacker_count": total_ready,
        "has_ready_attacker": int(total_ready > 0),
        "active_attacker_energy_missing": max(
            0,
            SHADOW_BULLET_COST - int(out["self_active_dark_energy"]),
        ) if self_active_id == GRIMMSNARL_EX_ID else -1,
        "marnie_body_count": marnie_bodies,
        "has_impidimp_anywhere": int(
            field_counts[IMPIDIMP_ID] + hand_counts[IMPIDIMP_ID] > 0
        ),
        "has_grimmsnarl_in_hand": int(hand_counts[GRIMMSNARL_EX_ID] > 0),
        "candy_route_available": candy_route,
        "morgrem_route_available": morgrem_route,
        "punk_up_available": punk_up_available,
        "needs_first_marnie_body": int(marnie_bodies == 0),
        "needs_attacker": int(field_counts[GRIMMSNARL_EX_ID] == 0),
        "needs_attacker_energy": int(
            field_counts[GRIMMSNARL_EX_ID] > 0 and total_ready == 0
        ),
        "self_board_count": len(me_in_play),
        "self_last_body_risk": int(len(me_in_play) <= 1),
        "froslass_engine_count": field_counts[FROSLASS_ID],
        "has_froslass": int(field_counts[FROSLASS_ID] > 0),
        "snorunt_ready_to_evolve": int(
            field_counts[SNORUNT_ID] > 0 and hand_counts[FROSLASS_ID] > 0
        ),
        "munkidori_count": field_counts[MUNKIDORI_ID],
        "munkidori_ready_count": munkidori_ready,
        "has_ready_munkidori": int(munkidori_ready > 0),
        "self_ability_holder_count": sum(
            field_counts[card_id] for card_id in ABILITY_HOLDER_IDS
        ),
        "opp_ability_holder_count": sum(
            opp_field_counts[card_id] for card_id in ABILITY_HOLDER_IDS
        ),
        "opp_has_froslass": int(opp_field_counts[FROSLASS_ID] > 0),
        "opp_has_grimmsnarl_ex": int(
            opp_field_counts[GRIMMSNARL_EX_ID] > 0
        ),
        "opp_has_munkidori": int(opp_field_counts[MUNKIDORI_ID] > 0),
        "mirror_match_signal": int(
            opp_field_counts[GRIMMSNARL_EX_ID]
            + opp_field_counts[IMPIDIMP_ID]
            + opp_field_counts[MORGREM_ID] > 0
        ),
        "self_froslass_vulnerable_count": sum(
            field_counts[card_id] for card_id in FROSLASS_VULNERABLE_IDS
        ),
        "opp_damaged_bench_count": opp_damaged_bench,
        "visible_dark_energy_count": dark_seen,
        "dark_energy_remaining_estimate": dark_remaining,
        "dark_hit_probability_draw3": _at_least_one_probability(
            unseen, dark_remaining, min(3, self_deck_count)
        ),
        "dark_hit_probability_draw6": _at_least_one_probability(
            unseen, dark_remaining, min(6, self_deck_count)
        ),
        "deck_runway_margin": self_deck_count - self_prize_count - 3,
        "deck_pressure_risk": int(
            self_deck_count - self_prize_count - 3 <= 4
        ),
        "shadow_bullet_kills_active": int(
            total_ready > 0
            and 0 < float(out["opp_active_hp"]) <= SHADOW_BULLET_DAMAGE
        ),
        "opp_active_survives_shadow_bullet_by": max(
            0.0, float(out["opp_active_hp"]) - SHADOW_BULLET_DAMAGE
        ),
    })
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
    """Public action history and selection context."""
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    your = int(current.get("yourIndex", 0))
    logs = [e for e in (observation.get("logs") or []) if isinstance(e, dict)]
    last_turn_start = -1
    for index, event in enumerate(logs):
        if int(event.get("type", -1)) == 2:
            last_turn_start = index
    turn_logs = logs[last_turn_start:] if last_turn_start >= 0 else logs

    out: dict[str, float | int] = {
        "public_log_count": len(logs),
        "current_turn_log_count": len(turn_logs),
        "select_context_card_id": _first_nested_id(select.get("contextCard")),
        "select_effect_card_id": _first_nested_id(select.get("effect")),
        "select_deck_count": len(select.get("deck") or []),
        # Non-MAIN selects hand these back as scalars, not lists.
        "select_remain_damage_counter": _scalar(
            select.get("remainDamageCounter")
        ),
        "select_remain_energy_cost_count": _size(
            select.get("remainEnergyCost")
        ),
        "current_looking_count": len(current.get("looking") or []),
        "search_begin_input": int(bool(observation.get("search_begin_input"))),
    }
    for scope, events in (("history", logs), ("turn", turn_logs)):
        for log_type in TRACKED_LOG_TYPES:
            out[f"{scope}_self_log_type_{log_type}"] = sum(
                int(
                    int(e.get("type", -1)) == log_type
                    and int(e.get("playerIndex", -1)) == your
                )
                for e in events
            )
            out[f"{scope}_opp_log_type_{log_type}"] = sum(
                int(
                    int(e.get("type", -1)) == log_type
                    and int(e.get("playerIndex", -1)) == 1 - your
                )
                for e in events
            )
    for slot in range(6):
        event = logs[-1 - slot] if slot < len(logs) else {}
        player = int(event.get("playerIndex", -1))
        out[f"recent_log_{slot}_type"] = int(event.get("type", -1))
        out[f"recent_log_{slot}_player"] = (
            0 if player == your else 1 if player == 1 - your else -1
        )
        out[f"recent_log_{slot}_card_id"] = int(event.get("cardId", -1))
        out[f"recent_log_{slot}_from_area"] = int(event.get("fromArea", -1))
        out[f"recent_log_{slot}_to_area"] = int(event.get("toArea", -1))
    for card_id in KEY_CARD_IDS:
        out[f"turn_log_card_{card_id}_count"] = sum(
            int(e.get("cardId", -1) == card_id) for e in turn_logs
        )
        out[f"select_deck_card_{card_id}_count"] = sum(
            int(c.get("id", -1) == card_id)
            for c in (select.get("deck") or [])
            if isinstance(c, dict)
        )
    looking_ids = sorted(
        card_id
        for card_id in (
            _first_nested_id(v) for v in (current.get("looking") or [])
        )
        if card_id >= 0
    )
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
        dict(base_state) if base_state is not None else state_features(current)
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
    target_area = _int(option.get("inPlayArea"), _int(option.get("area")))
    target_energy = _energy_count(target)
    attack_id = _int(option.get("attackId"))
    opp_active_hp = float(out["opp_active_hp"])

    legal_options = list(select.get("option") or [])
    offered_action_counts = Counter(
        action_type(current, o, select) for o in legal_options
    )
    offered_card_counts = Counter(
        int((candidate_card(current, o, select) or {}).get("id", -1))
        for o in legal_options
    )
    preceding = (
        legal_options[:option_position]
        if 0 <= option_position <= len(legal_options)
        else []
    )
    preceding_action_counts = Counter(
        action_type(current, o, select) for o in preceding
    )
    preceding_card_counts = Counter(
        int((candidate_card(current, o, select) or {}).get("id", -1))
        for o in preceding
    )

    # Evolving into Grimmsnarl ex triggers Punk Up: up to five Basic {D}
    # out of the deck, attached anywhere. That is why this evolve outranks
    # a manual attachment even when the body is already on the field.
    evolve_into_attacker = int(action == "evolve" and card_id == GRIMMSNARL_EX_ID)
    candy_into_attacker = int(
        action == "item"
        and card_id == RARE_CANDY_ID
        and int(out["candy_route_available"]) == 1
    )
    triggers_punk_up = int(bool(evolve_into_attacker or candy_into_attacker))
    hand_cost = 2 if candy_into_attacker else int(option_type in (7, 8, 9))

    attacks_with_shadow_bullet = int(
        action == "attack" and attack_id == SHADOW_BULLET_ID
    )
    attack_kills_active = int(
        attacks_with_shadow_bullet and 0 < opp_active_hp <= SHADOW_BULLET_DAMAGE
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
            if option_position >= 0 else -1
        ),
        "candidate_raw_index": int(
            option.get("index") if option.get("index") is not None else -1
        ),
        "candidate_raw_inplay_index": int(
            option.get("inPlayIndex")
            if option.get("inPlayIndex") is not None else -1
        ),
        "candidate_raw_player_relative": (
            0 if int(
                option.get("playerIndex")
                if option.get("playerIndex") is not None else your
            ) == your else 1
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
        "candidate_attack_id": attack_id,
        "candidate_area": _int(option.get("area")),
        "candidate_inplay_area": _int(option.get("inPlayArea")),
        "candidate_target_id": target_id,
        "candidate_target_hp": float(target.get("hp", 0)),
        "candidate_target_max_hp": float(target.get("maxHp", 0)),
        "candidate_target_damage": max(
            0.0,
            float(target.get("maxHp", 0)) - float(target.get("hp", 0)),
        ),
        "candidate_target_energy": target_energy,
        "candidate_target_dark_energy": _dark_energy_count(target),
        "candidate_target_special_energy": _special_energy_count(target),
        "candidate_target_appear_this_turn": int(
            bool(target.get("appearThisTurn"))
        ),
        "candidate_target_evolution_depth": len(
            target.get("preEvolution") or []
        ),
        "candidate_target_is_active": int(target_area == AREA_ACTIVE),
        "candidate_target_is_bench": int(target_area == AREA_BENCH),
        "candidate_hand_cost": hand_cost,
        "candidate_is_marnie_line": int(card_id in MARNIE_LINE_IDS),
        "candidate_is_route_card": int(card_id in ROUTE_IDS),
        "candidate_is_impidimp": int(card_id == IMPIDIMP_ID),
        "candidate_is_morgrem": int(card_id == MORGREM_ID),
        "candidate_is_grimmsnarl": int(card_id == GRIMMSNARL_EX_ID),
        "candidate_is_rare_candy": int(card_id == RARE_CANDY_ID),
        "candidate_is_dark_energy": int(card_id == DARK_ENERGY_ID),
        "candidate_is_munkidori": int(card_id == MUNKIDORI_ID),
        "candidate_is_froslass": int(card_id == FROSLASS_ID),
        "candidate_is_snorunt": int(card_id == SNORUNT_ID),
        "target_is_impidimp": int(target_id == IMPIDIMP_ID),
        "target_is_morgrem": int(target_id == MORGREM_ID),
        "target_is_grimmsnarl": int(target_id == GRIMMSNARL_EX_ID),
        "target_is_munkidori": int(target_id == MUNKIDORI_ID),
        "target_is_snorunt": int(target_id == SNORUNT_ID),
        "evolve_into_attacker": evolve_into_attacker,
        "candy_into_attacker": candy_into_attacker,
        "triggers_punk_up": triggers_punk_up,
        "evolve_attacker_to_active": int(
            evolve_into_attacker and target_area == AREA_ACTIVE
        ),
        "evolve_attacker_to_bench": int(
            evolve_into_attacker and target_area == AREA_BENCH
        ),
        "evolve_froslass": int(action == "evolve" and card_id == FROSLASS_ID),
        "candidate_fills_first_body": int(
            action == "bench"
            and card_id == IMPIDIMP_ID
            and int(out["needs_first_marnie_body"]) == 1
        ),
        "candidate_fills_attacker": int(
            triggers_punk_up and int(out["needs_attacker"]) == 1
        ),
        "energy_target_is_attacker": int(
            action == "energy" and target_id == GRIMMSNARL_EX_ID
        ),
        "energy_completes_attack_cost": int(
            action == "energy"
            and target_id == GRIMMSNARL_EX_ID
            and target_energy + 1 >= SHADOW_BULLET_COST
        ),
        "energy_target_is_munkidori": int(
            action == "energy" and target_id == MUNKIDORI_ID
        ),
        "energy_enables_munkidori": int(
            action == "energy"
            and target_id == MUNKIDORI_ID
            and _dark_energy_count(target) == 0
        ),
        "attack_is_shadow_bullet": attacks_with_shadow_bullet,
        "attack_kills_active": attack_kills_active,
        "attack_overkill": (
            max(0.0, SHADOW_BULLET_DAMAGE - opp_active_hp)
            if attacks_with_shadow_bullet else 0.0
        ),
        "ability_is_munkidori": int(
            action == "ability" and card_id == MUNKIDORI_ID
        ),
        "ability_munkidori_finishes_bench": int(
            action == "ability"
            and card_id == MUNKIDORI_ID
            and int(out["opp_bench_snipe_range_count"]) > 0
        ),
        "boss_opp_bench_value": int(action == "boss") * int(
            out["opp_bench_count"]
        ),
        "boss_opp_bench_low_hp_value": int(action == "boss") * int(
            out["opp_bench_low_hp_count"]
        ),
        "boss_with_ready_attacker": int(action == "boss") * int(
            out["has_ready_attacker"]
        ),
        "retreat_active_damage_value": int(action == "retreat") * float(
            out["self_active_damage"]
        ),
        "retreat_to_ready_backup_value": int(action == "retreat") * int(
            out["backup_attacker_ready"]
        ),
        "retreat_status_value": int(action == "retreat") * int(
            out["self_status_count"]
        ),
        "supporter_after_supporter": int(
            action in ("supporter", "boss")
        ) * int(out["supporter_played"]),
        "stadium_replaces_own": int(
            action == "stadium" and int(out["spikemuth_gym_active"]) == 1
        ),
        "item_low_deck_risk": int(action == "item") * int(out["deck_low_5"]),
        "ability_low_deck_risk": int(action == "ability") * int(
            out["deck_low_5"]
        ),
        "optional_draw_under_deck_pressure": int(
            action in {"ability", "item", "supporter"}
            and int(out["deck_pressure_risk"]) == 1
        ),
        "end_with_ready_attacker_penalty": int(action == "end") * int(
            out["has_ready_attacker"]
        ),
        "end_with_lethal_available": int(action == "end") * int(
            out["shadow_bullet_kills_active"]
        ),
        "bench_early_game_value": int(action == "bench") * int(
            out["early_game"]
        ),
        "same_action_option_count": offered_action_counts[action],
        "same_card_option_count": offered_card_counts[card_id],
        "action_type": action,
    })
    for name in ACTION_TYPES:
        out[f"is_{name}"] = int(action == name)
        out[f"offered_{name}_count"] = offered_action_counts[name]
        out[f"candidate_is_only_{name}"] = int(
            action == name and offered_action_counts[name] == 1
        )
    for offered_card_id in KEY_CARD_IDS:
        out[f"offered_card_{offered_card_id}_count"] = (
            offered_card_counts[offered_card_id]
        )
    context_option_features(current, select, option, out)
    return out


def context_option_features(
    current: dict[str, Any],
    select: dict[str, Any],
    option: dict[str, Any],
    out: dict[str, float | int | str],
) -> None:
    """Columns for the non-MAIN selects v1 left to the rule policy.

    These are not filler decisions. Against the pinned teacher the inherited
    rule policy agreed 39.5% of the time on deck search (about 8 per game),
    64.5% on Adrena-Brain damage placement and 50.0% on counter removal, while
    MAIN agreement was 90.5%. The three biggest blocks are a card choice out of
    a revealed zone and two damage-counter target choices, so the features
    below describe the candidate card, what we already hold of it, and what
    happens to the target if the counters land on it.
    """
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    me = players[your] if your < len(players) else {}
    context = int(select.get("context", -1))
    option_type = int(option.get("type", -1))

    card, owner_is_self, area = resolve_option(current, select, option)
    card = card or {}
    card_id = int(card.get("id", -1))
    max_hp = float(card.get("maxHp", 0))
    hp = float(card.get("hp", 0))
    damage = max(0.0, max_hp - hp)

    hand_counts = Counter(int(c.get("id", -1)) for c in _cards(me, "hand"))
    field_counts = Counter(int(c.get("id", -1)) for c in _in_play(me))
    discard_counts = Counter(
        int(c.get("id", -1)) for c in _cards(me, "discard")
    )
    deck_cards = [
        c for c in (select.get("deck") or []) if isinstance(c, dict)
    ]
    deck_counts = Counter(int(c.get("id", -1)) for c in deck_cards)

    # Adrena-Brain moves up to three counters (30 damage); Shadow Bullet's
    # secondary hit is a flat 30 to one Benched Pokemon.
    pending = int(select.get("remainDamageCounter") or 0)
    swing = 10.0 * pending if pending else SHADOW_BULLET_BENCH_DAMAGE
    number = option.get("number")

    out.update({
        "ctx_option_type": option_type,
        "ctx_area": area,
        "ctx_owner_is_self": int(owner_is_self),
        "ctx_card_id": card_id,
        "ctx_card_resolved": int(card_id >= 0),
        "ctx_number": int(number) if isinstance(number, (int, float)) else -1,
        "ctx_serial": _int(option.get("serial")),
        "ctx_target_hp": hp,
        "ctx_target_max_hp": max_hp,
        "ctx_target_damage": damage,
        "ctx_target_energy": _energy_count(card),
        "ctx_target_dark_energy": _dark_energy_count(card),
        "ctx_target_is_active": int(area == AREA_ACTIVE),
        "ctx_target_is_bench": int(area == AREA_BENCH),
        "ctx_target_appear_this_turn": int(bool(card.get("appearThisTurn"))),
        "ctx_target_evolution_depth": len(card.get("preEvolution") or []),
        "ctx_target_is_ex": int(card_id == GRIMMSNARL_EX_ID),
        "ctx_target_is_ability_holder": int(card_id in ABILITY_HOLDER_IDS),
        # Does this much damage finish the target? The whole point of
        # Adrena-Brain and the Shadow Bullet snipe is converting a survivor.
        "ctx_target_dies_to_swing": int(0 < hp <= swing),
        "ctx_target_hp_after_swing": max(0.0, hp - swing),
        "ctx_target_survives_by": max(0.0, hp - swing),
        "ctx_pending_damage_counters": pending,
        # Search: what we already have decides what is worth fetching.
        "ctx_copies_in_hand": hand_counts[card_id] if card_id >= 0 else -1,
        "ctx_copies_in_field": field_counts[card_id] if card_id >= 0 else -1,
        "ctx_copies_in_discard": (
            discard_counts[card_id] if card_id >= 0 else -1
        ),
        "ctx_copies_in_zone": deck_counts[card_id] if card_id >= 0 else -1,
        "ctx_zone_size": len(deck_cards),
        "ctx_is_marnie_line": int(card_id in MARNIE_LINE_IDS),
        "ctx_is_basic_pokemon": int(card_id in BASIC_POKEMON_IDS),
        "ctx_is_energy": int(card_id == DARK_ENERGY_ID),
        "ctx_is_supporter": int(card_id in SUPPORTER_IDS),
        "ctx_is_item": int(card_id in ITEM_IDS),
        "ctx_is_stadium": int(card_id == SPIKEMUTH_GYM_ID),
        "ctx_is_rare_candy": int(card_id == RARE_CANDY_ID),
        "ctx_is_grimmsnarl": int(card_id == GRIMMSNARL_EX_ID),
        "ctx_is_boss": int(card_id == BOSS_ID),
        "ctx_completes_candy_route": int(
            card_id == GRIMMSNARL_EX_ID
            and hand_counts[RARE_CANDY_ID] > 0
            and field_counts[IMPIDIMP_ID] > 0
        ) or int(
            card_id == RARE_CANDY_ID
            and hand_counts[GRIMMSNARL_EX_ID] > 0
            and field_counts[IMPIDIMP_ID] > 0
        ),
        "ctx_fetches_missing_attacker": int(
            card_id in (GRIMMSNARL_EX_ID, MORGREM_ID)
            and int(out.get("needs_attacker", 0)) == 1
        ),
        "ctx_fetches_first_body": int(
            card_id == IMPIDIMP_ID
            and int(out.get("needs_first_marnie_body", 0)) == 1
        ),
        "ctx_is_search_context": int(context == CTX_TO_HAND),
        "ctx_is_damage_context": int(context in (
            CTX_DAMAGE_COUNTER, CTX_DAMAGE, CTX_REMOVE_DAMAGE_COUNTER,
            CTX_REMOVE_COUNTER_COUNT,
        )),
        "ctx_is_attach_context": int(context in (
            CTX_ATTACH_FROM, CTX_ATTACH_TO
        )),
    })

    offered = list(select.get("option") or [])
    resolved_ids = []
    for other in offered:
        other_card, _, _ = resolve_option(current, select, other)
        resolved_ids.append(int((other_card or {}).get("id", -1)))
    counts = Counter(resolved_ids)
    out["ctx_same_card_offered"] = counts[card_id] if card_id >= 0 else -1
    out["ctx_distinct_cards_offered"] = len(
        {value for value in resolved_ids if value >= 0}
    )
    for key_id in KEY_CARD_IDS:
        out[f"ctx_offered_{key_id}"] = counts[key_id]


LEAKAGE_DENYLIST = {
    "reward", "target_reward", "target_win", "target_loss", "result",
    "winner", "initial_deck_json", "deck_hash", "deck_type", "visualize",
    "leaderboard_rank", "submission_score", "team_id",
}


def assert_no_leakage(feature_columns: list[str]) -> None:
    leaked = sorted(set(feature_columns) & LEAKAGE_DENYLIST)
    if leaked:
        raise ValueError(f"Leakage columns present in policy features: {leaked}")
