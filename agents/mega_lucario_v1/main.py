"""PTCG AI Battle Challenge submission agent.

Strategy
--------
A deterministic, context-aware Mega Lucario ex / Hariyama rule agent.
It is based on the public organizer Mega Lucario baseline and public
meta observations, but the implementation below is newly organized for
safe team development:

* Uses the proven 60-card Mega Lucario shell in deck.csv.
* Separates MAIN actions from setup, search, discard, switch and target selects.
* Plans an attacker and a possible Boss's Orders target before ranking MAIN actions.
* Keeps Hariyama as a non-ex answer to Crustle's ex-damage wall.
* Returns only the required number of choices for optional multi-selects.
* Falls back to a structurally legal choice instead of crashing.

The official sample_submission/cg directory must be bundled beside this file.
Use scripts/build_submission.py from the repository root.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from cg.api import (  # type: ignore
    AreaType,
    CardType,
    EnergyType,
    OptionType,
    SelectContext,
    SelectType,
    all_card_data,
    to_observation_class,
)

# -----------------------------------------------------------------------------
# Deck and card constants
# -----------------------------------------------------------------------------

ROOT_CANDIDATES = (
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "",
    os.getcwd(),
    "/kaggle_simulations/agent",
)


def _load_deck() -> list[int]:
    for root in ROOT_CANDIDATES:
        if not root:
            continue
        path = os.path.join(root, "deck.csv")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8-sig") as f:
            deck = [int(line.strip()) for line in f if line.strip()]
        if len(deck) != 60:
            raise ValueError(f"deck.csv must contain exactly 60 IDs, got {len(deck)}")
        return deck
    raise FileNotFoundError("deck.csv was not found beside main.py")


DECK = _load_deck()

# Pokémon
MAKUHITA = 673
HARIYAMA = 674
LUNATONE = 675
SOLROCK = 676
RIOLU = 677
MEGA_LUCARIO_EX = 678
CRUSTLE = 345

# Trainers / Energy
DUSK_BALL = 1102
SWITCH = 1123
PREMIUM_POWER_PRO = 1141
FIGHTING_GONG = 1142
POKE_PAD = 1152
HERO_CAPE = 1159
BOSS_ORDERS = 1182
CARMINE = 1192
LILLIE_DETERMINATION = 1227
GRAVITY_MOUNTAIN = 1252
BASIC_FIGHTING_ENERGY = 6

# Known attack in the public baseline.
MEGA_BRAVE = 983

CARD_TABLE = {card.cardId: card for card in all_card_data()}


@dataclass
class AttackPlan:
    attacker_slot: int = -1  # 0=active, 1..=bench slot + 1
    target_slot: int = -1    # 0=active, 1..=bench slot + 1
    preferred_attack_id: int | None = None
    needs_energy: bool = False
    expected_damage: int = 0
    score: float = -1e18


_PLAN = AttackPlan()
_LAST_TURN = -1
_LAST_PLAYER = -1
_ABILITY_USED_THIS_TURN = False


# -----------------------------------------------------------------------------
# Defensive helpers
# -----------------------------------------------------------------------------


def _enum_is(value: Any, enum_cls: Any, name: str) -> bool:
    member = getattr(enum_cls, name, None)
    return member is not None and value == member


def _ctx_is(value: Any, *names: str) -> bool:
    return any(_enum_is(value, SelectContext, name) for name in names)


def _select_type_is(value: Any, *names: str) -> bool:
    return any(_enum_is(value, SelectType, name) for name in names)


def _opt_value(option: Any, field: str, default: Any = None) -> Any:
    return getattr(option, field, default)


def _safe_card(obs: Any, area: Any, index: int, player_index: int) -> Any | None:
    """Return a Card/Pokemon object from an engine area, or None."""
    try:
        player = obs.current.players[player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return player.hand[index]
        if area == AreaType.DISCARD:
            return player.discard[index]
        if area == AreaType.ACTIVE:
            return player.active[index]
        if area == AreaType.BENCH:
            return player.bench[index]
        if area == AreaType.PRIZE:
            return player.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
    except Exception:
        return None
    return None


def _board(player: Any) -> list[Any]:
    active = [player.active[0]] if getattr(player, "active", None) else []
    return [p for p in active + list(getattr(player, "bench", [])) if p is not None]


def _pokemon_prizes(pokemon: Any) -> int:
    data = CARD_TABLE.get(pokemon.id)
    if data is None:
        return 1
    prizes = 3 if getattr(data, "megaEx", False) else 2 if getattr(data, "ex", False) else 1
    for energy_card in getattr(pokemon, "energyCards", []):
        if getattr(energy_card, "id", None) == 12:  # Legacy Energy
            prizes -= 1
    return max(0, prizes)


def _pokemon_target_value(pokemon: Any) -> float:
    data = CARD_TABLE.get(pokemon.id)
    score = _pokemon_prizes(pokemon) * 1200.0
    score += len(getattr(pokemon, "energies", [])) * 140.0
    score += len(getattr(pokemon, "tools", [])) * 90.0
    score += float(getattr(pokemon, "hp", 0))
    if data is not None:
        if getattr(data, "stage2", False):
            score += 250
        elif getattr(data, "stage1", False):
            score += 130
    return score


def _legal_fallback(select: Any) -> list[int]:
    n = len(getattr(select, "option", []) or [])
    if n == 0:
        return []
    min_count = max(0, int(getattr(select, "minCount", 0) or 0))
    required = min(min_count, n)
    return list(range(required))


def _finalize_selection(select: Any, scored: list[tuple[float, int]]) -> list[int]:
    """Convert scored option indices into a legal list.

    Crucially, this does not blindly select maxCount options. Optional choices
    are included only when their score is positive; minCount is then satisfied.
    """
    n = len(select.option)
    if n == 0:
        return []

    min_count = max(0, int(select.minCount or 0))
    max_count = min(n, max(0, int(select.maxCount or 0)))
    if max_count == 0:
        return []

    scored.sort(key=lambda pair: (pair[0], -pair[1]), reverse=True)

    if max_count == 1:
        best_score, best_idx = scored[0]
        if min_count == 0 and best_score <= 0:
            return []
        return [best_idx]

    chosen = [idx for score, idx in scored if score > 0][:max_count]
    if len(chosen) < min_count:
        for _, idx in scored:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= min_count:
                break
    return chosen[:max_count]


# -----------------------------------------------------------------------------
# Board analysis and attack planning
# -----------------------------------------------------------------------------


def _count_zone(cards: Iterable[Any]) -> Counter[int]:
    return Counter(getattr(card, "id", -1) for card in cards if card is not None)


def _attacker_profile(pokemon: Any, field_counts: Counter[int]) -> list[tuple[int, int, int | None, bool]]:
    """Return (energy_required, base_damage, attack_id, is_ex_attack) candidates."""
    pid = pokemon.id
    if pid == MEGA_LUCARIO_EX:
        return [
            (1, 130, None, True),
            (2, 270, MEGA_BRAVE, True),
        ]
    if pid in (MAKUHITA, HARIYAMA):
        return [(3, 210, None, False)]
    if pid == SOLROCK and field_counts[LUNATONE] > 0:
        return [(1, 70, None, False)]
    return []


def _adjust_damage(base_damage: int, attacker: Any, target: Any) -> int:
    target_data = CARD_TABLE.get(target.id)
    if target_data is None:
        return base_damage

    # All attackers in this deck use Fighting damage in the public baseline.
    damage = base_damage
    if getattr(target_data, "weakness", None) == EnergyType.FIGHTING:
        damage *= 2
    elif getattr(target_data, "resistance", None) == EnergyType.FIGHTING:
        damage -= 30

    # Crustle's wall blanks damage from Pokémon ex / Mega ex.
    attacker_data = CARD_TABLE.get(attacker.id)
    if target.id == CRUSTLE and attacker_data is not None:
        if getattr(attacker_data, "ex", False) or getattr(attacker_data, "megaEx", False):
            return 0
    return max(0, damage)


def _build_attack_plan(obs: Any, hand_counts: Counter[int], field_counts: Counter[int]) -> AttackPlan:
    state = obs.current
    select = obs.select
    me = state.players[state.yourIndex]
    opponent = state.players[1 - state.yourIndex]

    can_switch = False
    can_boss = False
    for option in select.option:
        typ = option.type
        if typ == OptionType.RETREAT:
            can_switch = True
        elif typ == OptionType.PLAY:
            card = _safe_card(obs, AreaType.HAND, option.index, state.yourIndex)
            if card is None:
                continue
            if card.id == SWITCH:
                can_switch = True
            elif card.id == BOSS_ORDERS:
                can_boss = True

    own_slots = ([me.active[0]] if me.active else [None]) + list(me.bench)
    opp_slots = ([opponent.active[0]] if opponent.active else [None]) + list(opponent.bench)

    plan = AttackPlan()
    for attacker_slot, attacker in enumerate(own_slots):
        if attacker is None:
            continue
        if attacker_slot != 0 and not can_switch:
            continue

        current_energy = len(getattr(attacker, "energies", []))
        profiles = _attacker_profile(attacker, field_counts)
        if attacker.id == MAKUHITA:
            # Makuhita itself does not have Hariyama's 210-damage attack.
            # Treat it as an attacker only when a legal Hariyama evolution for
            # this exact slot is available in the current MAIN options.
            can_evolve_here = False
            for option in select.option:
                if option.type != OptionType.EVOLVE:
                    continue
                source = _safe_card(obs, AreaType.HAND, option.index, state.yourIndex)
                option_slot = 0 if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
                if source is not None and source.id == HARIYAMA and option_slot == attacker_slot:
                    can_evolve_here = True
                    break
            if not can_evolve_here:
                profiles = []

        for energy_required, base_damage, attack_id, _ in profiles:
            future_energy = current_energy
            needs_energy = False
            if future_energy < energy_required:
                can_attach = (
                    hand_counts[BASIC_FIGHTING_ENERGY] > 0
                    and not bool(getattr(state, "energyAttached", False))
                )
                if can_attach and future_energy + 1 >= energy_required:
                    future_energy += 1
                    needs_energy = True
                else:
                    continue

            for target_slot, target in enumerate(opp_slots):
                if target is None:
                    continue
                if target_slot != 0 and not can_boss:
                    continue

                damage = _adjust_damage(base_damage, attacker, target)
                if damage <= 0:
                    continue

                target_hp = max(1, int(getattr(target, "hp", 1)))
                prizes = _pokemon_prizes(target) if damage >= target_hp else 0
                score = _pokemon_target_value(target)
                if damage >= target_hp:
                    score += 2500 + prizes * 2500
                    if len(opponent.prize) <= prizes:
                        score += 50000
                else:
                    score *= min(1.0, damage / target_hp)

                if attacker_slot == 0:
                    score += 250
                if target_slot == 0:
                    score += 300
                if not needs_energy:
                    score += 100
                if attack_id == MEGA_BRAVE and len(me.prize) in (2, 3):
                    # Avoid exposing a 3-prize attacker unnecessarily near the end.
                    score -= 450

                if score > plan.score:
                    plan = AttackPlan(
                        attacker_slot=attacker_slot,
                        target_slot=target_slot,
                        preferred_attack_id=attack_id,
                        needs_energy=needs_energy,
                        expected_damage=damage,
                        score=score,
                    )
    return plan


def _energy_target_score(pokemon: Any, active: bool, plan: AttackPlan, slot: int) -> float:
    energy_count = len(getattr(pokemon, "energies", []))
    score = 5000.0 + (15 if active else 0)

    if slot == plan.attacker_slot and plan.needs_energy:
        score += 1500

    if pokemon.id in (RIOLU, MEGA_LUCARIO_EX):
        score += 300 if energy_count < 2 else -150
        if pokemon.id == MEGA_LUCARIO_EX:
            score += 20
    elif pokemon.id in (MAKUHITA, HARIYAMA):
        score += 320 if energy_count < 3 else -120
        if pokemon.id == HARIYAMA:
            score += 30
    elif pokemon.id == SOLROCK:
        score += 120 if energy_count < 1 else -250
    elif pokemon.id == LUNATONE:
        score -= 400
    return score


# -----------------------------------------------------------------------------
# Context-specific option scoring
# -----------------------------------------------------------------------------


def _option_card(obs: Any, option: Any, default_area: Any | None = None) -> Any | None:
    state = obs.current
    area = _opt_value(option, "area", default_area)
    index = _opt_value(option, "index", -1)
    player_index = _opt_value(option, "playerIndex", state.yourIndex)
    if area is None or index is None or index < 0:
        return None
    return _safe_card(obs, area, index, player_index)


def _main_score(
    obs: Any,
    option: Any,
    hand_counts: Counter[int],
    field_counts: Counter[int],
    discard_counts: Counter[int],
    plan: AttackPlan,
) -> float:
    state = obs.current
    me = state.players[state.yourIndex]
    typ = option.type

    if typ == OptionType.ABILITY:
        return 30000

    if typ == OptionType.EVOLVE:
        target = _safe_card(obs, option.inPlayArea, option.inPlayIndex, state.yourIndex)
        source = _safe_card(obs, AreaType.HAND, option.index, state.yourIndex)
        score = 20000.0
        if target is not None:
            score += len(getattr(target, "energies", [])) * 50
            if target.id == RIOLU:
                score += 1000
            elif target.id == MAKUHITA:
                # Hariyama is the key non-ex Crustle answer.
                opponent_active = state.players[1 - state.yourIndex].active[0]
                if opponent_active is not None and opponent_active.id == CRUSTLE:
                    score += 2500
        if source is not None and source.id == MEGA_LUCARIO_EX:
            score += 500
        return score

    if typ == OptionType.ATTACH:
        target = _safe_card(obs, option.inPlayArea, option.inPlayIndex, state.yourIndex)
        attached = _safe_card(obs, AreaType.HAND, option.index, state.yourIndex)
        if target is None or attached is None:
            return -100
        slot = 0 if option.inPlayArea == AreaType.ACTIVE else option.inPlayIndex + 1
        if attached.id == HERO_CAPE:
            score = 8500.0
            if target.id == MEGA_LUCARIO_EX:
                score += 600
            elif target.id == RIOLU:
                score += 300
            return score
        return _energy_target_score(target, option.inPlayArea == AreaType.ACTIVE, plan, slot)

    if typ == OptionType.PLAY:
        card = _safe_card(obs, AreaType.HAND, option.index, state.yourIndex)
        if card is None:
            return -100
        cid = card.id
        data = CARD_TABLE.get(cid)
        bench_space = max(0, int(getattr(me, "benchMax", 5)) - len([p for p in me.bench if p is not None]))

        if data is not None and data.cardType == CardType.POKEMON:
            if bench_space <= 0:
                return -1000
            if cid == RIOLU:
                return 15000 if field_counts[RIOLU] + field_counts[MEGA_LUCARIO_EX] < 2 else -200
            if cid == MAKUHITA:
                opponent_active = state.players[1 - state.yourIndex].active[0]
                crustle_need = opponent_active is not None and opponent_active.id == CRUSTLE
                return 14800 if crustle_need and field_counts[HARIYAMA] == 0 else 12500
            if cid == LUNATONE:
                return 12000 if field_counts[LUNATONE] == 0 else -300
            if cid == SOLROCK:
                return 12300 if field_counts[SOLROCK] == 0 else -300
            return 9000

        if cid == SWITCH:
            return 18000 if plan.attacker_slot > 0 else -500
        if cid == BOSS_ORDERS:
            return 17500 if plan.target_slot > 0 and not state.supporterPlayed else -500
        if cid == DUSK_BALL:
            need_evolution = field_counts[RIOLU] > 0 or field_counts[MAKUHITA] > 0
            return 13500 if need_evolution else 3500
        if cid == FIGHTING_GONG:
            needs_energy = plan.needs_energy or discard_counts[BASIC_FIGHTING_ENERGY] > 0
            return 14500 if needs_energy else 4500
        if cid == POKE_PAD:
            setup_need = (
                field_counts[RIOLU] + field_counts[MEGA_LUCARIO_EX] < 2
                or field_counts[SOLROCK] == 0
                or field_counts[LUNATONE] == 0
            )
            return 13200 if setup_need else 4200
        if cid == PREMIUM_POWER_PRO:
            return 9000 if plan.expected_damage > 0 else -300
        if cid == CARMINE:
            if state.supporterPlayed:
                return -500
            return 12500 if len(me.hand) <= 5 else 7500
        if cid == LILLIE_DETERMINATION:
            if state.supporterPlayed:
                return -500
            return 12800 if len(me.hand) <= 4 else 8200
        if cid == GRAVITY_MOUNTAIN:
            if state.stadiumPlayed:
                return -300
            return 5000
        return 3000

    if typ == OptionType.RETREAT:
        return 16000 if plan.attacker_slot > 0 else -700

    if typ == OptionType.ATTACK:
        score = 7000.0
        attack_id = _opt_value(option, "attackId")
        if plan.preferred_attack_id is not None and attack_id == plan.preferred_attack_id:
            score += 1500
        elif plan.preferred_attack_id is None:
            # Prefer the engine's first legal attack when no exact ID is planned.
            score += 100
        return score

    if typ == OptionType.END:
        return -1000

    if typ == OptionType.DISCARD:
        return 100

    return 0


def _setup_active_score(state: Any, card: Any) -> float:
    if card is None:
        return -100
    if card.id == SOLROCK:
        return 450 if state.firstPlayer != state.yourIndex else 250
    if card.id == RIOLU:
        return 350
    if card.id == MAKUHITA:
        return 180
    if card.id == LUNATONE:
        return 120
    return 50


def _bench_score(card: Any, field_counts: Counter[int]) -> float:
    if card is None:
        return -100
    if card.id == RIOLU:
        return 600 if field_counts[RIOLU] + field_counts[MEGA_LUCARIO_EX] < 2 else 50
    if card.id == MAKUHITA:
        return 520 if field_counts[MAKUHITA] + field_counts[HARIYAMA] < 1 else 80
    if card.id == LUNATONE:
        return 500 if field_counts[LUNATONE] == 0 else 40
    if card.id == SOLROCK:
        return 480 if field_counts[SOLROCK] == 0 else 40
    return 100


def _to_hand_score(card: Any, hand_counts: Counter[int], field_counts: Counter[int]) -> float:
    if card is None:
        return -100
    cid = card.id
    score = 500.0 - hand_counts[cid] * 120
    if cid == MEGA_LUCARIO_EX:
        score += 500 if field_counts[RIOLU] > field_counts[MEGA_LUCARIO_EX] else 50
    elif cid == HARIYAMA:
        score += 480 if field_counts[MAKUHITA] > field_counts[HARIYAMA] else 30
    elif cid == RIOLU:
        score += 420 if field_counts[RIOLU] + field_counts[MEGA_LUCARIO_EX] < 2 else -100
    elif cid == MAKUHITA:
        score += 360 if field_counts[MAKUHITA] + field_counts[HARIYAMA] < 1 else -80
    elif cid == LUNATONE:
        score += 330 if field_counts[LUNATONE] == 0 else -250
    elif cid == SOLROCK:
        score += 320 if field_counts[SOLROCK] == 0 else -250
    elif cid == BASIC_FIGHTING_ENERGY:
        score += 250
    elif cid in (DUSK_BALL, FIGHTING_GONG, POKE_PAD):
        score += 180
    return score


def _discard_score(card: Any, hand_counts: Counter[int], field_counts: Counter[int]) -> float:
    """Higher means safer to discard."""
    if card is None:
        return -1000
    cid = card.id
    score = 0.0

    # Extra copies are safer to discard.
    score += max(0, hand_counts[cid] - 1) * 150

    if cid == BASIC_FIGHTING_ENERGY:
        return score - 350
    if cid == MEGA_LUCARIO_EX:
        return score - (500 if field_counts[RIOLU] > field_counts[MEGA_LUCARIO_EX] else 100)
    if cid == HARIYAMA:
        return score - (450 if field_counts[MAKUHITA] > field_counts[HARIYAMA] else 80)
    if cid == RIOLU:
        return score - 320
    if cid == MAKUHITA:
        return score - 280
    if cid in (LUNATONE, SOLROCK):
        return score - (250 if field_counts[cid] == 0 else 20)
    if cid in (BOSS_ORDERS, SWITCH, HERO_CAPE):
        return score - 180
    if cid in (CARMINE, LILLIE_DETERMINATION):
        return score - 80
    if cid == GRAVITY_MOUNTAIN:
        return score + 100
    return score + 30


def _switch_score(card: Any, option: Any, plan: AttackPlan) -> float:
    if card is None:
        return -100
    area = _opt_value(option, "area")
    index = int(_opt_value(option, "index", 0))
    slot = 0 if area == AreaType.ACTIVE else index + 1
    energy = len(getattr(card, "energies", []))
    score = energy * 40.0
    if slot == plan.attacker_slot:
        score += 2000
    if card.id == MEGA_LUCARIO_EX:
        score += 250
    elif card.id == HARIYAMA:
        score += 220
    elif card.id == SOLROCK:
        score += 80
    return score


def _score_option(
    obs: Any,
    idx: int,
    option: Any,
    hand_counts: Counter[int],
    field_counts: Counter[int],
    discard_counts: Counter[int],
    plan: AttackPlan,
) -> float:
    select = obs.select
    state = obs.current
    context = select.context

    if _ctx_is(context, "MAIN"):
        return _main_score(obs, option, hand_counts, field_counts, discard_counts, plan)

    if _ctx_is(context, "SETUP_ACTIVE_POKEMON"):
        return _setup_active_score(state, _option_card(obs, option))

    if _ctx_is(context, "SETUP_BENCH_POKEMON", "TO_BENCH", "TO_FIELD"):
        return _bench_score(_option_card(obs, option), field_counts)

    if _ctx_is(context, "SWITCH", "TO_ACTIVE"):
        return _switch_score(_option_card(obs, option), option, plan)

    if _ctx_is(context, "TO_HAND", "LOOK"):
        return _to_hand_score(_option_card(obs, option), hand_counts, field_counts)

    if _ctx_is(context, "DISCARD", "DISCARD_CARD_OR_ATTACHED_CARD"):
        return _discard_score(_option_card(obs, option), hand_counts, field_counts)

    if _ctx_is(context, "ATTACH_TO", "ATTACH_FROM", "EFFECT_TARGET"):
        card = _option_card(obs, option)
        if card is not None and hasattr(card, "energies"):
            area = _opt_value(option, "area")
            index = int(_opt_value(option, "index", 0))
            slot = 0 if area == AreaType.ACTIVE else index + 1
            return _energy_target_score(card, area == AreaType.ACTIVE, plan, slot)
        if card is not None and card.id == BASIC_FIGHTING_ENERGY:
            return 500
        return _to_hand_score(card, hand_counts, field_counts)

    if _ctx_is(context, "EVOLVE", "EVOLVES_TO"):
        card = _option_card(obs, option)
        if card is None:
            return 0
        if card.id == MEGA_LUCARIO_EX:
            return 800
        if card.id == HARIYAMA:
            return 750
        return 100

    if _ctx_is(context, "TO_DECK", "TO_DECK_BOTTOM", "TO_PRIZE"):
        # These contexts generally ask which card to put away; reuse discard safety.
        return _discard_score(_option_card(obs, option), hand_counts, field_counts)

    if _ctx_is(context, "DAMAGE", "DAMAGE_COUNTER", "DAMAGE_COUNTER_ANY"):
        card = _option_card(obs, option)
        if card is None:
            return 0
        return _pokemon_target_value(card) - float(getattr(card, "hp", 0))

    typ = option.type
    if typ == OptionType.ATTACK:
        return 1000 + (200 if _opt_value(option, "attackId") == plan.preferred_attack_id else 0)
    if typ == OptionType.YES:
        return 100
    if typ == OptionType.NO:
        return 0
    if typ == OptionType.NUMBER:
        return float(_opt_value(option, "number", 0))
    if typ == OptionType.CARD:
        return _to_hand_score(_option_card(obs, option), hand_counts, field_counts)
    return 1.0 / (idx + 1)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def agent(obs_dict: dict[str, Any]) -> list[int]:
    global _PLAN, _LAST_TURN, _LAST_PLAYER, _ABILITY_USED_THIS_TURN

    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        # Deck selection can still be answered without parsing the observation.
        if obs_dict.get("select") is None:
            return list(DECK)
        raw_select = obs_dict.get("select") or {}
        min_count = max(0, int(raw_select.get("minCount", 0) or 0))
        option_count = len(raw_select.get("option") or [])
        return list(range(min(min_count, option_count)))

    if obs.select is None:
        return list(DECK)

    select = obs.select
    state = obs.current
    if state is None:
        return _legal_fallback(select)

    try:
        if state.turn < _LAST_TURN or state.yourIndex != _LAST_PLAYER:
            _PLAN = AttackPlan()
            _ABILITY_USED_THIS_TURN = False
        if state.turn != _LAST_TURN:
            _ABILITY_USED_THIS_TURN = False
        _LAST_TURN = state.turn
        _LAST_PLAYER = state.yourIndex

        me = state.players[state.yourIndex]
        hand = list(me.hand or [])
        field = _board(me)
        discard = list(me.discard or [])
        hand_counts = _count_zone(hand)
        field_counts = _count_zone(field)
        discard_counts = _count_zone(discard)

        if _ctx_is(select.context, "MAIN"):
            _PLAN = _build_attack_plan(obs, hand_counts, field_counts)

        # YES/NO and numeric selections are cleaner as dedicated paths.
        if _select_type_is(select.type, "YES_NO"):
            yes_idx = next((i for i, o in enumerate(select.option) if o.type == OptionType.YES), None)
            no_idx = next((i for i, o in enumerate(select.option) if o.type == OptionType.NO), None)
            # Going first supports evolution/setup; activate beneficial effects by default.
            if yes_idx is not None:
                return [yes_idx]
            return [no_idx] if no_idx is not None else _legal_fallback(select)

        if _select_type_is(select.type, "COUNT"):
            scored = [(float(_opt_value(o, "number", 0)), i) for i, o in enumerate(select.option)]
            return _finalize_selection(select, scored)

        scored = [
            (
                _score_option(
                    obs,
                    i,
                    option,
                    hand_counts,
                    field_counts,
                    discard_counts,
                    _PLAN,
                ),
                i,
            )
            for i, option in enumerate(select.option)
        ]
        result = _finalize_selection(select, scored)
        if result:
            chosen = select.option[result[0]]
            if _ctx_is(select.context, "MAIN") and chosen.type == OptionType.ABILITY:
                _ABILITY_USED_THIS_TURN = True
        return result
    except Exception:
        # A legal low-information choice is better than a crash/forfeit.
        return _legal_fallback(select)
