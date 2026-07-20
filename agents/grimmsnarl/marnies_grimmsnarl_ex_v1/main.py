"""Dedicated Marnie's Grimmsnarl ex / Froslass / Munkidori policy.

The list is reconstructed from saved upper-ladder replays.  The policy keeps an
attack reservation while coordinating Punk Up, Adrena-Brain and Shadow Bullet.
It intentionally has no opponent-name or hidden-deck branches.
"""
from __future__ import annotations

import os

from cg.api import AreaType, CardType, EnergyType, Observation, OptionType, Pokemon, SelectContext
from policy_base import (
    BasePolicy,
    card_table,
    get_card,
    make_agent,
    new_diag,
    normalize_selection,
    prize_count,
)


class C:
    DARKNESS = 7

    FROSLASS = 104
    MUNKIDORI = 112
    SHAYMIN = 343
    IMPIDIMP = 646
    MORGREM = 647
    GRIMMSNARL_EX = 648
    SNORUNT = 860

    RARE_CANDY = 1079
    UNFAIR_STAMP = 1080
    BUDDY_POFFIN = 1086
    NIGHT_STRETCHER = 1097
    POKE_PAD = 1152
    HANDHELD_FAN = 1161
    BOSSES_ORDERS = 1182
    PETREL = 1219
    LILLIE = 1227
    DAWN = 1231
    SPIKEMUTH_GYM = 1259
    BATTLE_CAGE = 1264


class A:
    FROST_SMASH = 131
    MIND_BEND = 141
    FILCH = 934
    IMPIDIMP_PUNCH = 935
    MORGREM_PUNCH = 936
    SHADOW_BULLET = 937
    SNORUNT_CHILLY = 1239


MARNIES_LINE = {C.IMPIDIMP, C.MORGREM, C.GRIMMSNARL_EX}
ATTACKERS = {C.GRIMMSNARL_EX, C.MORGREM}


def _resolve_deck_path() -> str:
    import sys

    candidates = []
    if "__file__" in globals():
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv"))
    candidates.extend(("deck.csv", "/kaggle_simulations/agent/deck.csv"))
    candidates.extend(os.path.join(path, "deck.csv") for path in sys.path if path)
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("deck.csv not found")


with open(_resolve_deck_path(), encoding="utf-8-sig") as _deck_file:
    MY_DECK = [int(line) for line in _deck_file.read().splitlines() if line.strip()]
if len(MY_DECK) != 60:
    raise ValueError(f"deck.csv must contain 60 cards, got {len(MY_DECK)}")


def _fresh_diag():
    diag = new_diag()
    diag.update(
        {
            "shadow_bullets": 0,
            "adrena_brains": 0,
            "punk_up_searches": 0,
            "punk_up_targets": {"current": 0, "backup": 0, "other": 0},
            "attackable_ends": 0,
            "retreats_to_attacker": 0,
        }
    )
    return diag


DIAG = _fresh_diag()


class GrimmsnarlPolicy(BasePolicy):
    ENERGY_TYPES = {C.DARKNESS}
    ATTACKER_IDS = ATTACKERS

    def __init__(self, obs: Observation):
        super().__init__(obs)
        self.effect_id = getattr(getattr(self.select, "effect", None), "id", None)

    def go_first(self) -> bool:
        return True

    # ----- compact board model -------------------------------------------------
    @staticmethod
    def damage_on(pokemon) -> int:
        if pokemon is None:
            return 0
        return max(0, int(getattr(pokemon, "maxHp", 0) or 0) - int(getattr(pokemon, "hp", 0) or 0))

    @staticmethod
    def has_rule_box(pokemon) -> bool:
        data = card_table.get(getattr(pokemon, "id", -1))
        return bool(data and (getattr(data, "ex", False) or getattr(data, "megaEx", False)))

    @staticmethod
    def has_ability(pokemon) -> bool:
        data = card_table.get(getattr(pokemon, "id", -1))
        return bool(data and (getattr(data, "skills", None) or []))

    def open_bench(self) -> int:
        return max(0, int(getattr(self.me, "benchMax", 5) or 5) - len(self.me.bench))

    def board_count(self) -> int:
        return len(self.me.active) + len(self.me.bench)

    def shadow_damage(self, target) -> int:
        if target is None:
            return 180
        data = card_table.get(target.id)
        damage = 180
        if data is not None and getattr(data, "weakness", None) == EnergyType.DARKNESS:
            damage *= 2
        if data is not None and getattr(data, "resistance", None) == EnergyType.DARKNESS:
            damage = max(0, damage - 30)
        return damage

    def active_shadow_ready(self) -> bool:
        active = self.me.active[0] if self.me.active else None
        return bool(active and active.id == C.GRIMMSNARL_EX and self.can_attack(active))

    def ready_grimms(self):
        return [p for p in self.my_board() if p is not None and p.id == C.GRIMMSNARL_EX and self.can_attack(p)]

    def powered_munkidori(self) -> bool:
        return any(
            p is not None and p.id == C.MUNKIDORI and EnergyType.DARKNESS in (p.energies or [])
            for p in self.my_board()
        )

    def eligible_impidimps(self):
        return [
            p
            for p in self.my_board()
            if p is not None
            and p.id == C.IMPIDIMP
            and not bool(getattr(p, "appearThisTurn", False))
        ]

    def useful_discard(self) -> bool:
        priorities = {C.GRIMMSNARL_EX, C.MORGREM, C.IMPIDIMP, C.MUNKIDORI, C.DARKNESS}
        return any(card.id in priorities for card in self.me.discard)

    def opponent_has_shaymin(self) -> bool:
        return any(p is not None and p.id == C.SHAYMIN for p in self.opponent.active + self.opponent.bench)

    def bench_damage_lands(self, target) -> bool:
        if target is None:
            return False
        if self.opponent_has_shaymin() and not self.has_rule_box(target):
            return False
        return True

    def counter_lands(self, target, is_bench: bool) -> bool:
        return not (is_bench and self.stadium_id == C.BATTLE_CAGE)

    # ----- search and setup -----------------------------------------------------
    def score_setup_active(self, card):
        if card is None:
            return 0
        return {
            C.IMPIDIMP: 500,
            C.SNORUNT: 350,
            C.MUNKIDORI: 100,
        }.get(card.id, 10)

    def score_play_poke(self, card):
        if self.open_bench() <= 0:
            return -1
        safety = 250_000 if self.board_count() <= 1 else 0
        if card.id == C.IMPIDIMP:
            return (735_000 + safety) if self.field[C.IMPIDIMP] < 2 else (620_000 if self.field[C.IMPIDIMP] < 3 else -1)
        if card.id == C.SNORUNT:
            return (700_000 + safety) if self.field[C.SNORUNT] + self.field[C.FROSLASS] == 0 else 520_000
        if card.id == C.MUNKIDORI:
            return (690_000 + safety) if self.field[C.MUNKIDORI] == 0 else (535_000 if self.field[C.MUNKIDORI] == 1 else -1)
        return -1

    def score_to_bench(self, card):
        if card is None:
            return 0
        if card.id == C.IMPIDIMP:
            return 500 if self.field[C.IMPIDIMP] < 2 else (330 if self.field[C.IMPIDIMP] < 3 else 40)
        if card.id == C.SNORUNT:
            return 420 if self.field[C.SNORUNT] + self.field[C.FROSLASS] == 0 else 180
        if card.id == C.MUNKIDORI:
            return 390 if self.field[C.MUNKIDORI] == 0 else 120
        return 1

    def score_search_target(self, card) -> int:
        cid = card.id
        effect = self.effect_id

        if effect == C.SPIKEMUTH_GYM:
            if cid == C.IMPIDIMP and self.board_count() <= 1 and self.open_bench():
                return 1_100
            if cid == C.GRIMMSNARL_EX:
                bases = self.field[C.IMPIDIMP] + self.field[C.MORGREM]
                return 900 if bases and self.hand[C.GRIMMSNARL_EX] == 0 else 520
            if cid == C.MORGREM:
                return 850 if self.field[C.IMPIDIMP] and self.hand[C.MORGREM] == 0 else 460
            if cid == C.IMPIDIMP:
                return 700 if self.field[C.IMPIDIMP] < 2 and self.open_bench() else 150

        if effect == C.POKE_PAD:
            if self.board_count() <= 1 and self.open_bench():
                return {
                    C.MUNKIDORI: 1_100,
                    C.IMPIDIMP: 1_050,
                    C.SNORUNT: 1_000,
                }.get(cid, 250)
            if cid == C.MORGREM:
                return 950 if self.field[C.IMPIDIMP] > self.field[C.MORGREM] else 500
            if cid == C.FROSLASS:
                return 820 if self.field[C.SNORUNT] else 280
            if cid == C.MUNKIDORI:
                return 780 if self.field[C.MUNKIDORI] == 0 else 360
            if cid == C.IMPIDIMP:
                return 700 if self.field[C.IMPIDIMP] < 2 and self.open_bench() else 180
            if cid == C.SNORUNT:
                return 620 if self.field[C.SNORUNT] + self.field[C.FROSLASS] == 0 else 160

        if effect == C.DAWN:
            return {
                C.GRIMMSNARL_EX: 1_000,
                C.MORGREM: 950,
                C.IMPIDIMP: 900,
                C.FROSLASS: 650,
                C.SNORUNT: 600,
                C.MUNKIDORI: 500,
            }.get(cid, 50)

        if effect == C.PETREL:
            if cid == C.BUDDY_POFFIN and self.board_count() <= 2 and self.open_bench():
                return 1_200
            if cid == C.RARE_CANDY and self.hand[C.GRIMMSNARL_EX] and self.eligible_impidimps():
                return 1_050
            if cid == C.BOSSES_ORDERS and self.active_shadow_ready() and self.best_boss_value() >= 10_000:
                return 1_000
            if cid == C.SPIKEMUTH_GYM and self.stadium_id != C.SPIKEMUTH_GYM:
                return 920
            if cid == C.BUDDY_POFFIN and self.open_bench() and self.field[C.IMPIDIMP] < 2:
                return 880
            if cid == C.NIGHT_STRETCHER and self.useful_discard():
                return 840
            if cid == C.UNFAIR_STAMP:
                return 760
            if cid == C.LILLIE:
                return 720 if self.me.handCount <= 4 else 400
            if cid == C.RARE_CANDY:
                return 650
            return 100

        if effect == C.NIGHT_STRETCHER:
            if cid == C.GRIMMSNARL_EX and (self.field[C.IMPIDIMP] or self.field[C.MORGREM]):
                return 1_000
            if cid == C.MORGREM and self.field[C.IMPIDIMP]:
                return 950
            if cid == C.MUNKIDORI and self.field[C.MUNKIDORI] == 0:
                return 880
            if cid == C.DARKNESS and (not self.powered_munkidori() or not self.ready_grimms()):
                return 820
            return {
                C.IMPIDIMP: 700,
                C.FROSLASS: 620,
                C.SNORUNT: 580,
                C.DARKNESS: 500,
            }.get(cid, 50)

        return 200 - self.hand[cid] * 30

    def score_to_hand(self, card):
        if card is None:
            return 0
        return self.score_search_target(card)

    # ----- Trainer use ----------------------------------------------------------
    def can_rare_candy_now(self) -> bool:
        return bool(self.hand[C.GRIMMSNARL_EX] and self.eligible_impidimps())

    def needs_non_rule_search(self) -> bool:
        return bool(
            (self.field[C.IMPIDIMP] and self.hand[C.MORGREM] == 0)
            or (self.field[C.SNORUNT] and self.hand[C.FROSLASS] == 0)
            or self.field[C.MUNKIDORI] == 0
        )

    def score_play_trainer(self, card):
        cid = card.id
        if cid == C.RARE_CANDY:
            if not self.can_rare_candy_now():
                return -1
            return 850_000 if self.board_count() <= 1 else 890_000
        if cid == C.UNFAIR_STAMP:
            return 875_000 if self.opponent.handCount > 2 or self.me.handCount < 5 else 610_000
        if cid == C.BUDDY_POFFIN:
            if self.open_bench() <= 0 or self.me.deckCount <= 5:
                return -1
            missing = self.field[C.IMPIDIMP] < 2 or self.field[C.SNORUNT] + self.field[C.FROSLASS] == 0
            if self.board_count() <= 2:
                return 930_000
            return 790_000 if missing else (560_000 if self.open_bench() >= 2 else -1)
        if cid == C.POKE_PAD:
            if self.me.deckCount <= 7 or not self.needs_non_rule_search():
                return -1
            return 860_000 if self.board_count() <= 1 else 720_000
        if cid == C.NIGHT_STRETCHER:
            return (875_000 if self.board_count() <= 1 else 740_000) if self.useful_discard() else -1
        if cid == C.HANDHELD_FAN:
            active = self.me.active[0] if self.me.active else None
            return 775_000 if active and active.id == C.GRIMMSNARL_EX and not (active.tools or []) else -1
        if cid == C.BOSSES_ORDERS:
            value = self.best_boss_value()
            return 865_000 + value if not self.state.supporterPlayed and value >= 10_000 else -1
        if cid == C.PETREL:
            if self.state.supporterPlayed or self.me.deckCount <= 7:
                return -1
            return 715_000 if self.me.handCount <= 6 or not self.ready_grimms() else 545_000
        if cid == C.LILLIE:
            if self.state.supporterPlayed:
                return -1
            if self.me.handCount <= 3:
                return 805_000
            if len(self.me.prize) == 6 and self.me.handCount <= 6:
                return 770_000
            return 610_000 if self.me.handCount <= 5 and not self.active_shadow_ready() else -1
        if cid == C.DAWN:
            if self.state.supporterPlayed or self.me.deckCount <= 7:
                return -1
            missing_line = not self.ready_grimms() and (
                self.field[C.IMPIDIMP] + self.field[C.MORGREM] == 0
                or self.hand[C.MORGREM] == 0
                or self.hand[C.GRIMMSNARL_EX] == 0
            )
            return 820_000 if missing_line else 575_000
        if cid == C.SPIKEMUTH_GYM:
            if self.state.stadiumPlayed or self.stadium_id == C.SPIKEMUTH_GYM:
                return -1
            return 785_000
        return -1

    # ----- evolution and energy -------------------------------------------------
    def score_evolve(self, option):
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None:
            return 0
        if card.id == C.GRIMMSNARL_EX:
            bonus = self.energy_count(target) * 1_500 if isinstance(target, Pokemon) else 0
            if option.inPlayArea == AreaType.ACTIVE:
                bonus += 1_000
            return 900_000 + bonus
        if card.id == C.MORGREM:
            return 850_000 + (1_000 if option.inPlayArea == AreaType.ACTIVE else 0)
        if card.id == C.FROSLASS:
            ability_targets = sum(self.has_ability(p) for p in self.opponent.active + self.opponent.bench)
            return 760_000 + ability_targets * 12_000
        return 400_000

    def punk_target_score(self, pokemon, area) -> int:
        if pokemon is None or pokemon.id not in MARNIES_LINE:
            return -1
        energy = self.energy_count(pokemon)
        is_active = area == AreaType.ACTIVE
        if pokemon.id == C.GRIMMSNARL_EX and energy < 2:
            return 1_000_000 + energy * 10_000 + (2_000 if is_active else 0)
        # Concentrate three energies on one evolution base: two survive a one-energy retreat.
        if pokemon.id in (C.MORGREM, C.IMPIDIMP) and energy < 3:
            stage_bonus = 8_000 if pokemon.id == C.MORGREM else 0
            return 900_000 + energy * 10_000 + stage_bonus + (4_000 if is_active else 0)
        if pokemon.id == C.GRIMMSNARL_EX:
            return 200_000
        return 100_000 + energy

    def score_attach(self, option):
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return -1
        active = self.me.active[0] if self.me.active else None
        is_active = option.inPlayArea == AreaType.ACTIVE

        # If a powered Grimmsnarl is trapped on the Bench, first buy the retreat.
        if is_active and active.id != C.GRIMMSNARL_EX and self.ready_grimms():
            return 990_000
        if pokemon.id == C.GRIMMSNARL_EX and self.energy_count(pokemon) < 2:
            return 980_000 + self.energy_count(pokemon) * 1_000
        if pokemon.id == C.MUNKIDORI and EnergyType.DARKNESS not in (pokemon.energies or []):
            return 780_000 if self.active_shadow_ready() else 650_000
        if pokemon.id == C.MORGREM and self.energy_count(pokemon) < 2:
            return 700_000 + self.energy_count(pokemon) * 1_000
        if pokemon.id == C.IMPIDIMP and is_active and self.energy_count(pokemon) == 0:
            return 620_000
        return -1

    # ----- abilities and their selections --------------------------------------
    def score_ability(self, option):
        source = get_card(self.obs, option.area, option.index, self.my_index)
        if source is None:
            return -1
        if source.id == C.MUNKIDORI:
            movable = max((self.damage_on(p) for p in self.my_board()), default=0)
            return 880_000 if movable > 0 else -1
        if source.id == C.SPIKEMUTH_GYM:
            wanted = (
                (self.field[C.IMPIDIMP] + self.field[C.MORGREM] and self.hand[C.GRIMMSNARL_EX] == 0)
                or (self.field[C.IMPIDIMP] and self.hand[C.MORGREM] == 0)
                or (self.field[C.IMPIDIMP] < 2 and self.open_bench())
            )
            return 830_000 if wanted else 600_000
        return -1

    def score_counter_source(self, pokemon) -> int:
        damage = self.damage_on(pokemon)
        if damage <= 0:
            return -1
        keep_alive = prize_count(pokemon) * 500
        if pokemon.id == C.GRIMMSNARL_EX:
            keep_alive += 1_500
        return damage * 20 + keep_alive

    def counter_target_score(self, pokemon, is_bench: bool) -> int:
        if pokemon is None or not self.counter_lands(pokemon, is_bench):
            return -1
        hp = int(getattr(pokemon, "hp", 0) or 0)
        score = prize_count(pokemon) * 700 + (450 if self.has_ability(pokemon) else 0)
        if hp <= 30:
            score += 12_000
        elif hp <= 60:
            score += 6_000
        elif hp <= 90:
            score += 2_500
        active = self.opponent.active[0] if self.opponent.active else None
        if pokemon is active and self.active_shadow_ready():
            damage = self.shadow_damage(pokemon)
            if hp > damage and hp - 30 <= damage:
                score += 20_000
        return score - hp

    def spread_target_score(self, pokemon, is_bench: bool) -> int:
        if pokemon is None:
            return -1
        hp = int(getattr(pokemon, "hp", 0) or 0)
        if is_bench and not self.bench_damage_lands(pokemon):
            return -1
        score = prize_count(pokemon) * 800 + (500 if self.has_ability(pokemon) else 0)
        if hp <= 30:
            score += 15_000
        elif hp <= 60:
            score += 7_000
        elif hp <= 90:
            score += 3_000
        elif hp <= 120:
            score += 1_200
        return score - hp

    def score_spread_target(self, card):
        is_bench = card in self.opponent.bench
        if self.effect_id == C.MUNKIDORI:
            return self.counter_target_score(card, is_bench)
        if self.effect_id == C.GRIMMSNARL_EX:
            return self.spread_target_score(card, is_bench)
        return super().score_spread_target(card)

    def score_card(self, option):
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        context = self.context

        if context == SelectContext.ATTACH_TO and self.effect_id == C.HANDHELD_FAN:
            if isinstance(card, Pokemon):
                return 1_000 if option.area == AreaType.ACTIVE and card.id == C.GRIMMSNARL_EX else 10

        if context == SelectContext.ATTACH_FROM and self.effect_id == C.GRIMMSNARL_EX:
            return self.punk_target_score(card, option.area)

        if context == SelectContext.REMOVE_DAMAGE_COUNTER and self.effect_id == C.MUNKIDORI:
            return self.score_counter_source(card) if isinstance(card, Pokemon) else -1

        if context in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY, SelectContext.DAMAGE):
            if isinstance(card, Pokemon) and option.playerIndex == self.op_index:
                return self.score_spread_target(card)

        return super().score_card(option)

    # ----- prize route, retreat, attack -----------------------------------------
    def best_boss_value(self) -> int:
        if not self.active_shadow_ready():
            return -1
        best = -1
        for pokemon in self.opponent.bench:
            damage = self.shadow_damage(pokemon)
            if pokemon.hp <= damage:
                best = max(best, 10_000 + prize_count(pokemon) * 2_000 - pokemon.hp)
        return best

    def gust_value(self, card):
        if not isinstance(card, Pokemon):
            return 0
        damage = self.shadow_damage(card)
        if self.active_shadow_ready() and card.hp <= damage:
            return 20_000 + prize_count(card) * 2_000 - card.hp
        return prize_count(card) * 500 + (300 if self.has_ability(card) else 0) - card.hp

    def score_active_choice(self, option, card):
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex == self.op_index:
            return self.gust_value(card)
        if card.id == C.GRIMMSNARL_EX and self.can_attack(card):
            return 20_000 + card.hp
        if card.id == C.GRIMMSNARL_EX:
            return 8_000 + card.hp
        if card.id == C.MORGREM:
            return 5_000 + self.energy_count(card) * 500
        if card.id == C.IMPIDIMP:
            return 3_000 + self.energy_count(card) * 300
        if card.id == C.SNORUNT:
            return 1_500
        return 500

    def score_retreat(self):
        active = self.me.active[0] if self.me.active else None
        if active is None or self.active_shadow_ready():
            return -1
        if self.ready_grimms():
            return 995_000
        return -1

    def score_attack(self, option):
        active = self.me.active[0] if self.me.active else None
        target = self.opponent.active[0] if self.opponent.active else None
        if active is None:
            return -1
        attack_id = option.attackId
        if attack_id == A.SHADOW_BULLET:
            damage = self.shadow_damage(target)
            score = 760_000
            if target is not None and target.hp <= damage:
                score += 30_000 + prize_count(target) * 5_000
                if prize_count(target) >= len(self.me.prize):
                    score += 100_000
            return score
        if attack_id == A.MORGREM_PUNCH:
            return 620_000 + (20_000 if target is not None and target.hp <= 60 else 0)
        if attack_id == A.FILCH:
            return 575_000 if self.me.deckCount > 3 else -1
        if attack_id == A.IMPIDIMP_PUNCH:
            return 595_000 + (20_000 if target is not None and target.hp <= 10 else 0)
        if attack_id in (A.FROST_SMASH, A.MIND_BEND):
            return 605_000 + (20_000 if target is not None and target.hp <= 60 else 0)
        if attack_id == A.SNORUNT_CHILLY:
            return 560_000 + (20_000 if target is not None and target.hp <= 10 else 0)
        return 500_000

    def choose(self):
        ranked, scores = self.rank()
        chosen = normalize_selection(ranked, scores, self.select)
        if not chosen:
            return chosen

        if self.context == SelectContext.MAIN:
            option = self.select.option[chosen[0]]
            if option.type == OptionType.ATTACK and option.attackId == A.SHADOW_BULLET:
                DIAG["shadow_bullets"] += 1
            elif option.type == OptionType.ABILITY:
                source = get_card(self.obs, option.area, option.index, self.my_index)
                if source is not None and source.id == C.MUNKIDORI:
                    DIAG["adrena_brains"] += 1
            elif option.type == OptionType.RETREAT:
                DIAG["retreats_to_attacker"] += 1
            elif option.type == OptionType.END and self.active_shadow_ready():
                DIAG["attackable_ends"] += 1
        elif self.context == SelectContext.ATTACH_TO and self.effect_id == C.GRIMMSNARL_EX:
            DIAG["punk_up_searches"] += 1
        elif self.context == SelectContext.ATTACH_FROM and self.effect_id == C.GRIMMSNARL_EX:
            for index in chosen:
                option = self.select.option[index]
                target = get_card(self.obs, option.area, option.index, option.playerIndex)
                if target is None:
                    DIAG["punk_up_targets"]["other"] += 1
                elif target.id == C.GRIMMSNARL_EX:
                    DIAG["punk_up_targets"]["current"] += 1
                elif target.id in (C.MORGREM, C.IMPIDIMP):
                    DIAG["punk_up_targets"]["backup"] += 1
                else:
                    DIAG["punk_up_targets"]["other"] += 1
        return chosen


agent = make_agent(GrimmsnarlPolicy, MY_DECK, DIAG)
