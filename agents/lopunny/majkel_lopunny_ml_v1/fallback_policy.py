"""Optional deterministic Mega Lopunny baseline and arena ablation.

The learned policy is primary on teacher-distribution states. This policy
encodes interactions established by the exact deck and replay corpus:
Dudunsparce draw cycling, two Lopunny attackers, Air Balloon/Retreat to
activate Gale Thrust, Enriching Energy on an evolution, and attack continuity.
Set ``LOPUNNY_POLICY_MODE=rule`` to run it instead of the learned policy.
"""

from __future__ import annotations

from collections import Counter

from cg.api import AreaType, CardType, OptionType, Pokemon, SelectContext
from policy_base import (
    BasePolicy,
    card_table,
    get_card,
    make_agent,
    new_diag,
    prize_count,
)


class C:
    MIST = 11
    ENRICHING = 13
    SPIKY = 14
    DUDUNSPARCE = 66
    FAN_ROTOM = 174
    DUNSPARCE = 305
    BUNEARY = 848
    MEGA_LOPUNNY = 849
    BUDDY_POFFIN = 1086
    ULTRA_BALL = 1121
    POKEGEAR = 1122
    POKE_PAD = 1152
    AIR_BALLOON = 1174
    BOSS = 1182
    XEROSIC = 1197
    HILDA = 1225
    LILLIE = 1227
    WALLY = 1229


class A:
    DUDUN_LAND_CRUSH = 76
    FAN_ASSAULT_LANDING = 230
    DUN_TRADING_PLACES = 423
    DUN_RAM = 424
    BUNEARY_RUN_AROUND = 1223
    BUNEARY_KICK = 1224
    GALE_THRUST = 1225
    SPIKY_HOPPER = 1226


MY_DECK = [
    11, 11, 11, 11, 13, 14, 14, 14,
    66, 66, 66, 66, 174, 305, 305, 305, 305,
    848, 848, 848, 848, 849, 849, 849,
    1086, 1086, 1086, 1086, 1121, 1121, 1121, 1121,
    1122, 1122, 1122, 1122, 1152, 1152, 1152, 1152,
    1174, 1174, 1174, 1174, 1182, 1182, 1182, 1197,
    1225, 1225, 1225, 1225, 1227, 1227, 1227, 1227,
    1229, 1229, 1229, 1229,
]

ENERGY = {C.MIST, C.ENRICHING, C.SPIKY}
ATTACKERS = {C.BUNEARY, C.MEGA_LOPUNNY}
DIAG = new_diag()
DIAG.update({"chosen": Counter(), "retreat_routes": 0, "dudun_cycles": 0})


class LopunnyPolicy(BasePolicy):
    ENERGY_TYPES = ENERGY
    ATTACKER_IDS = ATTACKERS

    def __init__(self, obs):
        super().__init__(obs)
        effect = getattr(self.select, "effect", None)
        self.effect_id = getattr(effect, "id", -1) if effect is not None else -1

    def board_count(self):
        return len([pokemon for pokemon in self.my_board() if pokemon is not None])

    def count_field(self, card_id):
        return sum(pokemon is not None and pokemon.id == card_id for pokemon in self.my_board())

    def lopunnies(self):
        return [
            pokemon for pokemon in self.my_board()
            if pokemon is not None and pokemon.id == C.MEGA_LOPUNNY
        ]

    def ready_lopunnies(self):
        return [pokemon for pokemon in self.lopunnies() if self.energy_count(pokemon) >= 1]

    def ready_bench_lopunnies(self):
        return [
            pokemon for pokemon in (self.me.bench or [])
            if pokemon is not None
            and pokemon.id == C.MEGA_LOPUNNY
            and self.energy_count(pokemon) >= 1
        ]

    def active(self):
        return self.me.active[0] if self.me.active else None

    def opponent_active(self):
        return self.opponent.active[0] if self.opponent.active else None

    def go_first(self) -> bool:
        # All 194 observable IS_FIRST decisions selected YES.
        return True

    def score_setup_active(self, card):
        if card is None:
            return 0
        return {
            C.DUNSPARCE: 500,
            C.BUNEARY: 400,
            C.FAN_ROTOM: 250,
        }.get(card.id, 50)

    def score_play_poke(self, card):
        if card is None or len(self.me.bench or []) >= int(self.me.benchMax or 5):
            return -1
        if card.id == C.BUNEARY:
            copies = self.count_field(C.BUNEARY) + self.count_field(C.MEGA_LOPUNNY)
            return 680_000 - copies * 55_000
        if card.id == C.DUNSPARCE:
            copies = self.count_field(C.DUNSPARCE) + self.count_field(C.DUDUNSPARCE)
            return 620_000 - copies * 50_000
        if card.id == C.FAN_ROTOM:
            return 420_000 if self.count_field(C.FAN_ROTOM) == 0 else -1
        return 50_000

    def useful_supporter(self):
        if self.state.supporterPlayed:
            return False
        return any(self.hand[card_id] for card_id in (
            C.HILDA, C.LILLIE, C.WALLY, C.XEROSIC, C.BOSS
        ))

    def score_play_trainer(self, card):
        if card is None:
            return -1
        cid = card.id
        if cid == C.BUDDY_POFFIN:
            need = (
                self.count_field(C.BUNEARY) + self.count_field(C.MEGA_LOPUNNY) < 2
                or self.count_field(C.DUNSPARCE) + self.count_field(C.DUDUNSPARCE) < 2
            )
            return 590_000 if self.me.benchMax - len(self.me.bench) > 0 and need else 120_000
        if cid == C.ULTRA_BALL:
            needs_lopunny = self.count_field(C.BUNEARY) > 0 and self.hand[C.MEGA_LOPUNNY] == 0
            needs_dudun = self.count_field(C.DUNSPARCE) > 0 and self.hand[C.DUDUNSPARCE] == 0
            return 570_000 if len(self.me.hand) >= 3 and (needs_lopunny or needs_dudun) else 180_000
        if cid == C.POKEGEAR:
            return 530_000 if not self.state.supporterPlayed and not self.useful_supporter() else 210_000
        if cid == C.POKE_PAD:
            return 500_000 if self.me.deckCount > 5 else 160_000
        if cid == C.AIR_BALLOON:
            missing = any(not (pokemon.tools or []) for pokemon in self.my_board() if pokemon is not None)
            return 520_000 if missing else -1
        if cid == C.WALLY:
            damaged = max(
                (int(p.maxHp) - int(p.hp) for p in self.lopunnies()), default=0
            )
            return 610_000 + damaged * 100 if not self.state.supporterPlayed and damaged >= 60 else -1
        if cid == C.XEROSIC:
            return 560_000 if not self.state.supporterPlayed and self.opponent.handCount > 3 else -1
        if cid == C.LILLIE:
            return 555_000 if not self.state.supporterPlayed and self.me.handCount <= 6 else -1
        if cid == C.HILDA:
            return 550_000 if not self.state.supporterPlayed else -1
        if cid == C.BOSS:
            if self.state.supporterPlayed or not self.ready_lopunnies() or not self.opponent.bench:
                return -1
            active = self.opponent_active()
            best_bench = max(self.opponent.bench, key=lambda pokemon: (prize_count(pokemon), -pokemon.hp))
            active_value = prize_count(active) * 1000 - active.hp if active is not None else -1
            bench_value = prize_count(best_bench) * 1000 - best_bench.hp
            return 565_000 if bench_value > active_value + 200 else -1
        return 100_000

    def score_evolve(self, option):
        source = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if source is None or target is None:
            return -1
        if source.id == C.MEGA_LOPUNNY and target.id == C.BUNEARY:
            first = not self.lopunnies()
            active_bonus = 15_000 if option.inPlayArea == AreaType.ACTIVE else 0
            return 930_000 + int(first) * 60_000 + active_bonus
        if source.id == C.DUDUNSPARCE and target.id == C.DUNSPARCE:
            return 820_000 + (20_000 if option.inPlayArea == AreaType.ACTIVE else 0)
        return 100_000

    def score_ability(self, option):
        source = get_card(self.obs, option.area, option.index, self.my_index)
        if source is None:
            return -1
        if source.id == C.DUDUNSPARCE:
            if self.me.deckCount <= 5 or self.board_count() <= 2:
                return -1
            active_bonus = 100_000 if option.area == AreaType.ACTIVE and self.ready_bench_lopunnies() else 0
            return 850_000 + active_bonus
        return 300_000

    def score_attach(self, option):
        target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        source = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if not isinstance(target, Pokemon) or source is None:
            return -1
        energy_count = self.energy_count(target)
        is_active = option.inPlayArea == AreaType.ACTIVE
        if target.id == C.MEGA_LOPUNNY:
            if energy_count >= 2:
                return -1
            enriching = 75_000 if source.id == C.ENRICHING else 0
            first_energy = 80_000 if energy_count == 0 else 0
            return 760_000 + enriching + first_energy + int(is_active) * 20_000
        if target.id == C.BUNEARY and energy_count == 0 and source.id != C.ENRICHING:
            return 670_000 + int(is_active) * 10_000
        if target.id in (C.DUNSPARCE, C.DUDUNSPARCE) and energy_count == 0:
            return 300_000 + int(is_active) * 30_000
        return -1

    def score_retreat(self):
        active = self.active()
        if active is None or not self.ready_bench_lopunnies():
            return -1
        if active.id == C.MEGA_LOPUNNY:
            return 890_000  # new Active Lopunny gets Gale Thrust's +170
        return 900_000  # escape a pivot directly into the attacker

    def gale_bonus_observed(self):
        if bool(self.state.retreated):
            return True
        for log in self.obs.logs or []:
            if getattr(log, "type", -1) == 8 and getattr(log, "playerIndex", -1) == self.my_index:
                return True
        return False

    def score_attack(self, option):
        active = self.active()
        target = self.opponent_active()
        if active is None:
            return -1
        aid = option.attackId
        damage = 0
        if aid == A.GALE_THRUST:
            damage = 230 if self.gale_bonus_observed() else 60
        elif aid == A.SPIKY_HOPPER:
            damage = 160
        elif aid == A.DUDUN_LAND_CRUSH:
            damage = 90
        elif aid in (A.DUN_RAM, A.BUNEARY_KICK):
            damage = 20
        elif aid in (A.DUN_TRADING_PLACES, A.BUNEARY_RUN_AROUND):
            damage = 0
        elif aid == A.FAN_ASSAULT_LANDING:
            damage = 70 if self.state.stadium else 0
        score = 100_000 + damage * 100
        if target is not None and 0 < target.hp <= damage:
            score += 200_000 + prize_count(target) * 50_000
        if damage == 0 and aid not in (A.DUN_TRADING_PLACES, A.BUNEARY_RUN_AROUND):
            return -1
        return score

    def _search_value(self, card):
        if card is None:
            return 0
        cid = card.id
        if cid == C.MEGA_LOPUNNY:
            return 10_000 + self.count_field(C.BUNEARY) * 4_000 - self.hand[cid] * 1_000
        if cid == C.DUDUNSPARCE:
            return 8_000 + self.count_field(C.DUNSPARCE) * 2_000 - self.hand[cid] * 800
        if cid == C.BUNEARY:
            return 7_000 - (self.count_field(C.BUNEARY) + self.count_field(C.MEGA_LOPUNNY)) * 1_000
        if cid == C.DUNSPARCE:
            return 6_500 - (self.count_field(C.DUNSPARCE) + self.count_field(C.DUDUNSPARCE)) * 900
        if cid == C.ENRICHING:
            return 6_000
        if cid in (C.MIST, C.SPIKY):
            return 4_500
        if cid in (C.HILDA, C.LILLIE, C.WALLY):
            return 4_000
        return 1_000 - self.hand[cid] * 100

    def score_to_hand(self, card):
        return self._search_value(card)

    def score_to_bench(self, card):
        if card is None:
            return 0
        if card.id == C.BUNEARY:
            return 5_000 - (self.count_field(C.BUNEARY) + self.count_field(C.MEGA_LOPUNNY)) * 600
        if card.id == C.DUNSPARCE:
            return 4_500 - (self.count_field(C.DUNSPARCE) + self.count_field(C.DUDUNSPARCE)) * 500
        if card.id == C.FAN_ROTOM:
            return 2_000 if not self.count_field(C.FAN_ROTOM) else 100
        return 500

    def score_discard(self, card):
        if card is None:
            return 0
        cid = card.id
        if cid == C.MEGA_LOPUNNY and self.count_field(C.BUNEARY) > self.hand[cid]:
            return -5_000
        if cid == C.ENRICHING:
            return -4_000
        if cid in ENERGY and sum(self.hand[value] for value in ENERGY) <= 2:
            return -3_000
        if cid == C.BUNEARY and self.count_field(C.BUNEARY) + self.count_field(C.MEGA_LOPUNNY) < 2:
            return -2_500
        if self.hand[cid] >= 2:
            return 2_000 + self.hand[cid] * 100
        if cid in (C.POKEGEAR, C.POKE_PAD, C.AIR_BALLOON, C.XEROSIC, C.BOSS):
            return 800
        return 100

    def score_active_choice(self, option, card):
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex == self.op_index:
            return prize_count(card) * 10_000 - card.hp * 10 + self.energy_count(card) * 500
        if card.id == C.MEGA_LOPUNNY:
            return 20_000 + self.energy_count(card) * 3_000 + card.hp
        if card.id == C.DUNSPARCE:
            return 8_000 + card.hp
        if card.id == C.BUNEARY:
            return 6_000 + self.energy_count(card) * 1_000
        if card.id == C.DUDUNSPARCE:
            return 5_000
        return 1_000 + card.hp

    def score_card(self, option):
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if self.context == SelectContext.HEAL and isinstance(card, Pokemon):
            return (card.maxHp - card.hp) * 100 + int(card.id == C.MEGA_LOPUNNY) * 5_000
        if self.context == SelectContext.ATTACH_TO and self.effect_id == C.AIR_BALLOON:
            if isinstance(card, Pokemon) and not (card.tools or []):
                return 10_000 + int(card is self.active()) * 2_000 + int(card.id == C.MEGA_LOPUNNY) * 1_000
            return -1
        if self.context == SelectContext.TO_HAND:
            return self._search_value(card)
        return super().score_card(option)


agent = make_agent(LopunnyPolicy, MY_DECK, DIAG)


def diag_reset():
    DIAG.clear()
    DIAG.update({"chosen": Counter(), "retreat_routes": 0, "dudun_cycles": 0})


def diag_snapshot():
    return dict(DIAG)
