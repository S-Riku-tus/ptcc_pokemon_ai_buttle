"""Conservative deterministic fallback for exact-list Dragapult v1.

The learned ranker owns only supported, mandatory, single-pick decisions.
Multi-pick searches/discards, optional declines, unseen candidates, and model
failures come here.  This policy intentionally encodes mechanics and resource
safety rather than trying to outvote the learned pilot on normal decisions.
"""

from __future__ import annotations

from collections import Counter

from cg.api import AreaType, CardType, EnergyType, OptionType, Pokemon, SelectContext
from policy_base import BasePolicy, card_table, get_card, make_agent, new_diag, prize_count


class C:
    FIRE = 2
    PSYCHIC = 5
    DARK = 7
    MUNKIDORI = 112
    DREEPY = 119
    DRAKLOAK = 120
    DRAGAPULT = 121
    FEZANDIPITI = 140
    BUDEW = 235
    MEOWTH = 1071
    STAMP = 1080
    POFFIN = 1086
    STRETCHER = 1097
    HAMMER = 1120
    ULTRA_BALL = 1121
    POKE_PAD = 1152
    BOSS = 1182
    CRISPIN = 1198
    JUDGE = 1213
    LILLIE = 1227
    DAWN = 1231
    TOWER = 1246


class A:
    MIND_BEND = 141
    PETTY_GRUDGE = 150
    BITE = 151
    DRAGON_HEADBUTT = 152
    JET_HEADBUTT = 153
    PHANTOM_DIVE = 154
    CRUEL_ARROW = 183
    ITCHY_POLLEN = 323
    TUCK_TAIL = 1546


MY_DECK = [
    2, 2, 2, 2,
    5, 5, 5, 5,
    7, 7,
    112, 112,
    119, 119, 119, 119,
    120, 120, 120, 120,
    121, 121, 121,
    140,
    235, 235,
    1071,
    1080,
    1086, 1086, 1086, 1086,
    1097, 1097,
    1120, 1120, 1120, 1120,
    1121, 1121, 1121, 1121,
    1152, 1152, 1152, 1152,
    1182, 1182, 1182,
    1198, 1198, 1198,
    1213,
    1227, 1227, 1227, 1227,
    1231,
    1246, 1246,
]

ENERGY = {C.FIRE, C.PSYCHIC, C.DARK}
ATTACKERS = {
    C.MUNKIDORI, C.DREEPY, C.DRAKLOAK, C.DRAGAPULT,
    C.FEZANDIPITI, C.BUDEW, C.MEOWTH,
}
DIAG = new_diag()
DIAG.update({"chosen": Counter()})


class DragapultPolicy(BasePolicy):
    ENERGY_TYPES = ENERGY
    ATTACKER_IDS = ATTACKERS

    def active(self):
        return self.me.active[0] if self.me.active else None

    def opponent_active(self):
        return self.opponent.active[0] if self.opponent.active else None

    def count_line(self) -> int:
        return sum(
            pokemon is not None and pokemon.id in (C.DREEPY, C.DRAKLOAK, C.DRAGAPULT)
            for pokemon in self.my_board()
        )

    @staticmethod
    def has_type(pokemon, energy_type) -> bool:
        return pokemon is not None and energy_type in list(pokemon.energies or [])

    def phantom_ready(self, pokemon) -> bool:
        return (
            pokemon is not None
            and pokemon.id == C.DRAGAPULT
            and self.has_type(pokemon, EnergyType.FIRE)
            and self.has_type(pokemon, EnergyType.PSYCHIC)
        )

    def go_first(self) -> bool:
        # A Stage-2 deck values the extra evolution turn; this is only the
        # fallback until IS_FIRST has enough teacher support to be routed.
        return True

    def score_setup_active(self, card):
        if card is None:
            return 0
        return {
            C.BUDEW: 600,
            C.DREEPY: 500,
            C.MUNKIDORI: 180,
            C.FEZANDIPITI: 120,
            C.MEOWTH: 80,
        }.get(card.id, 10)

    def score_to_bench(self, card):
        if card is None:
            return 0
        existing = self.field[card.id]
        if card.id == C.DREEPY:
            return 10_000 - 1_500 * existing
        if card.id == C.BUDEW:
            return 7_000 - 5_000 * existing
        if card.id == C.MUNKIDORI:
            return 5_500 - 2_500 * existing
        if card.id == C.FEZANDIPITI:
            return 4_000 - 4_000 * existing
        if card.id == C.MEOWTH:
            return 3_000 - 3_000 * existing
        return 100

    def score_play_poke(self, card):
        if card is None or len(self.me.bench or []) >= int(self.me.benchMax or 5):
            return -1
        return self.score_to_bench(card) * 100

    def useful_supporter_in_hand(self) -> bool:
        return any(self.hand[cid] for cid in (C.CRISPIN, C.LILLIE, C.DAWN, C.JUDGE, C.BOSS))

    def score_play_trainer(self, card):
        if card is None:
            return -1
        cid = card.id
        bench_space = int(self.me.benchMax or 5) - len(self.me.bench or [])
        if cid == C.POFFIN:
            return 610_000 if bench_space > 0 and self.count_line() < 3 else 80_000
        if cid == C.ULTRA_BALL:
            missing_piece = (
                (self.field[C.DREEPY] and not (self.hand[C.DRAKLOAK] or self.hand[C.DRAGAPULT]))
                or (self.field[C.DRAKLOAK] and not self.hand[C.DRAGAPULT])
            )
            return 590_000 if len(self.me.hand or []) >= 3 and missing_piece else 180_000
        if cid == C.POKE_PAD:
            return 555_000 if self.me.deckCount > 5 else 100_000
        if cid == C.STRETCHER:
            useful = any(self.discard[x] for x in (C.DREEPY, C.DRAKLOAK, C.DRAGAPULT, C.FIRE, C.PSYCHIC))
            return 535_000 if useful else -1
        if cid == C.HAMMER:
            return 510_000 if any(p.energies for p in self.opponent.active + self.opponent.bench) else 120_000
        if cid == C.STAMP:
            return 640_000
        if cid == C.TOWER:
            opponent_tools = sum(len(p.tools or []) for p in self.opponent.active + self.opponent.bench)
            own_tools = sum(len(p.tools or []) for p in self.me.active + self.me.bench)
            return 500_000 + 30_000 * (opponent_tools - own_tools)
        if self.state.supporterPlayed:
            return -1
        if cid == C.CRISPIN:
            needs_route_energy = any(
                p.id in (C.DREEPY, C.DRAKLOAK, C.DRAGAPULT)
                and not (self.has_type(p, EnergyType.FIRE) and self.has_type(p, EnergyType.PSYCHIC))
                for p in self.my_board()
            )
            return 630_000 if needs_route_energy else 260_000
        if cid == C.DAWN:
            return 615_000 if self.count_line() < 3 else 240_000
        if cid == C.LILLIE:
            return 600_000 if self.me.handCount <= 6 else 160_000
        if cid == C.JUDGE:
            return 560_000 if self.opponent.handCount > max(4, self.me.handCount) else 210_000
        if cid == C.BOSS:
            if not any(self.phantom_ready(p) for p in self.my_board()) or not self.opponent.bench:
                return -1
            active = self.opponent_active()
            best_bench = max((prize_count(p) * 1000 - p.hp for p in self.opponent.bench), default=0)
            active_value = prize_count(active) * 1000 - active.hp if active else 0
            return 620_000 if best_bench > active_value else 250_000
        return 50_000

    def score_evolve(self, option):
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if card is None or target is None:
            return -1
        if card.id == C.DRAGAPULT:
            energy_bonus = 30_000 * self.energy_count(target)
            active_bonus = 25_000 if option.inPlayArea == AreaType.ACTIVE else 0
            return 900_000 + energy_bonus + active_bonus
        if card.id == C.DRAKLOAK:
            return 780_000 + 20_000 * self.energy_count(target)
        return 100_000

    def score_attach(self, option):
        target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        source = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        if not isinstance(target, Pokemon) or source is None:
            return -1
        if target.id == C.MUNKIDORI and source.id == C.DARK:
            return 860_000 if not self.has_type(target, EnergyType.DARKNESS) else -1
        if target.id in (C.DREEPY, C.DRAKLOAK, C.DRAGAPULT):
            if source.id == C.FIRE and not self.has_type(target, EnergyType.FIRE):
                return 840_000 + 5_000 * self.energy_count(target)
            if source.id == C.PSYCHIC and not self.has_type(target, EnergyType.PSYCHIC):
                return 850_000 + 5_000 * self.energy_count(target)
            return -1
        return super().score_attach(option)

    def score_card(self, option):
        card = get_card(self.obs, option.area, option.index, option.playerIndex)
        if card is None:
            return super().score_card(option)
        if self.context == SelectContext.ATTACH_TO and isinstance(card, Pokemon):
            if card.id in (C.DREEPY, C.DRAKLOAK, C.DRAGAPULT):
                missing = int(not self.has_type(card, EnergyType.FIRE)) + int(
                    not self.has_type(card, EnergyType.PSYCHIC)
                )
                return 20_000 + missing * 5_000 + self.energy_count(card) * 100
            if card.id == C.MUNKIDORI and not self.has_type(card, EnergyType.DARKNESS):
                return 18_000
        if self.context == SelectContext.REMOVE_DAMAGE_COUNTER and isinstance(card, Pokemon):
            return (card.maxHp - card.hp) * 100 + (2_000 if card.id == C.DRAGAPULT else 0)
        if self.context in (
            SelectContext.DAMAGE_COUNTER,
            SelectContext.DAMAGE_COUNTER_ANY,
            SelectContext.DAMAGE,
        ) and isinstance(card, Pokemon) and option.playerIndex == self.op_index:
            return self.score_spread_target(card)
        return super().score_card(option)

    def score_to_hand(self, card):
        if card is None:
            return 0
        priority = {
            C.DRAGAPULT: 10_000,
            C.DRAKLOAK: 9_000,
            C.DREEPY: 8_000,
            C.PSYCHIC: 7_500,
            C.FIRE: 7_300,
            C.DARK: 6_800,
            C.LILLIE: 6_500,
            C.CRISPIN: 6_400,
            C.DAWN: 6_300,
            C.BOSS: 5_800,
        }.get(card.id, 2_000)
        return priority - 400 * self.hand[card.id]

    def score_discard(self, card):
        if card is None:
            return 0
        cid = card.id
        if cid == C.DARK and self.field[C.MUNKIDORI] and not any(
            self.has_type(p, EnergyType.DARKNESS) for p in self.my_board() if p.id == C.MUNKIDORI
        ):
            return -10_000
        if cid in (C.DREEPY, C.DRAKLOAK, C.DRAGAPULT, C.FIRE, C.PSYCHIC):
            return -8_000 + 2_000 * max(0, self.hand[cid] - 1)
        if self.hand[cid] >= 2:
            return 4_000 + 500 * self.hand[cid]
        if cid in (C.HAMMER, C.TOWER, C.JUDGE):
            return 2_500
        return 100

    def score_putback(self, card):
        if card is None:
            return 0
        # Recon Directive: the lower-value card goes to the bottom.
        return -self.score_to_hand(card)

    def score_ability(self, option):
        holder = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if holder is not None and holder.id == C.MUNKIDORI:
            return 930_000 if any(p.hp < p.maxHp for p in self.my_board()) else -1
        if holder is not None and holder.id == C.DRAKLOAK:
            return 920_000 if self.me.deckCount > 1 else -1
        return 700_000

    def score_active_choice(self, option, card):
        if not isinstance(card, Pokemon):
            return 0
        if option.playerIndex == self.op_index:
            return self.gust_value(card)
        if self.phantom_ready(card):
            return 20_000
        if card.id == C.BUDEW and self.state.turn <= 4:
            return 10_000
        return super().score_active_choice(option, card)

    def score_spread_target(self, card):
        if not isinstance(card, Pokemon):
            return 0
        hp = int(card.hp)
        ko_bonus = 20_000 if 0 < hp <= 60 else 0
        return ko_bonus + prize_count(card) * 2_000 - hp * 15

    def score_attack(self, option):
        active = self.opponent_active()
        hp = int(active.hp) if active is not None else 0
        damage = {
            A.MIND_BEND: 60,
            A.PETTY_GRUDGE: 10,
            A.BITE: 40,
            A.DRAGON_HEADBUTT: 70,
            A.JET_HEADBUTT: 70,
            A.PHANTOM_DIVE: 200,
            A.CRUEL_ARROW: 100,
            A.ITCHY_POLLEN: 10,
            A.TUCK_TAIL: 60,
        }.get(option.attackId, 0)
        ko = 500_000 + 80_000 * prize_count(active) if active is not None and 0 < hp <= damage else 0
        if option.attackId == A.PHANTOM_DIVE:
            return 900_000 + ko + 20_000 * sum(0 < p.hp <= 60 for p in self.opponent.bench)
        if option.attackId == A.ITCHY_POLLEN:
            return 760_000 + ko if self.state.turn <= 5 else 430_000 + ko
        if option.attackId == A.JET_HEADBUTT:
            return 650_000 + ko
        if option.attackId == A.CRUEL_ARROW:
            return 620_000 + ko
        return 500_000 + damage * 100 + ko


agent = make_agent(DragapultPolicy, MY_DECK, DIAG)


def diag_reset():
    DIAG.clear()
    DIAG.update(new_diag())
    DIAG.update({"chosen": Counter()})


def diag_snapshot():
    return dict(DIAG)
