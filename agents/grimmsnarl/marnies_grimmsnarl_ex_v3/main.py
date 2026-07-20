"""Marnie's Grimmsnarl ex v3: hand-denial + wider spread over the v2 route model.

The 60-card list is unchanged.  v3 keeps v2's 908-replay route/target model and
adds the two changes that survived a large-sample A/B (4000 games vs two tuned
Alakazam agents; see VALIDATION_REPORT):

  - Unfair Stamp is played as a free Item to strip a large opposing hand *before*
    attacking.  v2 ranked it below Shadow Bullet, so it attacked-and-ended and
    effectively never stamped; but Stamp resets the opponent to two cards, which
    directly starves hand-size damage/draw engines (Alakazam's Powerful Hand does
    2 damage counters per card in the attacker's hand).  This was the measured
    driver of v3's win-rate gain and it generalised across both opponents.

  - A second Munkidori is opened up against a racing opponent (not only on a fully
    developed board), matching the archetype's multi-Munkidori relocation/heal
    engine.  This measured neutral vs the tested Alakazam agents but is retained
    for its damage-spread value against the wider field.

Changes tested and rejected: gating Froslass development and adding an explicit
"heal the threatened attacker" bias both *lowered* the win rate against the
hand-scaling opponent and were dropped.
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

    # Frequent opposing route pieces observed in the attached upper-ladder sample.
    FEZANDIPITI_EX = 140
    ABRA = 741
    KADABRA = 742
    ALAKAZAM = 743
    TAROUNTULA = 400
    SPIDOPS = 401
    ROCKET_MEWTWO_EX = 431
    ROCKET_ARTICUNO = 414
    CYNTHIA_GIBLE = 379
    CYNTHIA_GABITE = 380
    CYNTHIA_GARCHOMP_EX = 381
    CYNTHIA_ROSELIA = 341
    CYNTHIA_ROSERADE = 342

    DUNSPARCE_A = 65
    DUDUNSPARCE = 66
    DUNSPARCE_B = 305
    DWEBBLE = 344
    CRUSTLE = 345
    DRAKLOAK = 120
    DRAGAPULT_EX = 121
    MEGA_KANGASKHAN_EX = 756


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
            "attack_reservation_active": 0,
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
        # Crustle's protective Ability prevents attack damage from Pokémon ex.
        # Shadow Bullet may still place its Bench damage, but the Active damage is 0.
        if target.id == C.CRUSTLE:
            return 0
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
        priorities = {C.GRIMMSNARL_EX, C.MORGREM, C.IMPIDIMP, C.MUNKIDORI, C.SNORUNT, C.FROSLASS, C.DARKNESS}
        return any(card.id in priorities for card in self.me.discard)

    def opponent_ids(self):
        return {p.id for p in self.opponent.active + self.opponent.bench if p is not None}

    def cynthia_pressure(self) -> bool:
        return bool(self.opponent_ids() & {
            C.CYNTHIA_GIBLE, C.CYNTHIA_GABITE, C.CYNTHIA_GARCHOMP_EX,
            C.CYNTHIA_ROSELIA, C.CYNTHIA_ROSERADE,
        })

    def spidops_pressure(self) -> bool:
        return bool(self.opponent_ids() & {C.TAROUNTULA, C.SPIDOPS, C.ROCKET_MEWTWO_EX, C.ROCKET_ARTICUNO})

    def opp_is_racing(self) -> bool:
        """The opponent is applying a fast clock: a large held hand (draw/combo
        fuel, or Alakazam's hand-size damage) or a visible Alakazam draw line."""
        return getattr(self.opponent, "handCount", 0) >= 6 or bool(
            self.opponent_ids() & {C.ABRA, C.KADABRA, C.ALAKAZAM})

    def direct_candy_route(self) -> bool:
        """The visible cards already contain the Impidimp-Candy-Grimmsnarl route."""
        return bool(self.field[C.IMPIDIMP] and self.hand[C.RARE_CANDY] and self.hand[C.GRIMMSNARL_EX])

    def needs_morgrem_bridge(self) -> bool:
        return bool(self.field[C.IMPIDIMP] and self.hand[C.MORGREM] == 0 and not self.direct_candy_route())

    def grim_route_visible(self) -> bool:
        if self.field[C.GRIMMSNARL_EX] or (self.hand[C.GRIMMSNARL_EX] and self.field[C.MORGREM]):
            return True
        if self.direct_candy_route():
            return True
        return bool(self.field[C.IMPIDIMP] and self.hand[C.MORGREM])

    def line_bodies(self):
        return [p for p in self.my_board() if p is not None and p.id in MARNIES_LINE]

    def backup_route_score(self) -> int:
        """Approximate how close the second Shadow Bullet attacker is (public state only)."""
        active = self.me.active[0] if self.me.active else None
        best = 0
        for pokemon in self.line_bodies():
            if pokemon is active and pokemon.id == C.GRIMMSNARL_EX:
                continue
            energy = self.energy_count(pokemon)
            if pokemon.id == C.GRIMMSNARL_EX:
                score = 5 if energy >= 2 else 3 + energy
            elif pokemon.id == C.MORGREM:
                score = 3 + min(2, energy)
                if self.hand[C.GRIMMSNARL_EX]:
                    score += 1
            else:
                score = 1 + min(2, energy)
                if self.hand[C.MORGREM] or (self.hand[C.RARE_CANDY] and self.hand[C.GRIMMSNARL_EX]):
                    score += 2
            best = max(best, score)
        return best

    def backup_is_close(self) -> bool:
        return self.backup_route_score() >= 4

    def active_ko_ready(self) -> bool:
        target = self.opponent.active[0] if self.opponent.active else None
        return bool(self.active_shadow_ready() and target is not None and target.hp <= self.shadow_damage(target))

    def reserve_adjust(self, card_id: int, score: int) -> int:
        """Once Shadow Bullet is live, avoid turning a fixed attack into endless setup."""
        if score < 0 or not self.active_shadow_ready():
            return score
        if card_id == C.BOSSES_ORDERS and self.best_boss_value() >= 10_000:
            return score
        if card_id == C.UNFAIR_STAMP:
            # Free Item; its own scoring already ranks it relative to the attack
            # (above Shadow Bullet only when the opponent holds a large hand).
            return score
        if card_id in (C.RARE_CANDY, C.NIGHT_STRETCHER) and not self.backup_is_close():
            return min(score, 754_000)
        return min(score, 735_000)

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
        # Preserve the only Impidimp when possible; the replay pilot used all three basics
        # as legal pivots rather than blindly exposing the evolution line.
        if card.id == C.IMPIDIMP:
            return 540 if self.hand[C.IMPIDIMP] >= 2 else 430
        if card.id == C.SNORUNT:
            return 500 if self.hand[C.IMPIDIMP] <= 1 else 390
        if card.id == C.MUNKIDORI:
            return 300 if self.hand[C.IMPIDIMP] and self.hand[C.SNORUNT] else 160
        return 10

    def score_play_poke(self, card):
        if self.open_bench() <= 0:
            return -1
        safety = 250_000 if self.board_count() <= 1 else 0
        extra_line = 40_000 if self.cynthia_pressure() or self.spidops_pressure() else 0
        if card.id == C.IMPIDIMP:
            score = (745_000 + safety + extra_line) if self.field[C.IMPIDIMP] < 2 else (635_000 + extra_line if self.field[C.IMPIDIMP] < 3 else -1)
        elif card.id == C.SNORUNT:
            score = (705_000 + safety) if self.field[C.SNORUNT] + self.field[C.FROSLASS] == 0 else 515_000
        elif card.id == C.MUNKIDORI:
            # A second Munkidori doubles the relocation/heal engine that keeps our
            # attacker alive and spreads the opponent's own damage back; open it up
            # against a racing opponent, not only on a fully developed board.
            if self.field[C.MUNKIDORI] == 0:
                score = 695_000 + safety
            elif self.field[C.MUNKIDORI] == 1 and self.open_bench() >= 1 and (self.opp_is_racing() or self.board_count() >= 4):
                score = 615_000 if self.opp_is_racing() else 525_000
            else:
                score = -1
        else:
            return -1
        return min(score, 730_000) if self.active_shadow_ready() else score

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
            # Route deficit first.  Munkidori was the most common upper-log target overall,
            # but not when the Marnie line itself was missing.
            no_line = self.field[C.IMPIDIMP] + self.field[C.MORGREM] + self.field[C.GRIMMSNARL_EX] == 0
            if no_line and cid == C.IMPIDIMP:
                return 1_250
            if self.needs_morgrem_bridge() and cid == C.MORGREM:
                return 1_150
            if self.board_count() <= 1 and self.open_bench():
                return {C.IMPIDIMP: 1_120, C.SNORUNT: 1_070, C.MUNKIDORI: 1_040}.get(cid, 250)
            if cid == C.MORGREM:
                return 1_000 if self.needs_morgrem_bridge() else 420
            if cid == C.FROSLASS:
                return 850 if self.field[C.SNORUNT] else 280
            if cid == C.MUNKIDORI:
                return 820 if self.field[C.MUNKIDORI] == 0 else 330
            if cid == C.IMPIDIMP:
                pressure = 120 if self.cynthia_pressure() or self.spidops_pressure() else 0
                return 730 + pressure if self.field[C.IMPIDIMP] < 2 and self.open_bench() else 180
            if cid == C.SNORUNT:
                return 650 if self.field[C.SNORUNT] + self.field[C.FROSLASS] == 0 else 160

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
            # Observed hierarchy: board establishment, missing evolution, low hand, recovery, exact KO.
            if cid == C.BUDDY_POFFIN and self.board_count() <= 2 and self.open_bench():
                return 1_260
            if cid == C.RARE_CANDY and self.hand[C.GRIMMSNARL_EX] and self.eligible_impidimps():
                return 1_180
            if cid == C.LILLIE and self.me.handCount <= 4:
                return 1_140
            if cid == C.SPIKEMUTH_GYM and self.stadium_id != C.SPIKEMUTH_GYM:
                return 1_090
            if cid == C.POKE_PAD and self.needs_non_rule_search():
                return 1_050
            if cid == C.NIGHT_STRETCHER and self.useful_discard():
                return 1_020
            if cid == C.BOSSES_ORDERS and self.active_shadow_ready() and self.best_boss_value() >= 10_000:
                return 1_000
            if cid == C.BUDDY_POFFIN and self.open_bench() and self.field[C.IMPIDIMP] < 2:
                return 960
            if cid == C.UNFAIR_STAMP and self.opponent.handCount >= 5:
                return 860
            if cid == C.PETREL and self.me.handCount <= 5:
                return 760
            if cid == C.DAWN and not self.ready_grimms():
                return 720
            return 100

        if effect == C.NIGHT_STRETCHER:
            # Upper-log recovery was Energy first, then Impidimp and Grimmsnarl.
            if cid == C.DARKNESS:
                if not self.powered_munkidori():
                    return 1_180
                if not self.backup_is_close():
                    return 1_080
                return 720
            if cid == C.IMPIDIMP and self.field[C.IMPIDIMP] + self.field[C.MORGREM] == 0 and self.open_bench():
                return 1_120
            if cid == C.GRIMMSNARL_EX and (self.field[C.IMPIDIMP] or self.field[C.MORGREM]):
                return 1_100
            if cid == C.MORGREM and self.field[C.IMPIDIMP]:
                return 1_040
            if cid == C.MUNKIDORI and self.field[C.MUNKIDORI] == 0:
                return 900
            return {C.SNORUNT: 700, C.FROSLASS: 680, C.IMPIDIMP: 650}.get(cid, 50)

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
            self.needs_morgrem_bridge()
            or (self.field[C.SNORUNT] and self.hand[C.FROSLASS] == 0)
            or self.field[C.MUNKIDORI] == 0
        )

    def score_play_trainer(self, card):
        cid = card.id
        score = -1
        if cid == C.RARE_CANDY:
            if self.can_rare_candy_now():
                score = 900_000 if not self.ready_grimms() else 780_000
        elif cid == C.UNFAIR_STAMP:
            # Unfair Stamp is a free Item: play it before the attack so we both
            # strip the opponent's hand and refuel ours, then still Shadow Bullet.
            # Its value scales with the opponent's hand size because several ladder
            # decks convert a large hand straight into damage/draw (Alakazam's
            # Powerful Hand is 2 damage counters per card in hand).  Sit it above
            # Shadow Bullet (780k) but below board development, so leftover-hand
            # disruption happens after we have banked our own setup this turn.
            opp_hand = self.opponent.handCount
            if opp_hand >= 6:
                score = 815_000
            elif opp_hand >= 4:
                score = 805_000
            elif self.me.handCount < 4:
                score = 690_000
            else:
                score = 250_000
        elif cid == C.BUDDY_POFFIN:
            if self.open_bench() > 0 and self.me.deckCount > 5:
                missing = self.field[C.IMPIDIMP] < 2 or self.field[C.SNORUNT] + self.field[C.FROSLASS] == 0
                score = 930_000 if self.board_count() <= 2 else (790_000 if missing else (560_000 if self.open_bench() >= 2 else -1))
        elif cid == C.POKE_PAD:
            if self.me.deckCount > 7 and self.needs_non_rule_search():
                score = 860_000 if self.board_count() <= 1 else 720_000
        elif cid == C.NIGHT_STRETCHER:
            if self.useful_discard():
                score = 875_000 if self.board_count() <= 1 else 750_000
        elif cid == C.HANDHELD_FAN:
            active = self.me.active[0] if self.me.active else None
            if active and active.id == C.GRIMMSNARL_EX and not (active.tools or []):
                score = 770_000
        elif cid == C.BOSSES_ORDERS:
            value = self.best_boss_value()
            if not self.state.supporterPlayed and value >= 10_000:
                score = 865_000 + value
        elif cid == C.PETREL:
            if not self.state.supporterPlayed and self.me.deckCount > 7:
                score = 715_000 if self.me.handCount <= 6 or not self.ready_grimms() else 545_000
        elif cid == C.LILLIE:
            if not self.state.supporterPlayed:
                if self.me.handCount <= 3:
                    score = 805_000
                elif len(self.me.prize) == 6 and self.me.handCount <= 6:
                    score = 770_000
                elif self.me.handCount <= 5 and not self.active_shadow_ready():
                    score = 610_000
        elif cid == C.DAWN:
            if not self.state.supporterPlayed and self.me.deckCount > 7:
                missing_line = not self.ready_grimms() and (
                    self.field[C.IMPIDIMP] + self.field[C.MORGREM] == 0
                    or self.needs_morgrem_bridge()
                    or self.hand[C.GRIMMSNARL_EX] == 0
                )
                score = 820_000 if missing_line else 575_000
        elif cid == C.SPIKEMUTH_GYM:
            if not self.state.stadiumPlayed and self.stadium_id != C.SPIKEMUTH_GYM:
                still_needed = not self.ready_grimms() or not self.backup_is_close()
                score = 785_000 if still_needed and self.me.deckCount > 5 else -1
        return self.reserve_adjust(cid, score)

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
            score = 900_000 + bonus
            # Building the second Grimmsnarl is still worth doing before attacking;
            # only a third completed line is pushed below the live Shadow Bullet.
            if self.active_shadow_ready() and self.field[C.GRIMMSNARL_EX] >= 2:
                score = min(score, 742_000)
            return score
        if card.id == C.MORGREM:
            score = 850_000 + (1_000 if option.inPlayArea == AreaType.ACTIVE else 0)
            if self.active_shadow_ready() and self.field[C.GRIMMSNARL_EX] + self.field[C.MORGREM] >= 2:
                score = min(score, 738_000)
            return score
        if card.id == C.FROSLASS:
            ability_targets = sum(self.has_ability(p) for p in self.opponent.active + self.opponent.bench)
            score = 760_000 + ability_targets * 12_000
            if self.active_shadow_ready():
                score = min(score, 736_000)
            return score
        return 400_000

    def punk_target_score(self, pokemon, area) -> int:
        if pokemon is None or pokemon.id not in MARNIES_LINE:
            return -1
        energy = self.energy_count(pokemon)
        is_active = area == AreaType.ACTIVE
        # 1) make the current attacker live; 2) make a distinct backup live on evolution;
        # 3) only then add a retreat buffer.  This avoids v1's one-body three-energy overfocus.
        if pokemon.id == C.GRIMMSNARL_EX and energy < 2:
            return 1_020_000 + energy * 12_000 + (3_000 if is_active else 0)
        if pokemon.id in (C.MORGREM, C.IMPIDIMP) and energy < 2:
            # Prefer the body that has a visible evolution route, rather than blindly
            # assuming Stage 1 is always closer.  The second pilot used Rare Candy much
            # more often and placed most Punk Up energy on Grimmsnarl or Impidimp.
            if pokemon.id == C.MORGREM:
                route_bonus = 12_000 if self.hand[C.GRIMMSNARL_EX] else 2_000
            else:
                route_bonus = 14_000 if self.hand[C.RARE_CANDY] and self.hand[C.GRIMMSNARL_EX] else (8_000 if self.hand[C.MORGREM] else 0)
            pressure_bonus = 8_000 if self.cynthia_pressure() or self.spidops_pressure() else 0
            return 960_000 + energy * 12_000 + route_bonus + pressure_bonus + (3_000 if is_active else 0)
        if pokemon.id in (C.MORGREM, C.IMPIDIMP) and energy == 2:
            # Third energy is a retreat/energy-denial buffer, not the primary plan.
            return 820_000 + (8_000 if pokemon.id == C.MORGREM else 0)
        if pokemon.id == C.GRIMMSNARL_EX and energy == 2:
            return 760_000 if is_active else 700_000
        return 100_000

    def score_attach(self, option):
        pokemon = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        if not isinstance(pokemon, Pokemon):
            return -1
        active = self.me.active[0] if self.me.active else None
        is_active = option.inPlayArea == AreaType.ACTIVE

        # If a powered Grimmsnarl is trapped on the Bench, first buy the retreat.
        if is_active and active is not None and active.id != C.GRIMMSNARL_EX and self.ready_grimms():
            return 990_000
        if pokemon.id == C.GRIMMSNARL_EX and self.energy_count(pokemon) < 2:
            return 980_000 + self.energy_count(pokemon) * 1_000
        if pokemon.id == C.MUNKIDORI and EnergyType.DARKNESS not in (pokemon.energies or []):
            # Across both pilots Munkidori was the most common manual attachment target.
            # Activate it as soon as a concrete Grimmsnarl route is visible; Punk Up will
            # normally supply the attacker's two Energy later.
            if self.active_shadow_ready() or self.backup_is_close() or self.grim_route_visible():
                return 805_000
            return 660_000
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
                or self.needs_morgrem_bridge()
                or (self.field[C.IMPIDIMP] < 2 and self.open_bench())
            )
            score = 830_000 if wanted else 600_000
            return min(score, 730_000) if self.active_shadow_ready() and self.backup_is_close() else score
        return -1

    def score_counter_source(self, pokemon) -> int:
        damage = self.damage_on(pokemon)
        if damage <= 0:
            return -1
        keep_alive = prize_count(pokemon) * 500
        hp = int(getattr(pokemon, "hp", 0) or 0)
        # Remove counters from a Pokémon that is about to be lost, especially the
        # powered Munkidori that enables future Adrena-Brain uses.
        if hp <= 30:
            keep_alive += 5_000
        elif hp <= 60:
            keep_alive += 2_000
        if pokemon.id == C.MUNKIDORI:
            keep_alive += 1_200
        if pokemon.id == C.GRIMMSNARL_EX:
            keep_alive += 1_500
        return damage * 20 + keep_alive

    def route_piece_bonus(self, pokemon) -> int:
        if pokemon is None:
            return 0
        # Large-sample target frequencies show that deleting the next evolution body
        # or a draw/protection engine is often worth more than placing 30 randomly.
        if pokemon.id in {
            C.ABRA, C.KADABRA, C.TAROUNTULA, C.CYNTHIA_GIBLE, C.CYNTHIA_GABITE,
            C.CYNTHIA_ROSELIA, C.DUNSPARCE_A, C.DUNSPARCE_B, C.DWEBBLE,
            C.DRAKLOAK, C.IMPIDIMP, C.MORGREM,
        }:
            return 1_900
        if pokemon.id in {
            C.FEZANDIPITI_EX, C.MUNKIDORI, C.SHAYMIN, C.ROCKET_ARTICUNO,
            C.DUDUNSPARCE,
        }:
            return 1_500
        if pokemon.id in {
            C.ROCKET_MEWTWO_EX, C.SPIDOPS, C.CYNTHIA_GARCHOMP_EX, C.CRUSTLE,
            C.MEGA_KANGASKHAN_EX, C.DRAGAPULT_EX, C.GRIMMSNARL_EX,
        }:
            return 1_200
        return 0

    def counter_target_score(self, pokemon, is_bench: bool) -> int:
        if pokemon is None or not self.counter_lands(pokemon, is_bench):
            return -1
        hp = int(getattr(pokemon, "hp", 0) or 0)
        score = prize_count(pokemon) * 750 + (500 if self.has_ability(pokemon) else 0) + self.route_piece_bonus(pokemon)
        if hp <= 30:
            score += 18_000
        elif hp <= 60:
            score += 8_000  # Shadow Bullet's next bench 30 can finish it.
        elif hp <= 90:
            score += 3_000
        active = self.opponent.active[0] if self.opponent.active else None
        if pokemon is active and self.active_shadow_ready():
            damage = self.shadow_damage(pokemon)
            if hp > damage and hp - 30 <= damage:
                score += 24_000
        return score - hp

    def spread_target_score(self, pokemon, is_bench: bool) -> int:
        if pokemon is None:
            return -1
        hp = int(getattr(pokemon, "hp", 0) or 0)
        if is_bench and not self.bench_damage_lands(pokemon):
            return -1
        score = prize_count(pokemon) * 850 + (550 if self.has_ability(pokemon) else 0) + self.route_piece_bonus(pokemon)
        if hp <= 30:
            score += 20_000
        elif hp <= 60:
            score += 9_000  # one Adrena-Brain or second bullet completes the prize.
        elif hp <= 90:
            score += 4_000
        elif hp <= 120:
            score += 1_500
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
        active = self.opponent.active[0] if self.opponent.active else None
        active_value = -1
        if active is not None and active.hp <= self.shadow_damage(active):
            active_value = prize_count(active) * 2_000 + self.route_piece_bonus(active)
        best = -1
        for pokemon in self.opponent.bench:
            if pokemon.hp <= self.shadow_damage(pokemon):
                candidate = 10_000 + prize_count(pokemon) * 2_000 + self.route_piece_bonus(pokemon) - pokemon.hp
                # Do not spend the supporter to replace an equal-or-better active KO.
                if active_value >= candidate - 10_000:
                    continue
                best = max(best, candidate)
        return best

    def gust_value(self, card):
        if not isinstance(card, Pokemon):
            return 0
        damage = self.shadow_damage(card)
        if self.active_shadow_ready() and card.hp <= damage:
            return 20_000 + prize_count(card) * 2_000 - card.hp
        return prize_count(card) * 500 + (300 if self.has_ability(card) else 0) + self.route_piece_bonus(card) - card.hp

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
            # Against an Active Crustle the 180 is prevented.  Keep the attack as a
            # possible Bench-30 route, but let Boss, Adrena-Brain and development win.
            score = 650_000 if damage == 0 else 780_000
            if target is not None and damage > 0 and target.hp <= damage:
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
