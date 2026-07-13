"""Reconstruction of kashiwashira's leaderboard-representative Team Rocket Spidops deck.

Deck source: submission 54603674 (rating 1255.2), reconstructed from 100 public replays.
The policy is not copied source code—the original agent code is unavailable. It is a clean-room
behavioral reconstruction based on the exact deck, action logs, search selections, attachment
sources/targets, and attack frequencies found in those replays.
"""
from __future__ import annotations

import os
from typing import Any

from cg.api import AreaType, CardType, Observation, OptionType, Pokemon, SelectContext
from policy_base import (
    ATTACK_TABLE,
    BasePolicy,
    CARD_TABLE,
    diag_snapshot as _diag_snapshot,
    get_card,
    make_agent,
    new_diag,
    prize_count,
)


class C:
    GRASS = 1
    ROCKET_ENERGY = 15
    TAROUNTULA = 400
    SPIDOPS = 401
    ARTICUNO = 414
    MEWTWO_EX = 431
    MIMIKYU = 434
    BUG_CATCHING_SET = 1094
    ULTRA_BALL = 1121
    TRANSCEIVER = 1134
    POKE_PAD = 1152
    HERO_CAPE = 1159
    BRAVE_BANGLE = 1175
    ARIANA = 1216
    ARCHER = 1217
    GIOVANNI = 1218
    PROTON = 1220
    LILLIE = 1227
    FACTORY = 1257


class A:
    TAROUNTULA = 559
    SPIDOPS = 560
    MEWTWO_EX = 608
    MIMIKYU = 154


ROCKET_POKEMON = {C.TAROUNTULA, C.SPIDOPS, C.ARTICUNO, C.MEWTWO_EX, C.MIMIKYU}
ENERGY_TYPES = {C.GRASS, C.ROCKET_ENERGY}
ATTACKER_IDS = {C.TAROUNTULA, C.SPIDOPS, C.MEWTWO_EX, C.MIMIKYU}
SUPPORTERS = {C.ARIANA, C.ARCHER, C.GIOVANNI, C.PROTON, C.LILLIE}
OPTIONAL_DECK_SPEND = {
    C.BUG_CATCHING_SET,
    C.ULTRA_BALL,
    C.TRANSCEIVER,
    C.POKE_PAD,
    C.ARIANA,
    C.PROTON,
    C.LILLIE,
    C.FACTORY,
}


class Phase:
    SETUP = "SETUP"
    ACCELERATE = "ACCELERATE"
    PRESSURE = "PRESSURE"
    RECOVER = "RECOVER"
    ENDGAME = "ENDGAME"


class Tier:
    WIN = 900_000
    KO = 800_000
    ENABLE_KO = 760_000
    ATTACK = 700_000
    ACCELERATE = 650_000
    BUILD = 580_000
    VALUE = 450_000
    DISRUPT = 350_000
    END = 0


def _resolve_deck_path() -> str:
    candidates: list[str] = []
    if "__file__" in globals():
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv"))
    candidates.extend(["deck.csv", "/kaggle_simulations/agent/deck.csv"])
    import sys

    candidates.extend(os.path.join(path, "deck.csv") for path in sys.path if path)
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("deck.csv not found")


with open(_resolve_deck_path(), encoding="utf-8") as deck_file:
    MY_DECK = [int(line.strip()) for line in deck_file if line.strip()]
if len(MY_DECK) != 60:
    raise ValueError(f"deck.csv must contain exactly 60 card IDs, got {len(MY_DECK)}")

DIAG = new_diag()
_DIAG = DIAG


def diag_reset() -> None:
    DIAG.clear()
    DIAG.update(new_diag())


def diag_snapshot() -> dict[str, Any]:
    return _diag_snapshot(DIAG)


class SpidopsPolicy(BasePolicy):
    ENERGY_TYPES = ENERGY_TYPES
    WILD_ENERGY_IDS = {C.ROCKET_ENERGY}
    ATTACKER_IDS = ATTACKER_IDS

    def go_first(self) -> bool:
        # Stage-1 evolution plus broad setup profits from the extra development turn.
        return True

    def is_mewtwo_bonus_selection(self) -> bool:
        effect = getattr(self.select, "effect", None)
        return getattr(effect, "id", None) == C.MEWTWO_EX and any(
            getattr(option, "energyIndex", None) is not None for option in (self.select.option or [])
        )

    def option_attached_energy(self, option: Any) -> tuple[Any | None, Any | None]:
        pokemon = get_card(
            self.obs,
            getattr(option, "area", None),
            getattr(option, "index", None),
            getattr(option, "playerIndex", self.my_index),
        )
        energy_index = getattr(option, "energyIndex", None)
        try:
            energy = (getattr(pokemon, "energyCards", None) or [])[energy_index]
        except (IndexError, TypeError):
            energy = None
        return energy, pokemon

    def mewtwo_bonus_choice(self) -> list[int]:
        """Choose the minimum recyclable Benched Energy needed to convert 160 into a KO.

        The original leader declined this optional effect most of the time. Across the available
        representative replays it used one Energy 19 times and two Energy twice, and the selected
        Energy was Basic Grass so Spidops could recover it later.
        """
        target = self.opponent_active()
        if target is None:
            return []
        base = self.mewtwo_damage_with_discards(0)
        needed = 0
        for count in (0, 1, 2):
            if self.mewtwo_damage_with_discards(count) >= target.hp:
                needed = count
                break
        else:
            return []
        if needed == 0:
            return []

        ranked: list[tuple[float, int]] = []
        winning = prize_count(target) >= len(self.me.prize or [])
        for index, option in enumerate(self.select.option or []):
            energy, pokemon = self.option_attached_energy(option)
            if energy is None or pokemon is None:
                continue
            energy_id = getattr(energy, "id", None)
            # Grass is recyclable. Preserve Rocket Energy except for a game-winning KO.
            if energy_id == C.GRASS:
                score = 10_000.0
            elif winning:
                score = 1_000.0
            else:
                continue
            # Prefer expendable support bodies; avoid stripping the next ready attacker.
            if getattr(pokemon, "id", None) in (C.ARTICUNO, C.MIMIKYU):
                score += 500
            elif getattr(pokemon, "id", None) == C.TAROUNTULA:
                score += 250
            elif getattr(pokemon, "id", None) == C.SPIDOPS and self.can_attack(pokemon):
                score -= 600
            ranked.append((score, index))
        ranked.sort(reverse=True)
        if len(ranked) < needed:
            return []
        return [index for _, index in ranked[:needed]]

    def choose(self) -> list[int]:
        if self.is_mewtwo_bonus_selection():
            return self.mewtwo_bonus_choice()
        return super().choose()

    # ----- state ------------------------------------------------------------
    def active(self) -> Any:
        return self.me.active[0] if self.me.active else None

    def opponent_active(self) -> Any:
        return self.opponent.active[0] if self.opponent.active else None

    def rocket_count(self) -> int:
        return sum(1 for p in self.my_board() if p is not None and p.id in ROCKET_POKEMON)

    def spidops_count(self) -> int:
        return self.field[C.SPIDOPS]

    def tarountula_lines(self) -> int:
        return self.field[C.TAROUNTULA] + self.field[C.SPIDOPS]

    def deck_floor(self) -> int:
        # The top agent accepted some deck-out risk. Keep only a small forced-draw buffer.
        return max(3, len(self.me.prize or []) + 1)

    def low_deck(self) -> bool:
        return self.me.deckCount <= self.deck_floor()

    def has_basic_grass_in_discard(self) -> bool:
        return self.discard[C.GRASS] > 0

    def has_ready_active(self) -> bool:
        p = self.active()
        return p is not None and self.can_attack(p)

    def phase(self) -> str:
        if self.low_deck():
            return Phase.ENDGAME
        if self.has_ready_active():
            return Phase.PRESSURE
        if self.field[C.SPIDOPS] > 0:
            return Phase.ACCELERATE
        if self.tarountula_lines() == 0 and self.discard[C.TAROUNTULA] > 0:
            return Phase.RECOVER
        return Phase.SETUP

    @staticmethod
    def spidops_damage_from_count(rocket_count: int) -> int:
        # The clean clusters are 60/90/120/150/180 for 2..6 Rocket bodies.
        return 30 * max(0, rocket_count)

    @staticmethod
    def mewtwo_damage_with_discards(discard_count: int) -> int:
        # Replay effect context 26: base 160, then +60 for each of up to two Benched Energy
        # discarded. The leader selected one Energy 19 times and two Energy twice.
        return 160 + 60 * max(0, min(2, discard_count))

    def active_has_tool(self, card_id: int) -> bool:
        active = self.active()
        return any(getattr(tool, "id", None) == card_id for tool in (getattr(active, "tools", None) or []))

    def target_is_ex(self, target: Any | None) -> bool:
        data = CARD_TABLE.get(getattr(target, "id", None))
        return bool(data is not None and (getattr(data, "ex", False) or getattr(data, "megaEx", False)))

    def bench_energy_options(self) -> list[tuple[Any, Any]]:
        result: list[tuple[Any, Any]] = []
        for pokemon in self.me.bench or []:
            if pokemon is None:
                continue
            for energy in getattr(pokemon, "energyCards", None) or []:
                result.append((energy, pokemon))
        return result

    def mewtwo_bonus_capacity(self, *, allow_rocket: bool = False) -> int:
        grass = sum(1 for energy, _ in self.bench_energy_options() if getattr(energy, "id", None) == C.GRASS)
        if allow_rocket:
            other = sum(1 for energy, _ in self.bench_energy_options() if getattr(energy, "id", None) != C.GRASS)
            return min(2, grass + other)
        return min(2, grass)

    def attack_damage(self, attack_id: int, target: Any | None = None) -> int:
        if attack_id == A.SPIDOPS:
            damage = self.spidops_damage_from_count(self.rocket_count())
            # In the replay clusters, every non-weakness +30 event coincided with Brave Bangle.
            if self.active_has_tool(C.BRAVE_BANGLE) and self.target_is_ex(target):
                damage += 30
            return damage
        if attack_id == A.TAROUNTULA:
            return 30
        if attack_id == A.MEWTWO_EX:
            # Only spend Benched Energy when the extra 60/120 can turn the attack into a KO.
            base = self.mewtwo_damage_with_discards(0)
            if target is None or base >= getattr(target, "hp", 10**9):
                return base
            grass_capacity = self.mewtwo_bonus_capacity()
            for count in range(1, grass_capacity + 1):
                damage = self.mewtwo_damage_with_discards(count)
                if damage >= target.hp:
                    return damage
            # For a game-winning KO, permit Team Rocket's Energy as a last-resort resource.
            if target is not None and prize_count(target) >= len(self.me.prize or []):
                return self.mewtwo_damage_with_discards(self.mewtwo_bonus_capacity(allow_rocket=True))
            return base
        if attack_id == A.MIMIKYU:
            return 200
        data = ATTACK_TABLE.get(attack_id)
        raw = getattr(data, "damage", 0) if data is not None else 0
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    def attack_option_exists(self) -> bool:
        return any(o.type == OptionType.ATTACK for o in (self.select.option or []))

    def winning_attack_offered(self) -> bool:
        target = self.opponent_active()
        if target is None:
            return False
        needed = len(self.me.prize or [])
        return any(
            o.type == OptionType.ATTACK
            and self.attack_damage(o.attackId, target) >= target.hp
            and prize_count(target) >= needed
            for o in (self.select.option or [])
        )

    def current_attack_ko(self, target: Any | None = None) -> bool:
        target = target or self.opponent_active()
        if target is None:
            return False
        return any(
            o.type == OptionType.ATTACK and self.attack_damage(o.attackId, target) >= target.hp
            for o in (self.select.option or [])
        )

    def adding_rocket_enables_ko(self) -> bool:
        active = self.active()
        target = self.opponent_active()
        if active is None or target is None or active.id != C.SPIDOPS:
            return False
        before = self.spidops_damage_from_count(self.rocket_count())
        after = self.spidops_damage_from_count(min(6, self.rocket_count() + 1))
        return before < target.hp <= after

    def supporter_played(self) -> bool:
        return bool(getattr(self.state, "supporterPlayed", False))

    # ----- main action scoring ---------------------------------------------
    def score(self, option: Any) -> float:
        raw = super().score(option)
        if self.context != SelectContext.MAIN:
            return raw

        if option.type == OptionType.END:
            if self.attack_option_exists() and any(
                self.attack_damage(o.attackId) > 0
                for o in (self.select.option or [])
                if o.type == OptionType.ATTACK
            ):
                return -1
            return Tier.END

        # Once a game-winning attack is available, do not give another action a chance to
        # invalidate it. Giovanni is handled separately only when it creates the winning target.
        if self.winning_attack_offered() and option.type != OptionType.ATTACK:
            return -1

        # At an extremely low deck count, block optional filtering/draw unless it directly fixes
        # a missing attacker. The guard is intentionally much looser than the late Alakazam builds.
        if self.low_deck() and option.type in (OptionType.PLAY, OptionType.ABILITY):
            card_id = self.option_card_id(option)
            if card_id in OPTIONAL_DECK_SPEND and not self.directly_builds_attacker(card_id):
                return -1
        return raw

    def option_card_id(self, option: Any) -> int | None:
        if option.type == OptionType.PLAY:
            card = get_card(self.obs, AreaType.HAND, getattr(option, "index", None), self.my_index)
        elif option.type == OptionType.ABILITY:
            card = get_card(
                self.obs,
                getattr(option, "area", None),
                getattr(option, "index", None),
                self.my_index,
            )
        else:
            card = None
        return getattr(card, "id", None)

    def directly_builds_attacker(self, card_id: int | None) -> bool:
        return card_id in {C.TAROUNTULA, C.SPIDOPS, C.BUG_CATCHING_SET, C.ULTRA_BALL, C.PROTON}

    def score_play_poke(self, card: Any) -> float:
        if card.id not in ROCKET_POKEMON:
            return -1
        if self.rocket_count() >= 6:
            return -1

        # Every Rocket body adds 30 to Spidops. When one more body reaches a KO, that is not
        # "setup"—it is an immediate damage action and must beat the attack currently offered.
        ko_bonus = Tier.ENABLE_KO if self.adding_rocket_enables_ko() else 0
        phase = self.phase()
        count = self.field[card.id]

        if card.id == C.TAROUNTULA:
            if self.tarountula_lines() == 0:
                base = Tier.ACCELERATE + 25_000
            elif self.tarountula_lines() == 1:
                base = Tier.BUILD + 45_000
            else:
                base = Tier.VALUE + 20_000
        elif card.id == C.ARTICUNO:
            # The first copy protects the wide one-prize board; the second is mainly +30 damage.
            base = Tier.BUILD + (40_000 if count == 0 else -20_000)
        elif card.id == C.MIMIKYU:
            base = Tier.BUILD + (25_000 if count == 0 else -10_000)
        else:  # Mewtwo ex
            base = Tier.BUILD + (30_000 if count == 0 else -25_000)

        if phase == Phase.PRESSURE and self.current_attack_ko():
            base = min(base, Tier.ATTACK - 20_000)
        return max(base, ko_bonus + 1 if ko_bonus else base)

    def score_play_trainer(self, card: Any) -> float:
        cid = card.id
        phase = self.phase()
        board = self.rocket_count()
        hand_count = self.me.handCount

        if cid == C.BUG_CATCHING_SET:
            if self.low_deck():
                return -1
            return Tier.ACCELERATE + 35_000 if self.tarountula_lines() < 2 else Tier.VALUE + 20_000

        if cid == C.ULTRA_BALL:
            need_spidops = self.field[C.TAROUNTULA] > 0 and self.hand[C.SPIDOPS] == 0
            return Tier.ACCELERATE + 55_000 if need_spidops else Tier.VALUE - 20_000

        if cid == C.TRANSCEIVER:
            if self.supporter_played() or self.low_deck():
                return -1
            if board < 5:
                return Tier.BUILD + 20_000
            return Tier.VALUE + 15_000

        if cid == C.POKE_PAD:
            if self.supporter_played() or self.low_deck():
                return -1
            return Tier.BUILD + 10_000 if board < 5 else Tier.VALUE + 10_000

        if cid == C.PROTON:
            if self.supporter_played():
                return -1
            # Searches up to three Rocket Pokémon; this is the strongest first-turn action.
            return Tier.ACCELERATE + 70_000 if board <= 3 else Tier.BUILD + 25_000

        if cid == C.GIOVANNI:
            if self.supporter_played():
                return -1
            best = max((self.gust_value(p) for p in self.opponent.bench or []), default=-1)
            if best >= Tier.WIN:
                return Tier.WIN + 10_000
            if best >= Tier.KO:
                # 83/100 observed Giovanni plays were followed by an attack and 82 by a KO.
                # Preserve the target's prize/energy value so a better bench KO can beat a
                # lower-value Active KO, rather than treating every gust KO identically.
                return best + 5_000
            return Tier.DISRUPT + max(0, best / 100)

        if cid == C.ARIANA:
            if self.supporter_played() or self.low_deck():
                return -1
            return Tier.BUILD + 35_000 if hand_count <= 5 else Tier.VALUE

        if cid == C.LILLIE:
            if self.supporter_played() or self.low_deck():
                return -1
            return Tier.BUILD + 20_000 if hand_count <= 5 else Tier.VALUE - 30_000

        if cid == C.ARCHER:
            if self.supporter_played():
                return -1
            opp_hand = self.opponent.handCount
            return Tier.DISRUPT + 30_000 if opp_hand >= 7 and phase == Phase.PRESSURE else Tier.DISRUPT - 40_000

        if cid == C.FACTORY:
            if self.stadium_id == C.FACTORY:
                return -1
            # Replacing an opposing stadium is valuable; otherwise play early while it can pay back.
            return Tier.BUILD + 15_000 if self.stadium_id is not None else Tier.VALUE + 25_000

        if cid in (C.HERO_CAPE, C.BRAVE_BANGLE):
            return Tier.BUILD + 5_000

        if cid in ENERGY_TYPES:
            return -1  # energy is handled by ATTACH options, never as a generic PLAY
        return Tier.VALUE - 60_000

    def score_evolve(self, option: Any) -> float:
        source = get_card(self.obs, AreaType.HAND, getattr(option, "index", None), self.my_index)
        target = get_card(
            self.obs,
            getattr(option, "inPlayArea", None),
            getattr(option, "inPlayIndex", None),
            self.my_index,
        )
        if source is None or target is None or source.id != C.SPIDOPS or target.id != C.TAROUNTULA:
            return -1
        active_bonus = 35_000 if getattr(option, "inPlayArea", None) == AreaType.ACTIVE else 0
        energy_bonus = len(getattr(target, "energies", None) or []) * 8_000
        return Tier.ACCELERATE + 80_000 + active_bonus + energy_bonus

    def score_attack(self, option: Any) -> float:
        target = self.opponent_active()
        damage = self.attack_damage(option.attackId, target)
        if damage <= 0:
            return -1
        if target is None:
            return Tier.ATTACK + damage
        prizes = prize_count(target)
        if damage >= target.hp and prizes >= len(self.me.prize or []):
            return Tier.WIN + prizes * 10_000
        if damage >= target.hp:
            return Tier.KO + prizes * 20_000 + damage
        # Preserve the observed early Tarountula pressure, but let setup/acceleration happen first.
        if option.attackId == A.TAROUNTULA:
            return Tier.ATTACK - 110_000 + damage
        if option.attackId == A.MIMIKYU:
            return Tier.ATTACK - 40_000 + damage
        return Tier.ATTACK + damage

    def score_ability(self, option: Any) -> float:
        card = get_card(
            self.obs,
            getattr(option, "area", None),
            getattr(option, "index", None),
            self.my_index,
        )
        if card is None:
            return -1
        if card.id == C.SPIDOPS:
            if not self.has_basic_grass_in_discard():
                return -1
            if self.can_attack(card):
                # Fuel another line first; a ready Spidops does not need speculative over-attachment.
                return Tier.VALUE - 30_000
            active_bonus = 20_000 if getattr(option, "area", None) == AreaType.ACTIVE else 0
            return Tier.ACCELERATE + 65_000 + active_bonus
        if card.id == C.FACTORY:
            if self.low_deck() or self.me.handCount >= 7:
                return -1
            return Tier.BUILD + 25_000 if self.me.handCount <= 4 else Tier.VALUE + 10_000
        return Tier.VALUE

    def score_attach(self, option: Any) -> float:
        target = get_card(
            self.obs,
            getattr(option, "inPlayArea", None),
            getattr(option, "inPlayIndex", None),
            self.my_index,
        )
        source = get_card(self.obs, AreaType.HAND, getattr(option, "index", None), self.my_index)
        if target is None or source is None:
            return -1

        if source.id == C.HERO_CAPE:
            if getattr(target, "tools", None):
                return -1
            # Observed targets: Mewtwo 40, Spidops 22, Tarountula 1.
            if target.id == C.MEWTWO_EX:
                return Tier.ACCELERATE + 15_000
            if target.id == C.SPIDOPS:
                return Tier.BUILD + 20_000
            return Tier.VALUE - 30_000

        if source.id == C.BRAVE_BANGLE:
            if getattr(target, "tools", None):
                return -1
            return Tier.ACCELERATE + 10_000 if target.id == C.SPIDOPS else -1

        if source.id not in ENERGY_TYPES:
            return -1
        if self.can_attack(target):
            return -1

        energy_count = len(getattr(target, "energyCards", None) or [])
        active_bonus = 20_000 if getattr(option, "inPlayArea", None) == AreaType.ACTIVE else 0

        if source.id == C.ROCKET_ENERGY:
            # In representative logs: 135 attachments to Mewtwo, 22 to Spidops, only 1 elsewhere.
            if target.id == C.MEWTWO_EX:
                return Tier.ACCELERATE + 80_000 - energy_count * 8_000 + active_bonus
            if target.id == C.SPIDOPS:
                return Tier.ACCELERATE + 55_000 - energy_count * 7_000 + active_bonus
            return -1

        # Basic Grass: Spidops/Tarountula first; Mewtwo can use it, Articuno is a low-priority bank.
        if target.id == C.SPIDOPS:
            return Tier.ACCELERATE + 70_000 - energy_count * 8_000 + active_bonus
        if target.id == C.TAROUNTULA:
            return Tier.ACCELERATE + 60_000 - energy_count * 10_000 + active_bonus
        if target.id == C.MEWTWO_EX:
            return Tier.BUILD + 30_000 - energy_count * 8_000 + active_bonus
        if target.id == C.ARTICUNO:
            return Tier.VALUE - 20_000 - energy_count * 10_000
        return -1

    def score_retreat(self) -> float:
        active = self.active()
        if active is None:
            return -1
        if self.can_attack(active):
            return -1
        ready = [p for p in self.me.bench or [] if p is not None and self.can_attack(p)]
        return Tier.ATTACK - 50_000 if ready else -1

    # ----- card/effect sub-selections --------------------------------------
    def effect_id(self) -> int | None:
        return getattr(getattr(self.select, "effect", None), "id", None)

    def score_to_hand(self, card: Any) -> float:
        effect = self.effect_id()
        cid = card.id

        if effect == C.PROTON:
            # Up to three Team Rocket Pokémon. Build two Spidops lines plus protection/body count.
            if cid == C.TAROUNTULA:
                return 1_000 - self.tarountula_lines() * 180
            if cid == C.ARTICUNO:
                return 860 if self.field[C.ARTICUNO] == 0 else 400
            if cid == C.MIMIKYU:
                return 780 if self.field[C.MIMIKYU] == 0 else 360
            if cid == C.MEWTWO_EX:
                return 740 if self.field[C.MEWTWO_EX] == 0 else 300
            return -1

        if effect == C.BUG_CATCHING_SET:
            if cid == C.SPIDOPS and self.field[C.TAROUNTULA] > 0:
                return 1_000
            if cid == C.TAROUNTULA and self.tarountula_lines() < 2:
                return 900
            if cid == C.GRASS:
                return 850 if self.hand[C.GRASS] == 0 else 500
            return 100 if cid in (C.SPIDOPS, C.TAROUNTULA) else -1

        if effect in (C.TRANSCEIVER, C.POKE_PAD):
            return self.supporter_search_value(cid)

        if effect == C.ULTRA_BALL:
            if cid == C.SPIDOPS and self.field[C.TAROUNTULA] > 0:
                return 1_100
            if cid == C.TAROUNTULA and self.tarountula_lines() == 0:
                return 1_000
            if cid == C.ARTICUNO and self.field[C.ARTICUNO] == 0:
                return 750
            if cid == C.MEWTWO_EX and self.field[C.MEWTWO_EX] == 0:
                return 700
            if cid == C.MIMIKYU and self.field[C.MIMIKYU] == 0:
                return 650
            return 100

        if cid == C.SPIDOPS:
            return 1_000 if self.field[C.TAROUNTULA] > 0 else 500
        if cid == C.TAROUNTULA:
            return 900 if self.tarountula_lines() < 2 else 300
        if cid == C.GRASS:
            return 700
        if cid in SUPPORTERS:
            return self.supporter_search_value(cid)
        return 200 - self.hand[cid] * 30

    def supporter_search_value(self, card_id: int) -> float:
        if card_id not in SUPPORTERS:
            return -1
        if self.supporter_played():
            return -1
        if card_id == C.GIOVANNI:
            best = max((self.gust_value(p) for p in self.opponent.bench or []), default=-1)
            if best >= 760_000:
                return 1_200
        if self.rocket_count() < 5 and card_id == C.PROTON:
            return 1_100
        if self.me.handCount <= 5 and card_id == C.ARIANA:
            return 950
        if self.me.handCount <= 4 and card_id == C.LILLIE:
            return 900
        if self.opponent.handCount >= 7 and card_id == C.ARCHER:
            return 700
        return {C.PROTON: 700, C.ARIANA: 620, C.GIOVANNI: 600, C.LILLIE: 520, C.ARCHER: 300}[card_id]

    def score_discard(self, option: Any, card: Any) -> float:
        cid = card.id
        # Ultra Ball and Mewtwo post-attack costs: basic Grass is intentionally recyclable by
        # Spidops, so it is the preferred energy to discard. Preserve Team Rocket Energy.
        if cid == C.GRASS:
            return 1_000 if self.field[C.SPIDOPS] > 0 else 250
        if cid == C.ROCKET_ENERGY:
            return -500
        if cid == C.FACTORY and (self.stadium_id == C.FACTORY or self.hand[cid] >= 2):
            return 700
        if cid in SUPPORTERS and self.hand[cid] >= 2:
            return 600
        if cid in ROCKET_POKEMON and self.hand[cid] >= 2:
            return 450
        return 50 if self.hand[cid] >= 2 else -100

    def score_attach_target(self, option: Any, card: Any) -> float:
        if card.id == C.SPIDOPS:
            return 1_000 if not self.can_attack(card) else 550
        if card.id == C.TAROUNTULA:
            return 900
        if card.id == C.MEWTWO_EX:
            return 800
        if card.id == C.ARTICUNO:
            return 300
        return 50

    def score_setup_active(self, card: Any) -> float:
        # One de-duplicated setup choice per representative replay (100 total):
        # Tarountula 42, Mimikyu 34, Articuno 14, Mewtwo ex 10.
        return {
            C.TAROUNTULA: 500,
            C.MIMIKYU: 400,
            C.ARTICUNO: 300,
            C.MEWTWO_EX: 200,
        }.get(card.id, 0)

    def score_to_bench(self, card: Any) -> float:
        if card.id not in ROCKET_POKEMON:
            return -1
        count = self.field[card.id]
        if card.id == C.TAROUNTULA:
            return 1_000 - self.tarountula_lines() * 180
        if card.id == C.MEWTWO_EX:
            return 850 if count == 0 else 320
        if card.id == C.ARTICUNO:
            return 820 if count == 0 else 420
        if card.id == C.MIMIKYU:
            return 720 if count == 0 else 350
        return 400

    def score_evolves_choice(self, card: Any) -> float:
        return 1_000 if card.id == C.SPIDOPS else 100

    def gust_value(self, card: Any) -> float:
        active = self.active()
        if active is None:
            return -1
        # Estimate the attack available after Giovanni. Spidops damage depends on our board, not
        # the target, so this is stable enough for target choice.
        if active.id == C.SPIDOPS:
            damage = self.spidops_damage_from_count(self.rocket_count())
        elif active.id == C.MEWTWO_EX:
            damage = 160
        elif active.id == C.TAROUNTULA:
            damage = 30
        elif active.id == C.MIMIKYU:
            damage = 200
        else:
            damage = 0
        value = prize_count(card) * 20_000 + len(getattr(card, "energies", None) or []) * 2_000
        if damage >= getattr(card, "hp", 10**9):
            if prize_count(card) >= len(self.me.prize or []):
                return Tier.WIN
            return Tier.KO + value
        return value - getattr(card, "hp", 0)

    def score_active_choice(self, option: Any, card: Any) -> float:
        if getattr(option, "playerIndex", self.my_index) == self.op_index:
            return self.gust_value(card)
        # Promote a ready attacker first, then a fueled evolution line, then a durable body.
        score = len(getattr(card, "energies", None) or []) * 120
        if self.can_attack(card):
            score += 1_000
        if card.id == C.SPIDOPS:
            score += 700
        elif card.id == C.MEWTWO_EX:
            score += 550
        elif card.id == C.TAROUNTULA and self.hand[C.SPIDOPS] > 0:
            score += 500
        elif card.id == C.MIMIKYU:
            score += 250
        return score + getattr(card, "hp", 0)


agent = make_agent(SpidopsPolicy, MY_DECK, DIAG)
