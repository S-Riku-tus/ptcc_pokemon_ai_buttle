"""Marnie's Grimmsnarl ex v6: wall-aware targeting + faster, safer opening.

v5 already sequences a route, protects a live Shadow Bullet, and recognises a few
damage-immune walls.  v6 acts on the ladder log (rating 723.3, 17-15) where the
remaining losses trace to four fixable errors, plus the two the pilot called out
directly:

* **Wall detection was incomplete.**  Marnie's Grimmsnarl ex is our only Shadow
  Bullet attacker and it is BOTH a Pokémon ex AND a Pokémon *with an Ability*
  (Punk Up).  Cornerstone Mask Ogerpon ex prevents all damage from "Pokémon that
  have an Ability", so it walls us exactly like Crustle/Sylveon — but v5 only
  looked for the *ex* wording and hammered it for 0.  v6 generalises the wall to
  cover both the ex clause and the ability clause (and Neutralization Zone).

* **We hammered walls for 0.**  A 0-damage Shadow Bullet now ranks *below ending
  the turn* unless its Bench-30 (optionally + a ready Adrena-Brain) actually takes
  a prize this turn.  Otherwise we keep developing and look for the Boss's Orders
  unlock instead of stalling.

* **Boss's Orders was too short-sighted.**  Boss is valued on a two-turn prize
  picture: it will not gust a low-prize chip target away from a high-prize Active
  that a two-attack Shadow Bullet sequence already KOs, but it *will* remove a key
  engine or unlock a wall.

* **Targeting priority.**  Bench-30 / Adrena-Brain / Boss targets follow a fixed
  meta ranking — anti-Grimmsnarl tech (Shaymin bench-lock, walls) → the
  opponent's main Pokémon (high-prize ex/mega, draw/damage engines) → their
  pre-evolutions → everything else.

* **Faster, safer opening.**  Explicit `first_attacker_eta` / `backup_attacker_eta`
  drive a T3 focus on completing the first Grimmsnarl before optional engine
  setup, the initial Active prefers Impidimp > Munkidori > Snorunt, and a lone
  board force-develops a Basic so a single KO cannot wipe us.

The deck list is unchanged from v4/v5 so results isolate policy quality.  The
implementation stays rule-based; target ranking is safety-critical and not
delegated to ML in this version.
"""
from __future__ import annotations

import os
from collections import defaultdict

from cg.api import AreaType, CardType, EnergyType, Observation, OptionType, Pokemon, SelectContext
from policy_base import (
    BasePolicy,
    attack_table,
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
    SYLVEON = 330
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
    NEUTRALIZATION_ZONE = 1247

    # Damage-immune Actives that stop Shadow Bullet.  Ogerpon is the v6 addition:
    # it blocks "Pokémon that have an Ability", and Grimmsnarl ex has Punk Up.
    CRUSTLE = 345
    DWEBBLE = 344            # evolves into Crustle -> remove the wall before it lands
    CORNERSTONE_OGERPON = 117

    # Opposing bench-damage blockers (stop Shadow Bullet's Bench-30).
    RABSCA = 74             # protects the opponent's whole Bench

    # Frequent opposing route pieces / engines observed in the ladder sample.
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
    DRAKLOAK = 120
    DRAGAPULT_EX = 121
    MEGA_KANGASKHAN_EX = 756
    MEGA_LUCARIO_EX = 678
    ARCHALUDON_EX = 190
    MEGA_STARMIE_EX = 1031
    IONO_BELLIBOLT_EX = 269
    IONO_KILOWATTREL = 271


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

# ── target-priority tiers (bench-30 / Adrena-Brain / Boss / gust) ──────────────
# Tier 1 — anti-Grimmsnarl tech.  Removing these directly restores our game plan.
ANTI_GRIMM_TECH = {C.SHAYMIN, C.RABSCA}                       # disable our Bench-30 snipes
WALL_TARGETS = {C.CRUSTLE, C.SYLVEON, C.CORNERSTONE_OGERPON, C.DWEBBLE}
# Tier 2 — opponent's main engines (attackers handled by ex/mega/stage2 flags).
OPP_ENGINES = {C.MUNKIDORI, C.FEZANDIPITI_EX, C.DUDUNSPARCE, C.ALAKAZAM, C.IONO_KILOWATTREL}
# Tier 3 — known pre-evolutions of meta lines (a floor so the tier survives even
# when the card database exposes no evolution text, e.g. the static test harness).
KNOWN_PREEVOS = {
    C.ABRA, C.KADABRA, C.TAROUNTULA, C.CYNTHIA_GIBLE, C.CYNTHIA_GABITE,
    C.CYNTHIA_ROSELIA, C.DUNSPARCE_A, C.DUNSPARCE_B, C.DRAKLOAK,
    C.IMPIDIMP, C.MORGREM,
}


def _compute_active_wall_blockers():
    """Opposing Actives whose Ability prevents ALL Shadow Bullet damage to them.

    Marnie's Grimmsnarl ex is our only Shadow Bullet attacker.  It is a Stage 2
    Pokémon ex that also has an Ability (Punk Up) and attacks with basic Darkness
    Energy.  So an Active wall stops us when its Ability prevents all damage from
    an attacker that is a *Pokémon ex* OR a *Pokémon with an Ability*.  Clauses
    that do not apply to us are excluded:

      * "Basic Pokémon ex"  (Farigiraf ex) — Grimmsnarl is Stage 2, not Basic
      * "Tera Pokémon"      (Milotic ex)   — Grimmsnarl is not Tera
      * "Special Energy"    (Carracosta)   — we attach only basic Darkness
      * bench-only wording                 — handled by ``bench_damage_lands``

    Crustle, Sylveon and Cornerstone Mask Ogerpon ex are always included so the
    guard works even when a skill's text is unavailable (static test harness).
    """
    blockers = {C.CRUSTLE, C.SYLVEON, C.CORNERSTONE_OGERPON}
    try:
        for data in card_table.values():
            for skill in (getattr(data, "skills", None) or []):
                text = getattr(skill, "text", "") or ""
                low = text.lower()
                if "prevent all damage" not in low or "this pok" not in low:
                    continue
                if "basic pokémon" in low or "basic pokemon" in low:
                    continue  # Farigiraf ex only stops Basic ex; we are Stage 2
                if "tera" in low:
                    continue  # Milotic ex only stops Tera Pokémon
                if "special energy" in low:
                    continue  # Carracosta needs a Special Energy attacker
                if "your bench" in low or "benched pok" in low:
                    continue  # bench-only protection, handled elsewhere
                ex_wall = "{ex}" in text or "pokémon ex" in low or "pokemon ex" in low
                ability_wall = "have an ability" in low
                if ex_wall or ability_wall:
                    blockers.add(getattr(data, "cardId", None))
    except Exception:
        pass
    blockers.discard(None)
    return blockers


EX_ACTIVE_BLOCKERS = _compute_active_wall_blockers()


def _build_evolution_maps():
    """Map, for every card, the strongest ex its evolution line can reach and the
    set of its descendant names, from the card database's ``evolvesFrom`` chain.

    Lets us treat any Basic/Stage 1 that leads to a multi-prize ex (e.g. Riolu ->
    Mega Lucario ex, Duraludon -> Archaludon ex) as a real pre-evolution snipe
    target without hard-coding every line.
    """
    children = defaultdict(list)         # evolvesFrom-name -> [cardId, ...]
    descendant_names = defaultdict(set)  # name -> {descendant names}
    prize_potential = {}                 # cardId -> best prize count reachable
    try:
        for data in card_table.values():
            ef = getattr(data, "evolvesFrom", None)
            if ef:
                children[ef].append(getattr(data, "cardId", None))

        def prize_of(d):
            return 3 if getattr(d, "megaEx", False) else 2 if getattr(d, "ex", False) else 1

        for data in card_table.values():
            name = getattr(data, "name", "") or ""
            best = prize_of(data)
            seen, names, stack = set(), set(), [name]
            while stack:
                current = stack.pop()
                for child_id in children.get(current, []):
                    child = card_table.get(child_id)
                    if child is None:
                        continue
                    child_name = getattr(child, "name", "") or ""
                    if child_name in seen:
                        continue
                    seen.add(child_name)
                    names.add(child_name)
                    best = max(best, prize_of(child))
                    stack.append(child_name)
            cid = getattr(data, "cardId", None)
            if cid is not None:
                prize_potential[cid] = best
            if name:
                descendant_names[name] |= names
    except Exception:
        pass
    return children, descendant_names, prize_potential


EVO_CHILDREN, EVO_DESCENDANT_NAMES, EVO_PRIZE_POTENTIAL = _build_evolution_maps()


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
            "walled_shadow_bullets": 0,
            "walled_bench_ko_attacks": 0,
            "locked_boss_unlocks": 0,
            "emergency_basics": 0,
            "bench_evolves_active_avail": 0,
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
        # Crustle / Sylveon (any-ex) and Cornerstone Mask Ogerpon ex (any Ability)
        # prevent ALL Shadow Bullet damage to themselves; the Bench-30 can still
        # land.  Neutralization Zone does the same to any non-Rule-Box Pokémon.
        if target.id in EX_ACTIVE_BLOCKERS:
            return 0
        if getattr(self, "stadium_id", 0) == C.NEUTRALIZATION_ZONE and not self.has_rule_box(target):
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

    # ----- damage-immune wall handling ("locked" state) -------------------------
    def active_target_immune_to_ex(self) -> bool:
        """The opponent's Active prevents all Shadow Bullet damage to itself."""
        opp = getattr(self, "opponent", None)
        opp_active = getattr(opp, "active", None) if opp is not None else None
        target = opp_active[0] if opp_active else None
        if target is None:
            return False
        if getattr(target, "id", None) in EX_ACTIVE_BLOCKERS:
            return True
        if getattr(self, "stadium_id", 0) == C.NEUTRALIZATION_ZONE and not self.has_rule_box(target):
            return True
        return False

    def live_attack_ready(self) -> bool:
        """Active Grimmsnarl can attack AND that attack is actually meaningful.

        A ready Shadow Bullet against a wall does 0; treating that as a reserved
        attack would wrongly suppress all pre-attack development.
        """
        return self.active_shadow_ready() and not self.active_target_immune_to_ex()

    def opp_active_max_damage(self) -> int:
        opp = getattr(self, "opponent", None)
        opp_active = getattr(opp, "active", None) if opp is not None else None
        target = opp_active[0] if opp_active else None
        if target is None:
            return 0
        best = 0
        for attack_id in self.payable_attacks(target):
            data = attack_table.get(attack_id)
            best = max(best, int(getattr(data, "damage", 0) or 0) if data is not None else 0)
        return best

    def opp_active_threatens(self, hp: int) -> bool:
        """The opponent's Active can attack this turn for at least ``hp`` damage."""
        return self.opp_active_max_damage() >= hp

    def bench_evolve_available(self, card_id: int) -> bool:
        """A legal evolve option places ``card_id`` onto one of our Bench Pokémon."""
        select = getattr(self, "select", None)
        for option in (getattr(select, "option", None) or []):
            if getattr(option, "type", None) != OptionType.EVOLVE:
                continue
            if getattr(option, "inPlayArea", None) != AreaType.BENCH:
                continue
            card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
            if card is not None and card.id == card_id:
                return True
        return False

    def _adrena_ready_bonus(self) -> int:
        """Extra Bench damage available this turn from a powered Munkidori."""
        if not self.powered_munkidori():
            return 0
        movable = max((self.damage_on(p) for p in self.my_board()), default=0)
        return min(30, (movable // 10) * 10)

    def walled_shadow_value(self) -> int:
        """Score of a 0-damage Shadow Bullet while the Active is a wall.

        Only worth ending the turn on when the Bench-30 (optionally + a ready
        Adrena-Brain) takes a prize NOW.  Otherwise return -1 so the attack ranks
        below END and we stop hammering the wall — development and the Boss unlock
        are already un-suppressed, so we keep building / gust a real target.
        """
        adrena = self._adrena_ready_bonus()
        best = -1
        for pokemon in (getattr(self.opponent, "bench", None) or []):
            if pokemon is None or not self.bench_damage_lands(pokemon):
                continue
            hp = int(getattr(pokemon, "hp", 0) or 0)
            if 0 < hp <= 30 + adrena:
                value = 720_000 + prize_count(pokemon) * 8_000 + self.target_priority_bonus(pokemon)
                best = max(best, value)
        return best

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

    # ----- attacker route ETA ---------------------------------------------------
    def first_attacker_eta(self) -> int:
        """Whole self-turns until a Grimmsnarl ex can use Shadow Bullet.

        0 = attackable this turn (Punk Up powers a fresh evolution the turn it
        evolves), 1 = completable next turn, 2 = only a partial line (needs a
        draw), 99 = no line piece in sight.  Public own state only.
        """
        board = self.my_board()
        if any(p is not None and p.id == C.GRIMMSNARL_EX and self.can_attack(p) for p in board):
            return 0
        # Evolve into Grimmsnarl ex this turn -> Punk Up fuels it -> attacks now.
        if self.hand[C.GRIMMSNARL_EX] and (self.field[C.MORGREM] or self.direct_candy_route()):
            return 0
        grimm_in_play = [p for p in board if p is not None and p.id == C.GRIMMSNARL_EX]
        if grimm_in_play:
            need = 2 - max(self.energy_count(p) for p in grimm_in_play)
            if need <= 0:
                return 0
            return 1 if need <= 1 and self.hand[C.DARKNESS] else 2
        if self.field[C.IMPIDIMP] and self.hand[C.GRIMMSNARL_EX] and (self.hand[C.MORGREM] or self.hand[C.RARE_CANDY]):
            return 1
        if (self.field[C.IMPIDIMP] or self.field[C.MORGREM] or self.hand[C.IMPIDIMP]
                or self.hand[C.MORGREM] or self.hand[C.GRIMMSNARL_EX]):
            return 2
        return 99

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

    def backup_attacker_eta(self) -> int:
        """ETA (self-turns) of a distinct SECOND Shadow Bullet attacker."""
        score = self.backup_route_score()
        if score >= 5:
            return 0
        if score >= 4:
            return 1
        if score >= 2:
            return 2
        return 99

    def backup_is_close(self) -> bool:
        return self.backup_attacker_eta() <= 1

    def opening_focus(self) -> bool:
        """Young board with no attacker yet: complete the line before optional setup."""
        try:
            return self.first_attacker_eta() >= 2 and not self.ready_grimms()
        except Exception:
            return False

    def active_ko_ready(self) -> bool:
        target = self.opponent.active[0] if self.opponent.active else None
        return bool(self.active_shadow_ready() and target is not None and target.hp <= self.shadow_damage(target))

    def unique_backup_route_in_hand(self) -> bool:
        """True when Stamp would shuffle away the only visible next attacker route."""
        routes = 0
        if self.hand[C.GRIMMSNARL_EX] and self.field[C.MORGREM]:
            routes += 1
        if self.direct_candy_route():
            routes += 1
        if self.hand[C.MORGREM] and self.hand[C.GRIMMSNARL_EX] and self.field[C.IMPIDIMP]:
            routes += 1
        return routes == 1 and not self.backup_is_close()

    def stamp_score(self) -> int:
        """Value Unfair Stamp using both hands and route preservation."""
        opp_hand = int(getattr(self.opponent, "handCount", 0) or 0)
        my_hand = int(getattr(self.me, "handCount", 0) or 0)
        if self.unique_backup_route_in_hand() and my_hand >= 6 and opp_hand <= 6:
            return 240_000
        # A large self-hand is a real cost: do not turn ten useful cards into five
        # merely to reduce a medium opposing hand.
        hand_delta = (opp_hand - 2) - max(0, my_hand - 5)
        alakazam_seen = bool(self.opponent_ids() & {C.ABRA, C.KADABRA, C.ALAKAZAM})
        if my_hand <= 4 and opp_hand >= 4:
            return 830_000
        if opp_hand >= 8:
            return 825_000 if hand_delta >= 1 else 790_000
        if alakazam_seen and opp_hand >= 6 and my_hand <= 6:
            return 812_000
        if opp_hand >= 6 and my_hand <= 5:
            return 800_000
        return 250_000

    def reserved_bench_slots(self) -> int:
        reserve = 0
        if self.field[C.IMPIDIMP] + self.field[C.MORGREM] + self.field[C.GRIMMSNARL_EX] == 0:
            reserve += 1
        if not self.backup_is_close() and self.field[C.IMPIDIMP] + self.field[C.MORGREM] < 2:
            reserve += 1
        return reserve

    def useful_second_munkidori(self) -> bool:
        if self.field[C.MUNKIDORI] != 1 or self.open_bench() <= self.reserved_bench_slots():
            return False
        if self.opening_focus():
            return False  # complete the first attacker before doubling the engine
        damaged = sum(self.damage_on(p) for p in self.my_board())
        powered = self.powered_munkidori()
        spread_route = any(int(getattr(p, "hp", 0) or 0) <= 60 for p in self.opponent.bench)
        return bool((powered and damaged >= 30) or spread_route or (self.backup_is_close() and self.field[C.FROSLASS]))

    def one_turn_route_value(self, pokemon, *, include_bullet: bool) -> int:
        """Approximate prize value reachable by Bullet + powered Munkidori this turn."""
        if pokemon is None:
            return 0
        hp = int(getattr(pokemon, "hp", 0) or 0)
        available = (30 if include_bullet else 0) + (30 if self.powered_munkidori() else 0)
        prizes = prize_count(pokemon)
        if available >= hp:
            return 22_000 + prizes * 16_000
        if available + 30 >= hp:
            return 7_000 + prizes * 2_500
        return prizes * 600

    def reserve_adjust(self, card_id: int, score: int) -> int:
        """Once Shadow Bullet is live, avoid turning a fixed attack into endless setup.

        When the Active is walled (0-damage Shadow Bullet), the attack is not
        "live": setup, backup building and the Boss unlock must NOT be suppressed.
        """
        if score < 0 or not self.live_attack_ready():
            return score
        if card_id == C.BOSSES_ORDERS and self.best_boss_value() >= 10_000:
            return score
        if card_id == C.UNFAIR_STAMP:
            return score
        # A Candy that completes the second attacker is BUILD_BACKUP, not optional
        # setup, and must be allowed before the live attack.
        if card_id == C.RARE_CANDY and self.can_rare_candy_now() and not self.backup_is_close():
            return max(score, 795_000)
        if card_id == C.NIGHT_STRETCHER and not self.backup_is_close():
            return min(score, 754_000)
        return min(score, 735_000)

    def opponent_has_bench_shield(self, target) -> bool:
        """An opposing Ability prevents our Bench-30 from landing on ``target``."""
        for p in (list(getattr(self.opponent, "active", None) or [])
                  + list(getattr(self.opponent, "bench", None) or [])):
            if p is None:
                continue
            if p.id == C.RABSCA:
                return True  # protects the opponent's whole Bench
            if p.id == C.SHAYMIN and not self.has_rule_box(target):
                return True  # protects non-Rule-Box Bench Pokémon
        return False

    def bench_damage_lands(self, target) -> bool:
        if target is None:
            return False
        # Crustle / Sylveon / Ogerpon prevent ALL damage from our ex — including
        # Shadow Bullet's Bench-30 — so a benched wall cannot be chipped.
        if getattr(target, "id", None) in EX_ACTIVE_BLOCKERS:
            return False
        # Neutralization Zone zeroes ex damage to non-Rule-Box Pokémon anywhere.
        if getattr(self, "stadium_id", 0) == C.NEUTRALIZATION_ZONE and not self.has_rule_box(target):
            return False
        # Shaymin / Rabsca shield the opposing Bench.  (Battle Cage does NOT stop
        # the Bench-30: its own text says "damage from attacks is still taken".)
        return not self.opponent_has_bench_shield(target)

    def counter_lands(self, target, is_bench: bool) -> bool:
        return not (is_bench and self.stadium_id == C.BATTLE_CAGE)

    # ----- target priority ------------------------------------------------------
    def opp_has_evolution_of(self, pokemon) -> bool:
        """The opponent already has an evolution of ``pokemon`` in play."""
        opp = getattr(self, "opponent", None)
        if opp is None:
            return False
        data = card_table.get(getattr(pokemon, "id", None))
        if data is None:
            return False
        name = getattr(data, "name", "") or ""
        descendants = EVO_DESCENDANT_NAMES.get(name)
        if not descendants:
            return False
        for p in (list(getattr(opp, "active", None) or []) + list(getattr(opp, "bench", None) or [])):
            if p is None or p is pokemon:
                continue
            pd = card_table.get(getattr(p, "id", None))
            if pd is not None and (getattr(pd, "name", "") or "") in descendants:
                return True
        return False

    def target_priority_bonus(self, pokemon) -> int:
        """Meta-aware target ranking for Bench-30 / Adrena-Brain / Boss / gust.

        Anti-Grimmsnarl tech -> opponent main Pokémon -> their pre-evolutions ->
        everything else.  Higher = more valuable to hit / gust / KO.
        """
        if pokemon is None:
            return 0
        cid = getattr(pokemon, "id", None)
        data = card_table.get(cid)
        # Tier 1 — anti-Grimmsnarl tech: removing it restores our game plan.
        if cid in ANTI_GRIMM_TECH:
            return 6_000                      # Shaymin/Rabsca: unlock all Bench-30 snipes
        if cid in WALL_TARGETS:
            return 5_000                      # walls (and Dwebble -> Crustle) that zero us
        # Tier 2 — opponent's main Pokémon: high-prize attackers and key engines.
        if data is not None and getattr(data, "megaEx", False):
            return 3_200
        if data is not None and getattr(data, "ex", False):
            return 2_600
        if cid in OPP_ENGINES:
            return 2_200
        if data is not None and getattr(data, "stage2", False):
            return 1_600
        # Tier 3 — pre-evolutions of main Pokémon (disrupt their setup).
        if self.opp_has_evolution_of(pokemon):
            return 1_400                      # its evolution is already on their board
        potential = EVO_PRIZE_POTENTIAL.get(cid, 1)
        if potential >= 3:
            return 1_200                      # leads to a Mega ex
        if potential >= 2 or cid in KNOWN_PREEVOS:
            return 900                        # leads to an ex / known meta pre-evo
        if data is not None and (getattr(data, "name", "") or "") in EVO_CHILDREN:
            return 500                        # generic pre-evolution
        # Tier 4 — everything else (small bump for ability engines).
        if data is not None and (getattr(data, "skills", None) or []):
            return 200
        return 0

    def route_piece_bonus(self, pokemon) -> int:
        """Backward-compatible alias for :meth:`target_priority_bonus`."""
        return self.target_priority_bonus(pokemon)

    def opponent_has_shaymin(self) -> bool:
        return any(p is not None and p.id == C.SHAYMIN for p in self.opponent.active + self.opponent.bench)

    # ----- search and setup -----------------------------------------------------
    def score_setup_active(self, card):
        if card is None:
            return 0
        # Ladder data: an Impidimp start is fastest (2.42 avg first attack, 66.7%
        # win) and a Snorunt start slowest (3.91, evolving Snorunt->Froslass in
        # the Active fights with clearing the Active and completing the attacker).
        # So Impidimp > Munkidori > Snorunt regardless of copies held.
        if card.id == C.IMPIDIMP:
            return 3_000 if self.hand[C.IMPIDIMP] >= 2 else 2_600
        if card.id == C.MUNKIDORI:
            return 1_800
        if card.id == C.SNORUNT:
            return 1_000
        return 300

    def score_play_poke(self, card):
        if self.open_bench() <= 0:
            return -1
        # A lone board loses to a single KO: force a Basic down before anything
        # else so we cannot be board-wiped (the Ceruledge loss in the log).
        lone = self.board_count() <= 1
        safety = 260_000 if lone else 0
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
            elif self.useful_second_munkidori():
                # PRE_ATTACK_SAFE: it has an immediate relocation/prize route and
                # does not consume a bench slot reserved for the Marnie line.
                score = 792_000 if self.active_shadow_ready() else 665_000
            else:
                score = -1
        else:
            return -1
        if lone and score > 0:
            return max(score, 940_000)  # never skip developing off a single body
        if self.live_attack_ready() and not (card.id == C.MUNKIDORI and self.useful_second_munkidori()):
            return min(score, 730_000)
        return score

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
        # While we still have no attacker, digging for the line beats optional
        # engine setup — the single biggest win correlator is attacking by T3.
        focus = 60 if self.opening_focus() else 0

        if effect == C.SPIKEMUTH_GYM:
            if cid == C.IMPIDIMP and self.board_count() <= 1 and self.open_bench():
                return 1_100
            if cid == C.GRIMMSNARL_EX:
                bases = self.field[C.IMPIDIMP] + self.field[C.MORGREM]
                return 900 if bases and self.hand[C.GRIMMSNARL_EX] == 0 else 520
            if cid == C.MORGREM:
                return 850 if self.needs_morgrem_bridge() else 360
            if cid == C.IMPIDIMP:
                return 700 if self.field[C.IMPIDIMP] < 2 and self.open_bench() else 150

        if effect == C.POKE_PAD:
            # Route deficit first.  Munkidori was the most common upper-log target overall,
            # but not when the Marnie line itself was missing.
            no_line = self.field[C.IMPIDIMP] + self.field[C.MORGREM] + self.field[C.GRIMMSNARL_EX] == 0
            if no_line and cid == C.IMPIDIMP:
                return 1_250 + focus
            if self.needs_morgrem_bridge() and cid == C.MORGREM:
                return 1_150 + focus
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
        no_line = self.field[C.IMPIDIMP] + self.field[C.MORGREM] + self.field[C.GRIMMSNARL_EX] == 0
        return bool(
            no_line
            or self.needs_morgrem_bridge()
            or (self.field[C.SNORUNT] and self.hand[C.FROSLASS] == 0)
            or self.field[C.MUNKIDORI] == 0
        )

    def score_play_trainer(self, card):
        cid = card.id
        score = -1
        if cid == C.RARE_CANDY:
            if self.can_rare_candy_now():
                if not self.ready_grimms():
                    score = 910_000
                elif not self.backup_is_close():
                    score = 805_000
                else:
                    score = 735_000
        elif cid == C.UNFAIR_STAMP:
            score = self.stamp_score()
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
                # A tool on the attacker is optional; do not let it delay the line.
                score = 620_000 if self.opening_focus() else 770_000
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
        is_active = option.inPlayArea == AreaType.ACTIVE
        if card.id == C.GRIMMSNARL_EX:
            bonus = self.energy_count(target) * 1_500 if isinstance(target, Pokemon) else 0
            # Punk Up fuels this Grimmsnarl the turn it evolves, so it normally
            # attacks immediately from the Active — keep the Active preference.
            # The exception is a damage-immune wall (Crustle/Sylveon/Ogerpon/
            # Neutralization Zone): a fresh 2-prize ex facing a wall it cannot hit
            # belongs on the Bench, where Boss's Orders can later open a real target.
            if is_active:
                if self.active_target_immune_to_ex() and self.bench_evolve_available(C.GRIMMSNARL_EX):
                    bonus -= 3_000
                else:
                    bonus += 1_000
            score = 900_000 + bonus
            # Building the second Grimmsnarl is still worth doing before attacking;
            # only a third completed line is pushed below a *live* Shadow Bullet.
            if self.live_attack_ready() and self.field[C.GRIMMSNARL_EX] >= 2:
                score = min(score, 742_000)
            return score
        if card.id == C.MORGREM:
            # Morgrem has no Punk Up and needs two Energy already attached to attack.
            # Evolving the Active is only justified when that Morgrem can attack this
            # turn; otherwise a fragile 100-HP body should be built on the Bench,
            # especially when the opponent's Active can KO it next turn.
            can_attack_now = isinstance(target, Pokemon) and self.energy_count(target) >= 2
            bench_alt = self.bench_evolve_available(C.MORGREM)
            if is_active and can_attack_now:
                score = 851_000
            elif is_active and not bench_alt and not self.opp_active_threatens(100):
                score = 850_000
            elif is_active and bench_alt:
                score = 820_000
            else:
                score = 850_000
            if self.live_attack_ready() and self.field[C.GRIMMSNARL_EX] + self.field[C.MORGREM] >= 2:
                score = min(score, 738_000)
            return score
        if card.id == C.FROSLASS:
            ability_targets = sum(self.has_ability(p) for p in self.opponent.active + self.opponent.bench)
            score = 760_000 + ability_targets * 12_000
            if self.opening_focus():
                score = min(score, 640_000)  # complete the attacker before the utility body
            if self.live_attack_ready():
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
            if self.live_attack_ready() and self.backup_is_close():
                return min(score, 730_000)
            return score
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

    def counter_target_score(self, pokemon, is_bench: bool) -> int:
        if pokemon is None or not self.counter_lands(pokemon, is_bench):
            return -1
        hp = int(getattr(pokemon, "hp", 0) or 0)
        score = prize_count(pokemon) * 750 + (500 if self.has_ability(pokemon) else 0) + self.target_priority_bonus(pokemon)
        score += self.one_turn_route_value(pokemon, include_bullet=False)
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
        score = prize_count(pokemon) * 850 + (550 if self.has_ability(pokemon) else 0) + self.target_priority_bonus(pokemon)
        score += self.one_turn_route_value(pokemon, include_bullet=True)
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
    def active_continuation_value(self) -> int:
        """Prize value of continuing to attack the CURRENT opposing Active.

        High for an immediate KO, and still high for a 2-3 prize ex/mega that a
        two-attack Shadow Bullet sequence already KOs — so Boss will not gust a
        1-prize chip target away from a route that is nearly finished.
        """
        if not self.active_shadow_ready():
            return -1
        active = self.opponent.active[0] if self.opponent.active else None
        if active is None:
            return -1
        damage = self.shadow_damage(active)
        if damage <= 0:
            return -1  # walled: no continuation value, unlock is free to fire
        hp = int(getattr(active, "hp", 0) or 0)
        prizes = prize_count(active)
        if hp <= damage:
            return 20_000 + prizes * 6_000 + self.route_piece_bonus(active)
        if hp - damage <= damage:
            # a two-attack KO in progress: protect the line, weighted by prizes
            return 8_000 + prizes * 5_000 + self.route_piece_bonus(active) // 2
        if prizes >= 2 and hp - damage <= 2 * damage:
            return 3_000 + prizes * 3_000
        return prizes * 1_000

    def best_boss_value(self) -> int:
        if not self.active_shadow_ready():
            return -1
        locked = self.active_target_immune_to_ex()
        continuation = -1 if locked else self.active_continuation_value()
        best = -1
        for pokemon in self.opponent.bench:
            if pokemon is None:
                continue
            damage = self.shadow_damage(pokemon)
            if damage <= 0:
                continue  # gusting a second wall up unlocks nothing
            hp = int(getattr(pokemon, "hp", 0) or 0)
            prizes = prize_count(pokemon)
            bonus = self.route_piece_bonus(pokemon)
            if hp <= damage:
                candidate = 12_000 + prizes * 6_000 + bonus - hp
            elif locked:
                # Unlock: turn 0 into real damage on a fresh target.
                candidate = 12_000 + prizes * 1_500 + bonus + min(damage, hp) // 10
            elif bonus >= 3_000:
                # Not a KO, but removing key anti-Grimmsnarl tech / an engine is
                # still a worthwhile two-turn play once gusted up.
                candidate = 4_000 + bonus // 2 + prizes * 1_500
            else:
                continue
            # Do not spend the supporter to trade an equal-or-better route on the
            # current Active (e.g. a 2-3 prize ex already at a two-attack KO) for a
            # lower-value gust.  When locked, continuation is -1 so any unlock wins.
            if not locked and candidate <= continuation + 6_000:
                continue
            best = max(best, candidate)
        return best

    def gust_value(self, card):
        if not isinstance(card, Pokemon):
            return 0
        damage = self.shadow_damage(card)
        if self.active_shadow_ready() and card.hp <= damage:
            return 20_000 + prize_count(card) * 2_000 + self.route_piece_bonus(card) - card.hp
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
            if damage <= 0:
                # WALL-LOCKED: the 180 is prevented.  Only attack when the Bench-30
                # (+ a ready Adrena-Brain) takes a prize this turn; otherwise this
                # ranks below END so we stop hammering and develop / find Boss.
                return self.walled_shadow_value()
            score = 780_000
            if target is not None and target.hp <= damage:
                score += 30_000 + prize_count(target) * 5_000
                if prize_count(target) >= len(self.me.prize):
                    score += 100_000
            return score
        if attack_id == A.MORGREM_PUNCH:
            # 60 for {D}{D}, and NOT blocked by ex/ability walls (Morgrem has no
            # Ability and is not an ex): a real action into a walled Active.
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
                if self.active_target_immune_to_ex():
                    DIAG["walled_shadow_bullets"] += 1
                    DIAG["walled_bench_ko_attacks"] += 1  # only reached when a Bench KO justified it
            elif option.type == OptionType.ABILITY:
                source = get_card(self.obs, option.area, option.index, self.my_index)
                if source is not None and source.id == C.MUNKIDORI:
                    DIAG["adrena_brains"] += 1
            elif option.type == OptionType.PLAY:
                played = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if played is not None and played.id == C.BOSSES_ORDERS and self.active_target_immune_to_ex():
                    DIAG["locked_boss_unlocks"] += 1
                elif played is not None and self.board_count() <= 1 and card_table.get(played.id) is not None \
                        and getattr(card_table.get(played.id), "cardType", None) == CardType.POKEMON:
                    DIAG["emergency_basics"] += 1
            elif option.type == OptionType.EVOLVE:
                card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if (card is not None and card.id in (C.MORGREM, C.GRIMMSNARL_EX)
                        and option.inPlayArea == AreaType.BENCH
                        and self.bench_evolve_available(card.id)):
                    DIAG["bench_evolves_active_avail"] += 1
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
