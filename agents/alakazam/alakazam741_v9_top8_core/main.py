# alakazam741_v9_top8_core — top-8 general core reproduction.
#
# Rebuilt from the 8-team / 793-game analysis. The v8 flat Tier score is replaced by an
# EXPLICIT per-turn state machine (one state per MAIN decision) plus an ATTACK RESERVATION
# discipline that reproduces the top pilots' "reserve the attack first, then only run the
# pre-attack actions that keep it, then always attack" line.
#
# Deliberately unchanged from v8: the 60-card deck (this experiment isolates LOGIC, not cards),
# the bundled generic BasePolicy/make_agent (energy discipline, PrizeTracker, fallback), and the
# core damage model. No Boss's Orders and no opponent-card / deck-name matchup branches exist.
#
# Removed vs v8:
#   * fixed LOW_DECK_COUNT / _deck_floor = max(8, prizes+3) / _deck_preserve big-hand floor
#     and the hard "hand>=12 and deck<=14" Dudunsparce cap — all replaced by a dynamic
#     turns_to_deckout vs turns_to_win model that also counts Dudunsparce/Sacred Ash returns.
#   * board-size targets: development is gated on missing ROLES, never on a body count.
from __future__ import annotations

import os

from cg.api import (
    AreaType, CardType, EnergyType, Observation, OptionType, Pokemon, SelectContext,
)
from policy_base import (
    BasePolicy, EFFECT_PREVENT_ENERGY, ENERGY_PROVIDES, ITEM_LOCK_IDS, card_table,
    get_card, make_agent, new_diag, prize_count,
)


# ── Card IDs (胡地小人 / Alakazam + Dudunsparce single-prize) ─────────────────
class C:
    ABRA = 741            # Basic -> Kadabra
    KADABRA = 742         # Stage1 (Psychic Draw +2 on evolve) -> Alakazam
    ALAKAZAM = 743        # Stage2 attacker: Powerful Hand = 20 dmg x cards in hand
    DUNSPARCE = 305       # Basic -> Dudunsparce
    DUDUNSPARCE = 66      # Stage1 draw engine (Run Away Draw: draw 3, then shuffle self+attached into deck)

    PSYCHIC_ENERGY = 5
    TELEPATH_ENERGY = 19  # special, provides {P}; on attach searches 2 basic {P} bodies to bench
    HYPER_AROMA = 1082    # ACE SPEC Item: search up to 3 Stage-1 cards to hand

    BUDDY_POFFIN = 1086
    POKE_PAD = 1152
    HILDA = 1225          # Supporter: search Evolution + Energy
    LILLIE = 1227         # Supporter: reset a thin hand, draw 6/8
    DAWN = 1231           # Supporter: search Basic+Stage1+Stage2
    RARE_CANDY = 1079
    XEROSIC = 1197        # Supporter: opponent discards to 3 cards
    BATTLE_CAGE = 1264    # Stadium: block bench damage counters
    ENHANCED_HAMMER = 1081  # Item: discard a Special Energy from opponent
    NIGHT_STRETCHER = 1097  # Item: discard -> hand (Pokémon or basic Energy)
    SACRED_ASH = 1129     # returns up to 5 Pokémon from discard to the deck
    LANA_AID = 1184       # Supporter: heal / return cards toward the deck


POWERFUL_HAND = 1072   # Alakazam: place 2 counters (20 dmg) per card in hand, on opp Active
SUPER_PSY_BOLT = 1071  # Kadabra: 30
ABRA_TELEPORT = 1070   # Abra: 10 + switch
DUDUN_LAND_CRUSH = 76  # Dudunsparce: 90 (rare)
DUNSPARCE_TRADE = 423  # Dunsparce: switch (0 cost)
DUNSPARCE_RAM = 424    # Dunsparce: 20

ALAKAZAM_IDS = {C.ALAKAZAM}
ATTACKER_IDS = {C.ALAKAZAM, C.KADABRA}
ENERGY_TYPES = {C.PSYCHIC_ENERGY, C.TELEPATH_ENERGY}

ALAKAZAM_LINE = (C.ABRA, C.KADABRA, C.ALAKAZAM)
ENGINE_LINE = (C.DUNSPARCE, C.DUDUNSPARCE)
RECOVERABLE = (C.ABRA, C.KADABRA, C.ALAKAZAM, C.DUNSPARCE, C.DUDUNSPARCE, C.PSYCHIC_ENERGY)
DRAW_SUPPORTERS = (C.HILDA, C.DAWN, C.LILLIE, C.XEROSIC)


class TurnState:
    SETUP = "SETUP"
    PRESSURE = "PRESSURE"
    RECOVER = "RECOVER"
    LOCKED = "LOCKED"
    ENDGAME = "ENDGAME"


class Tier:
    BLOCK = 0            # never (below END)
    WIN_OR_SURVIVE = 1   # game-winning KO / the one unlock that restores an attack
    PRE_ATTACK = 2       # gated action that keeps the reserved attack and improves the plan
    ATTACK = 3           # the reserved attack itself
    BUILD_ATTACKER = 4   # complete THIS turn's / next attacker
    BUILD_BACKUP = 5     # secure the following attacker
    SEARCH = 6           # optional dig
    DISRUPT = 7          # hammer / xerosic / stadium
    END = 8


TIER_BASE = {
    Tier.BLOCK: -1_000_000,
    Tier.WIN_OR_SURVIVE: 800_000,
    Tier.PRE_ATTACK: 700_000,
    Tier.ATTACK: 600_000,
    Tier.BUILD_ATTACKER: 500_000,
    Tier.BUILD_BACKUP: 400_000,
    Tier.SEARCH: 300_000,
    Tier.DISRUPT: 200_000,
    Tier.END: 0,
}
TIER_SPAN = 100_000


def _fresh_diag():
    d = new_diag()
    d.update({
        # per-MAIN-decision behavioral counters (cumulative across the process)
        "state": {TurnState.SETUP: 0, TurnState.PRESSURE: 0, TurnState.RECOVER: 0,
                  TurnState.LOCKED: 0, TurnState.ENDGAME: 0},
        "attack_reserved": 0,       # decisions where a >0-damage attack was on the table
        "attacks": 0,               # attacks actually chosen
        "alakazam_attacks": 0,      # attacks chosen while Alakazam is Active
        "zero_damage_attacks": 0,   # 0-damage attack chosen (must stay 0)
        "attackable_ends": 0,       # END chosen while a meaningful attack existed (must stay 0)
        "retreats": 0,              # RETREAT chosen
        "dudun_abilities": 0,       # Run Away Draw chosen
        "dudun_last_active_blocked": 0,  # last-body Run Away Draw refused (safety)
        "pre_attack_actions": 0,    # gated actions taken while an attack was reserved
    })
    return d


_DIAG = _fresh_diag()


def diag_reset():
    _DIAG.clear()
    _DIAG.update(_fresh_diag())


def diag_snapshot():
    snap = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DIAG.items()}
    snap["fallback_rate"] = (
        snap.get("policy_fallback", 0) + snap.get("obs_fallback", 0)
    ) / max(1, snap.get("decisions", 0))
    return snap


def _resolve_deck_path():
    import sys
    cands = []
    if "__file__" in globals():
        cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv"))
    cands += ["deck.csv", "/kaggle_simulations/agent/deck.csv"]
    cands += [os.path.join(p, "deck.csv") for p in sys.path if p]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("deck.csv not found")


DECK_PATH = _resolve_deck_path()
with open(DECK_PATH) as f:
    my_deck = [int(x) for x in f.read().splitlines() if x.strip()]
if len(my_deck) != 60:
    raise ValueError(f"deck.csv must have 60 ids, got {len(my_deck)}")


# ── Alakazam policy ──────────────────────────────────────────────────────────
class AlakazamPolicy(BasePolicy):
    ENERGY_TYPES = ENERGY_TYPES
    ATTACKER_IDS = ATTACKER_IDS

    def __init__(self, obs: Observation):
        super().__init__(obs)
        # Compute the turn state + attack reservation ONCE per decision (guarded so a
        # malformed sub-select can never break scoring — base make_agent still falls back).
        self._state = TurnState.SETUP
        self._plan = {"damage": 0, "kos": False}
        self._attack_reserved = False
        try:
            self._plan = self._compute_attack_plan()
            self._attack_reserved = self._plan["damage"] > 0
            self._state = self._classify_state()
        except Exception:
            pass

    def go_first(self) -> bool:
        # Setup deck: the top pilots take turn 1 to build toward a turn-2 attack.
        return True

    # —— abstract hook wrappers (dispatch lives in the methods below) ——
    def score_play_poke(self, card):
        return self._score_play_poke(card)

    def score_play_trainer(self, card):
        return self._score_play_trainer(card)

    def score_evolve(self, o):
        return self._score_evolve(o)

    def score_attack(self, o):
        return self._score_attack(o)

    def score_ability(self, o):
        return self._score_ability(o)

    def score_card(self, o):
        return self._score_card(o)

    def score_attach(self, o):
        return self._score_attach(o)

    def score_retreat(self):
        return self._score_retreat()

    # ── damage model ─────────────────────────────────────────────────────────
    def _effect_prevented(self, target):
        return self.effect_prevented(target)

    def _opp_active_has_prevent_energy(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is None:
            return False
        return any(getattr(e, "id", None) in EFFECT_PREVENT_ENERGY
                   for e in (getattr(opp, "energyCards", None) or []))

    def _alakazam_damage(self, attack_id, target):
        if target is None:
            return 0
        if attack_id == POWERFUL_HAND:
            # Places counters (an effect) — Weakness/Resistance do not apply, Mist etc. => 0.
            if self._effect_prevented(target):
                return 0
            return 20 * self.me.handCount
        if attack_id == SUPER_PSY_BOLT:
            dmg = 30
        elif attack_id == ABRA_TELEPORT:
            dmg = 10
        elif attack_id == DUNSPARCE_RAM:
            dmg = 20
        elif attack_id == DUDUN_LAND_CRUSH:
            dmg = 90
        else:
            return 0
        data = card_table.get(target.id)
        if data is not None:
            if data.weakness == EnergyType.PSYCHIC:
                dmg *= 2
            elif data.resistance == EnergyType.PSYCHIC:
                dmg = max(0, dmg - 30)
        return dmg

    def _active_best_dmg(self, target):
        active = self.me.active[0] if self.me.active else None
        if active is None or target is None or not self.can_attack(active):
            return 0
        if active.id == C.ALAKAZAM:
            return self._alakazam_damage(POWERFUL_HAND, target)
        if active.id == C.KADABRA:
            return self._alakazam_damage(SUPER_PSY_BOLT, target)
        if active.id == C.DUNSPARCE:
            return self._alakazam_damage(DUNSPARCE_RAM, target)
        if active.id == C.DUDUNSPARCE:
            return self._alakazam_damage(DUDUN_LAND_CRUSH, target)
        return 0

    # ── attack reservation ─────────────────────────────────────────────────────
    def _best_offered_attack_damage(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is None:
            return 0
        best = 0
        for o in (self.select.option or []):
            if o.type == OptionType.ATTACK:
                best = max(best, self._alakazam_damage(o.attackId, opp))
        return best

    def _compute_attack_plan(self):
        """The attack we RESERVE for the end of the turn: the best >0-damage attack currently
        offered, plus whether it KOs the opposing Active (used by the pre-attack gate to protect
        the current KO from a hand spend)."""
        opp = self.opponent.active[0] if self.opponent.active else None
        best = self._best_offered_attack_damage()
        plan = {"damage": best, "kos": False}
        if opp is not None and best > 0:
            plan["kos"] = best >= opp.hp
        return plan

    def _has_meaningful_attack_option(self):
        return any(o.type == OptionType.ATTACK and self._attack_damage_for_option(o) > 0
                   for o in (self.select.option or []))

    def _attack_damage_for_option(self, option):
        if option.type != OptionType.ATTACK:
            return 0
        opp = self.opponent.active[0] if self.opponent.active else None
        return self._alakazam_damage(option.attackId, opp) if opp is not None else 0

    # ── dynamic deck / endgame model ────────────────────────────────────────────
    def _deck_returns_available(self):
        """Cards that will come BACK to the deck and stay re-drawable — the 'effective deck'.
        Run Away Draw shuffles the Dudunsparce stack (Dudunsparce + its Dunsparce + attached)
        back in (~2 cards each); Sacred Ash returns up to 5 line Pokémon from the discard."""
        ret = 2 * self.field[C.DUDUNSPARCE]
        line_in_discard = sum(self.discard.get(x, 0) for x in
                              (C.ABRA, C.KADABRA, C.ALAKAZAM, C.DUNSPARCE, C.DUDUNSPARCE))
        if self.hand.get(C.SACRED_ASH, 0) and not self.state.supporterPlayed:
            ret += min(5, line_in_discard)
        return ret

    def _effective_deck(self):
        return self.me.deckCount + self._deck_returns_available()

    def _turns_to_win(self):
        """Our own turns still needed to take the remaining prizes. Top-8 games run ~1.4 own
        turns per prize (attack rate < 100% and a KO is not always available), so a RAW prize
        count under-reserves the deck and decks us out; scale it up, then credit a reachable
        multi-prize KO taken this turn."""
        prizes = len(self.me.prize)
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is not None and self._ko_active_reachable():
            prizes -= max(0, prize_count(opp) - 1)
        prizes = max(1, prizes)
        return max(1, (prizes * 10 + 6) // 7)   # ceil(prizes / 0.7)

    def _turns_to_deckout(self, extra_spend=0):
        """Turns we survive drawing 1/turn, counting returnable cards, after an optional spend."""
        return self._effective_deck() - extra_spend

    def _optional_spend_ok(self, cost=1, makes_lethal=False, secures_backup=False):
        """Replaces v8's fixed max(8, prizes+3) floor. An optional draw/search is allowed when it
        (1) makes THIS turn's KO, (2) secures the first missing backup Alakazam, or (3) still
        leaves us winning before we deck out (turns_to_deckout > turns_to_win)."""
        if makes_lethal and self.me.deckCount - cost >= 2:
            return True
        if secures_backup and self.me.deckCount - cost >= self._turns_to_win():
            return True
        return self._turns_to_deckout(cost) > self._turns_to_win()

    def _lethal_after_draw(self, drawn=3):
        opp = self.opponent.active[0] if self.opponent.active else None
        return (opp is not None and self._have_attacker()
                and not self._effect_prevented(opp)
                and 20 * (self.me.handCount + drawn) >= opp.hp)

    # ── piece / role predicates ──────────────────────────────────────────────
    def _ready_alakazam_attacker(self):
        return any(p is not None and p.id == C.ALAKAZAM and self.can_attack(p)
                   for p in self.my_board())

    def _have_attacker(self):
        a = self.me.active[0] if self.me.active else None
        return (a is not None and a.id in ALAKAZAM_IDS and self.energy_count(a) >= 1) \
            or self.bench_attacker_ready()

    def _has_backup_line(self):
        return any(p is not None and p.id in ALAKAZAM_LINE for p in self.me.bench)

    def _needs_first_backup(self):
        return self._ready_alakazam_attacker() and not self._has_backup_line()

    def _need_pieces(self):
        return self.field[C.ALAKAZAM] < 1

    def _open_bench(self):
        return sum(1 for p in self.me.bench if p is not None) < getattr(self.me, "benchMax", 5)

    def _psychic_in_hand(self):
        return any(ENERGY_PROVIDES.get(c.id) == EnergyType.PSYCHIC for c in self.me.hand)

    def _energy_starved(self):
        bodies = [p for p in self.my_board() if p is not None]
        has_alakazam = any(p.id in ALAKAZAM_IDS for p in bodies)
        coming = any(p.id == C.KADABRA for p in bodies) and self.hand[C.ALAKAZAM] > 0
        if not (has_alakazam or coming):
            return False
        if any(p.id in ALAKAZAM_IDS and self.can_attack(p) for p in bodies):
            return False
        return not self._psychic_in_hand()

    def _achievable_hand(self):
        extra = 0
        if (self.me.deckCount > 3
                and any(p is not None and p.id == C.DUDUNSPARCE for p in self.me.bench)):
            extra += 3
        if not self.state.supporterPlayed and (self.hand[C.HILDA] or self.hand[C.DAWN]):
            extra += 1
        return self.me.handCount + extra

    def _ko_active_reachable(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        return (opp is not None and self._have_attacker()
                and not self._effect_prevented(opp)
                and 20 * self._achievable_hand() >= opp.hp)

    def _lethal_now(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is None:
            return False
        return self._active_best_dmg(opp) >= max(1, opp.hp)

    def _holds_complete_route(self):
        has_abra = self.field[C.ABRA] > 0 or self.hand[C.ABRA] > 0
        candy_route = has_abra and self.hand[C.RARE_CANDY] > 0 and self.hand[C.ALAKAZAM] > 0
        kadabra_route = has_abra and self.hand[C.KADABRA] > 0 and (
            self.hand[C.ALAKAZAM] > 0 or self.field[C.ALAKAZAM] == 0)
        has_fuel = self._psychic_in_hand() or any(
            p is not None and p.id in ALAKAZAM_LINE and self.can_attack(p) for p in self.my_board())
        return (candy_route or kadabra_route) and has_fuel

    # ── state classification ────────────────────────────────────────────────
    def _classify_state(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        if self._ready_alakazam_attacker():
            if opp is not None and self._effect_prevented(opp) and self._active_best_dmg(opp) <= 0:
                return TurnState.LOCKED
            if len(self.me.prize) <= 1 or self._turns_to_deckout() <= self._turns_to_win() + 1:
                return TurnState.ENDGAME
            return TurnState.PRESSURE
        # No ready Alakazam attacker: are we REBUILDING after losing one, or still setting up?
        if self.discard.get(C.ALAKAZAM, 0) or self.discard.get(C.KADABRA, 0):
            return TurnState.RECOVER
        return TurnState.SETUP

    # ── pre-attack gate ────────────────────────────────────────────────────
    def _hand_delta(self, option):
        """Conservative net hand-size change of an option (for the lethal-preservation gate)."""
        t = option.type
        if t == OptionType.ABILITY:
            card = get_card(self.obs, option.area, option.index, self.my_index)
            if card is not None and card.id == C.DUDUNSPARCE and option.area == AreaType.BENCH:
                return 3
            return 0
        if t in (OptionType.EVOLVE, OptionType.ATTACH, OptionType.ENERGY):
            return -1
        if t == OptionType.PLAY:
            cid = self._play_card_id(option)
            if cid in (C.HILDA, C.DAWN, C.LANA_AID):
                return 1
            if cid == C.NIGHT_STRETCHER and self._night_stretcher_worthwhile():
                return 0
            return -1
        return 0

    def _preserves_attack(self, option):
        """Would executing `option` keep the reserved attack legal and its KO intact?
        Only the hand-consuming case can break Powerful Hand's damage, so that is what we check;
        energy/attacker-Active are unaffected by pre-attack development in this deck."""
        if not self._attack_reserved:
            return True
        delta = self._hand_delta(option)
        if delta >= 0:
            return True
        active = self.me.active[0] if self.me.active else None
        opp = self.opponent.active[0] if self.opponent.active else None
        if active is None or active.id != C.ALAKAZAM or opp is None:
            return True
        # Losing the CURRENT KO to a hand spend is forbidden.
        if self._plan["kos"] and 20 * (self.me.handCount + delta) < opp.hp:
            return False
        return True

    def _improves_plan(self, option):
        """Is `option` one of the four allowed pre-attack actions? (raise the KO/damage, secure
        the next attacker, grow the hand safely, or apply a maintained high-value disruption)."""
        t = option.type
        # Draw/search that raises Powerful Hand or reaches a KO.
        if self._optional_deck_spend(option):
            return True
        # A hand-neutral / hand-positive recovery play (Night Stretcher) that fixes a route.
        if t == OptionType.PLAY:
            cid = self._play_card_id(option)
            if cid == C.NIGHT_STRETCHER and self._night_stretcher_worthwhile():
                return True
            if cid == C.ENHANCED_HAMMER and self._enhanced_hammer_worthwhile():
                return True
            if cid == C.XEROSIC and self._xerosic_pre_attack():
                return True
            if cid == C.BATTLE_CAGE and self._battle_cage_worthwhile():
                return True
            # Secure the first missing backup attacker.
            if self._needs_first_backup() and self._open_bench():
                if cid == C.ABRA or cid == C.BUDDY_POFFIN:
                    return True
        if t == OptionType.EVOLVE:
            # Evolving toward / into the attacker improves the (current or next) plan.
            return self._score_evolve(option) > 0
        if t in (OptionType.ATTACH, OptionType.ENERGY):
            # Fueling a backup keeps the follow-up attacker coming.
            return self._score_attach(option) > 0
        if t == OptionType.ABILITY:
            card = get_card(self.obs, option.area, option.index, self.my_index)
            return card is not None and card.id == C.DUDUNSPARCE
        return False

    # ── MAIN score dispatch ────────────────────────────────────────────────
    def score(self, option):
        # ACTIVATE (Run Away Draw yes/no) lives on the real dispatch path.
        if self.context == SelectContext.ACTIVATE:
            if option.type == OptionType.YES:
                return 0 if not self._activate_draw_ok() else 1
            if option.type == OptionType.NO:
                return 1 if not self._activate_draw_ok() else 0
        raw = super().score(option)
        if self.context != SelectContext.MAIN:
            return raw
        return self._score_main(option, raw)

    def _activate_draw_ok(self):
        # Never draw ourselves toward a deckout when it does not make the current KO, and
        # never empty the board (Run Away Draw shuffles the acting Dudunsparce away).
        board_count = sum(p is not None for p in self.my_board())
        if board_count <= 1:
            return False
        return self._optional_spend_ok(cost=1, makes_lethal=self._lethal_after_draw())

    def _tier(self, tier, score=0):
        local = max(0, min(int(score or 0), TIER_SPAN - 1))
        return TIER_BASE[tier] + local

    def _play_card_id(self, option):
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        return getattr(card, "id", None)

    def _optional_deck_spend(self, option):
        """A draw/search action that both spends deck and improves the current KO / next
        attacker while keeping us on the safe side of the deckout race."""
        if option.type == OptionType.ABILITY:
            card = get_card(self.obs, option.area, option.index, self.my_index)
            if card is None or card.id != C.DUDUNSPARCE or option.area != AreaType.BENCH:
                return False
            return self._optional_spend_ok(cost=1, makes_lethal=self._lethal_after_draw(),
                                           secures_backup=self._needs_first_backup())
        if option.type != OptionType.PLAY:
            return False
        cid = self._play_card_id(option)
        if cid not in (C.BUDDY_POFFIN, C.POKE_PAD, C.HILDA, C.DAWN, C.LILLIE, C.HYPER_AROMA):
            return False
        opp = self.opponent.active[0] if self.opponent.active else None
        makes_lethal = (opp is not None and self._ko_active_reachable()
                        and 20 * self.me.handCount < opp.hp)
        return self._optional_spend_ok(cost=2, makes_lethal=makes_lethal,
                                       secures_backup=self._needs_first_backup())

    def _score_main(self, option, raw):
        if raw < 0:
            return self._tier(Tier.BLOCK)

        state = self._state
        t = option.type

        # 1) The reserved attack and game-winning KO.
        if t == OptionType.ATTACK:
            dmg = self._attack_damage_for_option(option)
            if dmg <= 0:
                return self._tier(Tier.BLOCK)
            opp = self.opponent.active[0] if self.opponent.active else None
            if opp is not None and dmg >= opp.hp and prize_count(opp) >= len(self.me.prize):
                return self._tier(Tier.WIN_OR_SURVIVE, raw)
            active = self.me.active[0] if self.me.active else None
            is_alakazam = active is not None and active.id == C.ALAKAZAM
            return self._tier(Tier.ATTACK if is_alakazam else Tier.DISRUPT, raw)

        # 2) END is forbidden while a meaningful attack is reserved.
        if t == OptionType.END:
            if self._attack_reserved and state in (TurnState.PRESSURE, TurnState.ENDGAME, TurnState.RECOVER):
                return self._tier(Tier.BLOCK)
            return self._tier(Tier.END, raw)

        # 3) A pre-attack action that breaks the reserved attack is forbidden.
        if self._attack_reserved and not self._preserves_attack(option):
            return self._tier(Tier.BLOCK)

        # 4) LOCKED: only the unlock, then a real attack; no aimless draw/attack.
        if state == TurnState.LOCKED:
            return self._score_locked(option, raw)

        # 5) While an attack is reserved, gated pre-attack actions sit ABOVE the attack.
        if self._attack_reserved and state in (TurnState.PRESSURE, TurnState.ENDGAME):
            if self._improves_plan(option):
                return self._tier(Tier.PRE_ATTACK, raw)
            # Everything else waits below the attack (so we attack instead of stalling).
            return self._tier(self._action_tier(option, state), min(raw, TIER_SPAN - 1))

        # 6) SETUP / RECOVER: build toward an attacker by role priority.
        return self._tier(self._action_tier(option, state), raw)

    def _score_locked(self, option, raw):
        cid = self._play_card_id(option) if option.type == OptionType.PLAY else None
        if cid == C.ENHANCED_HAMMER and self._enhanced_hammer_worthwhile():
            return self._tier(Tier.WIN_OR_SURVIVE, raw)
        if cid == C.BATTLE_CAGE and self._battle_cage_worthwhile():
            return self._tier(Tier.DISRUPT, raw)
        # Do not spend the deck or place 0-damage while locked.
        if self._optional_deck_spend(option) or (
                option.type == OptionType.ABILITY
                and getattr(get_card(self.obs, option.area, option.index, self.my_index), "id", None)
                == C.DUDUNSPARCE):
            return self._tier(Tier.BLOCK)
        # Building the next attacker line while locked is still fine.
        if option.type in (OptionType.EVOLVE, OptionType.ATTACH, OptionType.ENERGY):
            return self._tier(self._action_tier(option, TurnState.LOCKED), raw)
        if option.type == OptionType.PLAY:
            data = card_table.get(cid)
            if data is not None and data.cardType == CardType.POKEMON:
                return self._tier(self._action_tier(option, TurnState.LOCKED), raw)
        return self._tier(Tier.END if option.type == OptionType.END else Tier.DISRUPT, raw)

    def _action_tier(self, option, state):
        t = option.type
        if t == OptionType.END:
            return Tier.END
        if t == OptionType.RETREAT:
            return Tier.PRE_ATTACK if self.bench_attacker_ready() else Tier.BLOCK
        if t == OptionType.ATTACK:
            active = self.me.active[0] if self.me.active else None
            return Tier.ATTACK if active is not None and active.id == C.ALAKAZAM else Tier.DISRUPT
        if t == OptionType.EVOLVE:
            return Tier.BUILD_ATTACKER if not self._ready_alakazam_attacker() else Tier.BUILD_BACKUP
        if t in (OptionType.ATTACH, OptionType.ENERGY):
            return Tier.BUILD_ATTACKER if not self._ready_alakazam_attacker() else Tier.BUILD_BACKUP
        if t == OptionType.ABILITY:
            card = get_card(self.obs, option.area, option.index, self.my_index)
            if card is not None and card.id == C.DUDUNSPARCE:
                return Tier.SEARCH
            return Tier.DISRUPT
        if t == OptionType.PLAY:
            cid = self._play_card_id(option)
            if cid == C.RARE_CANDY:
                return Tier.BUILD_ATTACKER
            if cid in (C.NIGHT_STRETCHER, C.SACRED_ASH) and state == TurnState.RECOVER:
                return Tier.BUILD_ATTACKER
            if cid in (C.BUDDY_POFFIN, C.POKE_PAD, C.HILDA, C.DAWN, C.LILLIE, C.HYPER_AROMA):
                return Tier.SEARCH
            if cid == C.NIGHT_STRETCHER and self._night_stretcher_worthwhile():
                return Tier.BUILD_ATTACKER if not self._ready_alakazam_attacker() else Tier.BUILD_BACKUP
            if cid in (C.ENHANCED_HAMMER, C.XEROSIC, C.BATTLE_CAGE, C.SACRED_ASH, C.LANA_AID):
                return Tier.DISRUPT
            data = card_table.get(cid)
            if data is not None and data.cardType == CardType.POKEMON:
                return Tier.BUILD_ATTACKER if cid == C.ABRA else Tier.BUILD_BACKUP
            return Tier.BUILD_BACKUP
        return Tier.DISRUPT

    # ── card-specific worth predicates ────────────────────────────────────────
    def _enhanced_hammer_worthwhile(self):
        """Use it only to strip an effect-prevention energy that is locking Powerful Hand, when a
        ready Alakazam can then deal meaningful damage THIS turn. If the attack still can't
        happen (no ready attacker / hand too small), it is not used (no wasted Hammer)."""
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is None or not self._opp_active_has_prevent_energy():
            return False
        if not self._ready_alakazam_attacker():
            return False
        return 20 * max(0, self.me.handCount - 1) > 0

    def _xerosic_pre_attack(self):
        """Board-derived mirror disruption allowed BEFORE the attack: the opponent runs an
        Alakazam line and holds >=6 cards (its hand IS its Powerful Hand fuel). Never named by
        matchup. A non-mirror large hand does NOT preempt our attack — it is scored below it."""
        if self.state.supporterPlayed:
            return False
        opp_hand = getattr(self.opponent, "handCount", 0) or 0
        opp_board = [p for p in (self.opponent.active + self.opponent.bench) if p is not None]
        mirror = any(p.id in ALAKAZAM_LINE for p in opp_board)
        return mirror and opp_hand >= 6

    def _battle_cage_worthwhile(self):
        if self.state.stadiumPlayed or self.stadium_id == C.BATTLE_CAGE:
            return False
        if self.stadium_id and self.stadium_id != C.BATTLE_CAGE:
            return True   # overwrite a (potentially hostile) opposing stadium
        has_bench = any(p is not None for p in self.me.bench)
        return has_bench and self.opponent_threatens_bench()

    def _night_stretcher_target_score(self, cid):
        line_in_play = self.field[C.ABRA] + self.field[C.KADABRA] + self.field[C.ALAKAZAM]
        line_in_hand = self.hand[C.ABRA] + self.hand[C.KADABRA] + self.hand[C.ALAKAZAM]
        engine_in_play = self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE]
        if cid == C.PSYCHIC_ENERGY:
            return 1100 if self._energy_starved() else 180
        if cid == C.ALAKAZAM:
            if self.field[C.KADABRA] > 0:
                return 1050
            if self.field[C.ABRA] > 0 and self.hand[C.RARE_CANDY] > 0:
                return 1000
            if self.hand[C.KADABRA] > 0 and (self.field[C.ABRA] or self.hand[C.ABRA]):
                return 750
            return 240
        if cid == C.KADABRA:
            return 980 if (self.field[C.ABRA] > 0 or self.hand[C.ABRA] > 0) else 280
        if cid == C.ABRA:
            if self._needs_first_backup() and self._open_bench():
                return 1150
            if line_in_play + line_in_hand == 0 and self._open_bench():
                return 900
            return 360
        if cid == C.DUDUNSPARCE:
            return 700 if (self.field[C.DUNSPARCE] > 0 and self.field[C.DUDUNSPARCE] == 0) else 260
        if cid == C.DUNSPARCE:
            return 650 if (engine_in_play == 0 and self._open_bench()) else 220
        return -1

    def _night_stretcher_worthwhile(self):
        return max((self._night_stretcher_target_score(cid)
                    for cid in RECOVERABLE if self.discard.get(cid, 0)), default=-1) >= 600

    # ── abilities ─────────────────────────────────────────────────────────────
    def _score_ability(self, o):
        card = get_card(self.obs, o.area, o.index, self.my_index)
        if card is None:
            return 0
        if card.id == C.DUDUNSPARCE:
            board_count = sum(p is not None for p in self.my_board())
            # Hard invariant: never shuffle away the last Pokémon in play.
            if o.area != AreaType.BENCH and board_count <= 1:
                _DIAG["dudun_last_active_blocked"] += 1
                return -1
            if o.area != AreaType.BENCH:
                # ACTIVE Run Away Draw = cycle a weak Active away and promote a ready benched
                # attacker to swing this turn (repositioning, not filtering) — allowed whenever
                # a bench attacker is ready and the board survives.
                if self.bench_attacker_ready():
                    return 14000
                return -1
            # BENCHED copy = the draw engine. Draw when it is safe by the dynamic model AND it
            # actually helps (bigger hand => bigger Powerful Hand, or digs to the next attacker).
            if not self._optional_spend_ok(cost=1, makes_lethal=self._lethal_after_draw(),
                                           secures_backup=self._needs_first_backup()):
                return -1
            return 15000
        return 9000

    # ── play ─────────────────────────────────────────────────────────────────
    def score_play(self, o):
        card = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
        if card is None:
            return 0
        d = card_table.get(card.id)
        if d is None:
            return 0
        if d.cardType == CardType.POKEMON:
            return self._score_play_poke(card)
        base = self._score_play_trainer(card)
        # Spend Items before the one Supporter so we keep the Supporter slot flexible.
        if (base > 0 and d.cardType == CardType.ITEM and not self.state.supporterPlayed
                and any(self.hand[s] for s in DRAW_SUPPORTERS)):
            base += 900
        return base

    def _score_play_poke(self, card):
        cid = card.id
        n = self.field[cid]
        if cid == C.ABRA:
            line = self.field[C.ABRA] + self.field[C.KADABRA] + self.field[C.ALAKAZAM]
            if line >= 3:
                return 1500      # role filled; more bodies just shrink Powerful Hand
            return 20000 - 250 * n
        if cid == C.DUNSPARCE:
            if self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE] >= 2:
                return 1200      # cap at two engine bodies
            return 18500 - 250 * n
        return 14000 - 200 * n

    def _score_play_trainer(self, card):
        cid = card.id
        opp = self.opponent.active[0] if self.opponent.active else None
        draw_for_ko = (opp is not None and self._ko_active_reachable()
                       and 20 * self.me.handCount < opp.hp)
        need_backup = self._needs_first_backup()

        if cid == C.RARE_CANDY:
            if self.field[C.ABRA] >= 1 and self.hand[C.ALAKAZAM] >= 1:
                # Prefer the Kadabra bridge (its Psychic Draw refills the hand) when available;
                # Candy is the skip for when the bridge is missing or we need the attacker NOW.
                if self.hand[C.KADABRA] >= 1 and not need_backup and self.field[C.ALAKAZAM] == 0:
                    return 8000
                return 20500
            return -1

        if cid in (C.HILDA, C.DAWN, C.POKE_PAD, C.BUDDY_POFFIN):
            makes_lethal = draw_for_ko
            if not self._optional_spend_ok(cost=2, makes_lethal=makes_lethal,
                                           secures_backup=need_backup):
                return -1

        if cid == C.HYPER_AROMA:
            if self._item_locked() or not self._optional_spend_ok(cost=3):
                return -1
            if self.field[C.KADABRA] + self.field[C.ALAKAZAM] == 0:
                return 14500
            if self.field[C.DUDUNSPARCE] == 0 and self.field[C.DUNSPARCE] > 0:
                return 11500
            return 5000 if self._need_pieces() else 1200

        if cid == C.LILLIE:
            if self.state.supporterPlayed or self._lethal_now() or self._holds_complete_route():
                return -1
            if self._has_pre_lillie_action():
                return -1
            if not self._optional_spend_ok(cost=1):
                return -1
            target_hand = 8 if len(self.me.prize) == 6 else 6
            net_gain = target_hand - max(0, self.me.handCount - 1)
            missing_attack = not self._have_attacker() or self._energy_starved() or self._need_pieces()
            if self.me.handCount <= 4 and missing_attack and net_gain >= 2:
                return 13200 + net_gain * 100
            disrupted = getattr(self.opponent, "handCount", 0) <= 2 and self.me.handCount <= 5
            if disrupted and missing_attack and net_gain >= 2:
                return 9000 + net_gain * 100
            return -1

        if cid == C.HILDA:
            if self.state.supporterPlayed:
                return -1
            if draw_for_ko:
                return 14000
            return 12500 if self._need_pieces() else 5000

        if cid == C.DAWN:
            if self.state.supporterPlayed:
                return -1
            if draw_for_ko:
                return 13800
            return 12000 if self._need_pieces() else 7500

        if cid == C.BUDDY_POFFIN:
            # ROLE-based, not board-size based: only fetch a Pokémon while a role is missing.
            missing_role = (self._need_pieces() or need_backup
                            or self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE] == 0)
            if not missing_role or not self._open_bench():
                return 600
            return 13000

        if cid == C.POKE_PAD:
            return 8500 if self._need_pieces() else 3500

        if cid == C.XEROSIC:
            if self.state.supporterPlayed:
                return -1
            opp_hand = getattr(self.opponent, "handCount", 0) or 0
            opp_board = [p for p in (self.opponent.active + self.opponent.bench) if p is not None]
            mirror = any(p.id in ALAKAZAM_LINE for p in opp_board)
            if mirror and opp_hand >= 6:
                return 15500
            if opp_hand >= 8:
                return 13000
            if mirror and opp_hand >= 4:
                return 9000
            return 400

        if cid == C.ENHANCED_HAMMER:
            return 16000 if self._enhanced_hammer_worthwhile() else -1

        if cid == C.BATTLE_CAGE:
            if not self._battle_cage_worthwhile():
                return -1
            if self.stadium_id:
                return 12500
            return 6500 if self.opponent_threatens_bench() else 1800

        if cid == C.NIGHT_STRETCHER:
            if not self._night_stretcher_worthwhile():
                return -1
            best = max(self._night_stretcher_target_score(x) for x in RECOVERABLE
                       if self.discard.get(x, 0))
            return 7800 + best

        if cid == C.LANA_AID:
            if self.state.supporterPlayed:
                return -1
            # Only a genuine deckout race justifies Lana (returns cards to the deck).
            racing = self._turns_to_deckout() <= self._turns_to_win() + 2
            return 6000 if racing else 1500

        if cid == C.SACRED_ASH:
            line_in_discard = sum(self.discard.get(x, 0) for x in
                                  (C.ABRA, C.KADABRA, C.ALAKAZAM, C.DUNSPARCE, C.DUDUNSPARCE))
            racing = self._turns_to_deckout() <= self._turns_to_win() + 2
            if racing and line_in_discard >= 3:
                return 12000
            return 6000 if (racing and self.me.discard) else 200

        return 9000

    def _has_pre_lillie_action(self):
        for option in (self.select.option or []):
            if option.type == OptionType.PLAY:
                cid = self._play_card_id(option)
                if cid == C.LILLIE:
                    continue
                if cid == C.BUDDY_POFFIN:
                    missing_role = (self._need_pieces() or self._needs_first_backup()
                                    or self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE] == 0)
                    if missing_role and self._open_bench():
                        return True
                data = card_table.get(cid)
                if data is not None and data.cardType == CardType.POKEMON:
                    if cid in (C.ABRA, C.DUNSPARCE) and self._open_bench():
                        return True
                if cid == C.RARE_CANDY and self.field[C.ABRA] and self.hand[C.ALAKAZAM]:
                    return True
                if cid == C.ENHANCED_HAMMER and self._enhanced_hammer_worthwhile():
                    return True
            elif option.type == OptionType.EVOLVE and self._score_evolve(option) > 0:
                return True
            elif option.type in (OptionType.ATTACH, OptionType.ENERGY) and self._score_attach(option) > 0:
                return True
        return False

    def _item_locked(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        return opp is not None and opp.id in ITEM_LOCK_IDS

    # ── evolve ─────────────────────────────────────────────────────────────────
    def _score_evolve(self, o):
        target = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.my_index)
        if not isinstance(target, Pokemon):
            return 0
        card = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
        cid = card.id if card is not None else None
        if cid == C.ALAKAZAM:
            have = self.field[C.ALAKAZAM]
            if have == 0 or o.inPlayArea == AreaType.ACTIVE:
                return 21000
            return 4000       # a second bench Alakazam only spends a Powerful Hand card
        if cid == C.KADABRA:
            if (self.hand[C.ALAKAZAM] >= 1 or self.me.handCount <= 4
                    or self.field[C.ALAKAZAM] == 0):
                return 20000
            return 6000
        if cid == C.DUDUNSPARCE:
            return 19000
        return 18000

    # ── attach energy ──────────────────────────────────────────────────────────
    def _score_attach(self, o):
        p = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.my_index)
        if not isinstance(p, Pokemon):
            return 0
        src = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
        if not self.should_fuel(p):
            return -1
        if not self.attach_helps(p, src):
            return -1
        if p.id in ALAKAZAM_IDS:
            base = 8000 + (200 if o.inPlayArea == AreaType.ACTIVE else 0)
        elif p.id in (C.ABRA, C.KADABRA):
            base = 1500      # pre-fuel the line; energy carries through evolution
        else:
            return -1
        if src is not None and src.id == C.TELEPATH_ENERGY:
            # Telepath also benches 2 basic-{P} bodies: good early with room, wasteful when full.
            if self._open_bench() and self._optional_spend_ok(cost=2):
                base += 250
            else:
                base -= 150
        return base

    # ── retreat ──────────────────────────────────────────────────────────────
    def _score_retreat(self):
        active = self.me.active[0] if self.me.active else None
        opp = self.opponent.active[0] if self.opponent.active else None
        if active is None or opp is None:
            return -1
        # Only retreat to REACH a meaningful attack: the current Active can't attack and a
        # benched attacker can. Never retreat FROM an attack-capable Alakazam (never "shield up").
        if active.id in ALAKAZAM_IDS and self.can_attack(active):
            return -1
        if active.id not in ALAKAZAM_IDS:
            for p in self.me.bench:
                if p is not None and p.id in ALAKAZAM_IDS and self.energy_count(p) >= 1:
                    return 6000
        return -1

    # ── attack ─────────────────────────────────────────────────────────────────
    def _score_attack(self, o):
        active = self.me.active[0] if self.me.active else None
        opp = self.opponent.active[0] if self.opponent.active else None
        if active is None or opp is None:
            return 800
        aid = o.attackId
        if aid in (ABRA_TELEPORT, DUNSPARCE_TRADE):
            # Switch-attacks end the turn: only worth it to bring up a ready attacker.
            if (active.id not in ALAKAZAM_IDS and active.id != C.KADABRA
                    and self.bench_attacker_ready()):
                return 5000
            return 700
        dmg = self._alakazam_damage(aid, opp)
        if dmg <= 0:
            return 500
        if opp.hp <= dmg and prize_count(opp) >= len(self.me.prize):
            return 90000    # game-winning KO
        if opp.hp <= dmg:
            margin_spent = (active.id != C.ALAKAZAM or 20 * (self.me.handCount - 1) < opp.hp)
            if margin_spent:
                return 30000 + prize_count(opp) * 200
            return 6000 + prize_count(opp) * 300
        return 1000 + min(dmg, 320)

    # ── sub-selects ─────────────────────────────────────────────────────────────
    def _score_card(self, o):
        card = get_card(self.obs, o.area, o.index, o.playerIndex)
        if card is None:
            return 0
        ctx = self.context
        if o.playerIndex == self.op_index and not isinstance(card, Pokemon):
            context_card = getattr(self.select, "contextCard", None)
            data = card_table.get(card.id)
            is_energy_card = data is not None and data.cardType in (
                CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY)
            active_bonus = 500 if getattr(o, "inPlayArea", None) == AreaType.ACTIVE else 0
            if getattr(context_card, "id", None) == C.ENHANCED_HAMMER:
                if card.id in EFFECT_PREVENT_ENERGY:
                    return 2000 + active_bonus
                return 200 if data is not None and data.cardType == CardType.SPECIAL_ENERGY else -1
            if card.id in EFFECT_PREVENT_ENERGY:
                return 2000 + active_bonus
            return 300 + active_bonus if is_energy_card else 50
        if (ctx == SelectContext.TO_HAND
                and getattr(getattr(self.select, "contextCard", None), "id", None) == C.NIGHT_STRETCHER
                and getattr(o, "area", None) == AreaType.DISCARD):
            return self._night_stretcher_target_score(card.id)
        if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            return self._score_active_choice(o, card)
        if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup_active(card)
        if ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            return self._score_to_bench(card)
        if ctx == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if ctx == SelectContext.ATTACH_TO and isinstance(card, Pokemon):
            return self._score_attach_target(card, o.inPlayArea == AreaType.ACTIVE)
        if ctx in (SelectContext.ATTACH_FROM, SelectContext.TO_HAND_ENERGY):
            return 100 if self.is_energy(card.id) else 10
        if ctx in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
                   SelectContext.DISCARD_ENERGY, SelectContext.DISCARD_ENERGY_CARD):
            return self._score_discard(card)
        if ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
            if isinstance(card, Pokemon) and o.playerIndex == self.op_index:
                return 10000 + prize_count(card) * 1000 - getattr(card, "hp", 0)
            return 0
        if ctx in (SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM, SelectContext.TO_PRIZE):
            if getattr(o, "area", None) == AreaType.DISCARD:
                cid = card.id
                if cid in ALAKAZAM_LINE:
                    return 90
                if cid in ENGINE_LINE:
                    return 70
                d = card_table.get(cid)
                if d is not None and d.cardType == CardType.POKEMON:
                    return 30
                return 5
            return self._score_putback(card)
        return 0

    def _score_attach_target(self, p, is_active):
        if not self.should_fuel(p):
            return -1
        if p.id in ALAKAZAM_IDS:
            return 8000 + (200 if is_active else 0)
        if p.id in (C.ABRA, C.KADABRA):
            return 1500
        return -1

    def _score_active_choice(self, o, card):
        if not isinstance(card, Pokemon):
            return 0
        if o.playerIndex == self.op_index:
            return prize_count(card) * 1000 - getattr(card, "hp", 0) // 10
        if o.playerIndex != self.my_index:
            return 0
        score = len(card.energies) * 10
        if card.id in ALAKAZAM_IDS:
            score += 200
        elif card.id == C.KADABRA:
            score += 95
        elif card.id == C.ABRA:
            if self.hand[C.KADABRA] or (self.hand[C.ALAKAZAM] and self.hand[C.RARE_CANDY]):
                score += 110
            else:
                score += 25
        elif card.id == C.DUDUNSPARCE:
            score += 40
        score += getattr(card, "hp", 0) // 30
        return score + 1

    def _score_setup_active(self, card):
        if card is None:
            return 0
        if card.id == C.ABRA:
            return 50
        if card.id == C.DUNSPARCE:
            return 30
        return 5

    def _score_to_bench(self, card):
        if card is None:
            return 0
        d = card_table.get(card.id)
        if d is None or d.cardType != CardType.POKEMON:
            return 0
        cid = card.id
        n = self.field[cid]
        if cid == C.ABRA:
            return 200 - 30 * n
        if cid == C.DUNSPARCE:
            return 140 - 30 * n
        return 100 - 20 * n

    def _score_to_hand(self, card):
        if card is None:
            return 0
        cid = card.id
        cc = getattr(self.select, "contextCard", None)
        if getattr(cc, "id", None) == C.HYPER_AROMA:
            if cid == C.KADABRA:
                return 400 if self.field[C.KADABRA] + self.field[C.ALAKAZAM] == 0 else 250
            if cid == C.DUDUNSPARCE:
                return 360 if self.field[C.DUDUNSPARCE] == 0 else 180
            return 20
        score = 200 - self.hand[cid] * 40
        engine_online = self.field[C.DUDUNSPARCE] >= 1
        if cid == C.DUDUNSPARCE:
            score += 45 if not engine_online else -10
        elif cid == C.DUNSPARCE:
            score += 70 if self.field[C.DUDUNSPARCE] + self.field[C.DUNSPARCE] < 1 else -10
        elif cid == C.ABRA:
            score += 85 if self.field[C.ALAKAZAM] + self.field[C.KADABRA] + self.field[C.ABRA] < 3 else 10
        elif cid == C.KADABRA:
            score += 80
        elif cid == C.ALAKAZAM:
            score += 85 if self.hand[C.ALAKAZAM] == 0 else 40
        elif self.is_energy(cid):
            if self._energy_starved() and ENERGY_PROVIDES.get(cid) == EnergyType.PSYCHIC:
                score += 300
            else:
                score += 30
        return score

    def _score_discard(self, card):
        if card is None:
            return 0
        cid = card.id
        if self.is_energy(cid):
            return 20 if self.hand[cid] >= 3 else -40
        if self.hand[cid] >= 2:
            return 60
        if cid in ALAKAZAM_LINE or cid in ENGINE_LINE:
            return -50 if self.field[cid] == 0 else 5
        if cid in (C.HILDA, C.DAWN) and self.state.supporterPlayed:
            return 30
        return 0

    def _score_putback(self, card):
        if card is None:
            return 0
        cid = card.id
        if self.hand[cid] >= 2:
            return 70
        if cid in ALAKAZAM_LINE or cid in ENGINE_LINE:
            return -40 if self.field[cid] == 0 else 60
        return 10

    # ── whole-set selection (Hyper Aroma: pick a balanced 3-card package) ────────
    def custom_selection(self):
        context_card = getattr(self.select, "contextCard", None)
        if (self.context != SelectContext.TO_HAND
                or getattr(context_card, "id", None) != C.HYPER_AROMA):
            return None
        options = list(self.select.option or [])
        if not options:
            return []
        min_count = max(0, min(self.select.minCount, len(options)))
        max_count = max(min_count, min(self.select.maxCount, len(options)))
        candidates = []
        for idx, option in enumerate(options):
            card = get_card(self.obs, option.area, option.index, option.playerIndex)
            if card is not None:
                candidates.append((idx, card.id))
        if not candidates:
            return list(range(min_count))
        abra_waiting = max(0, self.field[C.ABRA] - self.field[C.KADABRA] - self.hand[C.KADABRA])
        dunsparce_waiting = max(0, self.field[C.DUNSPARCE] - self.field[C.DUDUNSPARCE]
                                - self.hand[C.DUDUNSPARCE])
        desired = {
            C.KADABRA: min(2, abra_waiting if abra_waiting else int(self.field[C.ALAKAZAM] == 0)),
            C.DUDUNSPARCE: min(2, dunsparce_waiting if dunsparce_waiting
                               else int(self.field[C.DUDUNSPARCE] == 0 and self.field[C.DUNSPARCE] > 0)),
        }
        picked = []
        picked_by_id = {C.KADABRA: 0, C.DUDUNSPARCE: 0}
        remaining = list(candidates)
        while remaining and len(picked) < max_count:
            def value(item):
                _, cid = item
                current = picked_by_id.get(cid, 0)
                need = desired.get(cid, 0)
                if current < need:
                    return 1000 - current * 60 + (30 if cid == C.KADABRA else 20)
                return 200 - current * 120 + (20 if cid == C.KADABRA else 10)
            best = max(remaining, key=value)
            remaining.remove(best)
            picked.append(best[0])
            if best[1] in picked_by_id:
                picked_by_id[best[1]] += 1
        if len(picked) < min_count:
            used = set(picked)
            for idx in range(len(options)):
                if idx not in used:
                    picked.append(idx)
                    used.add(idx)
                if len(picked) >= min_count:
                    break
        return picked

    # ── diagnostics recording ────────────────────────────────────────────────
    def choose(self):
        sel = super().choose()
        try:
            self._record(sel)
        except Exception:
            pass
        return sel

    def _record(self, sel):
        if self.context != SelectContext.MAIN or not sel:
            return
        opts = self.select.option or []
        idx = sel[0]
        if not (0 <= idx < len(opts)):
            return
        opt = opts[idx]
        _DIAG["state"][self._state] = _DIAG["state"].get(self._state, 0) + 1
        if self._attack_reserved:
            _DIAG["attack_reserved"] += 1
        t = opt.type
        if t == OptionType.ATTACK:
            _DIAG["attacks"] += 1
            active = self.me.active[0] if self.me.active else None
            if active is not None and active.id == C.ALAKAZAM:
                _DIAG["alakazam_attacks"] += 1
            if opt.attackId == POWERFUL_HAND and self._attack_damage_for_option(opt) <= 0:
                _DIAG["zero_damage_attacks"] += 1
        elif t == OptionType.END:
            if self._attack_reserved and self._has_meaningful_attack_option():
                _DIAG["attackable_ends"] += 1
        elif t == OptionType.RETREAT:
            _DIAG["retreats"] += 1
        elif t == OptionType.ABILITY:
            card = get_card(self.obs, opt.area, opt.index, self.my_index)
            if card is not None and card.id == C.DUDUNSPARCE:
                _DIAG["dudun_abilities"] += 1
        elif self._attack_reserved and self._state in (TurnState.PRESSURE, TurnState.ENDGAME):
            _DIAG["pre_attack_actions"] += 1


_agent = make_agent(AlakazamPolicy, my_deck, _DIAG)


def agent(obs_dict):
    return _agent(obs_dict)
