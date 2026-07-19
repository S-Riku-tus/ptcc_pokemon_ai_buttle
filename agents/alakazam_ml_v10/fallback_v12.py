# alakazam741_v12_top_sync — v11 safety core with top-log tempo and circulation.
#
# The policy keeps v11's attack reservation, legal fallback, PrizeTracker, deck validation,
# last-body Dudunsparce protection and energy-aware backup ETA.  The v12 layer separates the
# opening Abra-body target from completed attack routes, accelerates the first Alakazam with Rare
# Candy, stops optional search after a concrete KO/backup improvement, and gives
# Nighttime Mine, Fezandipiti ex, Genesect and the Dudunsparce loop explicit role gates.
from __future__ import annotations

import os
import re
from collections import Counter

from cg.api import (
    AreaType, CardType, EnergyType, Observation, OptionType, Pokemon, SelectContext,
)
from policy_base import (
    BasePolicy, ATTACK_COST_ENERGIES, EFFECT_PREVENT_ENERGY, ENERGY_PROVIDES,
    GLOBAL_EFFECT_PROTECTORS, attack_table, card_table, get_card, make_agent, new_diag,
    prize_count,
)


# ── Card IDs (胡地小人 / Alakazam + Dudunsparce single-prize) ─────────────────
class C:
    ABRA = 741            # Basic -> Kadabra
    KADABRA = 742         # Stage1 (Psychic Draw +2 on evolve) -> Alakazam
    ALAKAZAM = 743        # Stage2 attacker: Powerful Hand = 20 dmg x cards in hand
    DUNSPARCE = 305       # Basic -> Dudunsparce
    DUDUNSPARCE = 66      # Stage1 draw engine (Run Away Draw: draw 3, then shuffle self+attached into deck)
    FEZANDIPITI_EX = 140   # Basic ex: persistent Bench draw engine after an opposing-turn KO
    SHAYMIN = 343          # Basic bench-protection role Pokémon
    GENESECT = 142         # Basic bench tech: blocks opposing ACE SPEC while a Tool is attached

    PSYCHIC_ENERGY = 5
    ENRICHING_ENERGY = 13  # ACE SPEC: provides {C}; hand attachment draws 4
    TELEPATH_ENERGY = 19  # special, provides {P}; on attach searches 2 basic {P} bodies to bench

    BUDDY_POFFIN = 1086
    POKE_PAD = 1152
    HILDA = 1225          # Supporter: search Evolution + Energy
    DAWN = 1231           # Supporter: search Basic+Stage1+Stage2
    RARE_CANDY = 1079
    BOSS_ORDERS = 1182    # Supporter: gust an opposing Benched Pokémon
    XEROSIC = 1197        # Supporter: opponent discards to 3 cards
    NIGHTTIME_MINE = 1266 # Stadium: Tera attacks cost one additional {C}
    ENHANCED_HAMMER = 1081  # Item: discard a Special Energy from opponent
    NIGHT_STRETCHER = 1097  # Item: discard -> hand (Pokémon or basic Energy)
    MAX_ROD = 1110         # ACE SPEC: up to 5 Pokémon/basic Energy, discard -> hand
    SACRED_ASH = 1129     # returns up to 5 Pokémon from discard to the deck
    LUCKY_HELMET = 1156   # Tool: draw 2 when the Active is damaged by an attack
    LANA_AID = 1184       # Supporter: heal / return cards toward the deck


POWERFUL_HAND = 1072   # Alakazam: place 2 counters (20 dmg) per card in hand, on opp Active
SUPER_PSY_BOLT = 1071  # Kadabra: 30
ABRA_TELEPORT = 1070   # Abra: 10 + switch
DUDUN_LAND_CRUSH = 76  # Dudunsparce: 90 (rare)
DUNSPARCE_TRADE = 423  # Dunsparce: switch (0 cost)
DUNSPARCE_RAM = 424    # Dunsparce: 20
FEZANDIPITI_ATTACK = 183  # 100 damage; may target an opposing Pokémon

ALAKAZAM_IDS = {C.ALAKAZAM}
ATTACKER_IDS = {C.ALAKAZAM, C.KADABRA}
ENERGY_TYPES = {C.PSYCHIC_ENERGY, C.TELEPATH_ENERGY}

ALAKAZAM_LINE = (C.ABRA, C.KADABRA, C.ALAKAZAM)
ENGINE_LINE = (C.DUNSPARCE, C.DUDUNSPARCE)
RECOVERABLE = (C.ABRA, C.KADABRA, C.ALAKAZAM, C.DUNSPARCE, C.DUDUNSPARCE,
               C.FEZANDIPITI_EX, C.GENESECT, C.PSYCHIC_ENERGY)
DRAW_SUPPORTERS = (C.HILDA, C.DAWN, C.XEROSIC)



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


# Per-turn scratch survives the engine's repeated policy rebuilds.  Once Boss has resolved, all
# optional MAIN actions are suppressed so the promised same-turn attack cannot be deferred.
_TURN_STATE = {
    "turn": None,
    "boss_committed": False,
    "last_opp_prizes": None,
    "ko_last_opponent_turn": False,
}


def _turn_sync(turn, opponent_prizes=None):
    if _TURN_STATE["turn"] != turn:
        previous_turn = _TURN_STATE["turn"]
        previous_prizes = _TURN_STATE["last_opp_prizes"]
        new_game = previous_turn is not None and turn is not None and turn < previous_turn
        _TURN_STATE["turn"] = turn
        _TURN_STATE["boss_committed"] = False
        _TURN_STATE["ko_last_opponent_turn"] = bool(
            not new_game and previous_prizes is not None and opponent_prizes is not None
            and opponent_prizes < previous_prizes
        )
        if opponent_prizes is not None:
            _TURN_STATE["last_opp_prizes"] = opponent_prizes


def _turn_boss_mark(turn):
    _turn_sync(turn)
    _TURN_STATE["boss_committed"] = True


def _turn_boss_committed(turn):
    _turn_sync(turn)
    return _TURN_STATE["boss_committed"]


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
        "fezandipiti_attacks": 0,    # fallback attacks chosen while Fezandipiti ex is Active
        "fezandipiti_abilities": 0,  # conditional draw ability uses
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

# Engine card-data versions expose the ACE flag under different attribute names, so keep the
# exact submitted ACE id as a schema-independent guard.
KNOWN_ACE_SPEC_IDS = {C.MAX_ROD}

def _is_ace_spec(card_id):
    data = card_table.get(card_id)
    if card_id in KNOWN_ACE_SPEC_IDS:
        return True
    if data is None:
        return False
    return bool(
        getattr(data, "aceSpec", False)
        or getattr(data, "isAceSpec", False)
        or getattr(data, "isAce", False)
    )

def _validate_deck(deck):
    if len(deck) != 60:
        raise ValueError(f"deck.csv must have 60 ids, got {len(deck)}")
    counts = Counter(deck)
    unknown = sorted(cid for cid in counts if cid not in card_table)
    if unknown:
        raise ValueError(f"deck.csv contains unknown card ids: {unknown}")
    over_limit = []
    for cid, count in counts.items():
        data = card_table[cid]
        if data.cardType != CardType.BASIC_ENERGY and count > 4:
            over_limit.append((cid, count))
    if over_limit:
        raise ValueError(f"deck.csv exceeds the four-copy limit: {over_limit}")
    ace_cards = [cid for cid in deck if _is_ace_spec(cid)]
    if len(ace_cards) > 1:
        raise ValueError(f"deck.csv contains multiple ACE SPEC cards: {ace_cards}")

_validate_deck(my_deck)


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
        self._turn = getattr(self.state, "turn", None)
        _turn_sync(self._turn, len(self.opponent.prize))
        try:
            self._plan = self._compute_attack_plan()
            self._attack_reserved = self._plan["damage"] > 0
            self._state = self._classify_state()
        except Exception:
            pass

    def go_first(self) -> bool:
        # Setup deck: the top pilots take turn 1 to build toward a turn-2 attack.
        return True

    def provided_by(self, src, target):
        """Enriching Energy always provides exactly one {C}; its draw-4 is resolved by the
        engine on hand attachment and is represented separately by `_hand_delta`."""
        if src is not None and src.id == C.ENRICHING_ENERGY:
            return [EnergyType.COLORLESS]
        return super().provided_by(src, target)

    # ── opening / phase helpers ────────────────────────────────────────────
    def _opening_window(self):
        # Global turns 1 and 2 are the two players' first turns in the official engine.
        return (getattr(self.state, "turn", 0) or 0) <= 2

    def _abra_body_count(self):
        # Evolved bodies still represent an Abra that was successfully established.
        return self.field[C.ABRA] + self.field[C.KADABRA] + self.field[C.ALAKAZAM]

    def _desired_abra_bodies(self):
        # This is deliberately independent of route completion.  Three opening Abra bodies do
        # not imply three held Alakazam cards, three bridges, or three attack-ready routes.
        # Keep replacing bodies while the prize race is still long; dropping to two after turn 4
        # recreated the board-collapse failure as soon as the first attacker was KO'd.
        return 3 if self._early_game() or len(self.me.prize) >= 3 else 2

    def _needs_dunsparce_body(self):
        return (self._open_bench() and (
            self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE] < 1
            or self._engine_handoff_needed()
        ))

    def _engine_handoff_needed(self):
        """Place a fresh Dunsparce before cycling the only Dudunsparce when both actions are
        currently legal.  This keeps a draw-engine body and a board survivor in play."""
        if (not self._open_bench() or self.field[C.DUNSPARCE] > 0
                or self.field[C.DUDUNSPARCE] != 1):
            return False
        if self._stop_optional_draw():
            return False
        return any(
            option.type == OptionType.ABILITY
            and getattr(get_card(self.obs, option.area, option.index, self.my_index), "id", None)
            == C.DUDUNSPARCE
            for option in (self.select.option or [])
        )

    def _engine_cycle_missing(self):
        """A Run Away Draw cycle returns the whole stack to deck.  Re-establish one Dunsparce,
        but never reserve two bench slots unconditionally."""
        return self._needs_dunsparce_body()

    def _has_dunsparce_play_option(self):
        for option in (self.select.option or []):
            if option.type == OptionType.PLAY and self._play_card_id(option) == C.DUNSPARCE:
                return True
        return False

    def _needed_hand_for_active_ko(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        active = self.me.active[0] if self.me.active else None
        if (opp is None or active is None or active.id != C.ALAKAZAM
                or self._effect_prevented(opp)):
            return None
        return max(1, (max(0, getattr(opp, "hp", 0)) + 19) // 20)

    def _draw_needed_for_ko(self):
        needed = self._needed_hand_for_active_ko()
        return needed is not None and self.me.handCount < needed

    def _stop_optional_draw(self):
        """Hard tempo stop: an existing Active KO plus eta<=1 backup is better than more cards.
        The 18-card cap is an additional guard against the v11 overdraw pattern."""
        if not self._attack_reserved or not self._plan.get("kos"):
            return False
        return self._backup_eta() <= 1 or self.me.handCount >= 18

    def _enriching_attached(self, pokemon):
        return any(getattr(e, "id", None) == C.ENRICHING_ENERGY
                   for e in (getattr(pokemon, "energyCards", None) or []))

    def _primary_psychic_attach_available(self):
        """A Dunsparce draw attachment must never consume the one attachment window that makes
        the first/current Alakazam attack.  Detect the concrete offered Psychic attachment."""
        if self._ready_alakazam_attacker():
            return False
        for option in (self.select.option or []):
            if option.type not in (OptionType.ATTACH, OptionType.ENERGY):
                continue
            source = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
            target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
            if (source is not None and target is not None and target.id in ALAKAZAM_LINE
                    and ENERGY_PROVIDES.get(source.id) == EnergyType.PSYCHIC):
                return True
        return False

    def _attachment_enables_active_alakazam(self, option):
        if option.type not in (OptionType.ATTACH, OptionType.ENERGY):
            return False
        target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
        source = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        return (target is not None and source is not None
                and option.inPlayArea == AreaType.ACTIVE
                and target.id == C.ALAKAZAM and not self.can_attack(target)
                and ENERGY_PROVIDES.get(source.id) == EnergyType.PSYCHIC
                and self.attach_helps(target, source))

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
        elif attack_id == FEZANDIPITI_ATTACK:
            dmg = 100
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
        if active.id == C.FEZANDIPITI_EX:
            return self._alakazam_damage(FEZANDIPITI_ATTACK, target)
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

    @staticmethod
    def _prize_turn_budget(prizes):
        """Conservative own-turn budget for a remaining prize count."""
        prizes = max(1, int(prizes))
        return max(1, (prizes * 10 + 6) // 7)   # ceil(prizes / 0.7)

    def _turns_to_win_conservative(self):
        """Prize clock without crediting a hypothetical same-turn KO.

        Draw/search safety predicates call this version to avoid a dependency
        cycle through reachable-hand estimation.
        """
        return self._prize_turn_budget(len(self.me.prize))

    def _turns_to_win(self):
        """Our own turns still needed to take the remaining prizes. Top-8 games run ~1.4 own
        turns per prize (attack rate < 100% and a KO is not always available), so a RAW prize
        count under-reserves the deck and decks us out; scale it up, then credit a reachable
        multi-prize KO taken this turn."""
        prizes = len(self.me.prize)
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is not None and self._ko_active_reachable():
            prizes -= max(0, prize_count(opp) - 1)
        return self._prize_turn_budget(prizes)

    def _turns_to_deckout(self, extra_spend=0):
        """Turns we survive drawing 1/turn, counting returnable cards, after an optional spend."""
        return self._effective_deck() - extra_spend

    def _optional_spend_ok(self, cost=1, makes_lethal=False, secures_backup=False):
        """Replaces v8's fixed max(8, prizes+3) floor. An optional draw/search is allowed when it
        (1) makes THIS turn's KO, (2) secures the first missing backup Alakazam, or (3) still
        leaves us winning before we deck out (turns_to_deckout > turns_to_win)."""
        if makes_lethal and self.me.deckCount - cost >= 2:
            return True
        if secures_backup and self.me.deckCount - cost > self._turns_to_win():
            return True
        return self._turns_to_deckout(cost) > self._turns_to_win()

    def _lethal_after_draw(self, drawn=3):
        opp = self.opponent.active[0] if self.opponent.active else None
        return (opp is not None and self._ready_alakazam_attacker()
                and not self._effect_prevented(opp)
                and 20 * (self.me.handCount + drawn) >= opp.hp)

    # ── piece / role predicates ──────────────────────────────────────────────
    def _ready_alakazam_attacker(self):
        return any(p is not None and p.id == C.ALAKAZAM and self.can_attack(p)
                   for p in self.my_board())

    def _backup_eta(self, recovered_cid=None, additions=None):
        """Turns until a FOLLOW-UP attacker can attack.

        ``recovered_cid`` remains for Night Stretcher compatibility. ``additions`` is a
        hypothetical hand Counter used by search-card evaluation. Only ETA <= 1 counts as a
        secured follow-up; merely adding another Basic is not treated as a completed backup.
        """
        bench = [p for p in self.me.bench if p is not None]
        if any(p.id in ALAKAZAM_IDS and self.can_attack(p) for p in bench):
            return 0

        extra = Counter(additions or {})
        if recovered_cid is not None:
            extra[recovered_cid] += 1

        def h(cid):
            return self.hand[cid] + extra[cid]

        psychic_available = self._psychic_in_hand() or any(
            extra[cid] > 0 and ENERGY_PROVIDES.get(cid) == EnergyType.PSYCHIC
            for cid in ENERGY_TYPES
        )
        can_energy = (not getattr(self.state, "energyAttached", False)) and psychic_available

        def fueled(pokemon):
            return self.energy_count(pokemon) >= 1

        # One legal enabling step from a body already on the bench.
        for pokemon in bench:
            if pokemon.id in ALAKAZAM_IDS and can_energy:
                return 1
        for pokemon in bench:
            if (pokemon.id == C.KADABRA and h(C.ALAKAZAM) > 0
                    and (fueled(pokemon) or can_energy)):
                return 1
        for pokemon in bench:
            if (pokemon.id == C.ABRA and h(C.RARE_CANDY) > 0 and h(C.ALAKAZAM) > 0
                    and (fueled(pokemon) or can_energy)):
                return 1

        # A body exists, but more than one enabling step or a future Energy is still needed.
        if any(p.id in (C.KADABRA, C.ALAKAZAM) for p in bench):
            return 2
        if any(p.id == C.ABRA for p in bench):
            if h(C.KADABRA) > 0 or (h(C.RARE_CANDY) > 0 and h(C.ALAKAZAM) > 0):
                return 2
            return 3
        if self._open_bench() and h(C.ABRA) > 0:
            return 3
        return 99

    def _needs_first_backup(self):
        """We have a ready attacker but no sufficient (eta<=1) follow-up yet."""
        return self._ready_alakazam_attacker() and self._backup_eta() > 1

    def _backup_energy_short(self):
        """A follow-up evolution line is present but its ONLY missing piece is energy — then
        securing energy outranks extra development / disruption (item 2)."""
        bench = [p for p in self.me.bench if p is not None]
        has_advanced = any(p.id in (C.KADABRA, C.ALAKAZAM) for p in bench)
        abra_ready = any(p.id == C.ABRA for p in bench) and (
            self.hand[C.KADABRA] > 0 or (self.hand[C.RARE_CANDY] > 0 and self.hand[C.ALAKAZAM] > 0))
        if not (has_advanced or abra_ready):
            return False
        fueled = any(p.id in ALAKAZAM_LINE and self.energy_count(p) >= 1 for p in bench)
        return not (self._psychic_in_hand() or fueled)

    @staticmethod
    def _search_deck_cost(cid):
        """Maximum number of cards a search removes from the deck when it resolves."""
        return {
            C.BUDDY_POFFIN: 2,
            C.POKE_PAD: 1,
            C.HILDA: 2,
            C.DAWN: 3,
        }.get(cid, 0)

    @staticmethod
    def _search_hand_delta(cid):
        """Net hand-size change after the complete search effect resolves."""
        return {
            C.BUDDY_POFFIN: -1,  # searched Basics go directly to the Bench
            C.POKE_PAD: 0,       # spend the Item, take one Pokémon to hand
            C.HILDA: 1,          # spend Supporter, take Evolution + Energy
            C.DAWN: 2,           # spend Supporter, take Basic + Stage1 + Stage2
        }.get(cid, -1)

    def _search_secures_backup(self, cid):
        """Whether the actual cards a search can add reduce backup ETA to <= 1.

        The previous implementation passed ``secures_backup=True`` merely because a backup was
        missing. That let any vaguely relevant search bypass deckout safety even when it only
        fetched an Abra that was still two or three turns from attacking.
        """
        if self._backup_eta() <= 1:
            return False

        def available(card_id):
            return self._maybe_in_deck(card_id)

        additions = []
        if cid == C.POKE_PAD:
            additions = [Counter({card_id: 1}) for card_id in self._legal_search_targets(cid)
                         if available(card_id)]
        elif cid == C.HILDA:
            evolutions = [card_id for card_id in (C.KADABRA, C.ALAKAZAM, C.DUDUNSPARCE)
                          if available(card_id)]
            energies = [card_id for card_id in ENERGY_TYPES if available(card_id)]
            additions = [Counter({evo: 1, energy: 1})
                         for evo in evolutions for energy in energies]
        elif cid == C.DAWN:
            basics = [card_id for card_id in (C.ABRA, C.DUNSPARCE, C.GENESECT)
                      if available(card_id)] or [None]
            stage1 = [card_id for card_id in (C.KADABRA, C.DUDUNSPARCE)
                      if available(card_id)] or [None]
            stage2 = [C.ALAKAZAM] if available(C.ALAKAZAM) else [None]
            for basic in basics:
                for middle in stage1:
                    for final in stage2:
                        cards = [card_id for card_id in (basic, middle, final)
                                 if card_id is not None]
                        additions.append(Counter(cards))
        # Buddy-Buddy Poffin creates bodies, but never an eta<=1 attacker by itself.
        return any(self._backup_eta(additions=extra) <= 1 for extra in additions)

    def _search_makes_lethal(self, cid):
        opp = self.opponent.active[0] if self.opponent.active else None
        if (opp is None or not self._ready_alakazam_attacker()
                or self._effect_prevented(opp)):
            return False
        before = 20 * self.me.handCount
        after = 20 * max(0, self.me.handCount + self._search_hand_delta(cid))
        return before < opp.hp <= after

    def _open_bench_slots(self):
        return max(0, getattr(self.me, "benchMax", 5)
                   - sum(1 for pokemon in self.me.bench if pokemon is not None))

    def _essential_bench_reserve(self):
        """Slots that must remain for the first attack line and one draw-engine body."""
        reserve = 0
        if self._abra_body_count() == 0:
            reserve += 1
        if self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE] == 0:
            reserve += 1
        if self._ready_alakazam_attacker():
            bench_line = any(p is not None and p.id in ALAKAZAM_LINE for p in self.me.bench)
            reserve += int(not bench_line)
        return reserve

    def _early_game(self):
        turn = getattr(self.state, "turn", 0) or 0
        # Engine turn numbering differs between harnesses; <=4 safely covers both players'
        # first two turns and is only used to raise development, never to block an attack.
        return turn <= 4

    def _remaining_copy_budget(self, cid, deck_total):
        in_deck = self.copies_in_deck(cid)
        if in_deck is not None:
            return self.hand[cid] + in_deck
        # When prizes are unresolved, count only cards actually held.  Underestimating route depth
        # asks us to develop one extra body; overestimating it recreates the board-wipe bug.
        return self.hand[cid]

    def _independent_route_count(self):
        """How many Alakazam attack lines can still be represented independently.

        The old `Abra+Kadabra+Alakazam >= 3` cap confused bodies with complete routes.  Here each
        future body consumes a distinct Alakazam card and, for an Abra, a distinct bridge
        (Kadabra or Rare Candy).  This prevents Active Alakazam + one incomplete line from being
        treated as a deep board merely because three evolution bodies are visible.
        """
        alakazam_bodies = self.field[C.ALAKAZAM]
        ala_budget = self._remaining_copy_budget(C.ALAKAZAM, 4)
        kadabra_bodies = self.field[C.KADABRA]
        from_kadabra = min(kadabra_bodies, ala_budget)
        ala_budget -= from_kadabra

        bridge_budget = (self._remaining_copy_budget(C.KADABRA, 4)
                         + self._remaining_copy_budget(C.RARE_CANDY, 4))
        from_abra = min(self.field[C.ABRA], max(0, ala_budget), bridge_budget)
        return alakazam_bodies + from_kadabra + from_abra

    def _desired_route_depth(self):
        # Route depth is NOT the opening Abra-body target.  Before the first attacker, one legal
        # route is enough; after it exists, exactly one eta<=1 follow-up is the operational goal.
        return 2 if self.field[C.ALAKAZAM] > 0 else 1

    def _needs_route_depth(self):
        return self._open_bench() and self._independent_route_count() < self._desired_route_depth()

    def _needs_more_abra_body(self):
        return self._open_bench() and self._abra_body_count() < self._desired_abra_bodies()

    def _shaymin_worthwhile(self):
        if self.field[C.SHAYMIN] > 0 or not self._open_bench():
            return False
        damage = self._opponent_ready_bench_damage(allow_next_attachment=True)
        if damage <= 0:
            return False
        open_slots = getattr(self.me, "benchMax", 5) - sum(
            1 for p in self.me.bench if p is not None)
        reserved = max(0, self._desired_abra_bodies() - self._abra_body_count())
        reserved += int(self._needs_dunsparce_body())
        if open_slots <= reserved:
            return False
        vulnerable = []
        for pokemon in self.me.bench:
            if pokemon is None or pokemon.id == C.SHAYMIN:
                continue
            data = card_table.get(pokemon.id)
            rule_box = bool(data and (getattr(data, "ex", False)
                                      or getattr(data, "megaEx", False)))
            if (not rule_box and pokemon.id in ALAKAZAM_LINE + ENGINE_LINE
                    and getattr(pokemon, "hp", 0) <= damage):
                vulnerable.append(pokemon)
        if not vulnerable:
            return False
        # Flower Curtain itself must not erase the reserved Powerful Hand KO.
        opp = self.opponent.active[0] if self.opponent.active else None
        if (self._attack_reserved and self._plan.get("kos") and opp is not None
                and self.me.active and self.me.active[0].id == C.ALAKAZAM
                and 20 * max(0, self.me.handCount - 1) < opp.hp):
            return False
        return True

    def _ko_during_previous_opponent_turn(self):
        return bool(_TURN_STATE.get("ko_last_opponent_turn", False))

    def _fez_draw_needed(self):
        """Use Flip the Script as recovery, not as automatic overdraw after every KO."""
        if not self._ko_during_previous_opponent_turn() or self._stop_optional_draw():
            return False
        needs_cards = (
            self.me.handCount <= 12
            or not self._ready_alakazam_attacker()
            or self._backup_eta() > 1
            or self._draw_needed_for_ko()
        )
        if not needs_cards:
            return False
        turns_to_win = self._turns_to_win_conservative()
        return (self.me.deckCount >= max(3, turns_to_win + 1)
                and self._turns_to_deckout(extra_spend=3) > turns_to_win)

    def _fezandipiti_worthwhile(self):
        """Bench a naturally drawn Fezandipiti early, without consuming the last essential slot."""
        if self.field[C.FEZANDIPITI_EX] > 0 or not self._open_bench():
            return False
        if self._survival_bench_needed():
            return True
        return self._open_bench_slots() > self._essential_bench_reserve()

    def _need_pieces(self):
        return self.field[C.ALAKAZAM] < 1 and not self._holds_complete_route()

    def _open_bench(self):
        return sum(1 for p in self.me.bench if p is not None) < getattr(self.me, "benchMax", 5)

    def _midgame(self):
        return (getattr(self.state, "turn", 0) or 0) >= 3

    def _board_count(self):
        return sum(pokemon is not None for pokemon in self.my_board())

    def _survival_bench_needed(self):
        """Do not leave the only Pokémon in play exposed to an immediate board-out loss."""
        return self._open_bench() and self._board_count() <= 1

    def _visible_count(self, cid):
        cards = list(self.me.hand or []) + list(self.me.discard or [])
        for pokemon in self.my_board():
            if pokemon is None:
                continue
            cards.append(pokemon)
            cards.extend(getattr(pokemon, "preEvolution", None) or [])
            cards.extend(getattr(pokemon, "energyCards", None) or [])
            cards.extend(getattr(pokemon, "tools", None) or [])
        cards.extend(card for card in (self.state.stadium or [])
                     if getattr(card, "playerIndex", self.my_index) == self.my_index)
        return sum(getattr(card, "id", None) == cid for card in cards)

    def _maybe_in_deck(self, cid):
        """Best available target-count test.

        Once PrizeTracker has seen a deck search this is exact. Before then, it still rejects a
        search when all submitted copies are already visible, while treating prize ambiguity as
        unknown instead of inventing certainty.
        """
        exact = self.copies_in_deck(cid)
        if exact is not None:
            return exact > 0
        total = Counter(my_deck).get(cid, 0)
        return total > self._visible_count(cid)

    def _deck_has_any(self, card_ids):
        return any(self._maybe_in_deck(cid) for cid in set(card_ids))

    @staticmethod
    def _has_tool(pokemon):
        return bool(pokemon is not None and (getattr(pokemon, "tools", None) or []))

    def _opponent_ace_seen(self):
        visible = list(self.opponent.discard or [])
        for pokemon in self.opponent.active + self.opponent.bench:
            if pokemon is None:
                continue
            visible.extend(getattr(pokemon, "energyCards", None) or [])
            visible.extend(getattr(pokemon, "tools", None) or [])
        visible.extend(card for card in (self.state.stadium or [])
                       if getattr(card, "playerIndex", None) == self.op_index)
        return any(_is_ace_spec(getattr(card, "id", None)) for card in visible)

    def _genesect_worthwhile(self):
        if (self.field[C.GENESECT] > 0 or not self._open_bench()
                or self._opponent_ace_seen()):
            return False
        # Complete the ACE lock immediately; a Tool merely remaining in deck is not a plan.
        if self.hand[C.LUCKY_HELMET] <= 0:
            return False
        # Fezandipiti is the always-on recovery role requested for this build, so Genesect may not
        # consume its last possible Bench slot when both are naturally held.
        reserve = self._essential_bench_reserve()
        if self.hand[C.FEZANDIPITI_EX] > 0 and self.field[C.FEZANDIPITI_EX] == 0:
            reserve += 1
        return self._open_bench_slots() > reserve

    def _genesect_needs_helmet(self):
        return any(pokemon is not None and pokemon.id == C.GENESECT
                   and not self._has_tool(pokemon)
                   for pokemon in self.my_board()) and not self._opponent_ace_seen()

    def _lucky_helmet_worthwhile(self):
        if self._genesect_needs_helmet():
            return True
        active = self.me.active[0] if self.me.active else None
        return (active is not None and not self._has_tool(active)
                and active.id in (C.ALAKAZAM, C.KADABRA, C.DUDUNSPARCE))

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
        """Upper bound on hand size from actions that are *actually useful and legal now*.

        The old estimate counted Dawn/Hilda merely because they were in hand.  That made the
        deckout clock and KO reachability optimistic even when the relevant search targets were
        already exhausted.  Count at most one Supporter, and only when its search has a concrete
        goal in the current state.
        """
        extra = 0
        if (self.me.deckCount > 3
                and any(p is not None and p.id == C.DUDUNSPARCE for p in self.me.bench)):
            extra += 3
        if not self.state.supporterPlayed:
            if self.hand[C.DAWN] and self._search_card_has_goal(C.DAWN):
                extra += 2
            elif self.hand[C.HILDA] and self._search_card_has_goal(C.HILDA):
                extra += 1
        if (self._fez_draw_needed() and any(
                o.type == OptionType.ABILITY
                and getattr(get_card(self.obs, o.area, o.index, self.my_index), "id", None)
                == C.FEZANDIPITI_EX for o in (self.select.option or []))):
            extra += 3
        return self.me.handCount + extra

    def _ko_active_reachable(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        return (opp is not None and self._ready_alakazam_attacker()
                and not self._effect_prevented(opp)
                and 20 * self._achievable_hand() >= opp.hp)

    def _holds_complete_route(self):
        """Whether the current cards already contain a legal attack route.

        A hand-only Abra is never a complete route. The route must start from a body already in
        play and produce an energy-ready Alakazam no later than next turn.  Fuel is checked on the
        actual target body; energy attached to an unrelated Abra/Kadabra no longer validates an
        unfuelled Alakazam elsewhere.
        """
        board = [p for p in self.my_board() if p is not None]
        psychic = self._psychic_in_hand()

        # Already attacking, or an Alakazam that only needs the held energy.
        for p in board:
            if p.id == C.ALAKAZAM and (self.can_attack(p) or psychic):
                return True

        if self.hand[C.ALAKAZAM] <= 0:
            return False

        # Kadabra in play -> Alakazam this/next turn, with energy on that line or in hand.
        for p in board:
            if p.id == C.KADABRA and (self.energy_count(p) >= 1 or psychic):
                return True

        # Abra in play + Candy + Alakazam is a one-step route next turn at latest.
        for p in board:
            if (p.id == C.ABRA and self.hand[C.RARE_CANDY] > 0
                    and (self.energy_count(p) >= 1 or psychic)):
                return True

        # Abra -> Kadabra now, then Alakazam next turn.  Require an actually offered evolution;
        # merely holding all three stages in hand does not count, and a full bench cannot fake it.
        if self.hand[C.KADABRA] > 0 and psychic:
            for option in (self.select.option or []):
                if option.type != OptionType.EVOLVE:
                    continue
                target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
                evo = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
                if (target is not None and target.id == C.ABRA
                        and evo is not None and evo.id == C.KADABRA):
                    return True
        return False

    # ── state classification ────────────────────────────────────────────────
    def _classify_state(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        active = self.me.active[0] if self.me.active else None
        active_ready = active is not None and active.id in ATTACKER_IDS and self.can_attack(active)
        if active_ready:
            if (active.id == C.ALAKAZAM and opp is not None and self._effect_prevented(opp)
                    and self._active_best_dmg(opp) <= 0):
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
        """Net hand-size change after the complete action/effect resolves.

        Evolution draw was previously ignored, so Kadabra, Alakazam and Rare Candy could be
        falsely blocked as if they reduced Powerful Hand damage. The values below include their
        actual evolution draw effects.
        """
        t = option.type
        if t == OptionType.ABILITY:
            card = get_card(self.obs, option.area, option.index, self.my_index)
            if card is not None and card.id == C.DUDUNSPARCE and option.area == AreaType.BENCH:
                return 3
            if card is not None and card.id == C.FEZANDIPITI_EX:
                return 3
            return 0
        if t == OptionType.EVOLVE:
            evolution = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
            evolution_draw = {
                C.KADABRA: 2,
                C.ALAKAZAM: 3,
            }.get(getattr(evolution, "id", None), 0)
            return evolution_draw - 1
        if t in (OptionType.ATTACH, OptionType.ENERGY):
            src = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
            if src is not None and src.id == C.ENRICHING_ENERGY:
                return 3
            return -1
        if t == OptionType.PLAY:
            cid = self._play_card_id(option)
            if cid in (C.HILDA, C.DAWN, C.BUDDY_POFFIN, C.POKE_PAD):
                return self._search_hand_delta(cid)
            if cid == C.LANA_AID:
                return 1
            if cid == C.MAX_ROD and self._max_rod_worthwhile():
                return max(0, len(self._max_rod_ranked_targets()) - 1)
            if cid == C.RARE_CANDY:
                # Candy + Alakazam leave the hand, then Alakazam draws 3: net +1.
                return 1
            if cid == C.NIGHT_STRETCHER and self._night_stretcher_worthwhile():
                return 0
            return -1
        return 0

    def _optional_role_spend(self, option):
        """Purely optional board/Tool/Stadium spends with no immediate attack conversion."""
        if option.type != OptionType.PLAY:
            return False
        return self._play_card_id(option) in {
            C.FEZANDIPITI_EX,
            C.GENESECT,
            C.LUCKY_HELMET,
            C.NIGHTTIME_MINE,
            C.BUDDY_POFFIN,
        }

    def _preserves_attack(self, option):
        """Keep the reserved attack legal, lethal, and on the same practical KO clock.

        Besides preventing a current KO from disappearing, optional role cards must not turn a
        two-hit Powerful Hand line into a three-hit line.  This retains the requested early
        Fezandipiti policy in ordinary states while restoring v3-like attack conversion when the
        single-card hand spend materially changes the prize race.
        """
        if not self._attack_reserved:
            return True
        delta = self._hand_delta(option)
        if delta >= 0:
            return True
        active = self.me.active[0] if self.me.active else None
        opp = self.opponent.active[0] if self.opponent.active else None
        if active is None or active.id != C.ALAKAZAM or opp is None:
            return True
        current_damage = max(0, self._plan["damage"])
        post_damage = 20 * max(0, self.me.handCount + delta)
        if current_damage > 0 and post_damage <= 0:
            return False
        if self._plan["kos"] and post_damage < opp.hp:
            return False
        if self._optional_role_spend(option) and current_damage > 0 and post_damage > 0:
            current_hits = (opp.hp + current_damage - 1) // current_damage
            post_hits = (opp.hp + post_damage - 1) // post_damage
            if post_hits > current_hits:
                return False
        return True

    def _direct_backup_energy_action(self, option):
        """A deterministic action that fixes the *only* missing backup piece: Psychic energy."""
        if option.type in (OptionType.ATTACH, OptionType.ENERGY):
            target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
            src = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
            return (target is not None and option.inPlayArea == AreaType.BENCH
                    and target.id in ALAKAZAM_LINE and src is not None
                    and ENERGY_PROVIDES.get(src.id) == EnergyType.PSYCHIC
                    and self._score_attach(option) > 0)
        if option.type != OptionType.PLAY:
            return False
        cid = self._play_card_id(option)
        if cid == C.HILDA and not self.state.supporterPlayed:
            return True
        if cid == C.NIGHT_STRETCHER and self.discard.get(C.PSYCHIC_ENERGY, 0):
            return self._night_stretcher_secures_backup(C.PSYCHIC_ENERGY)
        return False

    def _direct_backup_energy_fix_available(self):
        return any(self._direct_backup_energy_action(o) for o in (self.select.option or []))

    def _improves_plan(self, option):
        """Whether an action may precede an already-reserved attack.

        When a backup line is complete except for energy, deterministic energy fixes exclusively
        outrank extra bodies/evolutions/disruption.  If no deterministic fix exists, only a safe
        draw/reset may dig for it; Poké Pad, Dawn and extra Abra cannot masquerade as energy help.
        """
        t = option.type
        if self._backup_energy_short():
            if self._direct_backup_energy_action(option):
                return True
            if self._direct_backup_energy_fix_available():
                return False
            if t == OptionType.ABILITY:
                card = get_card(self.obs, option.area, option.index, self.my_index)
                return (card is not None and card.id in (C.DUDUNSPARCE, C.FEZANDIPITI_EX)
                        and self._optional_deck_spend(option))
            return False

        # Draw/search that raises Powerful Hand or reaches a KO.
        if self._optional_deck_spend(option):
            return True
        # A hand-neutral / hand-positive recovery play (Night Stretcher) that fixes a route.
        if t == OptionType.PLAY:
            cid = self._play_card_id(option)
            data = card_table.get(cid)
            if (data is not None and data.cardType == CardType.POKEMON
                    and self._survival_bench_needed()):
                return True
            if cid == C.MAX_ROD and self._max_rod_worthwhile():
                return True
            if cid == C.LUCKY_HELMET and self._lucky_helmet_worthwhile():
                return True
            if cid == C.NIGHT_STRETCHER and self._night_stretcher_worthwhile():
                return True
            if cid == C.ENHANCED_HAMMER and self._enhanced_hammer_worthwhile():
                return True
            if cid == C.XEROSIC and self._xerosic_pre_attack():
                return True
            if cid == C.BOSS_ORDERS and self._boss_worthwhile():
                return True
            if cid == C.NIGHTTIME_MINE and self._nighttime_mine_worthwhile():
                return True
            # Secure the first backup and, especially when going second, a genuinely independent
            # second route.  Shaymin/Fez are allowed only when their role predicate is satisfied.
            if (self._needs_first_backup()
                    or (self._opening_window() and self._needs_more_abra_body())) and self._open_bench():
                if cid in (C.ABRA, C.BUDDY_POFFIN):
                    return True
            if cid == C.SHAYMIN and self._shaymin_worthwhile():
                return True
            if cid == C.FEZANDIPITI_EX and self._fezandipiti_worthwhile():
                return True
            if cid == C.DUNSPARCE and self._engine_handoff_needed():
                return True
        if t == OptionType.EVOLVE:
            # Evolving toward / into the attacker improves the (current or next) plan.
            return self._score_evolve(option) > 0
        if t in (OptionType.ATTACH, OptionType.ENERGY):
            # Fueling a backup keeps the follow-up attacker coming.
            return self._score_attach(option) > 0
        if t == OptionType.ABILITY:
            card = get_card(self.obs, option.area, option.index, self.my_index)
            return (card is not None and card.id in (C.DUDUNSPARCE, C.FEZANDIPITI_EX)
                    and self._optional_deck_spend(option))
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
        if board_count <= 1 or self._stop_optional_draw():
            return False
        return self._optional_spend_ok(cost=1, makes_lethal=self._lethal_after_draw())

    def _tier(self, tier, score=0):
        local = max(0, min(int(score or 0), TIER_SPAN - 1))
        return TIER_BASE[tier] + local

    def _play_card_id(self, option):
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        return getattr(card, "id", None)

    @staticmethod
    def _legal_search_targets(cid):
        if cid == C.BUDDY_POFFIN:
            return (C.ABRA, C.DUNSPARCE)
        if cid == C.POKE_PAD:
            return (C.ABRA, C.KADABRA, C.ALAKAZAM, C.DUNSPARCE, C.DUDUNSPARCE,
                    C.GENESECT)
        if cid == C.HILDA:
            return (C.KADABRA, C.ALAKAZAM, C.DUDUNSPARCE,
                    C.PSYCHIC_ENERGY, C.TELEPATH_ENERGY)
        if cid == C.DAWN:
            return (C.ABRA, C.DUNSPARCE, C.GENESECT,
                    C.KADABRA, C.DUDUNSPARCE, C.ALAKAZAM)
        return ()

    def _search_card_has_goal(self, cid):
        """Search is role-filling, never a generic 'card is playable' action."""
        if self._stop_optional_draw():
            return False
        missing_abra = self._needs_more_abra_body()
        missing_engine = self._engine_cycle_missing()
        goals = []

        def add(*card_ids):
            goals.extend(card_ids)

        if cid == C.BUDDY_POFFIN:
            if missing_abra:
                add(C.ABRA)
            if missing_engine:
                add(C.DUNSPARCE)
            if self._survival_bench_needed():
                add(C.ABRA, C.DUNSPARCE)
        elif cid == C.POKE_PAD:
            if missing_abra or self._need_pieces() or not self._ready_alakazam_attacker():
                add(C.ABRA, C.KADABRA, C.ALAKAZAM)
            if missing_engine:
                add(C.DUNSPARCE, C.DUDUNSPARCE)
            if self._genesect_worthwhile():
                add(C.GENESECT)
        elif cid == C.HILDA:
            if self._energy_starved() or self._backup_energy_short():
                add(C.PSYCHIC_ENERGY, C.TELEPATH_ENERGY)
            if self._need_pieces() or self._backup_eta() > 1:
                add(C.KADABRA, C.ALAKAZAM, C.DUDUNSPARCE)
        elif cid == C.DAWN:
            if missing_abra or self._need_pieces() or self._backup_eta() > 1:
                add(C.ABRA, C.KADABRA, C.ALAKAZAM)
            if missing_engine:
                add(C.DUNSPARCE, C.DUDUNSPARCE)

        if self._draw_needed_for_ko() and cid != C.BUDDY_POFFIN:
            add(*self._legal_search_targets(cid))

        if not goals or not self._deck_has_any(goals):
            return False
        if self._opening_window():
            return True
        if not self._ready_alakazam_attacker():
            return True
        if self._draw_needed_for_ko():
            return cid != C.BUDDY_POFFIN
        if self._backup_eta() > 1:
            return True
        if missing_engine:
            return cid in (C.POKE_PAD, C.DAWN, C.BUDDY_POFFIN)
        return False

    def _optional_deck_spend(self, option):
        """A draw/search action that improves damage or a concrete next attacker and remains
        safe in the deckout race."""
        if self._stop_optional_draw():
            return False
        if option.type == OptionType.ABILITY:
            card = get_card(self.obs, option.area, option.index, self.my_index)
            if card is None:
                return False
            if card.id == C.DUDUNSPARCE and option.area == AreaType.BENCH:
                goal = (self._draw_needed_for_ko() or not self._ready_alakazam_attacker()
                        or self._backup_eta() > 1)
                if not goal:
                    return False
                return self._optional_spend_ok(
                    cost=1, makes_lethal=self._lethal_after_draw(),
                    secures_backup=False)
            if card.id == C.FEZANDIPITI_EX:
                if not self._fez_draw_needed():
                    return False
                return self._optional_spend_ok(
                    cost=3, makes_lethal=self._lethal_after_draw(),
                    secures_backup=False)
            return False
        if option.type != OptionType.PLAY:
            return False
        cid = self._play_card_id(option)
        if cid not in (C.BUDDY_POFFIN, C.POKE_PAD, C.HILDA, C.DAWN):
            return False
        if not self._search_card_has_goal(cid):
            return False
        return self._optional_spend_ok(
            cost=self._search_deck_cost(cid),
            makes_lethal=self._search_makes_lethal(cid),
            secures_backup=self._search_secures_backup(cid),
        )

    def _score_main(self, option, raw):
        if raw < 0:
            return self._tier(Tier.BLOCK)

        state = self._state
        t = option.type

        if _turn_boss_committed(self._turn) and t != OptionType.ATTACK:
            return self._tier(Tier.BLOCK)

        # 1) The reserved attack and game-winning KO.
        if t == OptionType.ATTACK:
            dmg = self._attack_damage_for_option(option)
            if dmg <= 0:
                return self._tier(Tier.BLOCK)
            opp = self.opponent.active[0] if self.opponent.active else None
            if opp is not None and dmg >= opp.hp and prize_count(opp) >= len(self.me.prize):
                return self._tier(Tier.WIN_OR_SURVIVE, raw)
            active = self.me.active[0] if self.me.active else None
            is_real_attacker = active is not None and active.id in ATTACKER_IDS
            return self._tier(Tier.ATTACK if is_real_attacker else Tier.DISRUPT, raw)

        # 2) END is forbidden while a meaningful attack is reserved.
        if t == OptionType.END:
            if self._attack_reserved and self._has_meaningful_attack_option():
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
        if cid == C.BOSS_ORDERS and self._boss_worthwhile():
            return self._tier(Tier.WIN_OR_SURVIVE, raw)
        # Item 5: never pass a locked turn on Xerosic alone (no attack to accompany it).
        if cid == C.XEROSIC:
            return self._tier(Tier.BLOCK)
        if cid == C.NIGHTTIME_MINE and self._nighttime_mine_worthwhile():
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
            return Tier.ATTACK if active is not None and active.id in ATTACKER_IDS else Tier.DISRUPT
        if t == OptionType.EVOLVE:
            return Tier.BUILD_ATTACKER if not self._ready_alakazam_attacker() else Tier.BUILD_BACKUP
        if t in (OptionType.ATTACH, OptionType.ENERGY):
            if self._attachment_enables_active_alakazam(option):
                return Tier.WIN_OR_SURVIVE
            return Tier.BUILD_ATTACKER if not self._ready_alakazam_attacker() else Tier.BUILD_BACKUP
        if t == OptionType.ABILITY:
            card = get_card(self.obs, option.area, option.index, self.my_index)
            if card is not None and card.id in (C.DUDUNSPARCE, C.FEZANDIPITI_EX):
                return Tier.SEARCH
            return Tier.DISRUPT
        if t == OptionType.PLAY:
            cid = self._play_card_id(option)
            data = card_table.get(cid)
            if (data is not None and data.cardType == CardType.POKEMON
                    and self._survival_bench_needed()):
                return Tier.WIN_OR_SURVIVE
            if cid == C.RARE_CANDY:
                return Tier.BUILD_ATTACKER
            if cid in (C.NIGHT_STRETCHER, C.MAX_ROD, C.SACRED_ASH) and state == TurnState.RECOVER:
                return Tier.BUILD_ATTACKER
            if cid in (C.BUDDY_POFFIN, C.POKE_PAD, C.HILDA, C.DAWN):
                return Tier.SEARCH
            if cid == C.NIGHT_STRETCHER and self._night_stretcher_worthwhile():
                return Tier.BUILD_ATTACKER if not self._ready_alakazam_attacker() else Tier.BUILD_BACKUP
            if cid in (C.ENHANCED_HAMMER, C.XEROSIC, C.BOSS_ORDERS,
                       C.NIGHTTIME_MINE, C.SACRED_ASH, C.LANA_AID):
                return Tier.DISRUPT
            data = card_table.get(cid)
            if data is not None and data.cardType == CardType.POKEMON:
                return Tier.BUILD_ATTACKER if cid == C.ABRA else Tier.BUILD_BACKUP
            return Tier.BUILD_BACKUP
        return Tier.DISRUPT

    # ── card-specific worth predicates ────────────────────────────────────────
    def _candy_accelerates_first_attack(self):
        if self._ready_alakazam_attacker() or self.field[C.ALAKAZAM] > 0:
            return False
        if self.hand[C.RARE_CANDY] <= 0 or self.hand[C.ALAKAZAM] <= 0:
            return False
        psychic_attach = (not getattr(self.state, "energyAttached", False)
                          and self._psychic_in_hand())
        # If an in-play Kadabra already evolves and attacks now, Candy does not save a turn.
        for pokemon in self.my_board():
            if (pokemon is not None and pokemon.id == C.KADABRA
                    and (self.energy_count(pokemon) >= 1 or psychic_attach)):
                return False
        return any(p is not None and p.id == C.ABRA
                   and not getattr(p, "appearThisTurn", False)
                   and (self.energy_count(p) >= 1 or psychic_attach)
                   for p in self.my_board())

    def _dawn_hilda_scores(self):
        """Item 4: choose Dawn (Basic + Stage1 + Stage2 whole line) vs Hilda (one Evolution +
        one Energy) from the board deficit, not a fixed point gap. Returns (dawn, hilda)."""
        abra_present = self.field[C.ABRA] > 0 or self.hand[C.ABRA] > 0
        kadabra_present = self.field[C.KADABRA] > 0 or self.hand[C.KADABRA] > 0
        alakazam_card = self.field[C.ALAKAZAM] > 0 or self.hand[C.ALAKAZAM] > 0
        candy_route = self.hand[C.RARE_CANDY] > 0 and self.field[C.ABRA] > 0
        psychic = self._psychic_in_hand()
        # A usable progression toward an Alakazam already in hand/play?
        evo_route = alakazam_card and (kadabra_present or candy_route or self.field[C.KADABRA] > 0)
        no_line = (not abra_present and self.field[C.KADABRA] == 0 and self.field[C.ALAKAZAM] == 0)

        if no_line:
            return 13500, 8000            # no ケーシィ / no line at all -> Dawn (fetch the basic)
        if self._ready_alakazam_attacker() and self._backup_energy_short():
            return 7000, 13500            # attacker up but backup 超エネルギー missing -> Hilda
        if self._ready_alakazam_attacker() and self._needs_route_depth():
            return (13200, 9000) if psychic else (8500, 13000)
        if abra_present and psychic and not evo_route:
            return 13000, 8500            # ケーシィ + fuel but evolution line missing -> Dawn
        if abra_present and (not psychic or not evo_route):
            return 8000, 12800            # ケーシィ but energy / a single evolution card short -> Hilda
        return 7500, 7000                 # line + fuel basically present: mild dig, slight Dawn lean

    def _pokepad_target_score(self, cid):
        """Item 3: Poké Pad fetches by ROUTE DEFICIT. Fetch the missing evolution stage; never a
        redundant same-name copy when the line is already present; and NEVER fetch Alakazam while
        no Abra exists to build it under."""
        abra_present = self.field[C.ABRA] > 0 or self.hand[C.ABRA] > 0
        kadabra_present = self.field[C.KADABRA] > 0 or self.hand[C.KADABRA] > 0
        alakazam_present = self.field[C.ALAKAZAM] > 0 or self.hand[C.ALAKAZAM] > 0
        candy_route = self.hand[C.RARE_CANDY] > 0 and (self.field[C.ABRA] > 0 or self.hand[C.ABRA] > 0)
        engine = self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE]

        if cid == C.ABRA:
            if not abra_present:
                return 1200                # no ケーシィ anywhere -> top priority
            return 850 if self._needs_more_abra_body() else 120
        if cid == C.KADABRA:
            if abra_present and not kadabra_present and not candy_route:
                return 1000                # ケーシィ present but no ユンゲラー / アメ bridge yet
            return 120
        if cid == C.ALAKAZAM:
            if not abra_present:
                return -1                  # forbid フーディン-first while no ケーシィ exists
            if not alakazam_present and (candy_route or kadabra_present):
                return 1050                # bridge present, フーディン missing -> fetch it
            return 150
        if cid == C.DUNSPARCE:
            return 700 if engine == 0 else 90
        if cid == C.DUDUNSPARCE:
            return 400 if (self.field[C.DUNSPARCE] > 0 and self.field[C.DUDUNSPARCE] == 0) else 80
        if cid == C.SHAYMIN:
            return 1150 if self._shaymin_worthwhile() else 20
        if cid == C.FEZANDIPITI_EX:
            return 950 if self._fezandipiti_worthwhile() else 30
        if cid == C.GENESECT:
            return 900 if self._genesect_worthwhile() else 20
        return 50

    @staticmethod
    def _energy_deficit(attached, cost):
        """Minimum additional single-provision energies needed to pay `cost`."""
        from collections import Counter
        have = Counter(attached)
        colorless = 0
        missing = 0
        for req in cost:
            if req == EnergyType.COLORLESS:
                colorless += 1
            elif have.get(req, 0) > 0:
                have[req] -= 1
            else:
                missing += 1
        missing += max(0, colorless - sum(have.values()))
        return missing

    def _boss_resolving(self):
        return getattr(getattr(self.select, "contextCard", None), "id", None) == C.BOSS_ORDERS

    def _damage_after_boss_spend(self, target):
        active = self.me.active[0] if self.me.active else None
        if active is None or target is None or not self.can_attack(active):
            return 0
        if active.id == C.ALAKAZAM:
            spend = 0 if self._boss_resolving() else 1
            return (0 if self._effect_prevented(target)
                    else 20 * max(0, self.me.handCount - spend))
        if active.id == C.FEZANDIPITI_EX:
            return self._alakazam_damage(FEZANDIPITI_ATTACK, target)
        if active.id == C.KADABRA:
            return self._alakazam_damage(SUPER_PSY_BOLT, target)
        return self._active_best_dmg(target)

    def _boss_target_score(self, target):
        if target is None or (self.state.supporterPlayed and not self._boss_resolving()):
            return -1
        damage = self._damage_after_boss_spend(target)
        hp = getattr(target, "hp", 0)
        if damage <= 0 or damage < hp:
            return -1
        active_opp = self.opponent.active[0] if self.opponent.active else None
        current_damage = self._damage_after_boss_spend(active_opp)
        current_ko = bool(active_opp is not None and current_damage >= active_opp.hp)
        # If the current Active was already a KO, spending Boss may not erase it through the
        # Powerful Hand hand-size loss.
        if self._plan.get("kos") and not current_ko:
            return -1
        prizes = prize_count(target)
        current_prizes = prize_count(active_opp) if current_ko and active_opp is not None else 0
        data = card_table.get(target.id)
        winning = prizes >= len(self.me.prize)
        more_prizes = prizes > current_prizes
        two_prize = prizes >= 2
        protector = target.id in GLOBAL_EFFECT_PROTECTORS or target.id == 675  # Team Rocket's Articuno
        main_attacker = bool(self.can_attack(target) or (data and (data.attacks or [])
                             and (getattr(data, "ex", False) or getattr(data, "stage2", False))))
        scarce_line = bool(data and (getattr(data, "stage1", False)
                                     or getattr(data, "stage2", False)))
        if not (winning or more_prizes or two_prize or protector or main_attacker or scarce_line):
            return -1
        score = 2000 + prizes * 1800 + max(0, current_damage - hp)
        score += 12000 if winning else 0
        score += 4500 if protector else 0
        score += 4000 if target.id == 675 else 0
        score += 2600 if main_attacker else 0
        score += 1600 if scarce_line else 0
        score += 1200 if more_prizes else 0
        return score

    def _boss_worthwhile(self):
        if self.state.supporterPlayed:
            return False
        return any(self._boss_target_score(p) > 0
                   for p in self.opponent.bench if p is not None)

    def _special_energy_intrinsic_value(self, energy_id):
        data = card_table.get(energy_id)
        if data is None or data.cardType != CardType.SPECIAL_ENERGY:
            return 0
        value = 1000
        text = " ".join((getattr(s, "text", "") or "").lower()
                        for s in (getattr(data, "skills", None) or []))
        if energy_id in EFFECT_PREVENT_ENERGY:
            value += 9000
        if getattr(data, "aceSpec", False):
            value += 4500
        if any(word in text for word in ("prevent", "cost", "damage", "attach", "discard")):
            value += 2200
        return value

    def _hammer_target_values(self):
        """Value every attached Special Energy by actual effect and tempo, including a real
        one-energy attack stop.  This replaces v11's fixed two-energy-deficit requirement."""
        values = {}
        active_target = self.opponent.active[0] if self.opponent.active else None
        meaningful_attack = (self._has_meaningful_attack_option()
                             or self._active_best_dmg(active_target) > 0)
        for area, pokemon in ((AreaType.ACTIVE, self.opponent.active[0]
                               if self.opponent.active else None),):
            if pokemon is None:
                continue
            data = card_table.get(pokemon.id)
            attached = list(getattr(pokemon, "energies", None) or [])
            was_payable = self.can_attack(pokemon)
            for energy in (getattr(pokemon, "energyCards", None) or []):
                eid = getattr(energy, "id", None)
                base = self._special_energy_intrinsic_value(eid)
                if base <= 0:
                    continue
                post = list(attached)
                provided = ENERGY_PROVIDES.get(eid, EnergyType.COLORLESS)
                if provided in post:
                    post.remove(provided)
                elif post:
                    post.pop()
                deficits = [self._energy_deficit(post, ATTACK_COST_ENERGIES.get(aid, []))
                            for aid in (data.attacks or [])
                            if aid in ATTACK_COST_ENERGIES] if data is not None else []
                deficit = min(deficits) if deficits else 0
                value = base
                if area == AreaType.ACTIVE and was_payable and deficit >= 1:
                    value += 6000 + 1200 * min(deficit, 2)
                if eid in EFFECT_PREVENT_ENERGY:
                    value += 8000
                if meaningful_attack or eid in EFFECT_PREVENT_ENERGY:
                    values[eid] = max(values.get(eid, 0), value)

        # Removing a special energy from an identified follow-up is useful when we also KO the
        # current Active; it reduces the opponent's next-turn energy without delaying our attack.
        if self._plan.get("kos"):
            for pokemon in self.opponent.bench:
                if pokemon is None:
                    continue
                data = card_table.get(pokemon.id)
                if data is None or not (data.attacks or []):
                    continue
                for energy in (getattr(pokemon, "energyCards", None) or []):
                    eid = getattr(energy, "id", None)
                    base = self._special_energy_intrinsic_value(eid)
                    if base > 0:
                        values[eid] = max(values.get(eid, 0), base + 3000)
        return values

    def _enhanced_hammer_worthwhile(self):
        if not self._hammer_target_values():
            return False
        # The surrounding pre-attack gate separately rejects any Hammer that spends away the
        # current Powerful Hand KO or a meaningful attack.
        return self._has_meaningful_attack_option() or self._opp_active_has_prevent_energy()

    def _xerosic_pre_attack(self):
        """Allow non-mirror Xerosic only when it is attached to a confirmed attack and disrupts
        a visibly hand-dependent evolution/combo turn more than a Boss KO would."""
        if self.state.supporterPlayed or self._state == TurnState.LOCKED:
            return False
        opp_hand = getattr(self.opponent, "handCount", 0) or 0
        opp_board = [p for p in (self.opponent.active + self.opponent.bench) if p is not None]
        mirror = any(p.id in ALAKAZAM_LINE for p in opp_board)
        if opp_hand < 6 or not self._has_meaningful_attack_option():
            return False
        if self._turns_to_deckout() <= self._turns_to_win() + 1:
            return False
        if self._boss_worthwhile():
            return False
        active = self.me.active[0] if self.me.active else None
        opp_active = self.opponent.active[0] if self.opponent.active else None
        if (self._plan.get("kos") and active is not None and active.id == C.ALAKAZAM
                and opp_active is not None
                and 20 * max(0, self.me.handCount - 1) < opp_active.hp):
            return False
        if mirror:
            return True
        hand_dependent = False
        for pokemon in opp_board:
            data = card_table.get(pokemon.id)
            if data is None:
                continue
            if getattr(data, "stage1", False) or getattr(data, "stage2", False):
                hand_dependent = True
            if getattr(data, "basic", False) and not self.can_attack(pokemon):
                hand_dependent = True
        return hand_dependent

    @staticmethod
    def _bench_damage_amount(attack_id):
        """Best-effort amount an attack can put on one Benched Pokémon.

        Card text is the source of truth available to the policy.  Values in a sentence mentioning
        the bench are treated as damage; "damage counters" are converted at 10 damage each.
        Unknown bench attacks use a conservative 10-damage floor rather than triggering Cage for
        every merely potential bench attacker.
        """
        attack = attack_table.get(attack_id)
        text = (getattr(attack, "text", "") or "").replace("’", "'")
        best = 0
        for sentence in re.split(r"[.;]", text):
            low = sentence.lower()
            can_choose_any = ("opponent's pok" in low and "active" not in low)
            if "bench" not in low and not can_choose_any:
                continue
            # Flower Curtain prevents attack damage, not placed damage counters.
            if "damage counter" in low:
                continue
            nums = [int(x) for x in re.findall(r"\d+", sentence)]
            if not nums:
                continue
            amount = max(nums)
            best = max(best, amount)
        return best

    def _opponent_ready_bench_damage(self, allow_next_attachment=False):
        """Immediate bench damage from the opposing Active.  At most one missing energy may be
        treated as next-turn-ready; a merely potential benched attacker is ignored."""
        active = self.opponent.active[0] if self.opponent.active else None
        if active is None:
            return 0
        data = card_table.get(active.id)
        if data is None:
            return 0
        best = 0
        attached = list(getattr(active, "energies", None) or [])
        for aid in (data.attacks or []):
            amount = self._bench_damage_amount(aid)
            if amount <= 0:
                continue
            deficit = self._energy_deficit(attached, ATTACK_COST_ENERGIES.get(aid, []))
            if deficit == 0 or (allow_next_attachment and deficit == 1):
                best = max(best, amount)
        return best

    def _opp_has_tera(self):
        return any(bool(card_table.get(p.id) and getattr(card_table[p.id], "tera", False))
                   for p in self.opponent.active + self.opponent.bench if p is not None)

    def _nighttime_mine_tax_stops_active(self):
        active = self.opponent.active[0] if self.opponent.active else None
        data = card_table.get(active.id) if active is not None else None
        if data is None or not getattr(data, "tera", False):
            return False
        attached = list(getattr(active, "energies", None) or [])
        return any(self.can_pay(attached, ATTACK_COST_ENERGIES.get(aid, []))
                   and not self.can_pay(attached, ATTACK_COST_ENERGIES.get(aid, [])
                                        + [EnergyType.COLORLESS])
                   for aid in (data.attacks or []))

    def _nighttime_mine_worthwhile(self):
        if self.state.stadiumPlayed or self.stadium_id == C.NIGHTTIME_MINE:
            return False
        # The Stadium is worth a hand card only when the opposing Active is a Tera Pokémon whose
        # currently payable attack becomes unpayable from the +{C} tax. A Tera merely sitting on
        # the Bench is not an immediate effect and no longer triggers a speculative replacement.
        if not self._nighttime_mine_tax_stops_active():
            return False
        opp = self.opponent.active[0] if self.opponent.active else None
        if self._plan.get("kos"):
            if opp is not None and prize_count(opp) >= len(self.me.prize):
                return False
            active = self.me.active[0] if self.me.active else None
            if (active is not None and active.id == C.ALAKAZAM and opp is not None
                    and 20 * max(0, self.me.handCount - 1) < opp.hp):
                return False
        return self._turns_to_deckout() > 1

    def _night_stretcher_target_score(self, cid):
        line_in_play = self.field[C.ABRA] + self.field[C.KADABRA] + self.field[C.ALAKAZAM]
        line_in_hand = self.hand[C.ABRA] + self.hand[C.KADABRA] + self.hand[C.ALAKAZAM]
        engine_in_play = self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE]
        if cid == C.PSYCHIC_ENERGY:
            return 1100 if (self._energy_starved() or self._backup_energy_short()) else 180
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
        if cid == C.SHAYMIN:
            return 850 if self._shaymin_worthwhile() else 100
        return -1

    def _night_stretcher_direct_ko(self, cid):
        """Whether recovering exactly `cid` creates an immediate KO by itself."""
        if cid != C.PSYCHIC_ENERGY or getattr(self.state, "energyAttached", False):
            return False
        active = self.me.active[0] if self.me.active else None
        opp = self.opponent.active[0] if self.opponent.active else None
        return (active is not None and active.id == C.ALAKAZAM and not self.can_attack(active)
                and opp is not None and not self._effect_prevented(opp)
                and 20 * self.me.handCount >= opp.hp)

    def _night_stretcher_secures_backup(self, cid):
        before = self._backup_eta()
        after = self._backup_eta(recovered_cid=cid)
        return before > 1 and after <= 1

    def _night_stretcher_allowed_targets(self):
        allowed = []
        before = self._backup_eta()
        for cid in RECOVERABLE:
            if not self.discard.get(cid, 0):
                continue
            if self._night_stretcher_target_score(cid) < 600:
                continue
            after = self._backup_eta(recovered_cid=cid)
            direct = self._night_stretcher_direct_ko(cid)
            secures = self._night_stretcher_secures_backup(cid)
            first_attack_improves = (not self._ready_alakazam_attacker()
                                     and after < before and after <= 2)
            if not (direct or secures or first_attack_improves):
                continue
            allowed.append(cid)
        return allowed

    def _night_stretcher_worthwhile(self):
        return bool(self._night_stretcher_allowed_targets())

    def _max_rod_target_score(self, cid):
        if cid == C.PSYCHIC_ENERGY:
            return 1300 if (self._energy_starved() or self._backup_energy_short()) else 520
        if cid == C.ALAKAZAM:
            return 1250 if self.field[C.KADABRA] or (
                self.field[C.ABRA] and self.hand[C.RARE_CANDY]) else 720
        if cid == C.KADABRA:
            return 1120 if self.field[C.ABRA] else 650
        if cid == C.ABRA:
            return 1050 if self._needs_more_abra_body() and self._open_bench() else 480
        if cid == C.DUDUNSPARCE:
            return 920 if self.field[C.DUNSPARCE] else 540
        if cid == C.DUNSPARCE:
            return 900 if self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE] == 0 else 470
        if cid == C.FEZANDIPITI_EX:
            return 820 if (self.field[C.FEZANDIPITI_EX] == 0
                           and self._ko_during_previous_opponent_turn()
                           and self.me.handCount <= 12) else 250
        if cid == C.GENESECT:
            return 760 if self._genesect_worthwhile() else 300
        return -1

    def _max_rod_ranked_targets(self):
        ranked = []
        for cid in RECOVERABLE:
            score = self._max_rod_target_score(cid)
            for _ in range(self.discard.get(cid, 0)):
                ranked.append((score, cid))
        return [cid for score, cid in sorted(ranked, reverse=True) if score >= 300][:5]

    def _max_rod_worthwhile(self):
        targets = self._max_rod_ranked_targets()
        if not targets:
            return False
        scores = [self._max_rod_target_score(cid) for cid in targets]
        # One immediately critical recovery is enough; otherwise require two useful returns so
        # the ACE is not spent as a one-card, hand-neutral Night Stretcher.
        return max(scores) >= 900 or sum(score >= 450 for score in scores) >= 2

    # ── abilities ─────────────────────────────────────────────────────────────
    def _score_ability(self, o):
        card = get_card(self.obs, o.area, o.index, self.my_index)
        if card is None:
            return 0
        if card.id == C.FEZANDIPITI_EX:
            # The engine guarantees the previous-turn KO condition; the policy additionally
            # requires a real hand deficit, deck room, and no already-secured Active KO.
            if not self._fez_draw_needed() or not self._optional_spend_ok(
                    cost=3, makes_lethal=self._lethal_after_draw(),
                    secures_backup=False):
                return -1
            return 15200
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
            if not self._optional_deck_spend(o):
                return -1
            score = 15000 + (2600 if self._enriching_attached(card) else 0)
            if (self.field[C.DUNSPARCE] == 0 and self.field[C.DUDUNSPARCE] == 1
                    and self._has_dunsparce_play_option()):
                score -= 5000       # keep an engine body in play before cycling when possible
            return score
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
        if self._survival_bench_needed():
            # Survival is mandatory, but the body is not interchangeable.  Prefer the attacker
            # line, then the reusable draw engine, then the persistent recovery role.  Genesect
            # remains a last-resort body unless its Helmet lock can be completed immediately.
            survival_priority = {
                C.ABRA: 34000,
                C.DUNSPARCE: 33000,
                C.FEZANDIPITI_EX: 32000,
                C.GENESECT: 31000 if self.hand[C.LUCKY_HELMET] > 0 else 30000,
            }
            return survival_priority.get(cid, 29500) - 100 * n
        if cid == C.ABRA:
            if not self._needs_more_abra_body():
                return -1
            return 22200 - 350 * n
        if cid == C.DUNSPARCE:
            engine_count = self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE]
            if engine_count >= 1 and not self._engine_handoff_needed():
                return -1
            return (21500 if self._engine_handoff_needed() else 20500) - 250 * n
        if cid == C.SHAYMIN:
            return 19800 if self._shaymin_worthwhile() else -1
        if cid == C.FEZANDIPITI_EX:
            return 21800 if self._fezandipiti_worthwhile() else -1
        if cid == C.GENESECT:
            return 21200 if self._genesect_worthwhile() else -1
        return 14000 - 200 * n

    def _score_play_trainer(self, card):
        cid = card.id
        opp = self.opponent.active[0] if self.opponent.active else None
        need_backup = self._needs_first_backup()

        if cid == C.RARE_CANDY:
            if self.field[C.ABRA] >= 1 and self.hand[C.ALAKAZAM] >= 1:
                # First attacker: speed wins.  Later attackers: Kadabra's evolution draw is the
                # preferred bridge whenever there is no same-turn attack to unlock.
                if self._candy_accelerates_first_attack():
                    return 26000
                if self.hand[C.KADABRA] >= 1 and self.field[C.ALAKAZAM] > 0:
                    return 6500
                return 20500
            return -1

        if cid in (C.HILDA, C.DAWN, C.POKE_PAD, C.BUDDY_POFFIN):
            makes_lethal = self._search_makes_lethal(cid)
            if not makes_lethal and not self._search_card_has_goal(cid):
                return -1
            if not self._optional_spend_ok(
                    cost=self._search_deck_cost(cid),
                    makes_lethal=self._search_makes_lethal(cid),
                    secures_backup=self._search_secures_backup(cid)):
                return -1

        if cid in (C.HILDA, C.DAWN):
            if self.state.supporterPlayed:
                return -1
            if self._search_makes_lethal(cid):
                return 14000 if cid == C.HILDA else 13800
            dawn_score, hilda_score = self._dawn_hilda_scores()
            return hilda_score if cid == C.HILDA else dawn_score

        if cid == C.BUDDY_POFFIN:
            # ROLE-based, not board-size based: only fetch a Pokémon while a role is missing.
            missing_role = (self._need_pieces() or need_backup or self._needs_more_abra_body()
                            or self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE] == 0
                            or self._shaymin_worthwhile())
            if not missing_role or not self._open_bench():
                return 600
            return 13000

        if cid == C.POKE_PAD:
            # ROUTE-based: worth digging when a stage is actually missing (target chosen by
            # _pokepad_target_score); otherwise a low-value spare dig.
            return 9000 if (self._need_pieces() or self._needs_first_backup() or self._needs_route_depth()) else 3500

        if cid == C.XEROSIC:
            if self.state.supporterPlayed:
                return -1
            opp_board = [p for p in (self.opponent.active + self.opponent.bench) if p is not None]
            mirror = any(p.id in ALAKAZAM_LINE for p in opp_board)
            if not self._xerosic_pre_attack():
                return -1
            opp_hand = getattr(self.opponent, "handCount", 0) or 0
            return (16000 if mirror else 11800) + min(2000, max(0, opp_hand - 6) * 300)

        if cid == C.BOSS_ORDERS:
            if not self._boss_worthwhile():
                return -1
            return 19000 + max(
                self._boss_target_score(p) for p in self.opponent.bench if p is not None
            )

        if cid == C.ENHANCED_HAMMER:
            return 16000 if self._enhanced_hammer_worthwhile() else -1

        if cid == C.NIGHTTIME_MINE:
            if not self._nighttime_mine_worthwhile():
                return -1
            return 14500 if self._nighttime_mine_tax_stops_active() else 7600

        if cid == C.NIGHT_STRETCHER:
            targets = self._night_stretcher_allowed_targets()
            if not targets:
                return -1
            best = max(self._night_stretcher_target_score(x) for x in targets)
            return 7800 + best

        if cid == C.MAX_ROD:
            if not self._max_rod_worthwhile():
                return -1
            return 16000 + sum(self._max_rod_target_score(x)
                               for x in self._max_rod_ranked_targets())

        if cid == C.LUCKY_HELMET:
            return 15500 if self._lucky_helmet_worthwhile() else -1

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
            return 18500 if self._needs_route_depth() else 5000
        if cid == C.KADABRA:
            if target.id == C.ABRA and self._candy_accelerates_first_attack():
                return 7000
            if (self.hand[C.ALAKAZAM] >= 1 or self.me.handCount <= 4
                    or self.field[C.ALAKAZAM] == 0):
                return 20000
            return 17500 if self._ready_alakazam_attacker() else 6000
        if cid == C.DUDUNSPARCE:
            return 19000
        return 18000

    # ── attach energy ──────────────────────────────────────────────────────────
    def _score_attach(self, o):
        p = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.my_index)
        if not isinstance(p, Pokemon):
            return 0
        src = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
        if src is None:
            return -1
        enriching = src.id == C.ENRICHING_ENERGY

        if self._attachment_enables_active_alakazam(o):
            return 30000

        if enriching and self._stop_optional_draw():
            return -1

        if enriching and p.id == C.DUNSPARCE:
            # Draw 4 now, then recycle the ACE together with the evolved Dudunsparce stack.
            copies = self.copies_in_deck(C.DUDUNSPARCE)
            evolution_live = self.hand[C.DUDUNSPARCE] > 0 or copies is None or copies > 0
            if evolution_live:
                if self._primary_psychic_attach_available():
                    return 1200
                return 12500 + (500 if o.inPlayArea == AreaType.ACTIVE else 0)

        if p.id == C.FEZANDIPITI_EX:
            # Bench draw support only: never divert Energy from the Alakazam line.
            return -1
        else:
            if (enriching and o.inPlayArea == AreaType.ACTIVE and self.bench_attacker_ready()
                    and not self.can_attack(p)):
                # Emergency retreat fuel for a stranded Basic/line body.
                return 9800
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
        if src.id == C.TELEPATH_ENERGY:
            # Telepath also benches 2 basic-{P} bodies: good early with room, wasteful when full.
            if self._open_bench() and (self._needs_more_abra_body()
                                       or self._needs_dunsparce_body()
                                       or self._shaymin_worthwhile()):
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
        if aid == FEZANDIPITI_ATTACK:
            targets = [p for p in (self.opponent.active + self.opponent.bench) if p is not None]
            winning = any(getattr(p, "hp", 0) <= 100 and prize_count(p) >= len(self.me.prize)
                          for p in targets)
            if winning:
                return 90000
            ko_value = max((prize_count(p) for p in targets if getattr(p, "hp", 0) <= 100),
                           default=0)
            return 26000 + ko_value * 500 if ko_value else 1400
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
        if isinstance(card, Pokemon) and getattr(o, "energyIndex", None) is not None:
            energy_cards = getattr(card, "energyCards", None) or []
            idx = o.energyIndex
            card = energy_cards[idx] if 0 <= idx < len(energy_cards) else None
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
                    return 2400 + active_bonus
                value = self._hammer_target_values().get(card.id, 0)
                if value > 0:
                    return 1900 + value + active_bonus
                return 100 if data is not None and data.cardType == CardType.SPECIAL_ENERGY else -1
            if card.id in EFFECT_PREVENT_ENERGY:
                return 2000 + active_bonus
            return 300 + active_bonus if is_energy_card else 50
        if (ctx == SelectContext.TO_HAND
                and getattr(getattr(self.select, "contextCard", None), "id", None) == C.NIGHT_STRETCHER
                and getattr(o, "area", None) == AreaType.DISCARD):
            if card.id not in self._night_stretcher_allowed_targets():
                return -1
            return self._night_stretcher_target_score(card.id)
        if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            return self._score_active_choice(o, card)
        if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
            return self._score_setup_active(card)
        if ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            return self._score_to_bench(card)
        if ctx == SelectContext.TO_HAND:
            return self._score_to_hand(card)
        if ctx in (getattr(SelectContext, "EVOLVES_FROM", object()),
                   getattr(SelectContext, "EVOLVES_TO", object())):
            if card.id == C.ALAKAZAM:
                return 3000
            if card.id == C.ABRA:
                return 2200 if self._candy_accelerates_first_attack() else 1200
            return 500
        if ctx == SelectContext.ATTACH_TO and isinstance(card, Pokemon):
            return self._score_attach_target(card, o.inPlayArea == AreaType.ACTIVE)
        if ctx in (SelectContext.ATTACH_FROM, SelectContext.TO_HAND_ENERGY):
            return 100 if self.is_energy(card.id) else 10
        if ctx in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
                   SelectContext.DISCARD_ENERGY, SelectContext.DISCARD_ENERGY_CARD):
            return self._score_discard(card)
        if (ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY)
                or ctx == getattr(SelectContext, "DAMAGE", None)):
            if isinstance(card, Pokemon) and o.playerIndex == self.op_index:
                hp = getattr(card, "hp", 0)
                ko_bonus = 20000 if hp <= 100 else 0
                role_bonus = 5000 if card.id in GLOBAL_EFFECT_PROTECTORS else 0
                data = card_table.get(card.id)
                if data is not None and (getattr(data, "stage1", False)
                                         or getattr(data, "stage2", False)
                                         or getattr(data, "ex", False)):
                    role_bonus += 2500
                return 10000 + ko_bonus + role_bonus + prize_count(card) * 1000 - hp
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
        source = getattr(self.select, "contextCard", None)
        if getattr(source, "id", None) == C.LUCKY_HELMET:
            if self._has_tool(p):
                return -1
            if p.id == C.GENESECT and not self._opponent_ace_seen():
                return 25000
            if is_active and p.id == C.ALAKAZAM:
                return 18000
            if is_active and p.id in (C.KADABRA, C.DUDUNSPARCE):
                return 12000
            return -1
        enriching = getattr(source, "id", None) == C.ENRICHING_ENERGY
        if enriching and self._stop_optional_draw():
            return -1
        if enriching and p.id == C.DUNSPARCE:
            return 12500 + (500 if is_active else 0)
        if p.id == C.FEZANDIPITI_EX:
            return -1
        if enriching and is_active and self.bench_attacker_ready() and not self.can_attack(p):
            return 9800
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
            if getattr(getattr(self.select, "contextCard", None), "id", None) == C.BOSS_ORDERS:
                return self._boss_target_score(card)
            return prize_count(card) * 1000 - getattr(card, "hp", 0) // 10
        if o.playerIndex != self.my_index:
            return 0
        if card.id in (C.FEZANDIPITI_EX, C.GENESECT):
            # These Pokémon do their jobs from the Bench. Promote only when every other body is
            # unavailable and the selection is forced.
            return -500 + getattr(card, "hp", 0) // 100
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
        elif card.id == C.SHAYMIN:
            score += 10
        score += getattr(card, "hp", 0) // 30
        return score + 1

    def _score_setup_active(self, card):
        if card is None:
            return 0
        if card.id == C.ABRA:
            return 50
        if card.id == C.DUNSPARCE:
            return 30
        if card.id in (C.FEZANDIPITI_EX, C.GENESECT):
            return 1
        if card.id == C.SHAYMIN:
            return 15
        return 5

    def _score_to_bench(self, card):
        if card is None:
            return 0
        d = card_table.get(card.id)
        if d is None or d.cardType != CardType.POKEMON:
            return 0
        cid = card.id
        n = self.field[cid]
        if self._survival_bench_needed():
            return 1000 - 20 * n
        if cid == C.ABRA:
            return (420 if self._needs_more_abra_body() else 40) - 25 * n
        if cid == C.DUNSPARCE:
            return (340 if self._needs_dunsparce_body() else 45) - 30 * n
        if cid == C.SHAYMIN:
            return 300 if self._shaymin_worthwhile() else -1
        if cid == C.FEZANDIPITI_EX:
            return 380 if self._fezandipiti_worthwhile() else -1
        if cid == C.GENESECT:
            return 360 if self._genesect_worthwhile() else -1
        return 100 - 20 * n

    def _score_to_hand(self, card):
        if card is None:
            return 0
        cid = card.id
        cc = getattr(self.select, "contextCard", None)
        if getattr(cc, "id", None) == C.MAX_ROD:
            return self._max_rod_target_score(cid)
        if getattr(cc, "id", None) == C.POKE_PAD:
            return self._pokepad_target_score(cid)
        score = 200 - self.hand[cid] * 40
        engine_online = self.field[C.DUDUNSPARCE] >= 1
        if cid == C.DUDUNSPARCE:
            score += 45 if not engine_online else -10
        elif cid == C.DUNSPARCE:
            score += 70 if self.field[C.DUDUNSPARCE] + self.field[C.DUNSPARCE] < 1 else -10
        elif cid == C.ABRA:
            score += 130 if self._needs_more_abra_body() else 10
        elif cid == C.KADABRA:
            score += 80
        elif cid == C.ALAKAZAM:
            score += 85 if self.hand[C.ALAKAZAM] == 0 else 40
        elif cid == C.SHAYMIN:
            score += 170 if self._shaymin_worthwhile() else -30
        elif cid == C.FEZANDIPITI_EX:
            score += 140 if self._fezandipiti_worthwhile() else -20
        elif cid == C.GENESECT:
            score += 150 if self._genesect_worthwhile() else -40
        elif cid == C.ENRICHING_ENERGY:
            if self._stop_optional_draw():
                score -= 80
            elif self.field[C.DUNSPARCE] > 0:
                score += 180
            else:
                score += 60
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
        if cid in ALAKAZAM_LINE or cid in ENGINE_LINE or cid in (
                C.FEZANDIPITI_EX, C.SHAYMIN, C.GENESECT):
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
        if cid in ALAKAZAM_LINE or cid in ENGINE_LINE or cid in (
                C.FEZANDIPITI_EX, C.SHAYMIN, C.GENESECT):
            return -40 if self.field[cid] == 0 else 60
        return 10

    # ── whole-set selection for Basic-to-bench search effects ────────────────
    def custom_selection(self):
        context_card = getattr(self.select, "contextCard", None)
        if (self.context == SelectContext.TO_HAND
                and getattr(context_card, "id", None) == C.MAX_ROD):
            options = list(self.select.option or [])
            scored = []
            for idx, option in enumerate(options):
                card = get_card(self.obs, option.area, option.index, option.playerIndex)
                if card is not None:
                    scored.append((self._max_rod_target_score(card.id), idx))
            min_count = max(0, min(self.select.minCount, len(options)))
            max_count = max(min_count, min(self.select.maxCount, len(options)))
            picked = [idx for score, idx in sorted(scored, reverse=True) if score >= 300]
            if len(picked) < min_count:
                picked.extend(idx for score, idx in sorted(scored, reverse=True)
                              if idx not in picked)
            return picked[:max_count]
        setup = self.context == SelectContext.SETUP_BENCH_POKEMON
        searched = (self.context in (SelectContext.TO_BENCH, SelectContext.TO_FIELD)
                    and getattr(context_card, "id", None) in (C.BUDDY_POFFIN,
                                                               C.TELEPATH_ENERGY))
        if not (setup or searched):
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
        missing_abra = max(0, self._desired_abra_bodies() - self._abra_body_count())
        wanted = [C.ABRA] if missing_abra else []
        wanted.extend([C.ABRA] * max(0, missing_abra - 1))
        if self._needs_dunsparce_body():
            wanted.append(C.DUNSPARCE)
        if self._shaymin_worthwhile():
            wanted.append(C.SHAYMIN)
        picked = []
        remaining = list(candidates)
        for wanted_id in wanted:
            if len(picked) >= max_count:
                break
            found = next((item for item in remaining if item[1] == wanted_id), None)
            if found is None:
                continue
            remaining.remove(found)
            picked.append(found[0])
        if len(picked) < min_count:
            for idx, cid in sorted(remaining, key=lambda item: self._score_to_bench(
                    get_card(self.obs, options[item[0]].area, options[item[0]].index,
                             options[item[0]].playerIndex)), reverse=True):
                picked.append(idx)
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
            elif active is not None and active.id == C.FEZANDIPITI_EX:
                _DIAG["fezandipiti_attacks"] += 1
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
            elif card is not None and card.id == C.FEZANDIPITI_EX:
                _DIAG["fezandipiti_abilities"] += 1
        elif self._attack_reserved and self._state in (TurnState.PRESSURE, TurnState.ENDGAME):
            _DIAG["pre_attack_actions"] += 1
        if t == OptionType.PLAY and self._play_card_id(opt) == C.BOSS_ORDERS:
            _turn_boss_mark(self._turn)


_agent = make_agent(AlakazamPolicy, my_deck, _DIAG)


def agent(obs_dict):
    if isinstance(obs_dict, dict) and obs_dict.get("select") is None:
        _TURN_STATE.update({"turn": None, "boss_committed": False,
                            "last_opp_prizes": None,
                            "ko_last_opponent_turn": False})
    return _agent(obs_dict)
