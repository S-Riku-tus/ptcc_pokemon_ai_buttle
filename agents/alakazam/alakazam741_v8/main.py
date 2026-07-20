# alakazam741_v7.2 - v3 pressure core with targeted bench protection and recovery.
#
# Key changes:
# - Boss's Orders and all gust-specific logic removed.
# - Crushing Hammer removed; +1 Battle Cage and +1 Night Stretcher improve
#   bench protection, stadium control, and attacker continuity without coin flips.
# - Run Away Draw can never remove the last Pokemon in play.
# - Low-deck ACTIVATE refusal and lethal-preservation execute on the real score path.
# - Winning KO is immediate; safe positive-hand setup may precede other attacks.
# - Enhanced Hammer is used only when it removes effect prevention for an immediate KO.
# - MAIN actions are classified by one Phase and one priority Tier.
# - Lillie ordering and three-card Hyper Aroma set selection are explicit.
# - BasePolicy/make_agent own common dispatch, selection, fallback, energy, prize,
#   and board-wide effect-prevention logic.
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
    KADABRA = 742         # Stage1 (Psychic Draw on evolve) -> Alakazam
    ALAKAZAM = 743        # Stage2 attacker: Powerful Hand = 20 dmg x cards in hand
    DUNSPARCE = 305       # Basic -> Dudunsparce (7-06: switched to id305 per ladder-#1 Majkel1337's
                          # list — 70HP + Trading Places free switch; the attack-id constants
                          # 423/424 below always belonged to THIS printing, not id65)
    DUDUNSPARCE = 66      # Stage1 draw engine (Run Away Draw)

    PSYCHIC_ENERGY = 5
    TELEPATH_ENERGY = 19  # special, provides {P}
    HYPER_AROMA = 1082    # ACE SPEC Item: search 3 Stage-1 cards.

    BUDDY_POFFIN = 1086
    POKE_PAD = 1152
    HILDA = 1225          # Supporter: search Evolution + Energy
    LILLIE = 1227         # Lillie's Determination: reset a thin hand, draw 6/8.
    DAWN = 1231           # Supporter: search Basic+Stage1+Stage2
    RARE_CANDY = 1079
    XEROSIC = 1197        # v2: 相手は手札が3枚になるまで捨てる(ミラーのPowerful Hand潰し)
    BATTLE_CAGE = 1264    # Stadium: block bench damage counters
    ENHANCED_HAMMER = 1081  # Item: discard a Special Energy from opp (e.g. Mist Energy)
    NIGHT_STRETCHER = 1097
    SACRED_ASH = 1129
    LANA_AID = 1184


POWERFUL_HAND = 1072   # Alakazam 743: place 2 counters (20 dmg) per card in hand, on opp Active
SUPER_PSY_BOLT = 1071  # Kadabra: 30
ALAKAZAM_IDS = {C.ALAKAZAM}
ABRA_TELEPORT = 1070   # Abra: 10 + switch
DUDUN_LAND_CRUSH = 76  # Dudunsparce: 90 (rarely; engine instead)
DUNSPARCE_TRADE = 423  # Dunsparce: switch
DUNSPARCE_RAM = 424

ENERGY_TYPES = {C.PSYCHIC_ENERGY, C.TELEPATH_ENERGY}
ATTACKER_IDS = {C.ALAKAZAM, C.KADABRA}
LOW_DECK_COUNT = 6


class Phase:
    SETUP = "SETUP"
    PRESSURE = "PRESSURE"
    RECOVER = "RECOVER"
    LOCKED = "LOCKED"
    ENDGAME = "ENDGAME"


class Tier:
    BLOCK = 0
    WIN_OR_SURVIVE = 1
    PRE_ATTACK = 2
    ATTACK = 3
    BUILD_ATTACKER = 4
    BUILD_BACKUP = 5
    SEARCH = 6
    DISRUPT = 7
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

_DIAG = new_diag()


def diag_reset():
    _DIAG.clear()
    _DIAG.update(new_diag())


def diag_snapshot():
    snapshot = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DIAG.items()}
    snapshot['fallback_rate'] = (snapshot.get('policy_fallback', 0) + snapshot.get('obs_fallback', 0)) / max(1, snapshot.get('decisions', 0))
    return snapshot


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

    def go_first(self) -> bool:
        return True

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

    def _low_deck(self):
        return self.me.deckCount <= LOW_DECK_COUNT

    def _deck_preserve(self):
        """Don't mill ourselves out of a WINNING game (real-ladder bug: we filtered our
        deck to 0 while ahead enough to close). If we already have a powered attacker and a
        hand big enough to keep KO-ing (Powerful Hand = 20×hand), we don't NEED more cards —
        and once the deck is down to about the number of prizes we still have to take, every
        extra optional draw/filter risks decking out before the last prize. So: stop optional
        drawing and just attack ~1 KO per turn, keeping enough deck to draw 1/turn to the end."""
        if not self._have_attacker():
            return False
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is None:
            return False
        remaining_prizes = len(self.me.prize)                 # ≈ turns we still need
        big_hand = 20 * self.me.handCount >= max(opp.hp, 130)  # can essentially KO a body now
        deck_low = self.me.deckCount <= remaining_prizes + 4   # keep a draw-1/turn buffer
        return big_hand and deck_low

    # ── v2: ドローソースの上限/下限 (デッキ切れ負け防止) ──────────────────────
    def _deck_floor(self):
        """任意ドロー/デッキ消費を止めるフロア。毎ターンの強制ドロー分を確保する。
        v3: 実ログでデッキ切れ負け5件(サイド1-2枚まで来て山札0)。v2のフロア
        max(5, サイド+2) では「圧倒している試合の山札切れ」を防げなかったため
        max(8, サイド+3) に引き上げ = 実質『山札10枚前後で任意ドロー原則停止』。"""
        return max(8, len(self.me.prize) + 3)

    def _lethal_after_draw(self):
        """Run Away Draw(+3)でこのターンのPowerful Handが致死圏に届くか。"""
        opp = self.opponent.active[0] if self.opponent.active else None
        return (opp is not None and self._have_attacker()
                and not self._effect_prevented(opp)
                and 20 * (self.me.handCount + 3) >= opp.hp)

    def _deck_spend_ok(self, cost=1, allow_lethal=True):
        """任意のデッキ消費(ドロー/サーチ)を許可するか。
        下限: フロアを割り込む消費は、それが今ターンの致死に直結する時だけ許す。"""
        if self.me.deckCount - cost > self._deck_floor():
            return True
        if allow_lethal and self._lethal_after_draw() and self.me.deckCount - cost >= 2:
            return True
        return False

    def _psychic_in_hand(self):
        """A {P}-providing energy in hand (the ONLY kind that fuels our attacks — Enriching's
        Colorless does not). 'Energy in hand' that is just Enriching still leaves us starved."""
        return any(ENERGY_PROVIDES.get(c.id) == EnergyType.PSYCHIC for c in self.me.hand)

    def _energy_starved(self):
        """We have an Alakazam-line attacker in play (or a Kadabra + Alakazam in hand to
        evolve) that CAN'T attack, and no usable {P} energy in hand to fix it. With only 6
        energy in 60 cards, energy is the bottleneck — searches should grab a {P} energy."""
        bodies = [p for p in (self.me.active + self.me.bench) if p is not None]
        has_alakazam = any(p.id in ALAKAZAM_IDS for p in bodies)
        coming = any(p.id == C.KADABRA for p in bodies) and self.hand[C.ALAKAZAM] > 0
        if not (has_alakazam or coming):
            return False
        if any(p.id in ALAKAZAM_IDS and self.can_attack(p) for p in bodies):
            return False                       # already have an attacker that can actually attack
        return not self._psychic_in_hand()

    def _effect_prevented(self, target):
        """Deck-facing alias for the shared target/energy/board protection check."""
        return self.effect_prevented(target)

    def _opp_active_has_prevent_energy(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is None:
            return False
        return any(getattr(e, 'id', None) in EFFECT_PREVENT_ENERGY
                   for e in (getattr(opp, 'energyCards', None) or []))

    # — damage —
    def _alakazam_damage(self, attack_id, target):
        if target is None:
            return 0
        if attack_id == POWERFUL_HAND:
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
        return 0

    def _hand_delta(self, option_type, option):
        """Conservative net hand change used only by the lethal-preservation gate."""
        if option_type == OptionType.ABILITY:
            card = get_card(self.obs, option.area, option.index, self.my_index)
            if card is not None and card.id == C.DUDUNSPARCE and option.area == AreaType.BENCH:
                return 3
            return 0
        if option_type in (OptionType.EVOLVE, OptionType.ATTACH, OptionType.ENERGY):
            return -1
        if option_type == OptionType.PLAY:
            card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
            if card is not None and card.id in (C.HILDA, C.DAWN, C.LANA_AID):
                return 1
            # Night Stretcher replaces itself with a Pokémon/basic Energy from discard.
            # Treat it as hand-neutral only when a useful legal target is confirmed.
            if (card is not None and card.id == C.NIGHT_STRETCHER
                    and self._night_stretcher_worthwhile()):
                return 0
            return -1
        return 0

    def custom_selection(self):
        """Choose a balanced three-card Hyper Aroma package.

        Fixed per-card scores can select three Kadabra for one Abra.  This set-level
        selector fills current evolution roles first, then uses remaining slots as
        low-duplication backups.
        """
        context_card = getattr(self.select, 'contextCard', None)
        if (self.context != SelectContext.TO_HAND
                or getattr(context_card, 'id', None) != C.HYPER_AROMA):
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

        abra_waiting = max(0, self.field[C.ABRA] - self.field[C.KADABRA]
                           - self.hand[C.KADABRA])
        dunsparce_waiting = max(0, self.field[C.DUNSPARCE] - self.field[C.DUDUNSPARCE]
                               - self.hand[C.DUDUNSPARCE])
        desired = {
            C.KADABRA: min(2, abra_waiting if abra_waiting else int(self.field[C.ALAKAZAM] == 0)),
            C.DUDUNSPARCE: min(2, dunsparce_waiting if dunsparce_waiting else int(self.field[C.DUDUNSPARCE] == 0 and self.field[C.DUNSPARCE] > 0)),
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
                # Strongly prefer role diversity; a third identical copy is a fallback.
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

    def score(self, option):
        # v3 P0-2 must live on the actual dispatch path, not in a dead helper.
        if self.context == SelectContext.ACTIVATE and self.me.deckCount <= self._deck_floor():
            if option.type == OptionType.YES:
                return 0
            if option.type == OptionType.NO:
                return 1
        raw = super().score(option)
        if self.context != SelectContext.MAIN:
            return raw
        return self._score_main(option, raw)

    def _ready_alakazam_attacker(self):
        return any(p is not None and p.id == C.ALAKAZAM and self.can_attack(p)
                   for p in self.my_board())

    def _phase(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        if self._ready_alakazam_attacker():
            if opp is not None and self._effect_prevented(opp) and self._active_best_dmg(opp) <= 0:
                return Phase.LOCKED
            if self.me.deckCount <= self._deck_floor() + 1 or len(self.me.prize) <= 1:
                return Phase.ENDGAME
            return Phase.PRESSURE
        if (self.field[C.ALAKAZAM] or self.field[C.KADABRA]
                or self.discard.get(C.ALAKAZAM, 0) or self.discard.get(C.KADABRA, 0)):
            return Phase.RECOVER
        return Phase.SETUP

    def _tier(self, tier, score=0):
        local = max(0, min(int(score or 0), TIER_SPAN - 1))
        return TIER_BASE[tier] + local

    def _attack_damage_for_option(self, option):
        if option.type != OptionType.ATTACK:
            return 0
        opp = self.opponent.active[0] if self.opponent.active else None
        return self._alakazam_damage(option.attackId, opp) if opp is not None else 0

    def _has_meaningful_attack_option(self):
        return any(option.type == OptionType.ATTACK
                   and self._attack_damage_for_option(option) > 0
                   for option in (self.select.option or []))

    def _play_card_id(self, option):
        card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
        return getattr(card, 'id', None)

    def _optional_deck_spend(self, option):
        if option.type == OptionType.ABILITY:
            card = get_card(self.obs, option.area, option.index, self.my_index)
            return card is not None and card.id == C.DUDUNSPARCE and option.area == AreaType.BENCH
        if option.type != OptionType.PLAY:
            return False
        return self._play_card_id(option) in (
            C.BUDDY_POFFIN, C.POKE_PAD, C.HILDA, C.DAWN, C.LILLIE, C.HYPER_AROMA
        )

    def _pre_attack_ko_setup(self, option):
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is None or not self._ko_active_reachable() or self._active_best_dmg(opp) >= opp.hp:
            return False
        if option.type == OptionType.ABILITY:
            card = get_card(self.obs, option.area, option.index, self.my_index)
            return (card is not None and card.id == C.DUDUNSPARCE
                    and option.area == AreaType.BENCH and self._deck_spend_ok(cost=3))
        if option.type == OptionType.PLAY:
            return self._play_card_id(option) in (C.HILDA, C.DAWN, C.POKE_PAD, C.HYPER_AROMA)
        return False

    def _breaks_current_lethal(self, option):
        if not self._lethal_attack_offered() or self._hand_delta(option.type, option) >= 0:
            return False
        active = self.me.active[0] if self.me.active else None
        opp = self.opponent.active[0] if self.opponent.active else None
        return (active is not None and active.id == C.ALAKAZAM and opp is not None
                and 20 * (self.me.handCount - 1) < opp.hp)

    def _hammer_unlocks_attack(self):
        opp = self.opponent.active[0] if self.opponent.active else None
        return (opp is not None and self._opp_active_has_prevent_energy()
                and 20 * max(0, self.me.handCount - 1) > 0)

    def _has_backup_line(self):
        """A benched Abra-line body that can become the next attacker."""
        return any(p is not None and p.id in (C.ABRA, C.KADABRA, C.ALAKAZAM)
                   for p in self.me.bench)

    def _needs_first_backup(self):
        return self._ready_alakazam_attacker() and not self._has_backup_line()

    def _battle_cage_worthwhile(self):
        if self.state.stadiumPlayed or self.stadium_id == C.BATTLE_CAGE:
            return False
        if self.stadium_id and self.stadium_id != C.BATTLE_CAGE:
            return True
        has_bench = any(p is not None for p in self.me.bench)
        return has_bench and self.opponent_threatens_bench()

    def _night_stretcher_target_score(self, cid):
        """Value a legal Night Stretcher target by the route it completes."""
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
            if self.field[C.ABRA] > 0 or self.hand[C.ABRA] > 0:
                return 980
            return 280
        if cid == C.ABRA:
            if self._needs_first_backup() and self._open_bench():
                return 1150
            if line_in_play + line_in_hand == 0 and self._open_bench():
                return 900
            return 360
        if cid == C.DUDUNSPARCE:
            if self.field[C.DUNSPARCE] > 0 and self.field[C.DUDUNSPARCE] == 0:
                return 700
            return 260
        if cid == C.DUNSPARCE:
            if engine_in_play == 0 and self._open_bench():
                return 650
            return 220
        return -1

    def _night_stretcher_worthwhile(self):
        recoverable = (C.ABRA, C.KADABRA, C.ALAKAZAM,
                       C.DUNSPARCE, C.DUDUNSPARCE, C.PSYCHIC_ENERGY)
        return max((self._night_stretcher_target_score(cid)
                    for cid in recoverable if self.discard.get(cid, 0)),
                   default=-1) >= 600

    def _action_tier(self, option, phase):
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
                return Tier.PRE_ATTACK if phase in (Phase.PRESSURE, Phase.ENDGAME) else Tier.SEARCH
            return Tier.DISRUPT
        if t == OptionType.PLAY:
            cid = self._play_card_id(option)
            if cid in (C.RARE_CANDY,):
                return Tier.BUILD_ATTACKER
            if cid in (C.BUDDY_POFFIN, C.POKE_PAD, C.HILDA, C.DAWN, C.LILLIE, C.HYPER_AROMA):
                if (cid == C.BUDDY_POFFIN and phase in (Phase.PRESSURE, Phase.ENDGAME)
                        and self._needs_first_backup() and self._open_bench()):
                    return Tier.PRE_ATTACK
                return Tier.SEARCH
            if cid == C.NIGHT_STRETCHER and self._night_stretcher_worthwhile():
                if phase in (Phase.PRESSURE, Phase.ENDGAME) and self._needs_first_backup():
                    return Tier.PRE_ATTACK
                return Tier.BUILD_ATTACKER if not self._ready_alakazam_attacker() else Tier.BUILD_BACKUP
            if cid == C.BATTLE_CAGE and self._battle_cage_worthwhile():
                return Tier.PRE_ATTACK if phase in (Phase.PRESSURE, Phase.ENDGAME) else Tier.DISRUPT
            # Preserve Xerosic's existing mirror score across the Tier wrapper: only
            # a 6+ card Alakazam mirror hand may be disrupted before a normal attack.
            # Winning attacks and exact-KO preservation are still handled earlier
            # in _score_main(), so this does not override a current or game-winning KO.
            if cid == C.XEROSIC and phase in (Phase.PRESSURE, Phase.ENDGAME):
                opp_hand = getattr(self.opponent, "handCount", 0) or 0
                opp_board = [
                    p for p in (self.opponent.active + self.opponent.bench)
                    if p is not None
                ]
                mirror = any(p.id in (C.ABRA, C.KADABRA, C.ALAKAZAM)
                             for p in opp_board)
                if mirror and opp_hand >= 6:
                    return Tier.PRE_ATTACK
            if cid in (C.ENHANCED_HAMMER, C.XEROSIC, C.BATTLE_CAGE):
                return Tier.DISRUPT
            data = card_table.get(cid)
            if data is not None and data.cardType == CardType.POKEMON:
                if (cid == C.ABRA and phase in (Phase.PRESSURE, Phase.ENDGAME)
                        and self._needs_first_backup()):
                    return Tier.PRE_ATTACK
                return Tier.BUILD_ATTACKER if cid == C.ABRA else Tier.BUILD_BACKUP
            return Tier.BUILD_BACKUP
        return Tier.DISRUPT

    def _score_main(self, option, raw):
        if raw < 0:
            return self._tier(Tier.BLOCK)
        if self._breaks_current_lethal(option):
            return self._tier(Tier.BLOCK)

        phase = self._phase()
        if option.type == OptionType.ATTACK:
            dmg = self._attack_damage_for_option(option)
            if dmg <= 0:
                return self._tier(Tier.BLOCK)
            opp = self.opponent.active[0] if self.opponent.active else None
            if opp is not None and dmg >= opp.hp and prize_count(opp) >= len(self.me.prize):
                return self._tier(Tier.WIN_OR_SURVIVE, raw)
            return self._tier(self._action_tier(option, phase), raw)

        if option.type == OptionType.END:
            if phase in (Phase.PRESSURE, Phase.ENDGAME) and self._has_meaningful_attack_option():
                return self._tier(Tier.BLOCK)
            return self._tier(Tier.END, raw)

        if phase == Phase.LOCKED:
            cid = self._play_card_id(option) if option.type == OptionType.PLAY else None
            if cid == C.ENHANCED_HAMMER and self._hammer_unlocks_attack():
                return self._tier(Tier.WIN_OR_SURVIVE, raw)
            if self._optional_deck_spend(option):
                return self._tier(Tier.BLOCK)

        if self._pre_attack_ko_setup(option):
            return self._tier(Tier.PRE_ATTACK, raw)

        # While pressuring, a safe positive-hand Dudunsparce activation may happen before
        # a margin KO, but routine development and disruption stay below the attack tier.
        if (phase in (Phase.PRESSURE, Phase.ENDGAME)
                and option.type == OptionType.ABILITY
                and self._hand_delta(option.type, option) > 0
                and self._deck_spend_ok(cost=3)):
            return self._tier(Tier.PRE_ATTACK, raw)

        return self._tier(self._action_tier(option, phase), raw)

    def _item_locked(self):
        """Use only explicit opponent lock abilities; state inference caused self-KOs."""
        opp = self.opponent.active[0] if self.opponent.active else None
        return opp is not None and opp.id in ITEM_LOCK_IDS

    def _bench_attacker_ready(self):
        return self.bench_attacker_ready()

    # — abilities —
    def _score_ability(self, o):
        card = get_card(self.obs, o.area, o.index, self.my_index)
        if card is None:
            return 0
        if card.id == C.DUDUNSPARCE:
            # Hard invariant: never shuffle away the last Pokémon in play.  This is
            # checked before Item-lock or deck-budget heuristics so no inference can
            # re-enable the historical instant-loss line.
            board_count = sum(p is not None for p in self.my_board())
            if o.area != AreaType.BENCH and board_count <= 1:
                return -1
            if not self._deck_spend_ok(cost=3):
                return -1
            if o.area != AreaType.BENCH:
                # ACTIVE copy: CYCLE this weak active out and promote a ready benched
                # attacker (or escape Item-lock), then attack the same turn. This is
                # REPOSITIONING TO ATTACK, not filtering — so it is ALWAYS allowed, even in
                # deck-preserve mode (getting the powered Alakazam active to swing is the
                # whole point). Bug fixed: gating this on _deck_preserve stranded a powered
                # Alakazam on the bench (Dudunsparce active, 0 energy, can't retreat) -> no
                # attacks -> no_offense loss.
                if self.bench_attacker_ready():
                    return 14000
                return -1
            # BENCHED copy = the draw engine (pure filtering). Draw-engine decks WIN by
            # drawing aggressively (big hand = big Powerful Hand) — blanket deck-out guards
            # regressed cabt — so we draw, EXCEPT: when we already have a winning hand and the
            # deck is low, stop filtering ourselves out of a won game (real-ladder bug).
            if self._deck_preserve():
                return -1
            # NB: top pilots activate Run Away Draw ~1/4 as often as we did (MAIN ABILITY 163 vs
            # our 622) — but a blunt hand-cap gave ~0 divergence gain here and risks the documented
            # cabt regression (deck-out guards hurt cabt), so we keep the aggressive-draw identity
            # and leave "draw less" as a separate real-ladder A/B. Only the high-hand floor stays.
            # v3 P0-2: 高手札×低山札ガードを強化(14/12→12/14)。手札12枚=240ダメは
            # 既にワンパン圏。これ以上の圧縮はデッキ切れリスクだけが増える。
            if self.me.handCount >= 12 and self.me.deckCount <= 14:
                return -1
            return 15000
        return 9000

    # — play —
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
        # v2: グッズ優先シーケンス — サポート(1回/ターン)を残したまま先にグッズを消化する。
        if (base > 0 and d.cardType == CardType.ITEM
                and not self.state.supporterPlayed
                and any(self.hand[s] for s in (C.HILDA, C.DAWN, C.LILLIE, C.XEROSIC))):
            base += 900
        return base

    def _score_play_poke(self, card):
        cid = card.id; n = self.field[cid]
        if cid == C.ABRA:
            # Majkel (7-05, 7275 MAIN decisions): moderate bench — our #1 over-pick was
            # flooding bodies (PLAY:Dunsparce 548x / Abra 152x). 3 line bodies is plenty.
            line = self.field[C.ABRA] + self.field[C.KADABRA] + self.field[C.ALAKAZAM]
            if line >= 3:
                return 1500
            return 20000 - 250 * n
        if cid == C.DUNSPARCE:
            if self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE] >= 2:
                return 1200   # cap at 2 engine bodies
            return 18500 - 250 * n
        return 14000 - 200 * n

    def _alakazam_ready(self):
        a = self.me.active[0] if self.me.active else None
        return a is not None and a.id in ALAKAZAM_IDS and self.energy_count(a) >= 1

    def _need_pieces(self):
        return self.field[C.ALAKAZAM] < 1

    def _open_bench(self):
        return sum(1 for p in self.me.bench if p is not None) < getattr(self.me, "benchMax", 5)

    def _achievable_hand(self):
        """Biggest hand we can realistically reach THIS turn (Powerful Hand = 20×hand):
        current hand + Run Away Draw (+3) + one draw/search Supporter (~+1 net)."""
        extra = 0
        if self.me.deckCount > 7 and any(p is not None and p.id == C.DUDUNSPARCE for p in self.me.bench):
            extra += 3
        if not self.state.supporterPlayed and (self.hand[C.HILDA] or self.hand[C.DAWN]):
            extra += 1
        return self.me.handCount + extra

    def _have_attacker(self):
        a = self.me.active[0] if self.me.active else None
        return (a is not None and a.id in ALAKAZAM_IDS and self.energy_count(a) >= 1) or self.bench_attacker_ready()

    def _lethal_now(self):
        """今の自分Activeの攻撃(Powerful Hand / Psychic / Super Psy Bolt)で相手Activeを
        このまま倒せるか。v3: 20×手札の743専用判定から _active_best_dmg ベースに一般化
        (245のPsychicによる致死も拾う。Mist等の効果防止は damage 計算側で0になる)。"""
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is None:
            return False
        return self._active_best_dmg(opp) >= max(1, opp.hp)

    def _lethal_attack_offered(self):
        """致死 かつ 実際にダメージの出る攻撃オプションがこのselectで提示されているか。
        (マヒ/眠り等で攻撃自体が提示されない時に展開行動まで封じないためのガード)"""
        if not self._lethal_now():
            return False
        return any(o.type == OptionType.ATTACK
                   and o.attackId in (POWERFUL_HAND, SUPER_PSY_BOLT)
                   for o in (self.select.option or []))

    def _ko_active_reachable(self):
        """Can Powerful Hand KO the opponent's ACTIVE this turn — now, or after the
        drawing still available to us? (Each turn, aim to KO the best target: usually
        the dangerous active attacker, by pumping the hand to lethal.)"""
        opp = self.opponent.active[0] if self.opponent.active else None
        return (opp is not None and self._have_attacker()
                and not self._effect_prevented(opp)        # Mist Energy etc. → 0, don't chase it
                and 20 * self._achievable_hand() >= opp.hp)

    def _holds_complete_route(self):
        has_abra = self.field[C.ABRA] > 0 or self.hand[C.ABRA] > 0
        candy_route = has_abra and self.hand[C.RARE_CANDY] > 0 and self.hand[C.ALAKAZAM] > 0
        kadabra_route = has_abra and self.hand[C.KADABRA] > 0 and (
            self.hand[C.ALAKAZAM] > 0 or self.field[C.ALAKAZAM] == 0
        )
        has_fuel = self._psychic_in_hand() or any(
            p is not None and p.id in (C.ABRA, C.KADABRA, C.ALAKAZAM) and self.can_attack(p)
            for p in self.my_board()
        )
        return (candy_route or kadabra_route) and has_fuel

    def _has_pre_lillie_action(self):
        """Whether a deterministic board-improving action should happen before Lillie."""
        for option in (self.select.option or []):
            if option.type == OptionType.PLAY:
                cid = self._play_card_id(option)
                if cid == C.LILLIE:
                    continue
                if cid == C.BUDDY_POFFIN:
                    bodies = (self.field[C.ABRA] + self.field[C.KADABRA]
                              + self.field[C.ALAKAZAM] + self.field[C.DUNSPARCE]
                              + self.field[C.DUDUNSPARCE])
                    if bodies < 4 and self._open_bench():
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

    def _enhanced_hammer_worthwhile(self):
        """Use Enhanced Hammer only when it immediately unlocks a KO."""
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is None or not self._opp_active_has_prevent_energy():
            return False
        # Playing the Item costs one hand card, so evaluate the post-play Powerful Hand.
        return 20 * max(0, self.me.handCount - 1) >= opp.hp

    def _score_play_trainer(self, card):
        cid = card.id
        if cid == C.RARE_CANDY:
            if self.field[C.ABRA] >= 1 and self.hand[C.ALAKAZAM] >= 1:
                # Majkel: step-evolve through Kadabra when possible — its Psychic Draw (+3
                # cards) beats the Candy skip (we over-played Candy 341x). Candy is for
                # when the Kadabra bridge is missing.
                if self.hand[C.KADABRA] >= 1:
                    return 8000   # prefer the Kadabra bridge, but Candy is still fine tempo
                return 20500
            return -1
        opp_active = self.opponent.active[0] if self.opponent.active else None
        # Each turn, if we can KO the dangerous Active this turn by drawing up to a lethal
        # Powerful Hand, DRAW toward it (a draw Supporter beats gusting a weaker target).
        draw_for_ko = (opp_active is not None and self._ko_active_reachable()
                       and 20 * self.me.handCount < opp_active.hp)
        # Winning + deck low: stop spending the deck on optional search supporters.
        if cid in (C.HILDA, C.DAWN, C.POKE_PAD) and self._deck_preserve():
            return -1
        # v2: 負けている時もデッキ切れは即負け。フロアを割るデッキ消費は致死直結時のみ。
        if cid in (C.HILDA, C.DAWN, C.POKE_PAD, C.BUDDY_POFFIN) \
                and not self._deck_spend_ok(cost=2):
            return -1
        if cid == C.HYPER_AROMA:
            if self._item_locked():
                return -1
            if not self._deck_spend_ok(cost=3):
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
            # Majkel plays Dawn broadly (214x divergent) — +3 hand = +60 Powerful Hand
            return 12000 if self._need_pieces() else 7500
        if cid == C.BUDDY_POFFIN:
            bodies = (self.field[C.ABRA] + self.field[C.KADABRA] + self.field[C.ALAKAZAM]
                      + self.field[C.DUNSPARCE]
                      + self.field[C.DUDUNSPARCE])
            if bodies >= 4 or not self._open_bench():
                return 600   # board is set — a Poffin now is -20 Powerful Hand for nothing
            return 13000
        if cid == C.POKE_PAD:
            # Majkel keeps digging with it after setup too — every deck→hand card
            # is +20 Powerful Hand (but below Poffin/supporters)
            return 8500 if self._need_pieces() else 3500
        if cid == C.XEROSIC:
            # v2: 相手の手札を3枚に。ミラー(相手もPowerful Hand)では手札=火力なので
            # 相手の手札が肥えた時に最優先で撃つ。他デッキ相手でも大量ハンドには妨害価値。
            if self.state.supporterPlayed:
                return -1
            opp_hand = getattr(self.opponent, "handCount", 0) or 0
            opp_board = [p for p in (self.opponent.active + self.opponent.bench)
                         if p is not None]
            mirror = any(p.id in (C.ABRA, C.KADABRA, C.ALAKAZAM)
                         for p in opp_board)
            # v3 P1-3: 3枚体制になったので発動閾値を緩和。ミラー(相手もPowerful Hand)
            # では手札=火力なので、手札6+で最優先。一位デッキもクセロシキ3枚採用。
            if mirror and opp_hand >= 6:
                return 15500        # 20×(手札-3超分)を丸ごと削る — ドローより優先
            if opp_hand >= 8:
                return 13000
            if mirror and opp_hand >= 4:
                return 9000
            return 400              # 手札が細い相手には温存
        if cid == C.ENHANCED_HAMMER:
            return 16000 if self._enhanced_hammer_worthwhile() else -1
        if cid == C.BATTLE_CAGE:
            if not self._battle_cage_worthwhile():
                return -1
            # v3 P1-2: 敵スタジアムが出ている(監視塔=無色特性無効でノコッチ停止、
            # Full Metal Lab=火力-30、Spikemuth Gym等)なら即座に張り替える。
            # 実ログ: 敵スタジアム下の試合が多数(Full Metal Lab 14試合など)。3枚体制。
            if self.stadium_id:
                return 12500
            # 場が空なら従来通り: ベンチ攻撃対面で先張り、それ以外は温存気味
            return 6500 if self.opponent_threatens_bench() else 1800
        if cid == C.NIGHT_STRETCHER:
            if not self._night_stretcher_worthwhile():
                return -1
            best = max(self._night_stretcher_target_score(cid2)
                       for cid2 in (C.ABRA, C.KADABRA, C.ALAKAZAM,
                                    C.DUNSPARCE, C.DUDUNSPARCE, C.PSYCHIC_ENERGY)
                       if self.discard.get(cid2, 0))
            return 7800 + best
        if cid == C.LANA_AID:
            if self.state.supporterPlayed:
                return -1
            return 6000 if self._low_deck() else 1500
        if cid == C.SACRED_ASH:
            # v3 P0-3: 山札にポケモン5枚を戻す=唯一の「山札回復」札。デッキ切れ負け5件
            # への直接対策として、山札14枚以下+トラッシュにライン3枚以上で最優先級に昇格。
            line_in_discard = sum(self.discard.get(x, 0) for x in
                                  (C.ABRA, C.KADABRA, C.ALAKAZAM,
                                   C.DUNSPARCE, C.DUDUNSPARCE))
            if self.me.deckCount <= 14 and line_in_discard >= 3:
                return 12000
            return 6000 if self._low_deck() and self.me.discard else 200
        return 9000

    # — evolve —
    def _score_evolve(self, o):
        target = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.my_index)
        if not isinstance(target, Pokemon):
            return 0
        card = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
        cid = card.id if card is not None else None
        if cid == C.ALAKAZAM:
            # One attacking Alakazam at a time — each extra evolve burns a hand card
            # (-20 Powerful Hand). Majkel does evolve the ACTIVE Kadabra (fresh attacker)
            # even with one Alakazam up, but doesn't stack bench Alakazams.
            have = self.field[C.ALAKAZAM]
            if have == 0 or o.inPlayArea == AreaType.ACTIVE:
                return 21000
            return 4000
        if cid == C.KADABRA:
            # JIT (Majkel 7-06: his 237 vs our 1120): evolve when BRIDGING to Alakazam or
            # when the hand needs the +3 draw — otherwise the piece is safer in hand
            # (on board it's Grimmsnarl-snipe/Froslass-chip bait, in hand it's +20 dmg).
            if self.hand[C.ALAKAZAM] >= 1 or self.me.handCount <= 4                     or self.field[C.ALAKAZAM] == 0:
                return 20000
            return 6000
        if cid == C.DUDUNSPARCE:
            return 19000
        return 18000

    # — attach energy —
    def _score_attach(self, o):
        p = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.my_index)
        if not isinstance(p, Pokemon):
            return 0
        src = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
        # GENERAL RULE (type-aware): attach only while the body still can't pay an attack;
        # once it CAN attack, hold the rest (fuels a backup AND +20 Powerful Hand per card).
        if not self.should_fuel(p):
            return -1
        if not self.attach_helps(p, src):
            return -1
        base = 0
        if p.id in ALAKAZAM_IDS:
            base = 8000 + (200 if o.inPlayArea == AreaType.ACTIVE else 0)
        elif p.id in (C.ABRA, C.KADABRA):
            base = 1500           # pre-fuel the line (energy carries through evolution)
        else:
            return -1             # non-attacker -> don't waste energy, hold it
        # v2: テレパスサイコエネルギーは貼るとデッキから基本{P}ポケモン2体をベンチへ。
        # ベンチが空いていて序盤(デッキ余裕あり)は素の超エネより優先、終盤/満員時は逆。
        if src is not None and src.id == C.TELEPATH_ENERGY:
            if self._open_bench() and self._deck_spend_ok(cost=2, allow_lethal=False):
                base += 250
            else:
                base -= 150
        return base

    # — retreat —
    def _score_retreat(self):
        active = self.me.active[0] if self.me.active else None
        opp = self.opponent.active[0] if self.opponent.active else None
        if active is None or opp is None:
            return -1
        if active.id not in ALAKAZAM_IDS:
            for p in self.me.bench:
                if p is not None and p.id in ALAKAZAM_IDS and self.energy_count(p) >= 1:
                    return 6000
        return -1

    # — attack —
    def _score_attack(self, o):
        active = self.me.active[0] if self.me.active else None
        opp = self.opponent.active[0] if self.opponent.active else None
        if active is None or opp is None:
            return 800
        aid = o.attackId
        if aid in (ABRA_TELEPORT, DUNSPARCE_TRADE):
            # These switch the Active with a benched Pokémon (ends the turn). Only worth
            # it to bring up a ready attacker when the current Active isn't one and we
            # can't otherwise swap (Issue 1) — otherwise it's just a wasted reposition.
            if active.id not in ALAKAZAM_IDS and active.id != C.KADABRA and self.bench_attacker_ready():
                return 5000
            return 700
        # Score this specific attack by its own damage.
        dmg = self._alakazam_damage(aid, opp)
        if dmg <= 0:
            return 500
        # Lethal: if this KO takes our last remaining prize(s), it wins the game now.
        if opp.hp <= dmg and prize_count(opp) >= len(self.me.prize):
            return 90000
        # v3 P0-1: KO可能時の攻撃スコアは2段階。
        #  - 致死ギリギリ(これ以上の手札消費で致死を失う) → 30000: 展開抑制と対で必ず攻撃
        #  - 余裕あり → 6000+α: 先に余剰マージンで展開させ、残ったら攻撃で〆る
        #    (v2の4200では夜のタンカ7500等に割り込まれてKOを逃すことがあった)
        if opp.hp <= dmg:
            margin_spent = (active.id != C.ALAKAZAM
                            or 20 * (self.me.handCount - 1) < opp.hp)
            if margin_spent:
                return 30000 + prize_count(opp) * 200
            return 6000 + prize_count(opp) * 300
        score = 1000 + min(dmg, 320)
        return score

    # — sub-selects —
    def _score_card(self, o):
        card = get_card(self.obs, o.area, o.index, o.playerIndex)
        if card is None:
            return 0
        ctx = self.context
        # Opponent card targeting (e.g. Enhanced Hammer: discard a Special Energy from
        # opp) — strip the Mist/Rock that's blocking Powerful Hand, prefer the Active.
        if o.playerIndex == self.op_index and not isinstance(card, Pokemon):
            context_card = getattr(self.select, 'contextCard', None)
            data = card_table.get(card.id)
            is_energy_card = data is not None and data.cardType in (
                CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY
            )
            active_bonus = 500 if getattr(o, 'inPlayArea', None) == AreaType.ACTIVE else 0
            if getattr(context_card, 'id', None) == C.ENHANCED_HAMMER:
                if card.id in EFFECT_PREVENT_ENERGY:
                    return 2000 + active_bonus
                return 200 if data is not None and data.cardType == CardType.SPECIAL_ENERGY else -1
            if card.id in EFFECT_PREVENT_ENERGY:
                return 2000 + active_bonus
            return 300 + active_bonus if is_energy_card else 50
        if (ctx == SelectContext.TO_HAND
                and getattr(getattr(self.select, 'contextCard', None), 'id', None) == C.NIGHT_STRETCHER
                and getattr(o, 'area', None) == AreaType.DISCARD):
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
            # Sacred Ash (TO_DECK from the DISCARD pile): recycle ALL 5 slots with line
            # pokemon — Majkel fills it (his 5-card picks vs our 3; TO_DECK agree 12%).
            if getattr(o, 'area', None) == AreaType.DISCARD:
                cid = card.id
                if cid in (C.ABRA, C.KADABRA, C.ALAKAZAM):
                    return 90
                if cid in (C.DUNSPARCE, C.DUDUNSPARCE):
                    return 70
                d = card_table.get(cid)
                if d is not None and d.cardType == CardType.POKEMON:
                    return 30
                return 5
            return self._score_putback(card)
        return 0

    def _score_attach_target(self, p, is_active):
        if not self.should_fuel(p):
            return -1             # already CAN attack (type-aware) -> don't over-fill
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
        # Promote (after a KO) the body that best keeps us in the game:
        #  1) a ready Alakazam (can Powerful Hand now) — energy bonus makes it top.
        #  2) any Alakazam (online next turn after we attach 1).
        #  3) the tankiest survivor (Dudunsparce 140 / Kadabra 80) so we don't just feed
        #     the opponent a free prize off a 50-HP Abra; a Kadabra can also evolve into
        #     Alakazam next turn. NEVER strand the win-con behind a fragile chump-promote.
        # Promotion order MEASURED against the Elo≥1150 Alakazam pool: they promote the
        # EVOLUTION LINE (Abra/Kadabra → becomes the Alakazam attacker), NOT the Dudunsparce
        # wall (a draw-engine dead end that can't pressure). We over-promoted Dudunsparce.
        score = len(card.energies) * 10
        if card.id in ALAKAZAM_IDS:
            score += 200         # a powered Alakazam = our attacker
        elif card.id == C.KADABRA:
            score += 95          # 80 HP, one evolve from Alakazam — keep the line going
        elif card.id == C.ABRA:
            # v2(ユーザー知見): 素のケーシィ(50HP)を前に出すのは無料サイド献上。
            if self.hand[C.KADABRA] or (self.hand[C.ALAKAZAM] and self.hand[C.RARE_CANDY]):
                score += 110     # すぐ進化して攻撃ラインに乗る
            else:
                score += 25      # 進化できない裸のケーシィはダンスパ系の後ろ          # continues the line to Alakazam (top pilots promote it)
        elif card.id == C.DUDUNSPARCE:
            score += 40          # 140 HP wall but a dead end — don't strand the win-con
        score += getattr(card, 'hp', 0) // 30   # mild "promote the survivor" tiebreak
        return score + 1

    def _score_setup_active(self, card):
        # Opening-active choice. MEASURED (in-process cabt, 60 games vs Lucario):
        # opening Abra      -> 26% loss, 0 no-offense (evolves in place -> Alakazam fast)
        # opening Dunsparce -> 57% loss, 5 no-offense (70HP body, no attacker path)
        # So: Abra >> Dunsparce > (anything that can become an attacker) >> tech basics.
        if card is None:
            return 0
        if card.id == C.ABRA:
            return 50          # the evolution line -> Alakazam: always preferred
        if card.id == C.DUNSPARCE:
            return 30          # draw engine; digs into Abra but slow to pressure
        return 5

    def _score_to_bench(self, card):
        if card is None:
            return 0
        d = card_table.get(card.id)
        if d is None or d.cardType != CardType.POKEMON:
            return 0
        cid = card.id; n = self.field[cid]
        if cid == C.ABRA:
            return 200 - 30 * n   # Majkel benches Abra over Dunsparce ~3:1
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
        # Majkel (7-05, TO_HAND 1503 decisions): grab the ALAKAZAM LINE (Abra/Kadabra/
        # Alakazam, his 634 picks) — do NOT hoard Dudunsparce (our 379x over-grab; the
        # engine shuffles itself back and re-benching is cheap).
        engine_online = self.field[C.DUDUNSPARCE] >= 1
        if cid == C.DUDUNSPARCE:
            # Majkel doesn't re-fetch the self-recycling engine (his 79 vs our 427 grabs) —
            # the LINE pieces come first even when the engine is offline.
            score += 45 if not engine_online else -10
        elif cid == C.DUNSPARCE:
            score += 70 if self.field[C.DUDUNSPARCE] + self.field[C.DUNSPARCE] < 1 else -10
        elif cid == C.ABRA:
            score += 85 if self.field[C.ALAKAZAM] + self.field[C.KADABRA] + self.field[C.ABRA] < 3 else 10
        elif cid == C.KADABRA:
            score += 80
        elif cid == C.ALAKAZAM:
            # his #1 grab (336x): spares feed Sacred Ash recycling & the 2nd attacker
            score += 85 if self.hand[C.ALAKAZAM] == 0 else 40
        elif self.is_energy(cid):
            # When starved, fetch a {P} energy (the only kind that fuels our attacks) — an
            # Enriching (Colorless) doesn't help, so don't prioritise it.
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
        if cid in (C.ABRA, C.KADABRA, C.ALAKAZAM, C.DUNSPARCE, C.DUDUNSPARCE):
            return -50 if self.field[cid] == 0 else 5
        if cid in (C.HILDA, C.DAWN) and self.state.supporterPlayed:
            return 30
        return 0

    def _score_putback(self, card):
        # TO_DECK (Majkel agree 2%→): return SPARE line pieces to the deck freely — they're
        # re-searchable (Dawn/Poké Pad/Hilda); only protect a piece the board still lacks.
        if card is None:
            return 0
        cid = card.id
        if self.hand[cid] >= 2:
            return 70
        if cid in (C.ABRA, C.KADABRA, C.ALAKAZAM, C.DUNSPARCE, C.DUDUNSPARCE):
            return -40 if self.field[cid] == 0 else 60
        return 10



_agent = make_agent(AlakazamPolicy, my_deck, _DIAG)


def agent(obs_dict):
    return _agent(obs_dict)
