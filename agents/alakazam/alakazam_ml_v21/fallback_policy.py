# alakazam_ml_v21 - ladder-risk control and role-aware sequencing:
#   [P0] Never END while an offered attack makes real progress.
#   [P0] Treat KO promotion and Abra/Dunsparce end-turn switching as different decisions.
#   [P1] Stop optional draw after a concrete KO is secured unless continuity still needs it.
#   [P1] Price Fezandipiti ex exposure from the opponent's prize clock and immediate draw out.
#   [P1] Infer key evolution families from the visible board instead of relying only on card IDs.
# Parent tactical core: v20.
# Historical core notes:
# alakazam741_v3 - v2 + 67戦の実ラダーログ全数分析(sub54523210)に基づく改善:
#   [P0-1] 致死維持ゲート: 実ログで「致死圏なのに手札消費プレイで圏外に落ちる/攻撃せず
#          ターンを終える」パターンを多数検出。ただし即攻撃の強制はA/Bで悪化(41.5% vs v2)。
#          Powerful Handは手札を消費しない → 「ドロー/展開で手札を伸ばしてから終端で攻撃」
#          が正解。→ 手札を減らす行動は実行後も致死維持できる時のみ許可(_hand_delta)。
#          致死ギリギリではKO攻撃30000で必ず〆る。勝利KOは常に90000。マージン有はKO=6000
#          (夜のタンカ等に割り込まれてKOを逃す事故だけ防ぐ)。
#   [P0-2] 山札予算: デッキ切れ負け5件。フロア max(5,サイド+2)→max(8,サイド+3)、
#          Run Away Draw高手札ガード強化(手札12+&山札14-)、低山札時はACTIVATE(任意ドロー)辞退。
#   [P0-3] 聖なる灰を山札回復札として昇格(山札14枚以下+トラッシュにライン3枚以上→12000)。
#   [P0-4] _item_locked() をMAINコンテキスト限定に(サブ選択での常時誤検知を修正)。
#   [P1-1] デッキ: +2クセロシキ(→3, ミラー対策=一位デッキ準拠), +1バトルケージ(→2, 敵スタジアム
#          張り替え用) / -1夜のタンカ(→2), -1ヒカリ(→3), -1ポケパッド(→3)。
#          ※フーディン245/シェイミ343の投入はA/Bで悪化したため見送り(コードの対応ロジックは
#            温存: crustle 83%→99%はハンマー4枚維持の方が効いた)。同名カードは合計4枚まで
#            (743+245で5枚は不可)という制約も確認済み。
#   [P1-2] 敵スタジアム(ロケット団の監視塔=無色特性無効でノコッチ停止, Full Metal Lab等)を
#          バトルケージで即張り替え(12500)。
#   [P1-3] ミラー対策: 相手フーディンライン(ケーシィ/ユンゲラー/フーディン)のKO価値+300、
#          クセロシキ発動閾値緩和(ミラーは相手手札6+で最優先)。
#   [A/B結果] vs v2: 64.2%(500戦) / gen-alakazam 94% / crustle 92% / kangaskhan 96% /
#          grimmsnarl 81% / megastarmie 66% / vs v1 76%。ラダーメタはミラー31%が最大勢力。
# Base: alakazam741_v2 (wmh/ptcg-abc alakazam v3 divergence-mined). Self-contained.
from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict

from cg.api import (
    AreaType, Card, CardType, EnergyType, Observation, OptionType, Pokemon,
    SelectContext, all_card_data, all_attack, to_observation_class,
)


# ── Card IDs (胡地小人 / Alakazam + Dudunsparce single-prize) ─────────────────
class C:
    ABRA = 741            # Basic -> Kadabra
    KADABRA = 742         # Stage1 (Psychic Draw on evolve) -> Alakazam
    ALAKAZAM = 743        # Stage2 attacker: Powerful Hand = 20 dmg x cards in hand
    ALAKAZAM_PSY = 245    # Stage2 TECH (1x): Psychic = 10 + 50/energy on opp Active.
                          # It does DAMAGE (not counters) -> bypasses Mist Energy; punishes
                          # energy-loaded ex. Our answer to Mist decks (Dragapult/Crustle).
    DUNSPARCE = 305       # Basic -> Dudunsparce (7-06: switched to id305 per ladder-#1 Majkel1337's
                          # list — 70HP + Trading Places free switch; the attack-id constants
                          # 423/424 below always belonged to THIS printing, not id65)
    DUDUNSPARCE = 66      # Stage1 draw engine (Run Away Draw)
    FEZANDIPITI_EX = 140  # Two-prize draw engine; Cruel Arrow is a matchup-only route
    PSYDUCK = 858         # Damp (ability lock tech)
    SHAYMIN = 343         # Flower Curtain (protect non-Rule-Box bench)
    GENESECT = 142        # ACE Nullifier (with tool)

    PSYCHIC_ENERGY = 5
    TELEPATH_ENERGY = 19  # special, provides {P}
    ENRICHING_ENERGY = 13 # ACE SPEC energy
    MIST_ENERGY = 11      # prevents Powerful Hand's damage-counter effect

    BUDDY_POFFIN = 1086
    POKE_PAD = 1152
    HILDA = 1225          # Supporter: search Evolution + Energy
    DAWN = 1231           # Supporter: search Basic+Stage1+Stage2
    RARE_CANDY = 1079
    BOSS_ORDERS = 1182
    XEROSIC = 1197        # v2: 相手は手札が3枚になるまで捨てる(ミラーのPowerful Hand潰し)
    NIGHTTIME_MINE = 1266 # Stadium: Tera attacks cost one extra {C}
    ENHANCED_HAMMER = 1081  # Item: discard a Special Energy from opp (e.g. Mist Energy)
    LUCKY_HELMET = 1156   # Tool: draw 2 when damaged
    WONDROUS_PATCH = 1146
    NIGHT_STRETCHER = 1097
    SACRED_ASH = 1129
    LANA_AID = 1184


POWERFUL_HAND = 1072   # Alakazam 743: place 2 counters (20 dmg) per card in hand, on opp Active
PSYCHIC_ATK = 339      # Alakazam 245: 10 + 50 per energy on opp Active (DAMAGE; bypasses Mist)
STRANGE_HACKING = 338  # Alakazam 245: confuse + move opp's damage counters around
SUPER_PSY_BOLT = 1071  # Kadabra: 30
ALAKAZAM_IDS = {743, 245}   # both Stage-2 Alakazam attackers (Powerful Hand / Psychic)
ABRA_TELEPORT = 1070   # Abra: 10 + switch
DUDUN_LAND_CRUSH = 76  # Dudunsparce: 90 (rarely; engine instead)
DUNSPARCE_TRADE = 423  # Dunsparce: switch
DUNSPARCE_RAM = 424
FEZANDIPITI_ATTACK = 183  # Cruel Arrow: 100 to any opposing Pokemon for {C}{C}{C}

ENERGY_TYPES = {C.PSYCHIC_ENERGY, C.TELEPATH_ENERGY, C.ENRICHING_ENERGY}
ATTACKER_IDS = {C.ALAKAZAM, C.KADABRA}
ONE_ENERGY_PIVOT_IDS = {
    C.FEZANDIPITI_EX,
    C.DUNSPARCE,
    C.SHAYMIN,
    C.ABRA,
    C.KADABRA,
}
LOW_DECK_COUNT = 6
ENERGY_DIG_MIN_PROBABILITY = 0.50
TERMINAL_BOSS_SCORE = 1_000_000
BLOCKED_BY_TERMINAL_BOSS_SCORE = -1_000_000
V21_TARGET_ROUTES_ENABLED = os.environ.get(
    "ALAKAZAM_V21_TARGET_ROUTES",
    os.environ.get("ALAKAZAM_V20_TARGET_ROUTES", "1"),
) != "0"
V21_HAND_GATE_ENABLED = os.environ.get(
    "ALAKAZAM_V21_HAND_GATE",
    os.environ.get("ALAKAZAM_V20_HAND_GATE", "1"),
) != "0"
V21_HAND_SURPLUS = max(
    0,
    int(
        os.environ.get(
            "ALAKAZAM_V21_HAND_SURPLUS",
            os.environ.get("ALAKAZAM_V20_HAND_SURPLUS", "2"),
        )
    ),
)
pre_turn = -1


def _diag_template():
    return {
        "decisions": 0,
        "policy_ok": 0,
        "policy_fallback": 0,
        "obs_fallback": 0,
        "deck_returns": 0,
        "errors": {},
        "hammer_mist_targets": 0,
        "hammer_non_mist_targets": 0,
        "hammer_reserve_decisions": 0,
        "hammer_mist_memory_reserves": 0,
        "mist_seen_max": 0,
        "boss_active_ko_dominance_blocks": 0,
        "boss_reachable_active_ko_blocks": 0,
        "boss_prevented_target_blocks": 0,
        "boss_effect_lock_escapes": 0,
        "terminal_boss_opportunities": 0,
        "terminal_boss_plays": 0,
        "target_route_plans": 0,
        "target_route_boss_actions": 0,
        "target_satisfied_draw_blocks": 0,
        "dudun_continuity_draws": 0,
        "progress_attack_end_blocks": 0,
        "abra_switch_sacrifice_choices": 0,
        "ko_promotion_attacker_choices": 0,
        "ko_promotion_shield_choices": 0,
        "fez_late_bench_blocks": 0,
        "fez_recovery_exceptions": 0,
        "core_line_target_bonuses": 0,
        "active_route_plans": 0,
        "active_route_hammer_actions": 0,
        "active_route_draw_actions": 0,
        "terminal_win_attack_gates": 0,
        "terminal_pivot_abilities": 0,
        "dudun_block_only_body": 0,
        "dudun_block_no_bench": 0,
        "dudun_block_deck_floor": 0,
        "dudun_block_ability_lock": 0,
        "dudun_declined_by_score": 0,
        "dudun_used_for_terminal_pivot": 0,
        "protected_attack_fallbacks": 0,
        "attack_first_energy_attachments": 0,
        "enriching_attachments": 0,
        "enriching_cycle_attachments": 0,
        "fez_draw_only_actions": 0,
        "fez_pivot_actions": 0,
        "fez_alternate_attacker_actions": 0,
        "fez_do_not_bench_blocks": 0,
        "fez_active_stall_turns": 0,
        "support_pivot_attach_actions": 0,
        "support_pivot_retreat_actions": 0,
        "kadabra_dual_active_choices": 0,
        "kadabra_dual_bench_choices": 0,
        "boss_same_turn_plays": 0,
        "boss_two_hit_plays": 0,
        "emergency_energy_draws": 0,
        "emergency_energy_draw_declines": 0,
        "candy_first_attack_routes": 0,
        "candy_immediate_ko_routes": 0,
        "fez_energy_investments": 0,
        "fez_pivot_conversions": 0,
        "fez_attack_conversions": 0,
        "shaymin_threat_plays": 0,
        "xerosic_nonmirror_plays": 0,
        "attack_lock_retreats": 0,
        "temporary_immunity_heads": 0,
        "temporary_immunity_tails": 0,
        "temporary_immunity_blocks": 0,
        "pivot_route_attachments": 0,
        "pivot_route_retreats": 0,
        "articuno_breaker_entries": 0,
        "articuno_breaker_promotions": 0,
    }


_DIAG = _diag_template()
_V9_STATE = {
    "turn": None,
    "fez_active_serial": None,
    "fez_active_turns": 0,
    "attack_disabled_key": None,
    "attack_disabled_turn": None,
    "attack_disable_event": None,
    "temporary_immunity_key": None,
    "temporary_immunity_turn": None,
    "temporary_immunity_event": None,
    "temporary_immunity_pending": None,
    "articuno_breaker_armed": False,
    "planned_hammer_target_key": None,
    "own_ko_turn": None,
    # Public-information memory only. Card serials remain stable when a visible
    # Mist Energy moves from the field to the discard pile, so this counts
    # distinct copies without peeking at the opponent's hand or deck.
    "mist_seen_serials": set(),
}


def _diag_record_error(exc):
    k = type(exc).__name__ + ": " + str(exc)[:160]
    _DIAG["errors"][k] = _DIAG["errors"].get(k, 0) + 1


def diag_reset():
    _DIAG.clear()
    _DIAG.update(_diag_template())
    _V9_STATE.update({
        "turn": None, "fez_active_serial": None, "fez_active_turns": 0,
        "attack_disabled_key": None, "attack_disabled_turn": None,
        "attack_disable_event": None,
        "temporary_immunity_key": None, "temporary_immunity_turn": None,
        "temporary_immunity_event": None,
        "temporary_immunity_pending": None,
        "articuno_breaker_armed": False,
        "planned_hammer_target_key": None,
        "own_ko_turn": None,
        "mist_seen_serials": set(),
    })


def diag_snapshot():
    s = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DIAG.items()}
    s["fallback_rate"] = (s.get("policy_fallback", 0) + s.get("obs_fallback", 0)) / max(1, s["decisions"])
    return s


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
MY_DECK_COUNTS = Counter(my_deck)

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# Active-ability Item-lock cards (Tyranitar / Jellicent ex …). Some lock cards
# (e.g. Budew) carry the effect without an exposed skill, so we ALSO detect lock
# from game state (hold Items but none playable) — see AlakazamPolicy._item_locked.
ITEM_LOCK_IDS = set()
for _c in all_card:
    for _s in (_c.skills or []):
        _t = (_s.text or '')
        if 'Item' in _t and 'Active Spot' in _t and 'play' in _t and ('opponent' in _t or 'neither' in _t):
            ITEM_LOCK_IDS.add(_c.cardId)

# CRITICAL for Alakazam: Powerful Hand "places damage counters" = an EFFECT, so a
# target that "prevents all effects of attacks done to it" takes 0 from it.
#   - special energies that grant this (Mist Energy 11, Rock Fighting Energy 20)
#   - Pokémon/Tools whose own ability prevents effects of attacks done to itself
EFFECT_PREVENT_ENERGY = set()
EFFECT_PREVENT_SELF = set()
for _c in all_card:
    _ct = _c.cardType
    for _s in (_c.skills or []):
        _t = (_s.text or '')
        if 'effects of attacks' in _t and 'prevent' in _t.lower():
            if _ct in (CardType.SPECIAL_ENERGY, CardType.BASIC_ENERGY):
                EFFECT_PREVENT_ENERGY.add(_c.cardId)
            elif 'to this Pokémon' in _t or 'to this Pok' in _t:
                EFFECT_PREVENT_SELF.add(_c.cardId)

# GENERAL energy rule: attach only what an attack costs — never over-fill — UNLESS the attack
# scales with energy attached to ITSELF (then more = more damage). Disruption (energy removal)
# is handled automatically: it drops the count back below the need, so we just refill.
ATTACK_COST = {}                 # attackId -> number of energies in its cost
ATTACK_COST_ENERGIES = {}        # attackId -> list of required EnergyType (0=Colorless, 5=Psychic…)
SELF_SCALING_ATTACKS = set()     # attacks whose damage grows with energy on the attacker
ATTACK_TABLE = {}
for _a in all_attack():
    ATTACK_TABLE[_a.attackId] = _a
    ATTACK_COST[_a.attackId] = len(_a.energies or [])
    ATTACK_COST_ENERGIES[_a.attackId] = list(_a.energies or [])
    _t = (_a.text or '').lower()
    if 'for each' in _t and 'energy attached to this' in _t:
        SELF_SCALING_ATTACKS.add(_a.attackId)

# What TYPE each energy card provides (Enriching -> Colorless 0; Telepath/Basic {P} -> Psychic 5).
# Critical: attaching energy must satisfy the attack's TYPE requirement, not just its count.
ENERGY_PROVIDES = {}
for _c in all_card:
    if _c.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
        ENERGY_PROVIDES[_c.cardId] = getattr(_c, 'energyType', 0)

# Situational-tech triggers (only bench the tech when the opponent's board warrants it):
#   Shaymin (Flower Curtain) matters ONLY vs bench-damage (spread/snipe) attacks;
#   Psyduck (Damp) matters ONLY vs abilities that require KO-ing the user itself.
BENCH_DAMAGE_ATTACKS = set()
for _a in all_attack():
    _t = (_a.text or '').lower()
    if ('benched' in _t and 'damage' in _t) or ('to each of your opponent' in _t and 'damage' in _t):
        BENCH_DAMAGE_ATTACKS.add(_a.attackId)
SELF_KO_ABILITY_IDS = set()
for _c in all_card:
    for _s in (_c.skills or []):
        _t = (_s.text or '').lower()
        if 'knock out' in _t and ('this pokémon' in _t or 'this pokemon' in _t or 'itself' in _t):
            SELF_KO_ABILITY_IDS.add(_c.cardId)

# v8 Boss targets use the card IDs observed in the current 1.32 replay corpus.
# v7 still treated 675 as Team Rocket's Articuno even though it is now Lunatone;
# that stale mapping caused three low-value ladder gusts. Generic card-text and
# stage checks below remain the primary mechanism.
ROCKET_ARTICUNO_ID = 414
FROSLASS_ID = 104
GRIMMSNARL_EX_ID = 648
# Public-replay deck signatures.  These are used only to estimate whether an
# unseen Mist Energy is likely enough to reserve the final Enhanced Hammer.
MIST_HIGH_SIGNATURE_IDS = {
    344, 345,       # Dwebble / Crustle: 179 of Majkel's 184 Mist games
    878, 879, 304,  # Hop's Phantump / Trevenant / Snorlax Mist variants
}
MIST_MEDIUM_SIGNATURE_IDS = {
    1030, 1031,     # Mega Starmie variants sometimes include Mist
}
SPIDOPS_SIGNATURE_IDS = {400, 401, 414, 431, 432, 434}
BOSS_KEY_ROLE_BONUS = {
    ROCKET_ARTICUNO_ID: 7000,  # remove Rocket protection before attacking the core
    FROSLASS_ID: 3600,         # stop repeated Ability damage-counter pressure
    678: 3200,                 # Mega Lucario ex
    666: 3200,                 # Cinderace pressure attacker
    140: 2600,                 # Fezandipiti ex draw engine / two-prize target
    112: 1800,                 # Munkidori damage-moving engine
    121: 3000,                 # Dragapult ex
    345: 3000,                 # Crustle (Mist matchup signature / wall)
    756: 3000,                 # Hop's Trevenant main attacker
    878: 2200,                 # Hop's Phantump
    879: 3000,                 # Hop's Trevenant
    647: 2200,                 # Marnie's Morgrem
    GRIMMSNARL_EX_ID: 3000,
    C.ABRA: 900,
    C.KADABRA: 2600,
    C.ALAKAZAM: 3000,
    C.ALAKAZAM_PSY: 3000,
}

# Card IDs are useful for known metagame engines, but they cannot cover every
# ladder deck.  Build an evolution-family index from public card metadata so a
# newly seen Duraludon/Riolu/Abra can inherit the strategic value of the
# Archaludon/Lucario/Alakazam line it enables.
_EVOLVES_FROM_BY_NAME = {}
for _card in all_card:
    _name = (getattr(_card, "name", "") or "").strip().casefold()
    _parent = (getattr(_card, "evolvesFrom", "") or "").strip()
    if _name and _parent:
        _EVOLVES_FROM_BY_NAME.setdefault(_name, _parent)


def _evolution_root_name(card_data):
    if card_data is None:
        return ""
    name = (getattr(card_data, "name", "") or "").strip()
    parent = (getattr(card_data, "evolvesFrom", "") or "").strip()
    seen = set()
    while parent and parent.casefold() not in seen:
        seen.add(parent.casefold())
        name = parent
        parent = _EVOLVES_FROM_BY_NAME.get(parent.casefold(), "")
    return name.casefold()


EVOLUTION_ROOT_BY_ID = {
    card.cardId: _evolution_root_name(card)
    for card in all_card
    if _evolution_root_name(card)
}
EVOLUTION_LINE_CEILING = defaultdict(int)
for _card in all_card:
    _root = EVOLUTION_ROOT_BY_ID.get(_card.cardId, "")
    if not _root:
        continue
    _ceiling = (
        5200 if getattr(_card, "megaEx", False)
        else 4000 if getattr(_card, "ex", False)
        else 3000 if getattr(_card, "stage2", False)
        else 1200 if getattr(_card, "stage1", False)
        else 400
    )
    EVOLUTION_LINE_CEILING[_root] = max(
        EVOLUTION_LINE_CEILING[_root], _ceiling
    )


def _norm_card_text(value):
    return (value or "").replace("’", "'").lower()


# Preserve each protection Ability's actual scope. Team Rocket's Articuno, for
# example, protects Basic Team Rocket Pokémon only; the old set lost that scope
# and incorrectly treated Evolutions as protected (and sometimes gusted a
# protected Articuno into the Active spot).
_GLOBAL_EFFECT_PROTECTOR_IDS = set()
for _c in all_card:
    for _s in (_c.skills or []):
        _t = _norm_card_text(_s.text)
        if ('prevent all effects of attacks' in _t and 'done to your' in _t
                and 'to this pok' not in _t):
            _GLOBAL_EFFECT_PROTECTOR_IDS.add(_c.cardId)

GLOBAL_EFFECT_PROTECTORS = {}
for _c in all_card:
    if _c.cardId not in _GLOBAL_EFFECT_PROTECTOR_IDS:
        continue
    for _s in (_c.skills or []):
        _t = _norm_card_text(_s.text)
        if ('prevent all effects of attacks' not in _t or 'done to your' not in _t
                or 'to this pok' in _t):
            continue
        _scope = _t.split('done to your', 1)[1]

        def _make_protection_predicate(scope):
            def predicate(target_data):
                if 'benched' in scope:
                    return False
                if target_data is None:
                    return True
                if ('basic' in scope
                        and (getattr(target_data, 'stage1', False)
                             or getattr(target_data, 'stage2', False))):
                    return False
                if ('team rocket' in scope
                        and 'team rocket' not in _norm_card_text(
                            getattr(target_data, 'name', ''))):
                    return False
                return True
            return predicate

        GLOBAL_EFFECT_PROTECTORS[_c.cardId] = _make_protection_predicate(_scope)


# These effects are not exposed as durable status flags.  Recover them from the
# public attack/coin log and bind them to the exact in-play Pokemon identity.
ATTACK_DISABLE_ATTACKS = set()
TEMPORARY_ATTACK_IMMUNITY_ATTACKS = {}
for _attack_id, _attack in ATTACK_TABLE.items():
    _text = _norm_card_text(getattr(_attack, "text", ""))
    if ("during your opponent's next turn" in _text
            and "defending pok" in _text
            and ("can't use attacks" in _text or "cannot use attacks" in _text)):
        ATTACK_DISABLE_ATTACKS.add(_attack_id)
    _full_immunity = (
        "during your opponent's next turn" in _text
        and "prevent all damage from and effects of attacks done to this pok" in _text
    )
    if not _full_immunity:
        continue
    if "flip a coin" in _text and "if heads" in _text:
        TEMPORARY_ATTACK_IMMUNITY_ATTACKS[_attack_id] = "heads"
    elif _text.startswith("during your opponent's next turn"):
        TEMPORARY_ATTACK_IMMUNITY_ATTACKS[_attack_id] = "always"
TEMPORARY_ATTACK_IMMUNITY_ATTACKS.setdefault(1266, "heads")


# ── generic helpers (proven scaffolding) ─────────────────────────────────────
def normalize_selection(ranked, scores, select):
    n = len(select.option)
    minc = max(0, min(select.minCount, n)); maxc = max(minc, min(select.maxCount, n))
    out, seen = [], set()
    for i in ranked:
        if not (0 <= i < n) or i in seen:
            continue
        s = scores[i] if i < len(scores) else 0
        if s > 0 or len(out) < minc:
            out.append(i); seen.add(i)
        if len(out) >= maxc:
            break
    for i in range(n):
        if len(out) >= minc:
            break
        if i not in seen:
            out.append(i); seen.add(i)
    return out


def _legal_fallback(select):
    try:
        n = len(select.option); return list(range(min(max(0, select.minCount), n)))
    except Exception:
        return []


def _legal_fallback_from_dict(obs_dict):
    try:
        sel = obs_dict.get("select") or {}
        return list(range(min(max(0, sel.get("minCount", 0)), len(sel.get("option") or []))))
    except Exception:
        return []


def _safe_get(seq, i):
    try:
        if seq is None or i is None or i < 0 or i >= len(seq):
            return None
        return seq[i]
    except Exception:
        return None


def get_card(obs, area, index, pi):
    try:
        player = obs.current.players[pi]
        match area:
            case AreaType.DECK: return _safe_get(getattr(obs.select, "deck", None), index)
            case AreaType.HAND: return _safe_get(getattr(player, "hand", None), index)
            case AreaType.DISCARD: return _safe_get(getattr(player, "discard", None), index)
            case AreaType.ACTIVE: return _safe_get(getattr(player, "active", None), index)
            case AreaType.BENCH: return _safe_get(getattr(player, "bench", None), index)
            case AreaType.PRIZE: return _safe_get(getattr(player, "prize", None), index)
            case AreaType.STADIUM: return _safe_get(getattr(obs.current, "stadium", None), index)
            case AreaType.LOOKING: return _safe_get(getattr(obs.current, "looking", None), index)
            case _: return None
    except Exception:
        return None


def prize_count(p):
    d = card_table.get(p.id)
    return (3 if d.megaEx else 2 if d.ex else 1) if d else 1


def is_energy(cid):
    d = card_table.get(cid)
    return cid in ENERGY_TYPES or (d is not None and d.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY))


def _pokemon_key(pokemon):
    if pokemon is None:
        return None
    return (getattr(pokemon, "serial", None), getattr(pokemon, "id", None))


# ── Alakazam policy ──────────────────────────────────────────────────────────
class AlakazamPolicy:
    def __init__(self, obs: Observation):
        self.obs = obs
        self.state = obs.current
        self.select = obs.select
        self.context = self.select.context
        self.my_index = self.state.yourIndex
        self.op_index = 1 - self.my_index
        self.me = self.state.players[self.my_index]
        self.opponent = self.state.players[self.op_index]
        self.stadium_id = self.state.stadium[0].id if self.state.stadium else 0
        self.field = defaultdict(int)
        self.hand = defaultdict(int)
        self.discard = defaultdict(int)
        for p in self._my_board():
            if p is not None:
                self.field[p.id] += 1
        for c in self.me.hand:
            self.hand[c.id] += 1
        for c in self.me.discard:
            self.discard[c.id] += 1
        # A policy object represents one immutable selection state. Route planning
        # is deliberately cached because every offered option asks the same small
        # bitmask search several times while being scored.
        self._ko_route_cache = {}
        self._chosen_ko_plan_cache = None
        self._chosen_ko_plan_cached = False

    def _my_board(self):
        return self.me.active + self.me.bench

    def _board_body_count(self):
        """Number of Pokémon that will remain as legal board bodies.

        Run Away Draw shuffles Dudunsparce itself away. Using it while this count is
        one loses immediately, regardless of Item lock, deck pressure, or whether the
        Dudunsparce is Active. This is an absolute safety invariant in v6.
        """
        return sum(1 for p in self._my_board() if p is not None)

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

    def _visible_own_count(self, card_id):
        count = self.hand.get(card_id, 0) + self.discard.get(card_id, 0)
        for pokemon in self._my_board():
            if pokemon is None:
                continue
            count += int(pokemon.id == card_id)
            count += sum(getattr(card, "id", None) == card_id
                         for card in (getattr(pokemon, "preEvolution", None) or []))
            count += sum(getattr(card, "id", None) == card_id
                         for card in (getattr(pokemon, "energyCards", None) or []))
        return count

    @staticmethod
    def _at_least_one_probability(population, successes, draws):
        population = max(0, int(population))
        successes = max(0, min(int(successes), population))
        draws = max(0, min(int(draws), population))
        if draws == 0 or successes == 0:
            return 0.0
        failures = population - successes
        if failures < draws:
            return 1.0
        return 1.0 - math.comb(failures, draws) / math.comb(population, draws)

    def _draw_probability(self, card_id, draws):
        remaining = max(0, MY_DECK_COUNTS.get(card_id, 0) - self._visible_own_count(card_id))
        unseen = self.me.deckCount + len(self.me.prize)
        return self._at_least_one_probability(unseen, remaining, min(draws, self.me.deckCount))

    def _energy_draw_probability(self, draws):
        psychic_ids = [
            card_id for card_id in MY_DECK_COUNTS
            if ENERGY_PROVIDES.get(card_id) == EnergyType.PSYCHIC
        ]
        remaining = sum(
            max(0, MY_DECK_COUNTS[card_id] - self._visible_own_count(card_id))
            for card_id in psychic_ids
        )
        unseen = self.me.deckCount + len(self.me.prize)
        return self._at_least_one_probability(unseen, remaining, min(draws, self.me.deckCount))

    def _opponent_can_ko_active(self):
        opponent = self.opponent.active[0] if self.opponent.active else None
        active = self.me.active[0] if self.me.active else None
        if opponent is None or active is None or not self._can_attack(opponent):
            return False
        if opponent.id == C.ALAKAZAM:
            return 20 * getattr(self.opponent, "handCount", 0) >= active.hp
        data = card_table.get(opponent.id)
        return any(
            ATTACK_TABLE.get(attack_id) is not None
            and getattr(ATTACK_TABLE[attack_id], "damage", 0) >= active.hp
            for attack_id in (getattr(data, "attacks", None) or [])
        )

    def _emergency_energy_draw(self, draws=3):
        if getattr(self.state, "energyAttached", False) or self._psychic_in_hand():
            return False
        if not self._energy_starved():
            return False
        if not any(p is not None and p.id in ALAKAZAM_IDS for p in self._my_board()):
            return False
        probability = self._energy_draw_probability(draws)
        urgent = len(self.opponent.prize) <= 2 or self._opponent_can_ko_active()
        return (self.me.deckCount <= 8 and urgent
                and probability >= ENERGY_DIG_MIN_PROBABILITY and self.me.deckCount >= 2)

    def _activate_draw_count(self):
        effect_id = getattr(self._context_effect_card(), "id", None)
        if effect_id == C.KADABRA:
            return 2
        if effect_id in (C.ALAKAZAM, C.DUDUNSPARCE, C.FEZANDIPITI_EX):
            return 3
        if effect_id == C.ENRICHING_ENERGY:
            return 4
        return 3

    def _activate_draw_ok(self):
        draws = self._activate_draw_count()
        return self._emergency_energy_draw(draws) or self._deck_spend_ok(
            cost=draws, allow_lethal=True
        )

    def _hand_size(self):
        return self.me.handCount

    def _energy_count(self, p):
        return len(p.energies) if p is not None else 0

    @staticmethod
    def _can_pay(attached, cost):
        """Can `attached` (list of EnergyType) pay `cost` (list of EnergyType, 0=Colorless)?
        Specific-type requirements must be met by that exact type; Colorless by anything left."""
        from collections import Counter
        have = Counter(attached)
        colorless = 0
        for req in cost:
            if req == EnergyType.COLORLESS:
                colorless += 1
            elif have.get(req, 0) > 0:
                have[req] -= 1
            else:
                return False            # e.g. a Psychic requirement with only Colorless attached
        return sum(have.values()) >= colorless

    def _can_attack(self, p):
        """TYPE-AWARE: can p actually pay one of its attacks with its currently attached
        energy? (1 Enriching = Colorless does NOT pay Powerful Hand's Psychic cost.)"""
        c = card_table.get(p.id)
        if c is None:
            return False
        attached = list(p.energies or [])
        return any(aid in ATTACK_COST_ENERGIES and self._can_pay(attached, ATTACK_COST_ENERGIES[aid])
                   for aid in (c.attacks or []))

    def _should_fuel(self, p):
        """Attach more energy ONLY while p still can't pay an attack (type-aware), so we never
        over-fill — UNLESS an attack scales with its own energy (then keep attaching)."""
        c = card_table.get(p.id)
        if c is None or not (c.attacks or []):
            return False
        if any(aid in SELF_SCALING_ATTACKS for aid in c.attacks):
            return True
        return not self._can_attack(p)

    def _attach_helps(self, p, src):
        """Would attaching energy `src` actually let p pay an attack it currently can't?
        (A Colorless Enriching onto a Psychic-needing Alakazam does NOT help -> don't waste it.)"""
        if src is None:
            return True
        prov = ENERGY_PROVIDES.get(src.id)
        if prov is None:
            return True
        new = list(p.energies or []) + [prov]
        c = card_table.get(p.id)
        return any(aid in ATTACK_COST_ENERGIES and self._can_pay(new, ATTACK_COST_ENERGIES[aid])
                   for aid in (c.attacks or []))

    def _opp_threatens_bench(self):
        """A payable opposing Active can KO a protected body within two attacks."""
        damage = self._opponent_ready_bench_damage()
        if damage <= 0:
            return False
        for pokemon in self.me.bench:
            if pokemon is None or pokemon.id == C.SHAYMIN:
                continue
            data = card_table.get(pokemon.id)
            rule_box = bool(data is not None and (
                getattr(data, "ex", False) or getattr(data, "megaEx", False)
            ))
            if not rule_box and getattr(pokemon, "hp", 0) <= 2 * damage:
                return True
        return False

    def _opp_has_tera(self):
        return any(
            bool(card_table.get(p.id) and getattr(card_table[p.id], "tera", False))
            for p in self.opponent.active + self.opponent.bench
            if p is not None
        )

    def _nighttime_mine_tax_stops_active(self):
        active = self.opponent.active[0] if self.opponent.active else None
        data = card_table.get(active.id) if active is not None else None
        if data is None or not getattr(data, "tera", False):
            return False
        attached = [
            ENERGY_PROVIDES.get(getattr(card, "id", None), EnergyType.COLORLESS)
            for card in (getattr(active, "energyCards", None) or [])
        ]
        if not attached:
            attached = list(getattr(active, "energies", None) or [])
        return any(
            self._can_pay(attached, ATTACK_COST_ENERGIES.get(attack_id, []))
            and not self._can_pay(
                attached,
                ATTACK_COST_ENERGIES.get(attack_id, []) + [EnergyType.COLORLESS],
            )
            for attack_id in (data.attacks or [])
        )

    def _nighttime_mine_worthwhile(self):
        if self.state.stadiumPlayed or self.stadium_id == C.NIGHTTIME_MINE:
            return False
        if self.me.deckCount <= 1:
            return False

        # Keep the card when spending it would drop Powerful Hand below the
        # current Active KO. The Stadium has no value after a winning attack.
        active = self.me.active[0] if self.me.active else None
        opponent = self.opponent.active[0] if self.opponent.active else None
        if (
            active is not None
            and active.id == C.ALAKAZAM
            and opponent is not None
            and self._can_attack(active)
            and 20 * max(0, self.me.handCount - 1) < opponent.hp
            <= 20 * self.me.handCount
        ):
            return False
        if self._opp_has_tera():
            return True

        # An opposing Stadium is worth replacing only after the opponent has
        # developed an attacker; otherwise retain one Powerful Hand card.
        return bool(
            self.stadium_id
            and any(
                p is not None and self._can_attack(p)
                for p in self.opponent.active + self.opponent.bench
            )
        )

    @staticmethod
    def _bench_damage_amount(attack):
        text = (getattr(attack, "text", "") or "").lower()
        values = [
            int(match.group(1))
            for match in re.finditer(
                r"(\d+) damage to (?:1 of |each of )?your opponent['’]s benched", text
            )
        ]
        return max(values, default=0)

    def _attack_payable_after_one(self, pokemon, attack_id):
        cost = ATTACK_COST_ENERGIES.get(attack_id, [])
        attached = list(getattr(pokemon, "energies", None) or [])
        if self._can_pay(attached, cost):
            return True
        provisions = set(cost) | {EnergyType.COLORLESS}
        return any(self._can_pay(attached + [energy_type], cost)
                   for energy_type in provisions)

    def _opponent_ready_bench_damage(self):
        active = self.opponent.active[0] if self.opponent.active else None
        data = card_table.get(active.id) if active is not None else None
        if active is None or data is None:
            return 0
        best = 0
        for attack_id in (data.attacks or []):
            if attack_id not in BENCH_DAMAGE_ATTACKS:
                continue
            if not self._attack_payable_after_one(active, attack_id):
                continue
            best = max(best, self._bench_damage_amount(ATTACK_TABLE.get(attack_id)))
        return best

    def _opp_has_froslass(self):
        """Froslass stacks one damage counter on every Ability Pokemon at checkup."""
        return any(p is not None and p.id == FROSLASS_ID
                   for p in (self.opponent.active + self.opponent.bench))

    def _opp_has_self_ko_ability(self):
        """Opponent has an ability that KOs the user itself -> Psyduck (Damp) matters."""
        return any(p is not None and p.id in SELF_KO_ABILITY_IDS
                   for p in (self.opponent.active + self.opponent.bench))

    def _visible_opponent_ids(self):
        ids = {
            p.id for p in (self.opponent.active + self.opponent.bench)
            if p is not None
        }
        ids.update(c.id for c in (self.opponent.discard or []) if c is not None)
        for pokemon in (self.opponent.active + self.opponent.bench):
            if pokemon is None:
                continue
            ids.update(
                e.id for e in (getattr(pokemon, "energyCards", None) or [])
                if e is not None
            )
        return ids

    def _mist_seen_count(self):
        """Distinct publicly revealed Mist Energy copies seen this game."""
        return len(_V9_STATE.get("mist_seen_serials", set()))

    def _mist_probability(self):
        """Estimate whether another unseen Mist remains, using only public zones.

        Unlike v11, one discarded Mist no longer pins the probability at 1.0 for
        the rest of the game. The prior is archetype-aware and decays as distinct
        copies are revealed. This reserves only the *last* Hammer when justified.
        """
        visible = self._visible_opponent_ids()
        seen = self._mist_seen_count()
        if visible & MIST_HIGH_SIGNATURE_IDS:
            # Crustle / Hop Trevenant commonly expose a full four-copy package.
            return (0.90, 0.78, 0.58, 0.34, 0.0)[min(seen, 4)]
        if visible & MIST_MEDIUM_SIGNATURE_IDS:
            # Starmie variants are mixed and usually lighter on Mist.
            return (0.50, 0.32, 0.0)[min(seen, 2)]
        if seen >= 2:
            return 0.0
        if seen == 1:
            return 0.35
        return 0.08

    def _attached_special_energy_entries(self):
        entries = []
        board = ((AreaType.ACTIVE, self.opponent.active)
                 , (AreaType.BENCH, self.opponent.bench))
        for area, pokemon_list in board:
            for pokemon in (pokemon_list or []):
                if pokemon is None:
                    continue
                for energy in (getattr(pokemon, "energyCards", None) or []):
                    data = card_table.get(getattr(energy, "id", None))
                    if data is not None and data.cardType == CardType.SPECIAL_ENERGY:
                        entries.append((area, pokemon, energy))
        return entries

    def _removing_energy_stops_attack(self, pokemon, energy):
        if not self.opponent.active or pokemon is not self.opponent.active[0]:
            return False
        data = card_table.get(pokemon.id)
        if data is None or not self._can_attack(pokemon):
            return False
        attached = list(getattr(pokemon, "energies", None) or [])
        provided = ENERGY_PROVIDES.get(getattr(energy, "id", None), EnergyType.COLORLESS)
        if provided in attached:
            attached.remove(provided)
        elif attached:
            attached.pop()
        return not any(
            attack_id in ATTACK_COST_ENERGIES
            and self._can_pay(attached, ATTACK_COST_ENERGIES[attack_id])
            for attack_id in (data.attacks or [])
        )

    def _non_mist_hammer_exception(self):
        """A last Hammer may be spent early only for concrete immediate tempo."""
        for area, pokemon, energy in self._attached_special_energy_entries():
            if energy.id == C.MIST_ENERGY:
                continue
            if area == AreaType.ACTIVE and self._removing_energy_stops_attack(pokemon, energy):
                return True
            # Grow Grass supplies +20 HP. Removing it KOs a body with <=20 HP left.
            if energy.id == 18 and getattr(pokemon, "hp", 0) <= 20:
                return True
        return False

    def _should_reserve_last_hammer(self):
        attached = self._attached_special_energy_entries()
        if any(energy.id in EFFECT_PREVENT_ENERGY for _, _, energy in attached):
            return False
        if self._mist_probability() < 0.30:
            return False
        # Four Hammers are in the fixed v9 deck. Three in discard plus one in hand
        # means this is the actual final answer to a future Mist attachment.
        if self.hand[C.ENHANCED_HAMMER] != 1 or self.discard[C.ENHANCED_HAMMER] < 3:
            return False
        return not self._non_mist_hammer_exception()

    def _should_reserve_hammer_for_seen_mist(self):
        """After Mist is public, reserve every remaining Hammer for likely copies.

        v13 only protected the fourth Hammer, so the ladder agent spent earlier
        copies on Telepath/other Special Energy 12 times after Mist had already
        appeared.  Once the matchup has demonstrated Mist, Hammer's deck role is
        effect-lock removal.  Reservation ends when the archetype-aware public-copy
        estimate says no likely copy remains, or while a prevention Energy is
        currently attached and can be removed now.
        """
        if self._mist_seen_count() <= 0:
            return False
        attached = self._attached_special_energy_entries()
        if any(energy.id in EFFECT_PREVENT_ENERGY for _, _, energy in attached):
            return False
        return self._mist_probability() >= 0.30

    def _hammer_target_score(self, energy, pokemon, area):
        """Dedicated Hammer target head: Mist first, concrete tempo second."""
        if energy is None:
            return -1
        planned_target = _V9_STATE.get("planned_hammer_target_key")
        if planned_target is not None and _pokemon_key(pokemon) != planned_target:
            return -1
        planned_bonus = 50000 if planned_target is not None else 0
        active_bonus = 500 if area == AreaType.ACTIVE else 0
        if energy.id in EFFECT_PREVENT_ENERGY:
            mist_bonus = 4000 if energy.id == C.MIST_ENERGY else 2500
            return 16000 + mist_bonus + active_bonus + planned_bonus
        data = card_table.get(energy.id)
        if data is None or data.cardType != CardType.SPECIAL_ENERGY:
            return -1
        value = 500 + active_bonus
        text = " ".join(
            (getattr(skill, "text", "") or "").lower()
            for skill in (getattr(data, "skills", None) or [])
        )
        if getattr(data, "aceSpec", False):
            value += 3000
        if any(word in text for word in ("damage", "attach", "prevent", "cost")):
            value += 1000
        if pokemon is not None and self._removing_energy_stops_attack(pokemon, energy):
            value += 9000
        if energy.id == 18 and pokemon is not None and getattr(pokemon, "hp", 0) <= 20:
            value += 20000
        return value + planned_bonus

    def _opponent_can_ko_fez_next_turn(self):
        active = self.opponent.active[0] if self.opponent.active else None
        data = card_table.get(active.id) if active is not None else None
        if active is None or data is None or not self._can_attack(active):
            return False
        base_damage = max(
            (int(getattr(ATTACK_TABLE.get(aid), "damage", 0) or 0)
             for aid in (data.attacks or [])),
            default=0,
        )
        if base_damage >= 210:
            return True
        # Variable-damage ex attacks commonly report zero base damage. Requiring an
        # already powered ex keeps this conservative instead of guessing from hand.
        return bool(
            (getattr(data, "ex", False) or getattr(data, "megaEx", False))
            and self._energy_count(active) >= 2
        )

    def _fez_recovery_window(self):
        own_ko_turn = _V9_STATE.get("own_ko_turn")
        turn = getattr(self.state, "turn", None)
        return bool(
            own_ko_turn is not None
            and turn is not None
            and int(turn) in (int(own_ko_turn), int(own_ko_turn) + 1)
        )

    def _fez_draw_creates_ko(self):
        """Playing Fez (-1), then Flip the Script (+3), supplies net +2 hand."""
        if not self._fez_recovery_window() or self.me.deckCount < 3:
            return False
        active = self.me.active[0] if self.me.active else None
        target = self.opponent.active[0] if self.opponent.active else None
        if (
            active is None
            or active.id != C.ALAKAZAM
            or not self._can_attack(active)
            or target is None
            or self._effect_prevented(target)
        ):
            return False
        current = 20 * self.me.handCount
        after_draw = 20 * (self.me.handCount + 2)
        return current < target.hp <= after_draw

    def _fez_entry_urgent(self):
        if not self._fez_recovery_window() or not self._deck_spend_ok(cost=3):
            return False
        if self._fez_draw_creates_ko() or self._emergency_energy_draw(3):
            return True
        return bool(
            self._ready_alakazam_count() == 0
            and self.me.handCount <= 4
            and self.field[C.DUDUNSPARCE] == 0
        )

    def _fez_two_prize_exposure(self):
        remaining = len(self.opponent.prize)
        if remaining <= 3:
            return not self._fez_entry_urgent()
        return remaining <= 4 and self._opponent_can_ko_fez_next_turn()

    def _fez_alternate_matchup(self):
        visible = self._visible_opponent_ids()
        if visible & SPIDOPS_SIGNATURE_IDS:
            return True
        opponent = self.opponent.active[0] if self.opponent.active else None
        if (opponent is not None and self._effect_prevented(opponent)
                and self.hand[C.ENHANCED_HAMMER] == 0):
            return True
        return any(
            p is not None and getattr(p, "hp", 0) <= 100 and prize_count(p) >= 2
            for p in (self.opponent.active + self.opponent.bench)
        )

    def _fez_attack_goal(self):
        if self._articuno_breaker_mode():
            return True
        return any(
            p is not None and getattr(p, "hp", 0) <= 100
            and (prize_count(p) >= 2 or self._boss_role_bonus(p) >= 1000
                 or prize_count(p) >= len(self.me.prize))
            for p in (self.opponent.active + self.opponent.bench)
        )

    def _fez_energy_eta(self, pokemon):
        missing = max(0, 3 - self._energy_count(pokemon))
        energy_in_hand = sum(1 for card in (self.me.hand or []) if is_energy(card.id))
        return missing if energy_in_hand >= missing else 99

    def _fez_mode(self, pokemon=None, *, for_bench=False):
        """DO_NOT_BENCH / DRAW_ONLY / PIVOT / ALTERNATE_ATTACKER."""
        if self._articuno_breaker_mode():
            if for_bench or (pokemon is not None and pokemon.id == C.FEZANDIPITI_EX):
                return "ALTERNATE_ATTACKER"
        if for_bench and self._fez_two_prize_exposure():
            return "DO_NOT_BENCH"
        if pokemon is not None and pokemon.id == C.FEZANDIPITI_EX:
            if (self._fez_alternate_matchup() and self._fez_attack_goal()
                    and self._fez_energy_eta(pokemon) <= 1):
                return "ALTERNATE_ATTACKER"
            active = self.me.active[0] if self.me.active else None
            if active is pokemon and self._bench_attacker_ready():
                return "PIVOT"
        return "DRAW_ONLY"

    def _fez_bench_worthwhile(self):
        if self.field[C.FEZANDIPITI_EX] > 0 or not self._open_bench():
            return False
        if self._articuno_breaker_mode():
            return True
        if self._fez_mode(for_bench=True) == "DO_NOT_BENCH":
            return False
        if self._board_body_count() <= 1:
            return True
        open_slots = getattr(self.me, "benchMax", 5) - sum(
            p is not None for p in self.me.bench
        )
        reserve = int(
            self.field[C.ABRA] + self.field[C.KADABRA] + self.field[C.ALAKAZAM]
            + self.field[C.ALAKAZAM_PSY] == 0
        )
        reserve += int(self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE] == 0)
        return open_slots > reserve

    def _fez_target_score(self, pokemon):
        if pokemon is None:
            return -1
        if self._temporary_attack_immunity_applies(pokemon):
            return -1
        hp = getattr(pokemon, "hp", 0)
        winning = hp <= 100 and prize_count(pokemon) >= len(self.me.prize)
        ko = hp <= 100
        score = prize_count(pokemon) * 1200 + self._boss_role_bonus(pokemon)
        if pokemon.id == ROCKET_ARTICUNO_ID:
            score += 30000
        if winning:
            score += 90000
        elif ko:
            score += 22000
        else:
            score -= hp
        return score

    def _energy_in_hand(self):
        return any(is_energy(c.id) for c in self.me.hand)

    def _psychic_in_hand(self):
        """A {P}-providing energy in hand (the ONLY kind that fuels our attacks — Enriching's
        Colorless does not). 'Energy in hand' that is just Enriching still leaves us starved."""
        return any(ENERGY_PROVIDES.get(c.id) == EnergyType.PSYCHIC for c in self.me.hand)

    def _active_alakazam_can_be_fueled(self):
        """A hand attachment can turn the Active Alakazam into an attacker now."""
        active = self.me.active[0] if self.me.active else None
        if (active is None or active.id not in ALAKAZAM_IDS
                or self._can_attack(active)
                or getattr(self.state, "energyAttached", False)):
            return False
        return any(
            ENERGY_PROVIDES.get(getattr(card, "id", None)) == EnergyType.PSYCHIC
            and self._attach_helps(active, card)
            for card in (self.me.hand or [])
        )

    def _active_alakazam_attack_route_offered(self):
        """Final MAIN safety check: attack now, or fuel the Active to attack now."""
        active = self.me.active[0] if self.me.active else None
        if active is None or active.id not in ALAKAZAM_IDS:
            return False
        if any(option.type == OptionType.ATTACK for option in (self.select.option or [])):
            return True
        if getattr(self.state, "energyAttached", False):
            return False
        for option in (self.select.option or []):
            if option.type not in (OptionType.ATTACH, OptionType.ENERGY):
                continue
            if (getattr(option, "inPlayArea", None) != AreaType.ACTIVE
                    or getattr(option, "inPlayIndex", None) != 0):
                continue
            source = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
            if self._attach_helps(active, source):
                return True
        return False

    def _attack_disabled_on_active(self):
        """Whether the previous attack disabled this exact Active Pokemon."""
        active = self.me.active[0] if self.me.active else None
        remembered = _V9_STATE.get("attack_disabled_key")
        if active is None or remembered is None or remembered != _pokemon_key(active):
            return False
        return not any(
            option.type == OptionType.ATTACK
            for option in (self.select.option or [])
        )

    def _energy_starved(self):
        """We have an Alakazam-line attacker in play (or a Kadabra + Alakazam in hand to
        evolve) that CAN'T attack, and no usable {P} energy in hand to fix it. With only 6
        energy in 60 cards, energy is the bottleneck — searches should grab a {P} energy."""
        bodies = [p for p in (self.me.active + self.me.bench) if p is not None]
        has_alakazam = any(p.id in ALAKAZAM_IDS for p in bodies)
        coming = any(p.id == C.KADABRA for p in bodies) and self.hand[C.ALAKAZAM] > 0
        if not (has_alakazam or coming):
            return False
        if any(p.id in ALAKAZAM_IDS and self._can_attack(p) for p in bodies):
            return False                       # already have an attacker that can actually attack
        return not self._psychic_in_hand()

    def _temporary_attack_immunity_applies(self, target):
        """Whether a previous attack fully protects this exact Active Pokemon."""
        if target is None:
            return False
        turn = getattr(self.state, "turn", None)
        valid_through = _V9_STATE.get("temporary_immunity_turn")
        if valid_through is None or turn is None or turn > valid_through:
            return False
        remembered = _V9_STATE.get("temporary_immunity_key")
        opponent_active = self.opponent.active[0] if self.opponent.active else None
        return bool(
            remembered is not None
            and remembered == _pokemon_key(target)
            and remembered == _pokemon_key(opponent_active)
        )

    def _non_energy_effect_prevented(self, target):
        """Whether a visible non-Energy card stops Powerful Hand's effect."""
        if target is None:
            return False
        if target.id in EFFECT_PREVENT_SELF:
            return True
        target_data = card_table.get(target.id)
        for protector in (self.opponent.active + self.opponent.bench):
            if protector is None:
                continue
            predicate = GLOBAL_EFFECT_PROTECTORS.get(protector.id)
            if predicate is not None and predicate(target_data):
                return True
        return False

    def _static_effect_prevented(self, target):
        """Whether visible cards statically stop Powerful Hand's attack effect."""
        if self._non_energy_effect_prevented(target):
            return True
        for e in (getattr(target, 'energyCards', None) or []):
            if getattr(e, 'id', None) in EFFECT_PREVENT_ENERGY:
                return True
        return False

    def _effect_prevented(self, target):
        """Whether Powerful Hand's damage-counter effect cannot affect ``target``."""
        return bool(
            target is not None
            and (self._temporary_attack_immunity_applies(target)
                 or self._static_effect_prevented(target))
        )

    def _rocket_articuno_in_play(self):
        return any(
            pokemon is not None and pokemon.id == ROCKET_ARTICUNO_ID
            for pokemon in (self.opponent.active + self.opponent.bench)
        )

    def _articuno_protection_applies(self, target):
        if target is None or not self._rocket_articuno_in_play():
            return False
        predicate = GLOBAL_EFFECT_PROTECTORS.get(ROCKET_ARTICUNO_ID)
        return bool(predicate is not None and predicate(card_table.get(target.id)))

    def _articuno_active_lock(self):
        active = self.opponent.active[0] if self.opponent.active else None
        return self._articuno_protection_applies(active)

    def _articuno_breaker_required(self):
        """All visible opposing targets are protected while Articuno is live."""
        if not self._articuno_active_lock():
            return False
        targets = [
            pokemon for pokemon in (self.opponent.active + self.opponent.bench)
            if pokemon is not None
        ]
        return bool(targets) and all(
            self._static_effect_prevented(pokemon) for pokemon in targets
        )

    def _articuno_breaker_mode(self):
        """Persist the Fez answer after an all-Basic Rocket lock is observed."""
        if not self._rocket_articuno_in_play():
            _V9_STATE["articuno_breaker_armed"] = False
            return False
        if self._articuno_breaker_required():
            if not _V9_STATE.get("articuno_breaker_armed"):
                _DIAG["articuno_breaker_entries"] += 1
            _V9_STATE["articuno_breaker_armed"] = True
        return bool(_V9_STATE.get("articuno_breaker_armed"))

    def _fez_breaker_energy_needed(self):
        if not self._articuno_breaker_mode():
            return False
        return any(
            pokemon is not None
            and pokemon.id == C.FEZANDIPITI_EX
            and self._energy_count(pokemon) < 3
            for pokemon in self._my_board()
        )

    def _articuno_escape_target(self, target):
        return bool(
            target is not None
            and self._articuno_active_lock()
            and not self._effect_prevented(target)
        )

    def _opp_active_has_prevent_energy(self):
        """Opponent's Active has Mist/Rock-Fighting special energy blocking Powerful
        Hand — Enhanced Hammer should strip it before we attack."""
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is None:
            return False
        return any(getattr(e, 'id', None) in EFFECT_PREVENT_ENERGY
                   for e in (getattr(opp, 'energyCards', None) or []))

    # — damage —
    def _alakazam_damage(self, attack_id, target):
        if target is None:
            return 0
        if self._temporary_attack_immunity_applies(target):
            return 0
        if attack_id == POWERFUL_HAND:
            if self._effect_prevented(target):
                return 0                     # Mist Energy etc. negates "place counters"
            return 20 * self._hand_size()    # counter placement -> no weakness
        if attack_id == FEZANDIPITI_ATTACK:
            # Cruel Arrow is a fixed 100 and may target the Bench, where Weakness
            # and Resistance are explicitly not applied.
            return 100
        if attack_id == PSYCHIC_ATK:
            # 245 Alakazam: 10 + 50 per energy on opp Active. This is DAMAGE, so it goes
            # THROUGH Mist Energy and applies Weakness — our answer to Mist/energy decks.
            dmg = 10 + 50 * len(target.energies)
        elif attack_id == SUPER_PSY_BOLT:
            dmg = 30
        elif attack_id == ABRA_TELEPORT:
            dmg = 10
        elif attack_id == DUNSPARCE_RAM:
            dmg = 20
        elif attack_id == DUDUN_LAND_CRUSH:
            dmg = 90
        else:
            dmg = 0
        od = card_table.get(target.id)
        if od is not None:
            if od.weakness == EnergyType.PSYCHIC:
                dmg *= 2
            elif od.resistance == EnergyType.PSYCHIC:
                dmg = max(0, dmg - 30)
        return dmg

    def _active_best_dmg(self, target):
        a = self.me.active[0] if self.me.active else None
        if a is None or target is None:
            return 0
        if a.id == C.FEZANDIPITI_EX and self._can_attack(a):
            return self._alakazam_damage(FEZANDIPITI_ATTACK, target)
        if self._energy_count(a) >= 1:
            if a.id == C.ALAKAZAM:
                return self._alakazam_damage(POWERFUL_HAND, target)
            if a.id == C.ALAKAZAM_PSY:
                return self._alakazam_damage(PSYCHIC_ATK, target)
            if a.id == C.KADABRA:
                return self._alakazam_damage(SUPER_PSY_BOLT, target)
        return 0

    @staticmethod
    def _route_retreat_cost(pokemon):
        data = card_table.get(getattr(pokemon, "id", None))
        return max(0, int(getattr(data, "retreatCost", 0) or 0)) if data else 0

    def _can_attack_after_attachment(self, pokemon, source=None):
        data = card_table.get(getattr(pokemon, "id", None))
        if data is None:
            return False
        attached = list(getattr(pokemon, "energies", None) or [])
        provided = ENERGY_PROVIDES.get(getattr(source, "id", None))
        if provided is not None:
            attached.append(provided)
        return any(
            attack_id in ATTACK_COST_ENERGIES
            and self._can_pay(attached, ATTACK_COST_ENERGIES[attack_id])
            for attack_id in (getattr(data, "attacks", None) or [])
        )

    def _route_attack_damage(self, attacker, target, *, hand_adjust=0, source=None):
        """Damage for a concrete post-pivot attacker in a small virtual state."""
        if attacker is None or target is None:
            return 0
        current_active = self.me.active[0] if self.me.active else None
        if (_pokemon_key(attacker) == _pokemon_key(current_active)
                and self._attack_disabled_on_active()):
            return 0
        if not self._can_attack_after_attachment(attacker, source):
            return 0
        if self._temporary_attack_immunity_applies(target):
            return 0
        if attacker.id == C.ALAKAZAM:
            if self._effect_prevented(target):
                return 0
            return 20 * max(0, self._hand_size() + hand_adjust)
        if attacker.id == C.ALAKAZAM_PSY:
            return self._alakazam_damage(PSYCHIC_ATK, target)
        if attacker.id == C.KADABRA:
            return self._alakazam_damage(SUPER_PSY_BOLT, target)
        if attacker.id == C.FEZANDIPITI_EX:
            return self._alakazam_damage(FEZANDIPITI_ATTACK, target)
        if attacker.id == C.DUDUNSPARCE:
            return self._alakazam_damage(DUDUN_LAND_CRUSH, target)
        return 0

    def _candidate_attack_route(self, attacker, *, hand_adjust=0, source=None):
        if attacker is None:
            return None
        targets = (
            [p for p in (self.opponent.active + self.opponent.bench) if p is not None]
            if attacker.id == C.FEZANDIPITI_EX
            else [self.opponent.active[0] if self.opponent.active else None]
        )
        best = None
        for target in targets:
            damage = self._route_attack_damage(
                attacker, target, hand_adjust=hand_adjust, source=source
            )
            if damage <= 0:
                continue
            ko = damage >= getattr(target, "hp", 0)
            prizes = prize_count(target) if ko else 0
            winning = bool(ko and prizes >= len(self.me.prize))
            score = 2000 + min(400, damage)
            if ko:
                score += 36000 + prizes * 4500
            if winning:
                score += 90000
            if attacker.id == C.FEZANDIPITI_EX and target.id == ROCKET_ARTICUNO_ID:
                score += 12000
            if self._effect_prevented(
                    self.opponent.active[0] if self.opponent.active else None):
                score += 9000
            if attacker.id == C.FEZANDIPITI_EX and self._fez_two_prize_exposure():
                score -= 7000
            route = {
                "attacker": attacker,
                "target": target,
                "damage": damage,
                "ko": ko,
                "winning": winning,
                "score": score,
            }
            if best is None or route["score"] > best["score"]:
                best = route
        return best

    def _best_pivot_attack_route(self, *, assume_attach=False, source=None):
        """Evaluate attach -> retreat -> attack as one narrow tactical route."""
        active = self.me.active[0] if self.me.active else None
        if active is None or not any(p is not None for p in self.me.bench):
            return None
        breaker = self._articuno_breaker_mode()
        # v15 already owns the broad support-pivot policy.  The new normal-board
        # route only fills its missing Active-Alakazam immediate-KO case.
        if active.id not in ALAKAZAM_IDS and not breaker:
            return None
        shortfall = max(0, self._route_retreat_cost(active) - self._energy_count(active))
        if assume_attach:
            if (getattr(self.state, "energyAttached", False)
                    or shortfall != 1 or source is None):
                return None
            hand_adjust = 3 if getattr(source, "id", None) == C.ENRICHING_ENERGY else -1
            current = self._candidate_attack_route(
                active, hand_adjust=hand_adjust, source=source
            )
        else:
            if shortfall > 0:
                return None
            hand_adjust = 0
            current = self._candidate_attack_route(active)

        candidates = []
        for pokemon in self.me.bench:
            if pokemon is None:
                continue
            if breaker:
                if pokemon.id not in (*ALAKAZAM_IDS, C.FEZANDIPITI_EX):
                    continue
            elif pokemon.id not in ALAKAZAM_IDS:
                continue
            route = self._candidate_attack_route(pokemon, hand_adjust=hand_adjust)
            if route is None or (not breaker and not route["ko"]):
                continue
            candidates.append(route)
        if not candidates:
            return None
        best = max(candidates, key=lambda route: route["score"])
        current_score = current["score"] if current is not None else -1
        return best if best["score"] > current_score + 500 else None

    def _pivot_action_score(self, route):
        if route is None:
            return -1
        if route["winning"]:
            return 85000
        if route["ko"]:
            return 47000 + prize_count(route["target"]) * 500
        opponent = self.opponent.active[0] if self.opponent.active else None
        if self._attack_disabled_on_active() or self._effect_prevented(opponent):
            return 42000
        return 24000 + min(1200, route["damage"] * 3)

    def _pivot_attach_score(self, pokemon, area, source=None):
        active = self.me.active[0] if self.me.active else None
        if (pokemon is None or area != AreaType.ACTIVE
                or _pokemon_key(pokemon) != _pokemon_key(active)):
            return -1
        route = self._best_pivot_attack_route(assume_attach=True, source=source)
        if route is None:
            return -1
        basic_bonus = 100 if getattr(source, "id", None) == C.PSYCHIC_ENERGY else 0
        rich_penalty = 800 if getattr(source, "id", None) == C.ENRICHING_ENERGY else 0
        return self._pivot_action_score(route) + basic_bonus - rich_penalty

    def _context_effect_card(self):
        # Official 1.32 observations use `effect` for attached-card and many
        # Supporter sub-selections; older harnesses populated `contextCard`.
        return (getattr(self.select, "contextCard", None)
                or getattr(self.select, "effect", None))

    def _boss_resolving(self):
        return getattr(self._context_effect_card(), "id", None) == C.BOSS_ORDERS

    def _boss_damage_after_spend(self, target):
        """Damage available after committing Boss's Orders.

        Powerful Hand depends on hand size, so playing Boss costs 20 damage. During
        the subsequent opponent-target selection the card has already left the hand,
        therefore the extra spend is zero. We intentionally use the real current
        hand rather than an optimistic achievable-hand estimate: Boss consumes the
        Supporter for the turn and must create a concrete same-turn KO.
        """
        active = self.me.active[0] if self.me.active else None
        if active is None or target is None or not self._can_attack(active):
            return 0
        if active.id == C.ALAKAZAM:
            if self._effect_prevented(target):
                _DIAG["boss_prevented_target_blocks"] += 1
                return 0
            spend = 0 if self._boss_resolving() else 1
            return 20 * max(0, self.me.handCount - spend)
        return self._active_best_dmg(target)

    def _main_option_card(self, option):
        if getattr(option, "type", None) not in (
                OptionType.PLAY, OptionType.EVOLVE,
                OptionType.ATTACH, OptionType.ENERGY):
            return None
        return get_card(
            self.obs,
            AreaType.HAND,
            getattr(option, "index", None),
            self.my_index,
        )

    def _main_has_card_action(self, card_id, option_types):
        return any(
            getattr(option, "type", None) in option_types
            and getattr(self._main_option_card(option), "id", None) == card_id
            for option in (self.select.option or [])
        )

    def _boss_action_available(self):
        """Boss must be both legal and actually offered in this MAIN selection."""
        return bool(
            self.context == SelectContext.MAIN
            and not self.state.supporterPlayed
            and self._main_has_card_action(C.BOSS_ORDERS, (OptionType.PLAY,))
        )

    def _target_area(self, target):
        key = _pokemon_key(target)
        active = self.opponent.active[0] if self.opponent.active else None
        if key is not None and key == _pokemon_key(active):
            return AreaType.ACTIVE
        if any(
                pokemon is not None and key == _pokemon_key(pokemon)
                for pokemon in self.opponent.bench):
            return AreaType.BENCH
        return None

    @staticmethod
    def _attack_hand_required(target):
        """Minimum hand at attack time for 743's 20x Powerful Hand."""
        return math.ceil(max(1, int(getattr(target, "hp", 0) or 0)) / 20)

    @staticmethod
    def _guaranteed_hand_delta(kind):
        """Net hand change for route actions whose result is deterministic."""
        return {
            "dudun": 3,
            "enriching": 3,
            "evolve_alakazam": 2,
            "evolve_kadabra": 1,
            "rare_candy": 1,
            "dawn": 2,
            "hilda": 1,
            "boss": -1,
            "hammer": -1,
            "normal_attach": -1,
        }.get(kind, 0)

    def _target_priority_score(self, target, area=None):
        """Prize-race value only; route feasibility is evaluated separately."""
        if target is None:
            return -1
        area = area or self._target_area(target)
        prizes = prize_count(target)
        my_remaining = len(self.me.prize)
        opponent_remaining = len(self.opponent.prize)

        score = self._target_value(target) + prizes * 6000
        if prizes >= my_remaining:
            score += 100000
        # When behind in the prize race, removing a multi-prizer now matters more.
        if my_remaining > opponent_remaining:
            score += prizes * 1800
        elif my_remaining < opponent_remaining:
            score += prizes * 700
        # Active needs no Supporter and is the stable tiebreak when values are close.
        if area == AreaType.ACTIVE:
            score += 900
        # Low remaining HP is a tiebreak only, never a substitute for prize/role value.
        score += max(0, 500 - min(500, int(getattr(target, "hp", 0) or 0)))
        return score

    def _target_priority_list(self):
        """All visible targets in strategic order, independent of reachability."""
        entries = []
        active = self.opponent.active[0] if self.opponent.active else None
        if active is not None:
            entries.append({
                "target": active,
                "area": AreaType.ACTIVE,
                "score": self._target_priority_score(active, AreaType.ACTIVE),
            })
        for index, pokemon in enumerate(self.opponent.bench):
            if pokemon is None:
                continue
            entries.append({
                "target": pokemon,
                "area": AreaType.BENCH,
                "index": index,
                "score": self._target_priority_score(pokemon, AreaType.BENCH),
            })
        return sorted(
            entries,
            key=lambda entry: (
                -entry["score"],
                int(entry["area"] != AreaType.ACTIVE),
                int(getattr(entry["target"], "hp", 0) or 0),
                str(_pokemon_key(entry["target"])),
            ),
        )

    def _target_has_prevent_energy(self, target):
        return any(
            getattr(energy, "id", None) in EFFECT_PREVENT_ENERGY
            for energy in (getattr(target, "energyCards", None) or [])
        )

    def _terminal_boss_targets(self):
        """Legal immediate Boss KOs that take exactly the last 2/3 prize cards."""
        remaining = len(self.me.prize)
        if remaining not in (2, 3) or not self._boss_action_available():
            return []
        targets = []
        for pokemon in self.opponent.bench:
            if pokemon is None or prize_count(pokemon) < remaining:
                continue
            damage = self._boss_offered_attack_damage(pokemon)
            if damage >= max(1, int(getattr(pokemon, "hp", 0) or 0)):
                targets.append(pokemon)
        return sorted(
            targets,
            key=lambda pokemon: (
                -self._target_priority_score(pokemon, AreaType.BENCH),
                int(getattr(pokemon, "hp", 0) or 0),
                str(_pokemon_key(pokemon)),
            ),
        )

    def _terminal_boss_gate_score(self, option):
        targets = self._terminal_boss_targets()
        if not targets:
            return None
        card = self._main_option_card(option)
        if (getattr(option, "type", None) == OptionType.PLAY
                and getattr(card, "id", None) == C.BOSS_ORDERS):
            return (
                TERMINAL_BOSS_SCORE
                + self._target_priority_score(targets[0], AreaType.BENCH)
            )
        return BLOCKED_BY_TERMINAL_BOSS_SCORE

    def _boss_offered_attack_damage(self, target):
        """Damage after Boss, restricted to attacks actually offered this selection."""
        active = self.me.active[0] if self.me.active else None
        if active is None or target is None:
            return 0
        best = 0
        for option in (self.select.option or []):
            if getattr(option, "type", None) != OptionType.ATTACK:
                continue
            attack_id = getattr(option, "attackId", None)
            if active.id == C.ALAKAZAM and attack_id == POWERFUL_HAND:
                if self._effect_prevented(target):
                    continue
                damage = 20 * max(0, self.me.handCount - 1)
            else:
                damage = self._alakazam_damage(attack_id, target)
            best = max(best, damage)
        return best

    def _dudun_ability_options(self):
        options = []
        for option in (self.select.option or []):
            if getattr(option, "type", None) != OptionType.ABILITY:
                continue
            card = get_card(
                self.obs,
                getattr(option, "area", None),
                getattr(option, "index", None),
                self.my_index,
            )
            if card is not None and card.id == C.DUDUNSPARCE:
                options.append(option)
        return options

    def _active_route_atoms(self):
        """Guaranteed hand-growth actions offered in the current MAIN state."""
        atoms = []

        for number, _option in enumerate(self._dudun_ability_options()):
            atoms.append({
                "kind": "dudun",
                "delta": self._guaranteed_hand_delta("dudun"),
                "deck": 3,
                "supporter": False,
                "target": None,
                "identity": ("dudun", number),
                "cards": {},
            })

        if self._main_has_card_action(
                C.ENRICHING_ENERGY, (OptionType.ATTACH, OptionType.ENERGY)):
            atoms.append({
                "kind": "enriching",
                "delta": self._guaranteed_hand_delta("enriching"),
                "deck": 4,
                "supporter": False,
                "target": None,
                "identity": ("enriching", 0),
                "cards": {C.ENRICHING_ENERGY: 1},
            })

        seen_evolution_sources = set()
        for option in (self.select.option or []):
            if getattr(option, "type", None) != OptionType.EVOLVE:
                continue
            card = self._main_option_card(option)
            if card is None or card.id not in (C.ALAKAZAM, C.KADABRA):
                continue
            source = getattr(option, "index", None)
            if source in seen_evolution_sources:
                continue
            seen_evolution_sources.add(source)
            if card.id == C.ALAKAZAM:
                kind, deck = "evolve_alakazam", 3
            else:
                kind, deck = "evolve_kadabra", 2
            atoms.append({
                "kind": kind,
                "delta": self._guaranteed_hand_delta(kind),
                "deck": deck,
                "supporter": False,
                "target": (
                    getattr(option, "inPlayArea", None),
                    getattr(option, "inPlayIndex", None),
                ),
                "identity": (kind, source),
                "cards": {card.id: 1},
            })

        if (self._main_has_card_action(C.RARE_CANDY, (OptionType.PLAY,))
                and self.hand[C.ALAKAZAM] > 0):
            atoms.append({
                "kind": "rare_candy",
                "delta": self._guaranteed_hand_delta("rare_candy"),
                "deck": 3,
                "supporter": False,
                "target": None,
                "identity": ("rare_candy", 0),
                "cards": {C.RARE_CANDY: 1, C.ALAKAZAM: 1},
            })

        if not self.state.supporterPlayed:
            if self._main_has_card_action(C.DAWN, (OptionType.PLAY,)):
                atoms.append({
                    "kind": "dawn",
                    "delta": self._guaranteed_hand_delta("dawn"),
                    "deck": 3,
                    "supporter": True,
                    "target": None,
                    "identity": ("dawn", 0),
                    "cards": {C.DAWN: 1},
                })
            if self._main_has_card_action(C.HILDA, (OptionType.PLAY,)):
                atoms.append({
                    "kind": "hilda",
                    "delta": self._guaranteed_hand_delta("hilda"),
                    "deck": 2,
                    "supporter": True,
                    "target": None,
                    "identity": ("hilda", 0),
                    "cards": {C.HILDA: 1},
                })
        return atoms

    def _ko_route_plan(self, target):
        """Smallest deterministic legal route to KO one concrete visible target.

        Active routes are free to use one draw/search Supporter. Bench routes reserve
        that Supporter slot for Boss and charge its -1 hand cost. Unknown draws are
        never assumed; only guaranteed hand deltas from currently offered actions are
        searched.
        """
        target_area = self._target_area(target)
        needs_boss = target_area == AreaType.BENCH
        active = self.me.active[0] if self.me.active else None
        empty = {
            "reachable": False,
            "ko": False,
            "winning": False,
            "target": target,
            "target_area": target_area,
            "needs_boss": needs_boss,
            "damage": 0,
            "hand": self.me.handCount,
            "required_hand": self._attack_hand_required(target) if target is not None else 0,
            "actions": frozenset(),
            "next_actions": frozenset(),
            "action_count": 0,
            "deck_cost": 0,
        }

        cache = getattr(self, "_ko_route_cache", None)
        if cache is None:
            cache = {}
            self._ko_route_cache = cache
        cache_key = (id(target), target_area)
        if cache_key in cache:
            return cache[cache_key]

        if (getattr(self, "context", None) != SelectContext.MAIN
                or target is None or active is None
                or target_area is None
                or active.id != C.ALAKAZAM or not self._can_attack(active)
                or self._temporary_attack_immunity_applies(target)
                or (needs_boss and not self._boss_action_available())):
            cache[cache_key] = empty
            return empty

        blocked = self._effect_prevented(target)
        hammer_ready = (
            blocked
            and not self._non_energy_effect_prevented(target)
            and self._target_has_prevent_energy(target)
            and self._main_has_card_action(C.ENHANCED_HAMMER, (OptionType.PLAY,))
        )
        if blocked and not hammer_ready:
            cache[cache_key] = empty
            return empty

        mandatory_actions = set()
        base_hand = self.me.handCount
        if needs_boss:
            mandatory_actions.add("boss")
            base_hand += self._guaranteed_hand_delta("boss")
        if hammer_ready:
            mandatory_actions.add("hammer")
            base_hand += self._guaranteed_hand_delta("hammer")
        immediate_damage = 20 * max(0, base_hand)
        target_prizes = prize_count(target)
        wins_on_ko = (
            target_prizes >= len(self.me.prize)
            or (
                target_area == AreaType.ACTIVE
                and not any(pokemon is not None for pokemon in self.opponent.bench)
            )
        )
        if immediate_damage >= target.hp:
            actions = frozenset(mandatory_actions)
            non_boss_actions = actions - {"boss"}
            result = {
                **empty,
                "reachable": True,
                "ko": True,
                "winning": wins_on_ko,
                "damage": immediate_damage,
                "hand": base_hand,
                "actions": actions,
                "next_actions": (
                    non_boss_actions if non_boss_actions
                    else (frozenset({"boss"}) if needs_boss else frozenset())
                ),
                "action_count": len(mandatory_actions),
            }
            cache[cache_key] = result
            return result

        atoms = [
            atom for atom in self._active_route_atoms()
            if not (needs_boss and atom["supporter"])
        ]
        best = None
        for mask in range(1 << len(atoms)):
            chosen = [
                atoms[index] for index in range(len(atoms))
                if mask & (1 << index)
            ]
            if sum(int(atom["supporter"]) for atom in chosen) > 1:
                continue
            kinds = {atom["kind"] for atom in chosen}
            if "rare_candy" in kinds and "evolve_kadabra" in kinds:
                evolved_abras = {
                    atom["target"] for atom in chosen
                    if atom["kind"] == "evolve_kadabra"
                }
                candy_abras = set()
                if (self.me.active and self.me.active[0] is not None
                        and self.me.active[0].id == C.ABRA
                        and not getattr(self.me.active[0], "appearThisTurn", False)):
                    candy_abras.add((AreaType.ACTIVE, 0))
                candy_abras.update(
                    (AreaType.BENCH, index)
                    for index, pokemon in enumerate(self.me.bench)
                    if pokemon is not None
                    and pokemon.id == C.ABRA
                    and not getattr(pokemon, "appearThisTurn", False)
                )
                if not candy_abras - evolved_abras:
                    continue
            targets = [
                atom["target"] for atom in chosen
                if atom["target"] is not None
            ]
            if len(targets) != len(set(targets)):
                continue
            card_costs = Counter()
            for atom in chosen:
                card_costs.update(atom["cards"])
            if any(count > self.hand[card_id]
                   for card_id, count in card_costs.items()):
                continue
            deck_cost = sum(atom["deck"] for atom in chosen)
            if deck_cost > self.me.deckCount:
                continue
            hand = base_hand + sum(atom["delta"] for atom in chosen)
            damage = 20 * max(0, hand)
            if damage < target.hp:
                continue
            actions = mandatory_actions | {atom["kind"] for atom in chosen}
            non_boss_actions = frozenset(actions - {"boss"})
            action_count = len(chosen) + len(mandatory_actions)
            supporter_cost = sum(int(atom["supporter"]) for atom in chosen)
            overkill = max(0, damage - target.hp)
            key = (action_count, deck_cost, supporter_cost, overkill)
            if best is None or key < best[0]:
                best = (key, {
                    **empty,
                    "reachable": True,
                    "ko": True,
                    "winning": wins_on_ko,
                    "damage": damage,
                    "hand": hand,
                    "actions": frozenset(actions),
                    "next_actions": (
                        non_boss_actions if non_boss_actions
                        else (frozenset({"boss"}) if needs_boss else frozenset())
                    ),
                    "action_count": action_count,
                    "deck_cost": deck_cost,
                })
        result = best[1] if best is not None else empty
        cache[cache_key] = result
        return result

    def _active_route_plan(self):
        """Compatibility wrapper for the v18 Active-only callers/tests."""
        target = self.opponent.active[0] if self.opponent.active else None
        return self._ko_route_plan(target)

    def _chosen_ko_plan(self):
        """Try targets in strategic order and return the first reachable KO route."""
        if getattr(self, "_chosen_ko_plan_cached", False):
            return getattr(self, "_chosen_ko_plan_cache", None)
        plan = None
        entries = self._target_priority_list()
        if not V21_TARGET_ROUTES_ENABLED:
            entries = [
                entry for entry in entries
                if entry["area"] == AreaType.ACTIVE
            ]
        for entry in entries:
            candidate = self._ko_route_plan(entry["target"])
            if not candidate["ko"]:
                continue
            plan = {
                **candidate,
                "priority": entry["score"],
            }
            break
        self._chosen_ko_plan_cache = plan
        self._chosen_ko_plan_cached = True
        return plan

    def _boss_active_reachable_damage(self, target):
        """Compatibility wrapper backed by the legal full-turn route planner."""
        active = self.opponent.active[0] if self.opponent.active else None
        if target is None or active is None or _pokemon_key(target) != _pokemon_key(active):
            return 0
        return self._active_route_plan()["damage"]

    def _active_route_action_score(self, kind):
        plan = self._active_route_plan()
        if not plan["ko"] or kind not in plan["next_actions"]:
            return -1
        if plan["winning"]:
            return 88000
        target = self.opponent.active[0]
        return 47000 + prize_count(target) * 900

    def _chosen_route_action_score(self, kind):
        plan = self._chosen_ko_plan()
        if plan is None or kind not in plan["next_actions"]:
            return -1
        if plan["winning"]:
            return 88000
        return 47000 + prize_count(plan["target"]) * 900

    def _draw_redundant_for_chosen_target(self, kind):
        """Whether optional draw has no concrete job before the selected KO.

        v20 waited for five surplus cards.  In the public v20 run that still
        allowed repeated Run Away Draw cycles in already won turns and eight
        deck-out losses.  v21 stops a benched Dudunsparce as soon as the KO is
        secured, except when that draw is the only realistic way to build a
        replacement attacker.  Other draw/search actions retain a small buffer
        because they may also establish the next evolution line.
        """
        plan = self._chosen_ko_plan()
        secured = bool(
            V21_HAND_GATE_ENABLED
            and plan is not None
            and plan["ko"]
            # Boss itself may remain; every other prerequisite must be complete.
            and not (plan["actions"] - {"boss"})
            and kind not in plan["actions"]
            and self._ready_alakazam_count() > 0
            and not self._energy_starved()
        )
        if not secured:
            return False
        surplus = plan["hand"] - plan["required_hand"]
        if kind != "dudun":
            return surplus >= V21_HAND_SURPLUS

        future_line = (
            self.field[C.ABRA]
            + self.field[C.KADABRA]
            + max(0, self._ready_alakazam_count() - 1)
        )
        continuity_crisis = bool(
            not plan["winning"]
            and self._ready_alakazam_count() == 1
            and self._backup_eta() > 1
            and future_line == 0
            and self.me.deckCount > self._deck_floor() + 6
        )
        return not continuity_crisis

    def _rocket_evolution_escape_target(self, pokemon):
        """An evolved Team Rocket Pokemon is outside Articuno's Basic-only shield."""
        if pokemon is None or not self._rocket_articuno_in_play():
            return False
        data = card_table.get(pokemon.id)
        return bool(
            data is not None
            and (getattr(data, "stage1", False) or getattr(data, "stage2", False))
            and "team rocket" in _norm_card_text(getattr(data, "name", ""))
            and not self._effect_prevented(pokemon)
        )

    def _visible_evolution_family_scores(self):
        cached = getattr(self, "_visible_family_score_cache", None)
        if cached is not None:
            return cached
        scores = defaultdict(int)
        counts = Counter()
        for pokemon in (self.opponent.active + self.opponent.bench):
            if pokemon is None:
                continue
            data = card_table.get(pokemon.id)
            root = EVOLUTION_ROOT_BY_ID.get(pokemon.id, "")
            if data is None or not root:
                continue
            counts[root] += 1
            body = (
                4800 if getattr(data, "megaEx", False)
                else 3700 if getattr(data, "ex", False)
                else 2800 if getattr(data, "stage2", False)
                else 1500 if getattr(data, "stage1", False)
                else 450
            )
            body += min(1800, self._energy_count(pokemon) * 600)
            body += min(500, len(getattr(pokemon, "tools", None) or []) * 250)
            scores[root] += body
        for root, count in counts.items():
            scores[root] += EVOLUTION_LINE_CEILING.get(root, 0) // 3
            scores[root] += max(0, count - 1) * 450
        self._visible_family_score_cache = dict(scores)
        return self._visible_family_score_cache

    def _evolution_line_role_bonus(self, pokemon):
        """Value an enabling member of the opponent's most important visible line."""
        if pokemon is None:
            return 0
        data = card_table.get(pokemon.id)
        root = EVOLUTION_ROOT_BY_ID.get(pokemon.id, "")
        if data is None or not root:
            return 0
        ceiling = EVOLUTION_LINE_CEILING.get(root, 0)
        if ceiling >= 5000:
            future = 1900
        elif ceiling >= 4000:
            future = 1500
        elif ceiling >= 3000:
            future = 1100
        elif ceiling >= 1200:
            future = 300
        else:
            future = 0
        # The future-value bonus is for the pieces that still enable the payoff,
        # not for paying twice for an ex/Mega body already priced by prizes.
        if getattr(data, "megaEx", False) or getattr(data, "ex", False):
            future //= 4
        elif getattr(data, "stage2", False):
            future //= 2

        visible = self._visible_evolution_family_scores()
        if visible:
            best = max(visible.values())
            if visible.get(root, 0) == best and best >= 1200:
                future += 1100
        return future

    @staticmethod
    def _engine_text_role_bonus(data):
        if data is None:
            return 0
        text = " ".join(
            _norm_card_text(getattr(skill, "text", ""))
            for skill in (getattr(data, "skills", None) or [])
        )
        bonus = 0
        if "search your deck" in text or ("attach" in text and "energy" in text):
            bonus = max(bonus, 1400)
        if "draw " in text:
            bonus = max(bonus, 1200)
        if "damage counter" in text and ("move" in text or "put" in text):
            bonus = max(bonus, 1300)
        if "switch in" in text and "opponent" in text:
            bonus = max(bonus, 1100)
        if "prevent all" in text:
            bonus = max(bonus, 1600)
        return bonus

    def _boss_role_bonus(self, p):
        if p is None:
            return 0
        bonus = BOSS_KEY_ROLE_BONUS.get(p.id, 0)
        if p.id in GLOBAL_EFFECT_PROTECTORS:
            bonus = max(bonus, 6500)
        if self._rocket_evolution_escape_target(p):
            bonus += 8500
        d = card_table.get(p.id)
        if d is None:
            return bonus
        bonus += self._evolution_line_role_bonus(p)
        bonus += self._engine_text_role_bonus(d)
        if getattr(d, 'stage2', False):
            bonus += 1800
        elif getattr(d, 'stage1', False):
            bonus += 1000
        if getattr(d, 'ex', False):
            bonus += 1600
        # A powered Pokémon with a real attack is generally a next-turn threat.
        if len(getattr(p, 'energies', []) or []) >= 1 and getattr(d, 'attacks', None):
            bonus += 900
        texts = ' '.join((getattr(skill, 'text', '') or '').lower()
                         for skill in (getattr(d, 'skills', None) or []))
        if 'prevent all effects of attacks' in texts or 'prevent all damage' in texts:
            bonus += 3000
        if ('cannot use' in texts or "can't use" in texts) and ('opponent' in texts):
            bonus += 1400
        return bonus

    def _target_value(self, p):
        """Strategic removal value, not merely the lowest HP target."""
        if p is None:
            return -1
        d = card_table.get(p.id)
        score = prize_count(p) * 2200
        score += len(getattr(p, 'energies', []) or []) * 380
        score += len(getattr(p, 'tools', []) or []) * 220
        score += self._boss_role_bonus(p)
        if d is not None:
            if getattr(d, 'stage2', False):
                score += 1200
            elif getattr(d, 'stage1', False):
                score += 700
        score += min(400, max(0, getattr(p, 'maxHp', getattr(p, 'hp', 0))) // 2)
        return score

    def _boss_two_hit_target_score(self, target, damage):
        """Allow only a sticky, prize-closing route to an explicit key target.

        v8 connected only 74.1% of Boss plays to an attack, down from v7's 89.2%.
        A generic two-hit target can retreat after the first hit, so v9 requires all
        of: the target closes the prize race, cannot currently pay its retreat cost,
        and is a three-prizer, protection engine, or powered main attacker.
        """
        if target is None or damage <= 0 or damage >= target.hp or 2 * damage < target.hp:
            return -1
        target_prizes = prize_count(target)
        if target_prizes < 2:
            return -1

        active = self.opponent.active[0] if self.opponent.active else None
        active_damage = self._active_best_dmg(active) if active is not None else 0
        active_ko = bool(active is not None and active_damage >= active.hp)
        active_prizes = prize_count(active) if active_ko and active is not None else 0
        remaining = len(self.me.prize)
        if active_ko and active_prizes >= remaining:
            return -1
        if active_ko and target_prizes <= active_prizes:
            return -1

        role = self._boss_role_bonus(target)
        data = card_table.get(target.id)
        attached = len(getattr(target, 'energies', []) or [])
        retreat_cost = int(getattr(data, 'retreatCost', 0) or 0) if data is not None else 0
        sticky = retreat_cost > attached
        protector = target.id in GLOBAL_EFFECT_PROTECTORS or target.id == ROCKET_ARTICUNO_ID
        main_attacker = bool(
            data is not None
            and getattr(data, 'attacks', None)
            and attached >= 2
            and (getattr(data, 'ex', False) or getattr(data, 'megaEx', False)
                 or getattr(data, 'stage2', False))
        )
        prize_closing = target_prizes >= remaining
        explicit_value = target_prizes >= 3 or protector or (main_attacker and role >= 3000)
        if not (sticky and prize_closing and explicit_value):
            return -1

        score = 6500 + self._target_value(target)
        score += target_prizes * 1200
        if target_prizes >= 3:
            score += 1800
        return score

    def _boss_effect_lock_escape_ko(self, target, damage=None):
        """Whether Boss converts a zero-effect Active attack into a Bench KO.

        A protected Active (Mist, Rock Fighting Energy, or a correctly scoped
        board protector) is otherwise a complete Powerful Hand lock.  In that
        state, gusting even a low-value one-prize Basic is strictly better when it
        yields a real KO; the normal anti-waste Boss filters must not reject it.
        """
        active = self.opponent.active[0] if self.opponent.active else None
        if (active is None or target is None or not self._effect_prevented(active)
                or self._effect_prevented(target)):
            return False
        if damage is None:
            damage = self._boss_damage_after_spend(target)
        return damage > 0 and damage >= getattr(target, "hp", 0)

    def _boss_target_score(self, target):
        """Score a better same-turn KO or a strict two-hit multi-prize route.

        This prevents the old behaviour of gusting a merely KO-able low-value Basic.
        A target must be a win, a higher-prize Pokémon, a protection/engine piece
        (notably Team Rocket's Articuno), a developed attacker, or an invested
        evolution line. If the opposing Active is already a KO, the Bench target must
        beat it by a real strategic margin.
        """
        if target is None or (self.state.supporterPlayed and not self._boss_resolving()):
            return -1
        damage = self._boss_damage_after_spend(target)
        if damage <= 0:
            return -1
        if damage < getattr(target, 'hp', 0):
            return self._boss_two_hit_target_score(target, damage)
        if self._boss_resolving():
            # The MAIN decision has already committed to a target-ranked KO route.
            # During Boss's sub-selection, keep every legal KO target comparable by
            # the same prize-race priority instead of reapplying the old anti-waste
            # filters (which could reject the chosen one-prize fallback target).
            return 3000 + self._target_priority_score(target, AreaType.BENCH)

        active = self.opponent.active[0] if self.opponent.active else None
        effect_lock_escape_ko = self._boss_effect_lock_escape_ko(target, damage)
        active_damage_without_boss = self._active_best_dmg(active) if active is not None else 0
        active_ko = bool(active is not None and active_damage_without_boss >= active.hp)
        active_plan = self._active_route_plan()
        reachable_active_ko = bool(active_plan["ko"])
        target_prizes = prize_count(target)
        active_prizes = prize_count(active) if active_ko and active is not None else 0
        target_role = self._boss_role_bonus(target)
        active_role = self._boss_role_bonus(active) if active is not None else 0
        d = card_table.get(target.id)
        winning = target_prizes >= len(self.me.prize)
        more_prizes = target_prizes > active_prizes
        key_role = target_role >= 2200
        developed = bool(
            len(getattr(target, 'energies', []) or []) >= 2
            or (d is not None and (getattr(d, 'stage1', False)
                                   or getattr(d, 'stage2', False)))
        )
        two_prize = target_prizes >= 2
        escape_lock_ko = self._articuno_escape_target(target) or effect_lock_escape_ko

        if not (winning or more_prizes or key_role or developed or two_prize
                or escape_lock_ko):
            return -1

        target_value = self._target_value(target)
        active_value = self._target_value(active) if active_ko else 0
        # Preserve v11's successful distinction between an actual Active KO and
        # a merely projected KO after optional future actions.  v13 treated both
        # as equally certain and skipped profitable Boss tempo too often.
        if active_ko:
            if active_prizes >= len(self.me.prize) and not winning:
                _DIAG["boss_active_ko_dominance_blocks"] += 1
                return -1
            if not winning:
                role_upgrade = target_role >= active_role + 1500
                if target_prizes < active_prizes and not role_upgrade:
                    _DIAG["boss_active_ko_dominance_blocks"] += 1
                    return -1
                if target_value < active_value + 700 and not role_upgrade:
                    _DIAG["boss_active_ko_dominance_blocks"] += 1
                    return -1
        elif reachable_active_ko:
            projected_prizes = prize_count(active)
            projected_winning = bool(active_plan["winning"])
            if projected_winning and not winning:
                _DIAG["boss_reachable_active_ko_blocks"] += 1
                return -1
            if projected_prizes > target_prizes and not winning:
                _DIAG["boss_reachable_active_ko_blocks"] += 1
                return -1
        # If Active is not KO-able, a gusted target still has to be strategically
        # meaningful; a random 1-prize Basic is not enough.
        if not active_ko and not (
                winning or two_prize or key_role or developed or escape_lock_ko):
            return -1

        score = 3000 + target_value
        if winning:
            score += 90000
        if more_prizes:
            score += 1800
        if escape_lock_ko:
            score += 7000
        if effect_lock_escape_ko:
            score += 5000
        if target.id == ROCKET_ARTICUNO_ID:
            score += 4000
        return score

    def _gust_ko_targets(self):
        return [p for p in self.opponent.bench
                if p is not None and self._boss_target_score(p) > 0]

    def _gust_value(self, p):
        return self._boss_target_score(p)

    def _hand_delta(self, t, o):
        """この行動で手札が何枚増減するか(概算)。PLAYのドロー系サポは正、
        グッズ/進化/エネ貼りは-1。Run Away Draw(ベンチ)は+3。"""
        if t == OptionType.ABILITY:
            return (
                self._guaranteed_hand_delta("dudun")
                if getattr(o, 'area', None) == AreaType.BENCH else 0
            )
        if t == OptionType.EVOLVE:
            card = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
            if card is not None and card.id == C.KADABRA:
                return self._guaranteed_hand_delta("evolve_kadabra")
            if card is not None and card.id in ALAKAZAM_IDS:
                return self._guaranteed_hand_delta("evolve_alakazam")
            return -1
        if t in (OptionType.ATTACH, OptionType.ENERGY):
            card = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
            if card is not None and card.id == C.ENRICHING_ENERGY:
                return self._guaranteed_hand_delta("enriching")
            return -1
        if t == OptionType.PLAY:
            card = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
            if card is not None and card.id == C.HILDA:
                return self._guaranteed_hand_delta("hilda")
            if card is not None and card.id == C.DAWN:
                return self._guaranteed_hand_delta("dawn")
            if card is not None and card.id == C.LANA_AID:
                return min(3, self._lana_recoverable_count()) - 1
            if card is not None and card.id == C.RARE_CANDY:
                return self._guaranteed_hand_delta("rare_candy")
            return -1
        return 0

    # — entry —
    def rank(self):
        if not self.select.option or self.select.maxCount == 0:
            return [], []
        scores = [self._score(o) for o in self.select.option]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked, scores

    def choose(self):
        ranked, scores = self.rank()
        selected = normalize_selection(ranked, scores, self.select)
        try:
            self._record_v9(selected)
        except Exception:
            pass
        return selected

    def _record_v18(self, index, option):
        if self.context != SelectContext.MAIN:
            return

        selected_card = None
        if option.type in (
                OptionType.PLAY, OptionType.EVOLVE,
                OptionType.ATTACH, OptionType.ENERGY):
            selected_card = self._main_option_card(option)

        dudun_options = self._dudun_ability_options()
        dudun_on_board = any(
            pokemon is not None and pokemon.id == C.DUDUNSPARCE
            for pokemon in self._my_board()
        )
        selected_dudun = option in dudun_options
        if dudun_on_board and not dudun_options:
            active = self.me.active[0] if self.me.active else None
            if self._board_body_count() <= 1:
                _DIAG["dudun_block_only_body"] += 1
            elif (active is not None and active.id == C.DUDUNSPARCE
                  and not any(pokemon is not None for pokemon in self.me.bench)):
                _DIAG["dudun_block_no_bench"] += 1
            elif (not self._deck_spend_ok(cost=3)
                  and not self._emergency_energy_draw(3)):
                _DIAG["dudun_block_deck_floor"] += 1
            else:
                _DIAG["dudun_block_ability_lock"] += 1
        elif dudun_options and not selected_dudun:
            _DIAG["dudun_declined_by_score"] += 1
            if self._draw_redundant_for_chosen_target("dudun"):
                _DIAG["target_satisfied_draw_blocks"] += 1
        elif selected_dudun and self._terminal_pivot_win(option):
            _DIAG["dudun_used_for_terminal_pivot"] += 1
            _DIAG["terminal_pivot_abilities"] += 1
        elif selected_dudun:
            plan = self._chosen_ko_plan()
            if (
                plan is not None
                and plan["ko"]
                and not (plan["actions"] - {"boss"})
                and not self._draw_redundant_for_chosen_target("dudun")
            ):
                _DIAG["dudun_continuity_draws"] += 1

        if option.type == OptionType.ATTACK and self._terminal_win_attack_offered():
            _DIAG["terminal_win_attack_gates"] += 1
        if option.type == OptionType.END and self._progress_attack_offered():
            _DIAG["progress_attack_end_blocks"] += 1

        terminal_targets = self._terminal_boss_targets()
        if terminal_targets:
            _DIAG["terminal_boss_opportunities"] += 1
            if (option.type == OptionType.PLAY
                    and getattr(selected_card, "id", None) == C.BOSS_ORDERS):
                _DIAG["terminal_boss_plays"] += 1

        route_kind = None
        if selected_dudun:
            route_kind = "dudun"
        elif option.type == OptionType.EVOLVE:
            route_kind = {
                C.ALAKAZAM: "evolve_alakazam",
                C.KADABRA: "evolve_kadabra",
            }.get(getattr(selected_card, "id", None))
        elif option.type in (OptionType.ATTACH, OptionType.ENERGY):
            if getattr(selected_card, "id", None) == C.ENRICHING_ENERGY:
                route_kind = "enriching"
        elif option.type == OptionType.PLAY:
            route_kind = {
                C.ENHANCED_HAMMER: "hammer",
                C.RARE_CANDY: "rare_candy",
                C.DAWN: "dawn",
                C.HILDA: "hilda",
                C.BOSS_ORDERS: "boss",
            }.get(getattr(selected_card, "id", None))

        plan = self._active_route_plan()
        if plan["ko"] and route_kind in plan["actions"]:
            _DIAG["active_route_plans"] += 1
            if route_kind == "hammer":
                _DIAG["active_route_hammer_actions"] += 1
            elif route_kind in {
                    "dudun", "enriching", "evolve_alakazam",
                    "evolve_kadabra", "rare_candy", "dawn", "hilda"}:
                _DIAG["active_route_draw_actions"] += 1
        chosen_plan = self._chosen_ko_plan()
        if chosen_plan is not None and route_kind in chosen_plan["next_actions"]:
            _DIAG["target_route_plans"] += 1
            if route_kind == "boss":
                _DIAG["target_route_boss_actions"] += 1

    def _record_v9(self, selected):
        if not selected or not self.select.option:
            return
        index = selected[0]
        if not (0 <= index < len(self.select.option)):
            return
        option = self.select.option[index]
        self._record_v18(index, option)

        # Enhanced Hammer's attached-energy selection is the metric the v8
        # report could not satisfy because energyIndex was previously ignored.
        if getattr(self._context_effect_card(), "id", None) == C.ENHANCED_HAMMER:
            owner = get_card(self.obs, option.area, option.index, option.playerIndex)
            energy_index = getattr(option, "energyIndex", None)
            cards = getattr(owner, "energyCards", None) or []
            energy = cards[energy_index] if energy_index is not None and 0 <= energy_index < len(cards) else None
            if energy is not None and energy.id == C.MIST_ENERGY:
                _DIAG["hammer_mist_targets"] += 1
            elif energy is not None:
                _DIAG["hammer_non_mist_targets"] += 1
            _V9_STATE["planned_hammer_target_key"] = None
            return

        if self.context == SelectContext.ACTIVATE:
            if self._emergency_energy_draw(self._activate_draw_count()):
                key = ("emergency_energy_draws"
                       if option.type == OptionType.YES
                       else "emergency_energy_draw_declines")
                _DIAG[key] += 1
            return
        if self.context == getattr(SelectContext, "TO_ACTIVE", object()):
            _V9_STATE["own_ko_turn"] = getattr(self.state, "turn", None)
            promoted = get_card(
                self.obs,
                getattr(option, "area", getattr(option, "inPlayArea", None)),
                getattr(option, "index", getattr(option, "inPlayIndex", None)),
                getattr(option, "playerIndex", self.my_index),
            )
            if (promoted is not None
                    and promoted.id == C.FEZANDIPITI_EX
                    and self._articuno_breaker_mode()
                    and self._can_attack(promoted)):
                _DIAG["articuno_breaker_promotions"] += 1
            if promoted is not None:
                if self._promotion_attacks_next_turn(promoted):
                    _DIAG["ko_promotion_attacker_choices"] += 1
                elif promoted.id in (C.DUNSPARCE, C.DUDUNSPARCE):
                    _DIAG["ko_promotion_shield_choices"] += 1
            return
        if self.context == SelectContext.SWITCH:
            switched = get_card(
                self.obs,
                getattr(option, "area", getattr(option, "inPlayArea", None)),
                getattr(option, "index", getattr(option, "inPlayIndex", None)),
                getattr(option, "playerIndex", self.my_index),
            )
            effect = self._context_effect_card()
            if (
                getattr(option, "playerIndex", self.my_index) == self.my_index
                and getattr(effect, "id", None) in (C.ABRA, C.DUNSPARCE)
                and switched is not None
                and switched.id in (C.DUNSPARCE, C.DUDUNSPARCE)
            ):
                _DIAG["abra_switch_sacrifice_choices"] += 1
            if (
                getattr(option, "playerIndex", self.my_index) == self.op_index
                and getattr(effect, "id", None) == C.BOSS_ORDERS
                and switched is not None
                and self._evolution_line_role_bonus(switched) > 0
            ):
                _DIAG["core_line_target_bonuses"] += 1
        if self.context != SelectContext.MAIN:
            return
        hammer_offered = any(
                candidate.type == OptionType.PLAY
                and getattr(get_card(self.obs, AreaType.HAND, candidate.index, self.my_index), "id", None)
                == C.ENHANCED_HAMMER
                for candidate in self.select.option
        )
        if hammer_offered and self._should_reserve_last_hammer():
            _DIAG["hammer_reserve_decisions"] += 1
        if hammer_offered and self._should_reserve_hammer_for_seen_mist():
            _DIAG["hammer_mist_memory_reserves"] += 1
        if any(
                candidate.type == OptionType.PLAY
                and getattr(get_card(self.obs, AreaType.HAND, candidate.index, self.my_index), "id", None)
                == C.FEZANDIPITI_EX
                for candidate in self.select.option
        ) and self._fez_mode(for_bench=True) == "DO_NOT_BENCH":
            _DIAG["fez_do_not_bench_blocks"] += 1
            if len(self.opponent.prize) <= 3:
                _DIAG["fez_late_bench_blocks"] += 1

        card = None
        if option.type in (OptionType.PLAY, OptionType.ATTACH, OptionType.ENERGY,
                           OptionType.EVOLVE):
            card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)

        if (option.type == OptionType.PLAY
                and getattr(card, "id", None) == C.ENHANCED_HAMMER):
            chosen_plan = self._chosen_ko_plan()
            if (chosen_plan is not None
                    and "hammer" in chosen_plan["next_actions"]):
                _V9_STATE["planned_hammer_target_key"] = _pokemon_key(
                    chosen_plan["target"]
                )
            else:
                _V9_STATE["planned_hammer_target_key"] = None
        elif _V9_STATE.get("planned_hammer_target_key") is not None:
            _V9_STATE["planned_hammer_target_key"] = None

        if option.type == OptionType.PLAY and getattr(card, "id", None) == C.FEZANDIPITI_EX:
            _DIAG["fez_draw_only_actions"] += 1
            if len(self.opponent.prize) <= 3 and self._fez_entry_urgent():
                _DIAG["fez_recovery_exceptions"] += 1
        elif option.type in (OptionType.ATTACH, OptionType.ENERGY):
            target = get_card(self.obs, option.inPlayArea, option.inPlayIndex, self.my_index)
            if (target is not None and target.id in ALAKAZAM_IDS
                    and option.inPlayArea == AreaType.ACTIVE
                    and self._attach_helps(target, card)):
                _DIAG["attack_first_energy_attachments"] += 1
            if getattr(card, "id", None) == C.ENRICHING_ENERGY:
                _DIAG["enriching_attachments"] += 1
                if target is not None and target.id in (C.DUNSPARCE, C.DUDUNSPARCE):
                    _DIAG["enriching_cycle_attachments"] += 1
            if target is not None and target.id == C.FEZANDIPITI_EX:
                mode = self._fez_mode(target)
                key = ("fez_pivot_actions" if mode == "PIVOT"
                       else "fez_alternate_attacker_actions")
                _DIAG[key] += 1
                _DIAG["fez_energy_investments"] += 1
            if target is not None and self._support_pivot_ready(target, option.inPlayArea):
                _DIAG["support_pivot_attach_actions"] += 1
            if (target is not None
                    and self._pivot_attach_score(target, option.inPlayArea, card) > 0):
                _DIAG["pivot_route_attachments"] += 1
        elif option.type == OptionType.ATTACK:
            active = self.me.active[0] if self.me.active else None
            if active is not None and active.id == C.FEZANDIPITI_EX:
                _DIAG["fez_alternate_attacker_actions"] += 1
                _DIAG["fez_attack_conversions"] += 1
            opponent = self.opponent.active[0] if self.opponent.active else None
            if (active is not None and active.id == C.ALAKAZAM
                    and opponent is not None and self._effect_prevented(opponent)):
                _DIAG["protected_attack_fallbacks"] += 1
        elif option.type == OptionType.RETREAT:
            active = self.me.active[0] if self.me.active else None
            if self._attack_disabled_on_active():
                _DIAG["attack_lock_retreats"] += 1
            if active is not None and active.id == C.FEZANDIPITI_EX:
                _DIAG["fez_pivot_actions"] += 1
                _DIAG["fez_pivot_conversions"] += 1
            if active is not None and active.id in ONE_ENERGY_PIVOT_IDS:
                _DIAG["support_pivot_retreat_actions"] += 1
            if self._best_pivot_attack_route() is not None:
                _DIAG["pivot_route_retreats"] += 1
        elif option.type == OptionType.ABILITY:
            ability_user = get_card(self.obs, option.area, option.index, self.my_index)
            if ability_user is not None and ability_user.id == C.FEZANDIPITI_EX:
                _DIAG["fez_draw_only_actions"] += 1

        if option.type == OptionType.EVOLVE and getattr(card, "id", None) == C.KADABRA:
            if (self._same_evolution_area_available(C.KADABRA, AreaType.ACTIVE)
                    and self._same_evolution_area_available(C.KADABRA, AreaType.BENCH)):
                key = ("kadabra_dual_active_choices"
                       if option.inPlayArea == AreaType.ACTIVE
                       else "kadabra_dual_bench_choices")
                _DIAG[key] += 1

        if option.type == OptionType.PLAY and getattr(card, "id", None) == C.RARE_CANDY:
            projection = self._candy_route_projection()
            if projection["available"]:
                _DIAG["candy_first_attack_routes"] += 1
            if projection["ko"]:
                _DIAG["candy_immediate_ko_routes"] += 1
        if (option.type == OptionType.PLAY
                and getattr(card, "id", None) == C.SHAYMIN
                and self._opp_threatens_bench()):
            _DIAG["shaymin_threat_plays"] += 1
        if (option.type == OptionType.PLAY
                and getattr(card, "id", None) == C.XEROSIC):
            mirror = any(
                pokemon is not None and pokemon.id in ALAKAZAM_IDS | {C.ABRA, C.KADABRA}
                for pokemon in (self.opponent.active + self.opponent.bench)
            )
            if not mirror:
                _DIAG["xerosic_nonmirror_plays"] += 1

        if option.type == OptionType.PLAY and getattr(card, "id", None) == C.BOSS_ORDERS:
            candidates = [
                pokemon for pokemon in self.opponent.bench
                if pokemon is not None and self._boss_target_score(pokemon) > 0
            ]
            if candidates:
                target = max(candidates, key=self._boss_target_score)
                damage = self._boss_damage_after_spend(target)
                key = "boss_same_turn_plays" if damage >= target.hp else "boss_two_hit_plays"
                _DIAG[key] += 1
                if self._boss_effect_lock_escape_ko(target, damage):
                    _DIAG["boss_effect_lock_escapes"] += 1

    def _score(self, o):
        t = o.type
        # First-or-second: GO FIRST. The Elo≥1150 Alakazam pool goes first 35/35 (unanimous) —
        # a setup/evolution deck wants the extra turn to build the Abra→Kadabra→Alakazam line and
        # get the Dudunsparce draw engine online before it has to attack. (Was hardcoded second.)
        if self.context == SelectContext.IS_FIRST:
            return 100 if t == OptionType.YES else 0
        if t == OptionType.NUMBER:
            return o.number if o.number is not None else 0
        # v3 P0-2: 山札がフロア以下の時、「発動しますか?」系の任意効果(ユンゲラーの
        # Psychic Draw等、多くがデッキ消費ドロー)は辞退して山札を守る。
        if self.context == SelectContext.ACTIVATE:
            activate = self._activate_draw_ok()
            if t == OptionType.YES:
                return 100 if activate else 0
            if t == OptionType.NO:
                return 0 if activate else 1
        if t == OptionType.YES:
            return 1
        if t == OptionType.NO:
            return 0
        if self.context == SelectContext.MAIN:
            terminal_boss_score = self._terminal_boss_gate_score(o)
            if terminal_boss_score is not None:
                return terminal_boss_score
        if (self.context == SelectContext.MAIN
                and t != OptionType.ATTACK
                and self._terminal_win_attack_offered()):
            return -1
        # ── v3 P0-1: 致死維持ゲート ──────────────────────────────────────────
        # A/B検証の学び: Powerful Handは手札を消費しないので「先に展開/ドローして
        # 手札を伸ばし、ターン終端で攻撃」が最適(即攻撃強制は勝率を下げた: 41.5%)。
        # 本当のバグは「手札消費プレイで致死圏を割る」「ターンを攻撃せず終える」の2つ。
        # → 手札を減らす行動は、実行後も致死が維持できる時だけ許可。
        #   致死ギリギリでは展開を止めて攻撃(30000)。勝利KOは常に即攻撃(90000)。
        if (self.context == SelectContext.MAIN
                and t in (OptionType.PLAY, OptionType.EVOLVE, OptionType.ATTACH,
                          OptionType.ENERGY, OptionType.ABILITY, OptionType.RETREAT)
                and self._lethal_attack_offered()):
            if t == OptionType.PLAY and not self.state.supporterPlayed:
                card = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
                if card is not None and card.id == C.BOSS_ORDERS:
                    best_boss = max(
                        (self._boss_target_score(p) for p in self.opponent.bench
                         if p is not None),
                        default=-1,
                    )
                    # A qualified Boss target is an intentional replacement for the
                    # current Active KO (e.g. Rocket Articuno protection engine), not
                    # an accidental lethal-breaking setup action.
                    if best_boss > 0:
                        return 35000 + min(12000, best_boss // 8)
            opp = self.opponent.active[0] if self.opponent.active else None
            a = self.me.active[0] if self.me.active else None
            # 手札枚数=火力なのは743のPowerful Handだけ。245/ユンゲラーの致死は手札非依存。
            if (opp is not None and a is not None and a.id == C.ALAKAZAM
                    and self._hand_delta(t, o) < 0
                    and 20 * (self.me.handCount - 1) < opp.hp):
                return 10          # このプレイで致死を失う → 攻撃を選ばせる
        if t == OptionType.CARD:
            return self._score_card(o)
        if t == getattr(OptionType, "ENERGY_CARD", object()):
            return self._score_card(o)
        if t == OptionType.PLAY:
            return self._score_play(o)
        if t in (OptionType.ENERGY, OptionType.ATTACH):
            # In MAIN these are attachment actions. In DISCARD_ENERGY and
            # related sub-selections, OptionType.ENERGY identifies an attached
            # Energy candidate and must be scored through energyIndex.
            if self.context != SelectContext.MAIN:
                return self._score_card(o)
            return self._score_attach(o)
        if t == OptionType.EVOLVE:
            return self._score_evolve(o)
        if t == OptionType.ABILITY:
            return self._score_ability(o)
        if t == OptionType.RETREAT:
            return self._score_retreat()
        if t == OptionType.ATTACK:
            return self._score_attack(o)
        if t == OptionType.END:
            # Final invariant: after all useful setup is exhausted, an attack that
            # makes any real progress must beat END.  Do not force a zero-effect
            # Powerful Hand into Mist/Fossil/Articuno protection.
            if self._progress_attack_offered():
                return -100000
            opponent = self.opponent.active[0] if self.opponent.active else None
            meaningful_route = (
                self._active_alakazam_attack_route_offered()
                and not self._effect_prevented(opponent)
            )
            return -1000 if meaningful_route else 0
        return 0

    def _item_locked(self):
        """Are we Item-locked (can't play Item cards)? Detect from a known lock
        ability on the opponent's Active, OR from game state: we hold Item card(s)
        but the engine offers no way to play any of them."""
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is not None and opp.id in ITEM_LOCK_IDS:
            return True
        # v3 P0-4: 状態ベースの検知は「PLAYオプションが並ぶMAIN」でだけ意味を持つ。
        # サブ選択(TO_HAND等)ではItemのPLAYが提示されないため常に誤検知していた。
        if self.context != SelectContext.MAIN:
            return False
        items = [c for c in self.me.hand
                 if card_table.get(c.id) is not None and card_table[c.id].cardType == CardType.ITEM]
        if not items:
            return False
        for o in self.select.option:
            if o.type == OptionType.PLAY:
                c = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
                if c is not None and card_table.get(c.id) is not None and card_table[c.id].cardType == CardType.ITEM:
                    return False   # an Item is playable → not locked
        return True

    def _bench_attacker_ready(self):
        """A benched Alakazam that already has the energy to attack (Powerful Hand
        needs 1 {P}). If one exists, we want IT active, not a Dunsparce/Dudunsparce."""
        return any(p is not None and p.id in ALAKAZAM_IDS and self._energy_count(p) >= 1
                   for p in self.me.bench)

    # — abilities —
    def _score_ability(self, o):
        card = get_card(self.obs, o.area, o.index, self.my_index)
        if card is None:
            return 0
        if card.id == C.DUDUNSPARCE:
            # ABSOLUTE v6 safety invariant: Run Away Draw shuffles this Pokémon
            # back into the deck. If it is our only body, activating it loses the
            # game immediately. No Item-lock/reposition/draw exception may bypass
            # this check.
            if self._board_body_count() <= 1:
                return -1
            if o.area != AreaType.BENCH and not any(
                    p is not None for p in self.me.bench):
                return -1
            # Run Away Draw: draw 3 + shuffle this Pokémon back into the deck.
            # v2: 固定フロア(7)→動的フロア(残サイド連動)。致死に届くドローだけ例外。
            if not self._deck_spend_ok(cost=3) and not self._emergency_energy_draw(3):
                return -1
            if self._terminal_pivot_win(o):
                return 88000
            route_score = self._chosen_route_action_score("dudun")
            if route_score > 0:
                return route_score
            if o.area != AreaType.BENCH:
                # ACTIVE copy: CYCLE this weak active out and promote a ready benched
                # attacker (or escape Item-lock), then attack the same turn. This is
                # REPOSITIONING TO ATTACK, not filtering — so it is ALWAYS allowed, even in
                # deck-preserve mode (getting the powered Alakazam active to swing is the
                # whole point). Bug fixed: gating this on _deck_preserve stranded a powered
                # Alakazam on the bench (Dudunsparce active, 0 energy, can't retreat) -> no
                # attacks -> no_offense loss.
                if self._item_locked() or self._bench_attacker_ready():
                    return 14000
                return -1
            # BENCHED copy = the draw engine (pure filtering). Draw-engine decks WIN by
            # drawing aggressively (big hand = big Powerful Hand) — blanket deck-out guards
            # regressed cabt — so we draw, EXCEPT: when we already have a winning hand and the
            # deck is low, stop filtering ourselves out of a won game (real-ladder bug).
            if self._deck_preserve():
                return -1
            # v20: once the selected KO route is already guaranteed without this
            # benched cycle, three more cards are overkill and extra deck exposure.
            # Active Dudunsparce remains exempt because its job is repositioning.
            if self._draw_redundant_for_chosen_target("dudun"):
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
    def _score_play(self, o):
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
                and any(self.hand[s] for s in (C.HILDA, C.DAWN, C.BOSS_ORDERS, C.XEROSIC))):
            base += 900
        return base

    def _score_play_poke(self, card):
        cid = card.id; n = self.field[cid]
        if cid == C.ABRA:
            # Majkel (7-05, 7275 MAIN decisions): moderate bench — our #1 over-pick was
            # flooding bodies (PLAY:Dunsparce 548x / Abra 152x). 3 line bodies is plenty.
            line = (self.field[C.ABRA] + self.field[C.KADABRA]
                    + self.field[C.ALAKAZAM] + self.field[C.ALAKAZAM_PSY])
            if line >= 3:
                return 1500
            return 20000 - 250 * n
        if cid == C.DUNSPARCE:
            if self._opp_has_froslass() and self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE] >= 1:
                return 300    # do not expose a second draw-engine line to repeated counters
            if self.field[C.DUNSPARCE] + self.field[C.DUDUNSPARCE] >= 2:
                return 1200   # cap at 2 engine bodies
            return 18500 - 250 * n
        if cid == C.FEZANDIPITI_EX:
            # Draw engine by default. It is never a generic seventh body: reserve
            # the first attack line and one draw-engine slot, and stop exposing a
            # two-prizer when the opponent can close on it.
            if not self._fez_bench_worthwhile():
                return -1
            if self._articuno_breaker_required():
                return 22000
            return 15200 if self._fez_alternate_matchup() else 12800
        if cid == C.SHAYMIN:
            # Flower Curtain protects the bench from attack damage -> bench it ONLY vs a
            # bench-damage (spread/snipe) opponent; otherwise it just clogs a bench slot.
            return 17000 if (n == 0 and self._opp_threatens_bench()) else -1
        if cid == C.PSYDUCK:
            # Damp only locks self-KO abilities (almost nothing in this meta) -> bench it
            # ONLY when the opponent actually has such an ability in play.
            return 9000 if (n == 0 and self._opp_has_self_ko_ability()) else -1
        if cid == C.GENESECT:
            return 9000 if n == 0 else -1
        return 14000 - 200 * n

    def _alakazam_ready(self):
        a = self.me.active[0] if self.me.active else None
        return a is not None and a.id in ALAKAZAM_IDS and self._energy_count(a) >= 1

    def _need_pieces(self):
        return self.field[C.ALAKAZAM] < 1

    def _open_bench(self):
        return sum(1 for p in self.me.bench if p is not None) < getattr(self.me, "benchMax", 5)

    def _achievable_hand(self):
        """Biggest hand we can realistically reach THIS turn (Powerful Hand = 20×hand):
        current hand + Run Away Draw (+3) + one guaranteed Supporter net gain."""
        extra = 0
        if self.me.deckCount > 7 and any(p is not None and p.id == C.DUDUNSPARCE for p in self.me.bench):
            extra += self._guaranteed_hand_delta("dudun")
        if not self.state.supporterPlayed:
            if self.hand[C.DAWN]:
                extra += self._guaranteed_hand_delta("dawn")
            elif self.hand[C.HILDA]:
                extra += self._guaranteed_hand_delta("hilda")
        return self.me.handCount + extra

    def _have_attacker(self):
        a = self.me.active[0] if self.me.active else None
        return (a is not None and a.id in ALAKAZAM_IDS and self._energy_count(a) >= 1) or self._bench_attacker_ready()

    def _ready_alakazam_count(self):
        return sum(
            p is not None and p.id in ALAKAZAM_IDS and self._can_attack(p)
            for p in self._my_board()
        )

    def _backup_eta(self):
        """Turns until a deterministic *benched* backup Alakazam can attack.

        v13 called this method from the Rich draw-stop rule without defining it,
        causing live policy fallbacks.  Keep the estimate conservative: only cards
        already public to us and a single normal next-turn evolution/attachment are
        counted.  Unknown draws never make the ETA optimistic.
        """
        psychic_available = self._psychic_in_hand()
        for pokemon in (self.me.bench or []):
            if pokemon is None:
                continue
            if pokemon.id in ALAKAZAM_IDS:
                if self._can_attack(pokemon):
                    return 0
                if psychic_available:
                    return 1
            if (pokemon.id == C.KADABRA and self.hand[C.ALAKAZAM] > 0
                    and (self._can_attack(pokemon) or psychic_available)):
                return 1
            if (pokemon.id == C.ABRA and self.hand[C.RARE_CANDY] > 0
                    and self.hand[C.ALAKAZAM] > 0
                    and (self._energy_count(pokemon) > 0 or psychic_available)):
                return 1
        return 99

    def _candy_route_projection(self):
        if self.hand[C.RARE_CANDY] <= 0 or self.hand[C.ALAKAZAM] <= 0:
            return {"available": False, "damage": 0, "ko": False}
        opponent = self.opponent.active[0] if self.opponent.active else None
        if opponent is None or self._effect_prevented(opponent):
            return {"available": False, "damage": 0, "ko": False}
        can_attach = not getattr(self.state, "energyAttached", False) and self._psychic_in_hand()
        candidates = [
            p for p in self._my_board()
            if p is not None and p.id == C.ABRA and not getattr(p, "appearThisTurn", False)
            and (self._energy_count(p) >= 1 or can_attach)
        ]
        if not candidates:
            return {"available": False, "damage": 0, "ko": False}
        post_hand = max(0, self.me.handCount - 2 + min(3, self.me.deckCount))
        if all(self._energy_count(p) == 0 for p in candidates):
            post_hand -= 1
        damage = 20 * max(0, post_hand)
        return {"available": True, "damage": damage, "ko": damage >= opponent.hp}

    def _candy_accelerates_first_attack(self):
        if self._ready_alakazam_count() > 0:
            return False
        projection = self._candy_route_projection()
        if not projection["available"]:
            return False
        can_attach = not getattr(self.state, "energyAttached", False) and self._psychic_in_hand()
        return not any(
            p is not None and p.id == C.KADABRA
            and (self._energy_count(p) >= 1 or can_attach)
            for p in self._my_board()
        )

    def _kadabra_draws_candy_for_active(self, option, target):
        if (getattr(option, "inPlayArea", None) != AreaType.BENCH
                or target is None or target.id != C.ABRA
                or self.hand[C.RARE_CANDY] > 0 or self.hand[C.ALAKAZAM] <= 0):
            return 0
        active = self.me.active[0] if self.me.active else None
        if (active is None or active.id != C.ABRA
                or getattr(active, "appearThisTurn", False)):
            return 0
        return 2200 + int(4000 * self._draw_probability(C.RARE_CANDY, 2))

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
                   and o.attackId in (POWERFUL_HAND, PSYCHIC_ATK, SUPER_PSY_BOLT)
                   for o in (self.select.option or []))

    def _progress_attack_offered(self):
        """An offered attack that produces damage/counters in the current state."""
        opponent = self.opponent.active[0] if self.opponent.active else None
        if opponent is None:
            return False
        for option in (self.select.option or []):
            if getattr(option, "type", None) != OptionType.ATTACK:
                continue
            attack_id = getattr(option, "attackId", None)
            if attack_id == FEZANDIPITI_ATTACK:
                if any(
                    pokemon is not None
                    and not self._temporary_attack_immunity_applies(pokemon)
                    for pokemon in (self.opponent.active + self.opponent.bench)
                ):
                    return True
                continue
            if self._alakazam_damage(attack_id, opponent) > 0:
                return True
        return False

    def _terminal_win_attack_offered(self):
        """An offered attack wins by prizes or by removing the final opposing body."""
        opponent = self.opponent.active[0] if self.opponent.active else None
        if opponent is None:
            return False
        board_wipe = not any(
            pokemon is not None for pokemon in self.opponent.bench
        )
        for option in (self.select.option or []):
            if getattr(option, "type", None) != OptionType.ATTACK:
                continue
            damage = self._alakazam_damage(
                getattr(option, "attackId", None), opponent
            )
            if damage < opponent.hp:
                continue
            if board_wipe or prize_count(opponent) >= len(self.me.prize):
                return True
        return False

    def _terminal_pivot_win(self, option=None):
        """Active Dudunsparce can draw, leave play, and promote a winning attacker."""
        active = self.me.active[0] if self.me.active else None
        opponent = self.opponent.active[0] if self.opponent.active else None
        if (active is None or active.id != C.DUDUNSPARCE or opponent is None
                or any(pokemon is not None for pokemon in self.opponent.bench)
                or self._board_body_count() <= 1):
            return False
        if option is not None and getattr(option, "area", None) == AreaType.BENCH:
            return False
        for pokemon in self.me.bench:
            if pokemon is None or pokemon.id not in ALAKAZAM_IDS:
                continue
            if not self._can_attack(pokemon) or self._effect_prevented(opponent):
                continue
            if pokemon.id == C.ALAKAZAM:
                damage = 20 * (self.me.handCount + 3)
            else:
                damage = self._route_attack_damage(pokemon, opponent)
            if damage >= opponent.hp:
                return True
        return False

    def _winning_gust_ready(self):
        """ベンチに「呼べば今の攻撃でKOでき、そのサイドで勝ち切れる」対象がいて、
        今のActiveのKOでは勝ち切れない場合のみTrue(致死中に許す唯一の展開行動)。"""
        opp = self.opponent.active[0] if self.opponent.active else None
        if opp is not None and prize_count(opp) >= len(self.me.prize):
            return False       # 今のActiveを倒せば勝ち — 呼ぶ必要なし
        return any(p is not None and self._active_best_dmg(p) >= p.hp
                   and prize_count(p) >= len(self.me.prize)
                   for p in self.opponent.bench)

    def _ko_active_reachable(self):
        """Can Powerful Hand KO the opponent's ACTIVE this turn — now, or after the
        drawing still available to us? (Each turn, aim to KO the best target: usually
        the dangerous active attacker, by pumping the hand to lethal.)"""
        opp = self.opponent.active[0] if self.opponent.active else None
        return (opp is not None and self._have_attacker()
                and not self._effect_prevented(opp)        # Mist Energy etc. → 0, don't chase it
                and 20 * self._achievable_hand() >= opp.hp)

    def _lana_recoverable_count(self):
        recoverable_ids = (
            C.PSYCHIC_ENERGY,
            C.ABRA,
            C.KADABRA,
            C.ALAKAZAM,
            C.DUNSPARCE,
            C.DUDUNSPARCE,
            C.SHAYMIN,
        )
        return sum(self.discard.get(card_id, 0) for card_id in recoverable_ids)

    def _score_play_trainer(self, card):
        cid = card.id
        route_kind = {
            C.ENHANCED_HAMMER: "hammer",
            C.RARE_CANDY: "rare_candy",
            C.DAWN: "dawn",
            C.HILDA: "hilda",
            C.BOSS_ORDERS: "boss",
        }.get(cid)
        if route_kind is not None:
            route_score = self._chosen_route_action_score(route_kind)
            if route_score > 0:
                return route_score
        ready = self._alakazam_ready()
        if cid == C.RARE_CANDY:
            if self.field[C.ABRA] >= 1 and self.hand[C.ALAKAZAM] >= 1:
                if (self._draw_redundant_for_chosen_target("rare_candy")
                        and self._backup_eta() <= 1):
                    return 4500
                projection = self._candy_route_projection()
                if projection["available"] and projection["ko"]:
                    return 33000
                if self._candy_accelerates_first_attack():
                    return 23500
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
        # Winning + deck low: stop spending the deck on draw/search supporters — preserve it
        # so we can draw 1/turn to the finish (Boss's Orders gust is still allowed below).
        if cid in (C.HILDA, C.DAWN, C.POKE_PAD) and self._deck_preserve():
            return -1
        # v2: 負けている時もデッキ切れは即負け。フロアを割るデッキ消費は致死直結時のみ。
        if cid in (C.HILDA, C.DAWN, C.POKE_PAD, C.BUDDY_POFFIN) \
                and not self._deck_spend_ok(cost=2):
            return -1
        if cid == C.HILDA:
            if self.state.supporterPlayed:
                return -1
            if self._draw_redundant_for_chosen_target("hilda"):
                return 5000 if self._need_pieces() else 2200
            if draw_for_ko:
                return 14000
            return 12500 if self._need_pieces() else 5000
        if cid == C.DAWN:
            if self.state.supporterPlayed:
                return -1
            if self._draw_redundant_for_chosen_target("dawn"):
                return 5000 if self._need_pieces() else 2200
            if draw_for_ko:
                return 13800
            # Majkel plays Dawn broadly (214x divergent) — +3 hand = +60 Powerful Hand
            return 12000 if self._need_pieces() else 7500
        if cid == C.BUDDY_POFFIN:
            bodies = (self.field[C.ABRA] + self.field[C.KADABRA] + self.field[C.ALAKAZAM]
                      + self.field[C.ALAKAZAM_PSY] + self.field[C.DUNSPARCE]
                      + self.field[C.DUDUNSPARCE])
            if bodies >= 4 or not self._open_bench():
                return 600   # board is set — a Poffin now is -20 Powerful Hand for nothing
            return 13000
        if cid == C.POKE_PAD:
            # Majkel keeps digging with it after setup too — every deck→hand card
            # is +20 Powerful Hand (but below Poffin/supporters)
            return 8500 if self._need_pieces() else 3500
        if cid == C.BOSS_ORDERS:
            if self.state.supporterPlayed:
                return -1
            values = [self._boss_target_score(p)
                      for p in self.opponent.bench if p is not None]
            best = max(values, default=-1)
            if best <= 0:
                return -1
            # Boss is a tactical KO card, not a generic switch effect. Place it
            # above ordinary setup only when a qualified target exists.
            return 18500 + min(12000, best // 8)
        if cid == C.XEROSIC:
            # v2: 相手の手札を3枚に。ミラー(相手もPowerful Hand)では手札=火力なので
            # 相手の手札が肥えた時に最優先で撃つ。他デッキ相手でも大量ハンドには妨害価値。
            if self.state.supporterPlayed:
                return -1
            opp_hand = getattr(self.opponent, "handCount", 0) or 0
            opp_board = [p for p in (self.opponent.active + self.opponent.bench)
                         if p is not None]
            mirror = any(p.id in (C.ABRA, C.KADABRA, C.ALAKAZAM, C.ALAKAZAM_PSY)
                         for p in opp_board)
            # v3 P1-3: 3枚体制になったので発動閾値を緩和。ミラー(相手もPowerful Hand)
            # では手札=火力なので、手札6+で最優先。一位デッキもクセロシキ3枚採用。
            if mirror and opp_hand >= 6:
                return 15500        # 20×(手札-3超分)を丸ごと削る — ドローより優先
            if mirror and opp_hand >= 4:
                return 9000
            developing = sum(
                1 for pokemon in opp_board
                if not getattr(card_table.get(pokemon.id), "stage2", False)
            )
            if opp_hand >= 9 and developing >= 2:
                return 13000
            return -1              # non-mirror disruption needs a concrete large-hand target
        if cid == C.ENHANCED_HAMMER:
            # Strip Mist/effect-prevention Energy anywhere on the board. A benched
            # Mist is still the highest-value target because it can become Active.
            attached = self._attached_special_energy_entries()
            if any(energy.id in EFFECT_PREVENT_ENERGY for _, _, energy in attached):
                return 17000
            # Once Mist has actually appeared, preserve every likely answer rather
            # than only the final Hammer (the v13 policy's observed failure).
            if self._should_reserve_hammer_for_seen_mist():
                return -1
            if self._should_reserve_last_hammer():
                return -1
            # otherwise only worth it if the opponent has any Special Energy to remove
            if any(card_table.get(getattr(e, 'id', None)) is not None
                   and card_table[e.id].cardType == CardType.SPECIAL_ENERGY
                   for p in (self.opponent.active + self.opponent.bench) if p is not None
                   for e in (getattr(p, 'energyCards', None) or [])):
                return 9500   # Majkel hammers special energy on sight (248x on 7-06)
            return -1
        if cid == C.NIGHTTIME_MINE:
            if not self._nighttime_mine_worthwhile():
                return -1
            return 14500 if self._nighttime_mine_tax_stops_active() else 7600
        if cid == C.LUCKY_HELMET:
            return 7000 if not ready else 1000
        if cid == C.NIGHT_STRETCHER:
            recoverable = (self.discard.get(C.ALAKAZAM, 0) or self.discard.get(C.ABRA, 0)
                           or self.discard.get(C.KADABRA, 0) or self.discard.get(C.DUNSPARCE, 0)
                           or self.discard.get(C.PSYCHIC_ENERGY, 0))
            return 7500 if recoverable else 300
        if cid == C.LANA_AID:
            if self.state.supporterPlayed:
                return -1
            recoverable = self._lana_recoverable_count()
            if recoverable <= 0:
                return -1
            if recoverable >= 3:
                return 16000 if self._low_deck() else 13200
            if recoverable == 2:
                return 10500
            return 4500
        if cid == C.SACRED_ASH:
            # v3 P0-3: 山札にポケモン5枚を戻す=唯一の「山札回復」札。デッキ切れ負け5件
            # への直接対策として、山札14枚以下+トラッシュにライン3枚以上で最優先級に昇格。
            line_in_discard = sum(self.discard.get(x, 0) for x in
                                  (C.ABRA, C.KADABRA, C.ALAKAZAM, C.ALAKAZAM_PSY,
                                   C.DUNSPARCE, C.DUDUNSPARCE))
            if self.me.deckCount <= 14 and line_in_discard >= 3:
                return 12000
            return 6000 if self._low_deck() and self.me.discard else 200
        if cid == C.WONDROUS_PATCH:
            return 7000 if self.discard.get(C.PSYCHIC_ENERGY, 0) and self._open_bench() else 300
        return 9000

    # — evolve —
    def _same_evolution_area_available(self, card_id, area):
        for option in (self.select.option or []):
            if (getattr(option, 'type', None) != OptionType.EVOLVE
                    or getattr(option, 'inPlayArea', None) != area):
                continue
            card = get_card(self.obs, AreaType.HAND, option.index, self.my_index)
            if card is not None and card.id == card_id:
                return True
        return False

    def _kadabra_target_bonus(self, o, target):
        """Use Bench as the default, with explicit Active tempo exceptions.

        v7 chose Active 35/35 times and v8 chose Bench 34/34 times, while the
        top reference used Bench 120/129. v9 keeps that prior but selects Active
        when evolution immediately creates the only meaningful attack, a KO, or
        enough extra HP to avoid leaving an unfuelled line stranded in front.
        """
        both_areas = (
            self._same_evolution_area_available(C.KADABRA, AreaType.ACTIVE)
            and self._same_evolution_area_available(C.KADABRA, AreaType.BENCH)
        )
        if not both_areas:
            return 0
        opponent = self.opponent.active[0] if self.opponent.active else None
        target_area = getattr(o, 'inPlayArea', None)
        if target_area == AreaType.ACTIVE:
            active_ready = self._energy_count(target) >= 1
            immediate_ko = bool(
                opponent is not None
                and active_ready
                and self._alakazam_damage(SUPER_PSY_BOLT, opponent) >= opponent.hp
            )
            if immediate_ko:
                return 2200
            if active_ready and not self._bench_attacker_ready():
                return 1300
            if (opponent is not None and not self._bench_attacker_ready()
                    and target.hp <= 30 and self._can_attack(opponent)):
                return 1050
            return 0
        if target_area != AreaType.BENCH:
            return 0

        bonus = 900
        if target.hp < getattr(target, 'maxHp', target.hp):
            bonus += 200
        if opponent is not None and self._can_attack(opponent):
            bonus += 250
        if self._opp_has_froslass():
            bonus += 100
        return bonus + self._kadabra_draws_candy_for_active(o, target)

    def _score_evolve(self, o):
        target = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.my_index)
        if not isinstance(target, Pokemon):
            return 0
        card = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
        cid = card.id if card is not None else None
        route_kind = {
            C.ALAKAZAM: "evolve_alakazam",
            C.KADABRA: "evolve_kadabra",
        }.get(cid)
        if route_kind is not None:
            route_score = self._chosen_route_action_score(route_kind)
            if route_score > 0:
                return route_score
            if (self._draw_redundant_for_chosen_target(route_kind)
                    and self._backup_eta() <= 1):
                return 5000
        if cid == C.ALAKAZAM_PSY:
            # The Psychic tech (bypasses Mist, punishes energy). Make THIS Alakazam only
            # when (a) the opp Active is Mist-protected AND we can't strip it (no Enhanced
            # Hammer in hand), or (b) it's heavily energy-loaded. Otherwise the 743 Powerful
            # Hand (after Enhanced Hammer if needed) is our higher-ceiling main attacker.
            opp = self.opponent.active[0] if self.opponent.active else None
            if opp is not None and ((self._effect_prevented(opp) and self.hand[C.ENHANCED_HAMMER] == 0)
                                    or len(opp.energies) >= 4):
                return 21500
            return 20400
        if cid == C.ALAKAZAM:
            # One attacking Alakazam at a time — each extra evolve burns a hand card
            # (-20 Powerful Hand). Majkel does evolve the ACTIVE Kadabra (fresh attacker)
            # even with one Alakazam up, but doesn't stack bench Alakazams.
            have = self.field[C.ALAKAZAM] + self.field[C.ALAKAZAM_PSY]
            if have == 0 or o.inPlayArea == AreaType.ACTIVE:
                return 21000
            return 4000
        if cid == C.KADABRA:
            # JIT (Majkel 7-06: his 237 vs our 1120): evolve when BRIDGING to Alakazam or
            # when the hand needs the +3 draw — otherwise the piece is safer in hand
            # (on board it's Grimmsnarl-snipe/Froslass-chip bait, in hand it's +20 dmg).
            target_bonus = self._kadabra_target_bonus(o, target)
            if self._candy_accelerates_first_attack():
                return 7000 + target_bonus
            if self.hand[C.ALAKAZAM] >= 1 or self.me.handCount <= 4                     or self.field[C.ALAKAZAM] + self.field[C.ALAKAZAM_PSY] == 0:
                return 20000 + target_bonus
            return 6000 + target_bonus
        if cid == C.DUDUNSPARCE:
            return 19000
        return 18000

    # ── v2: エンリッチ vs 超エネルギーの貼り分け ──────────────────────────────
    def _need_p_fuel(self):
        """{P}を貼りたいアタッカーがいて、手札に{P}系エネルギーを持っているか。"""
        holds_p = self.hand[C.PSYCHIC_ENERGY] or self.hand[C.TELEPATH_ENERGY]
        needs = any(p is not None and p.id in (*ALAKAZAM_IDS, C.ABRA, C.KADABRA)
                    and self._should_fuel(p) for p in self._my_board())
        return bool(holds_p and needs)

    def _enrich_cycle_ready(self, pokemon):
        if pokemon is None or pokemon.id not in (C.DUNSPARCE, C.DUDUNSPARCE):
            return False
        if pokemon.id == C.DUDUNSPARCE:
            return True
        return bool(
            self.hand[C.DUDUNSPARCE] > 0
            and not getattr(pokemon, "appearThisTurn", False)
        )

    def _enrich_draw_needed(self):
        if self._draw_redundant_for_chosen_target("enriching"):
            return False
        opponent = self.opponent.active[0] if self.opponent.active else None
        required = self._attack_hand_required(opponent) if opponent is not None else 0
        current_ready = self._ready_alakazam_count() > 0
        current_ko = bool(
            current_ready and opponent is not None and not self._effect_prevented(opponent)
            and self.me.handCount >= required
        )
        # Once current KO and a one-turn backup are both secured, avoid consuming
        # four more deck cards merely for overkill.
        if current_ko and self._backup_eta() <= 1 and not self._energy_starved():
            return False
        return bool(
            self.me.handCount <= max(8, required)
            or self._need_pieces()
            or self._energy_starved()
        )

    def _enriching_attach_score(self, pokemon, *, is_active=False):
        """Use Rich as controlled draw, while preserving immediate attack fuel.

        v14's 24,800-point attach/evolve/cycle route over-consumed the deck and
        doubled down on optional draw.  The v11 hand/fuel gate produced better
        attack continuity in fixed-deck ablations, so v15 restores that gate
        without removing v14's hard Active-attack invariant.
        """
        del is_active
        if pokemon is None or not self._deck_spend_ok(cost=4, allow_lethal=False):
            return -1
        if self._active_alakazam_can_be_fueled():
            return -1
        if self.field[C.DUDUNSPARCE] >= 1 and self.me.handCount >= 8:
            return -1
        if not self._enrich_draw_needed():
            return -1
        starving = self.me.handCount <= 5
        need_psychic_now = self._need_p_fuel()
        if starving:
            score = 7800 if need_psychic_now else 8300
        elif not need_psychic_now:
            score = 6500
        else:
            return -1
        if pokemon.id in (C.DUNSPARCE, C.DUDUNSPARCE):
            score += 150
        return score

    def _fez_attach_score(self, pokemon, is_active, source=None):
        if getattr(source, "id", None) == C.ENRICHING_ENERGY:
            if (self._articuno_breaker_mode()
                    and self._energy_count(pokemon) < 3
                    and self._deck_spend_ok(cost=4, allow_lethal=False)):
                return 16500 + self._energy_count(pokemon) * 600
            return self._enriching_attach_score(pokemon, is_active=is_active)
        mode = self._fez_mode(pokemon)
        attached = self._energy_count(pokemon)
        if mode == "PIVOT":
            return 17500 if is_active and attached == 0 else -1
        if mode == "ALTERNATE_ATTACKER":
            if self._articuno_breaker_mode() and attached < 3:
                return 16500 + (500 if is_active else 0) + attached * 600
            if self._fez_energy_eta(pokemon) <= 1:
                return 12500 + (500 if is_active else 0) + attached * 150
        return -1

    def _support_pivot_ready(self, pokemon, area):
        """A single attachment can turn a stranded Active into a same-turn attack.

        v9 correctly escaped Fezandipiti ex, but the ladder logs show the same
        failure after Shaymin/Dunsparce/Abra/Kadabra promotions.  Those cards all
        retreat for one Energy.  Dudunsparce is deliberately excluded: Run Away
        Draw is its cheaper escape route and preserves the attachment.
        """
        return bool(
            pokemon is not None
            and area == AreaType.ACTIVE
            and pokemon.id in ONE_ENERGY_PIVOT_IDS
            and self._energy_count(pokemon) == 0
            and self._bench_attacker_ready()
        )

    def _support_pivot_attach_score(self, pokemon, area, source=None):
        if not self._support_pivot_ready(pokemon, area):
            return -1
        # Basic Energy is marginally preferable: Telepath remains a searchable
        # attacker fuel and can force two additional deck pulls in a low-deck game.
        basic_bonus = 100 if getattr(source, "id", None) == C.PSYCHIC_ENERGY else 0
        return 17600 + basic_bonus

    # — attach energy —
    def _score_attach(self, o):
        p = get_card(self.obs, o.inPlayArea, o.inPlayIndex, self.my_index)
        if not isinstance(p, Pokemon):
            return 0
        src = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
        if getattr(src, "id", None) == C.ENRICHING_ENERGY:
            route_score = self._chosen_route_action_score("enriching")
            if route_score > 0:
                return route_score
        route_score = self._pivot_attach_score(p, o.inPlayArea, src)
        if (self._fez_breaker_energy_needed() and p.id != C.FEZANDIPITI_EX
                and route_score <= 0):
            return -1
        if (o.inPlayArea == AreaType.ACTIVE and p.id in ALAKAZAM_IDS
                and self._should_fuel(p) and self._attach_helps(p, src)):
            # This attachment creates a same-turn Alakazam attack. It must beat
            # Rich-cycle and backup-development attachments, which consume the
            # same once-per-turn resource.
            return max(28000, route_score)
        if p.id == C.FEZANDIPITI_EX:
            return max(
                route_score,
                self._fez_attach_score(p, o.inPlayArea == AreaType.ACTIVE, src),
            )
        if route_score > 0:
            return route_score
        pivot_score = self._support_pivot_attach_score(p, o.inPlayArea, src)
        if pivot_score > 0:
            return pivot_score
        # v2: エンリッチは「攻撃を可能にするか」の汎用ゲートを通さず専用ロジックで判断
        if src is not None and src.id == C.ENRICHING_ENERGY:
            return self._enriching_attach_score(
                p, is_active=o.inPlayArea == AreaType.ACTIVE
            )
        # GENERAL RULE (type-aware): attach only while the body still can't pay an attack;
        # once it CAN attack, hold the rest (fuels a backup AND +20 Powerful Hand per card).
        if not self._should_fuel(p):
            return -1
        if not self._attach_helps(p, src):
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
        if self._attack_disabled_on_active() and self._bench_attacker_ready():
            active_evolution = any(
                option.type == OptionType.EVOLVE
                and getattr(option, "inPlayArea", None) == AreaType.ACTIVE
                for option in (self.select.option or [])
            )
            return 17500 if active_evolution else 45000
        if active.id not in ALAKAZAM_IDS:
            for p in self.me.bench:
                if p is not None and p.id in ALAKAZAM_IDS and self._energy_count(p) >= 1:
                    # Once the one-Energy pivot has been paid, retreat before a
                    # low-value support attack can consume the turn.
                    return 16800 if active.id in ONE_ENERGY_PIVOT_IDS else 6000
        route = self._best_pivot_attack_route()
        if route is not None:
            return self._pivot_action_score(route)
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
            if active.id not in ALAKAZAM_IDS and active.id != C.KADABRA and self._bench_attacker_ready():
                return 5000
            return 700
        if aid == FEZANDIPITI_ATTACK:
            targets = [
                p for p in (self.opponent.active + self.opponent.bench)
                if p is not None
            ]
            best = max((self._fez_target_score(p) for p in targets), default=-1)
            if best < 0:
                _DIAG["temporary_immunity_blocks"] += 1
                return -1
            if any(getattr(p, "hp", 0) <= 100
                   and not self._temporary_attack_immunity_applies(p) for p in targets):
                return max(26000, best)
            return 1800 if self._fez_alternate_matchup() else 900
        # Score THIS specific attack by its own damage — not the best available attack.
        # (Strange Hacking 338 does 0 damage, just confuses; scoring it like Psychic made
        # the agent spam it: opponent can't attack, but we deal 0 → stall → we deck out.)
        dmg = self._alakazam_damage(aid, opp)
        if self._temporary_attack_immunity_applies(opp):
            _DIAG["temporary_immunity_blocks"] += 1
            return -1
        if aid == POWERFUL_HAND and self._effect_prevented(opp):
            return -1
        if aid == STRANGE_HACKING:
            # Utility only: worth a little to Confuse a threatening Active we can't yet KO,
            # but never over a real attack and never as a stall. Stays below END-beating
            # real attacks; above END so it's a last resort if nothing else can act.
            opp_dangerous = prize_count(opp) >= 2 and self._achievable_hand() * 20 < opp.hp
            return 600 if opp_dangerous else 200
        if dmg <= 0:
            return 500
        if (opp.hp <= dmg
                and not any(pokemon is not None for pokemon in self.opponent.bench)):
            return 95000
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
    def _score_rare_candy_selection(self, option, card):
        if card is None:
            return 0
        if card.id == C.ALAKAZAM:
            return 5000
        if not isinstance(card, Pokemon) or card.id != C.ABRA:
            return 0
        area = getattr(option, "area", getattr(option, "inPlayArea", None))
        fueled = self._energy_count(card) >= 1
        can_attach = not getattr(self.state, "energyAttached", False) and self._psychic_in_hand()
        score = 1000 + (2000 if fueled or can_attach else 0)
        if area == AreaType.ACTIVE:
            score += 2500 if fueled or can_attach else 300
        return score

    def _score_card(self, o):
        card = get_card(self.obs, o.area, o.index, o.playerIndex)
        attached_to = card if isinstance(card, Pokemon) else None
        energy_index = getattr(o, "energyIndex", None)
        if attached_to is not None and energy_index is not None:
            energy_cards = getattr(attached_to, "energyCards", None) or []
            card = energy_cards[energy_index] if 0 <= energy_index < len(energy_cards) else None
        if card is None:
            return 0
        ctx = self.context
        if (ctx in (getattr(SelectContext, "EVOLVES_FROM", object()),
                    getattr(SelectContext, "EVOLVES_TO", object()))
                and getattr(self._context_effect_card(), "id", None) == C.RARE_CANDY):
            return self._score_rare_candy_selection(o, card)
        # Opponent card targeting (e.g. Enhanced Hammer: discard a Special Energy from
        # opp) — strip the Mist/Rock that's blocking Powerful Hand, prefer the Active.
        if o.playerIndex == self.op_index and not isinstance(card, Pokemon):
            if getattr(self._context_effect_card(), "id", None) == C.ENHANCED_HAMMER:
                area = getattr(o, "area", None) or getattr(o, "inPlayArea", None)
                return self._hammer_target_score(card, attached_to, area)
            if card.id in EFFECT_PREVENT_ENERGY:
                return 2000 + (500 if getattr(o, 'inPlayArea', None) == AreaType.ACTIVE else 0)
            d = card_table.get(card.id)
            if d is not None and d.cardType == CardType.SPECIAL_ENERGY:
                return 300
            return 50
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
            return 100 if is_energy(card.id) else 10
        if ctx in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
                   SelectContext.DISCARD_ENERGY, SelectContext.DISCARD_ENERGY_CARD):
            return self._score_discard(card)
        if ctx in (SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY,
                   getattr(SelectContext, "DAMAGE", object())):
            if isinstance(card, Pokemon) and o.playerIndex == self.op_index:
                if ctx == getattr(SelectContext, "DAMAGE", None):
                    return self._fez_target_score(card)
                return 10000 + prize_count(card) * 1000 - getattr(card, "hp", 0)
            return 0
        if ctx in (SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM, SelectContext.TO_PRIZE):
            # Sacred Ash (TO_DECK from the DISCARD pile): recycle ALL 5 slots with line
            # pokemon — Majkel fills it (his 5-card picks vs our 3; TO_DECK agree 12%).
            if getattr(o, 'area', None) == AreaType.DISCARD:
                cid = card.id
                if cid in (C.ABRA, C.KADABRA, C.ALAKAZAM, C.ALAKAZAM_PSY):
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
        # v2: ATTACH_TO(貼り先選択)でも、貼るのがエンリッチなら専用ロジック
        cc = self._context_effect_card()
        area = AreaType.ACTIVE if is_active else AreaType.BENCH
        route_score = self._pivot_attach_score(p, area, cc)
        if (self._fez_breaker_energy_needed()
                and self.field[C.FEZANDIPITI_EX] > 0
                and p.id != C.FEZANDIPITI_EX
                and route_score <= 0):
            return -1
        if (cc is not None and is_active and p.id in ALAKAZAM_IDS and self._should_fuel(p)
                and self._attach_helps(p, cc)):
            return max(28000, route_score)
        if p.id == C.FEZANDIPITI_EX:
            return max(route_score, self._fez_attach_score(p, is_active, cc))
        if route_score > 0:
            return route_score
        pivot_score = self._support_pivot_attach_score(
            p,
            AreaType.ACTIVE if is_active else AreaType.BENCH,
            cc,
        )
        if pivot_score > 0:
            return pivot_score
        if cc is not None and getattr(cc, "id", None) == C.ENRICHING_ENERGY:
            return self._enriching_attach_score(p, is_active=is_active)
        if not self._should_fuel(p):
            return -1             # already CAN attack (type-aware) -> don't over-fill
        if p.id in ALAKAZAM_IDS:
            return 8000 + (200 if is_active else 0)
        if p.id in (C.ABRA, C.KADABRA):
            return 1500
        return -1

    def _opponent_can_ko_target_next_turn(self, target):
        if target is None:
            return False
        attacker = self.opponent.active[0] if self.opponent.active else None
        data = card_table.get(getattr(attacker, "id", None))
        if attacker is None or data is None or not self._can_attack(attacker):
            return False
        if attacker.id == C.ALAKAZAM:
            return 20 * int(getattr(self.opponent, "handCount", 0) or 0) >= target.hp
        fixed = max(
            (
                int(getattr(ATTACK_TABLE.get(attack_id), "damage", 0) or 0)
                for attack_id in (getattr(data, "attacks", None) or [])
            ),
            default=0,
        )
        if fixed >= target.hp:
            return True
        # Unknown/variable ex attacks often expose zero base damage in metadata.
        # Treat a visibly powered ex as lethal only to genuinely fragile bodies.
        return bool(
            (getattr(data, "ex", False) or getattr(data, "megaEx", False))
            and self._energy_count(attacker) >= 2
            and target.hp <= 140
        )

    def _promotion_attacks_next_turn(self, card):
        if card is None:
            return False
        if card.id in ALAKAZAM_IDS:
            return self._can_attack(card) or self._psychic_in_hand()
        opponent = self.opponent.active[0] if self.opponent.active else None
        return bool(
            opponent is not None
            and self._can_attack(card)
            and self._route_attack_damage(card, opponent) >= opponent.hp
        )

    def _score_end_turn_switch_choice(self, card):
        """Choose the shield left Active after Abra/Trading Places ends our turn."""
        threatened = self._opponent_can_ko_target_next_turn(card)
        opponent_prizes = len(self.opponent.prize)
        if card.id == C.DUNSPARCE:
            return 5200 - int(threatened) * 150
        if card.id == C.DUDUNSPARCE:
            return 4800 - int(threatened) * 150
        if card.id == C.FEZANDIPITI_EX:
            if opponent_prizes >= 5:
                return 4100 - int(threatened) * 300
            return 2100 if not threatened else -1200
        if card.id in ALAKAZAM_IDS:
            # Only expose the attacker when no disposable pivot exists and it is
            # expected to survive; otherwise preserve the deck's win condition.
            return 3000 if not threatened else -700
        if card.id == C.KADABRA:
            return 1800 if not threatened else -900
        if card.id == C.ABRA:
            return 1300 if not threatened else -1100
        if card.id in (C.PSYDUCK, C.SHAYMIN, C.GENESECT):
            return 900 if not threatened else 300
        return 1000 + getattr(card, "hp", 0) // 20 - int(threatened) * 500

    def _score_ko_promotion(self, card):
        """Promotion after our Active was KO'd: attacker now, otherwise a shield."""
        if self._promotion_attacks_next_turn(card):
            score = 12000
            if card.id in ALAKAZAM_IDS:
                score += 3000
            if (card.id == C.FEZANDIPITI_EX
                    and self._articuno_breaker_mode()):
                score += 12000
            return score

        if card.id == C.DUNSPARCE:
            return 8500
        if card.id == C.DUDUNSPARCE:
            return 8000
        if card.id == C.FEZANDIPITI_EX:
            threatened = self._opponent_can_ko_target_next_turn(card)
            if len(self.opponent.prize) >= 5:
                return 7000
            return 5200 if not threatened else 1200
        if card.id in (C.PSYDUCK, C.SHAYMIN, C.GENESECT):
            return 4300
        if card.id == C.KADABRA:
            return 3200 if not self._opponent_can_ko_target_next_turn(card) else -800
        if card.id == C.ABRA:
            return 2600 if not self._opponent_can_ko_target_next_turn(card) else -1000
        if card.id in ALAKAZAM_IDS:
            return 2400
        return 3500 + getattr(card, "hp", 0) // 20

    def _score_active_choice(self, o, card):
        if not isinstance(card, Pokemon):
            return 0
        if o.playerIndex == self.op_index:
            return self._gust_value(card)
        if o.playerIndex != self.my_index:
            return 0
        if self.context == getattr(SelectContext, "TO_ACTIVE", object()):
            return self._score_ko_promotion(card)
        effect = self._context_effect_card()
        if (
            self.context == SelectContext.SWITCH
            and getattr(effect, "id", None) in (C.ABRA, C.DUNSPARCE)
        ):
            return self._score_end_turn_switch_choice(card)
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
        elif card.id == C.FEZANDIPITI_EX:
            if self._can_attack(card) and any(
                    p is not None and getattr(p, "hp", 0) <= 100
                    for p in (self.opponent.active + self.opponent.bench)):
                score += 180
            else:
                score -= 200     # draw engine and two-prize liability stay on Bench
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
        elif card.id in (C.PSYDUCK, C.SHAYMIN, C.GENESECT):
            score -= 20          # tech bodies: don't promote into the attacker slot
        if (card.id == C.FEZANDIPITI_EX
                and self._articuno_breaker_mode()
                and self._can_attack(card)):
            score += 12000
        score += getattr(card, 'hp', 0) // 30   # mild "promote the survivor" tiebreak
        return score + 1

    def _score_setup_active(self, card):
        # Opening-active choice. MEASURED (in-process cabt, 60 games vs Lucario):
        # opening Abra      -> 26% loss, 0 no-offense (evolves in place -> Alakazam fast)
        # opening Dunsparce -> 57% loss, 5 no-offense (70HP body, no attacker path)
        # opening Psyduck/Genesect (pure tech) -> ~60% loss (fragile, can't ever attack).
        # So: Abra >> Dunsparce > (anything that can become an attacker) >> tech basics.
        # Tech basics (Psyduck 858 / Shaymin 343 / Genesect 142) have NO offensive line
        # and must be the last resort — opening them strands us with a dead active.
        if card is None:
            return 0
        if card.id == C.ABRA:
            return 40 if self.hand[C.DUNSPARCE] else 50
        if card.id == C.DUNSPARCE:
            return 70 if self.hand[C.ABRA] else 45
        if card.id in (C.FEZANDIPITI_EX, C.PSYDUCK, C.SHAYMIN, C.GENESECT):
            return 1           # pure tech, fragile, no attack -> last resort only
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
        if cid == C.FEZANDIPITI_EX:
            return 170 if self._fez_bench_worthwhile() else -1
        if cid == C.SHAYMIN:
            return 150 if (n == 0 and self._opp_threatens_bench()) else -1
        if cid == C.PSYDUCK:
            return 90 if (n == 0 and self._opp_has_self_ko_ability()) else -1
        return 100 - 20 * n

    def _score_to_hand(self, card):
        if card is None:
            return 0
        cid = card.id
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
        elif cid == C.FEZANDIPITI_EX:
            score += 100 if self._fez_bench_worthwhile() else -40
        elif cid == C.ENRICHING_ENERGY:
            score += 65   # ACE SPEC — Majkel grabs it 54x vs our 1x
        elif is_energy(cid):
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
        if is_energy(cid):
            return 20 if self.hand[cid] >= 3 else -40
        if self.hand[cid] >= 2:
            return 60
        if cid in (C.ABRA, C.KADABRA, C.ALAKAZAM, C.DUNSPARCE, C.DUDUNSPARCE,
                   C.FEZANDIPITI_EX):
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
        if cid in (C.ABRA, C.KADABRA, C.ALAKAZAM, C.DUNSPARCE, C.DUDUNSPARCE,
                   C.FEZANDIPITI_EX):
            return -40 if self.field[cid] == 0 else 60
        return 10



def _remember_visible_mist(obs):
    state = getattr(obs, "current", None)
    if state is None:
        return
    opponent = state.players[1 - state.yourIndex]
    visible = list(getattr(opponent, "discard", None) or [])
    for pokemon in (opponent.active + opponent.bench):
        if pokemon is not None:
            visible.extend(getattr(pokemon, "energyCards", None) or [])
    serials = _V9_STATE.setdefault("mist_seen_serials", set())
    fallback_count = 0
    for card in visible:
        if getattr(card, "id", None) != C.MIST_ENERGY:
            continue
        serial = getattr(card, "serial", None)
        if serial is None:
            fallback_count += 1
        else:
            serials.add(("serial", int(serial)))
    # Old harness stubs may omit serials. Preserve the largest visible count, but
    # do not repeatedly count the same public card on every decision.
    for index in range(fallback_count):
        serials.add(("visible_without_serial", index))
    _DIAG["mist_seen_max"] = max(_DIAG["mist_seen_max"], len(serials))


def _remember_attack_disable(obs):
    state = getattr(obs, "current", None)
    if state is None:
        return
    turn = getattr(state, "turn", None)
    me = state.players[state.yourIndex]
    active = me.active[0] if me.active else None
    active_key = _pokemon_key(active)
    if _V9_STATE.get("attack_disabled_turn") != turn:
        _V9_STATE.update({"attack_disabled_key": None, "attack_disabled_turn": None})
    for log in (getattr(obs, "logs", None) or []):
        if (getattr(log, "type", None) == 15
                and getattr(log, "playerIndex", None) != state.yourIndex
                and getattr(log, "attackId", None) in ATTACK_DISABLE_ATTACKS):
            event = (
                turn,
                getattr(log, "playerIndex", None),
                getattr(log, "serial", None),
                getattr(log, "attackId", None),
            )
            if event != _V9_STATE.get("attack_disable_event"):
                _V9_STATE.update({
                    "attack_disabled_key": active_key,
                    "attack_disabled_turn": turn,
                    "attack_disable_event": event,
                })
            break
    if (_V9_STATE.get("attack_disabled_key") is not None
            and _V9_STATE["attack_disabled_key"] != active_key):
        _V9_STATE.update({"attack_disabled_key": None, "attack_disabled_turn": None})


def _remember_temporary_attack_immunity(obs):
    """Pair a protecting attack with its coin result and remember its exact user."""
    state = getattr(obs, "current", None)
    if state is None:
        return
    turn = getattr(state, "turn", None)
    opponent_index = 1 - state.yourIndex
    opponent = state.players[opponent_index]
    opponent_active = opponent.active[0] if opponent.active else None
    active_key = _pokemon_key(opponent_active)

    valid_through = _V9_STATE.get("temporary_immunity_turn")
    if valid_through is not None and turn is not None and turn > valid_through:
        _V9_STATE.update({
            "temporary_immunity_key": None,
            "temporary_immunity_turn": None,
            "temporary_immunity_pending": None,
        })
    if (_V9_STATE.get("temporary_immunity_key") is not None
            and _V9_STATE["temporary_immunity_key"] != active_key):
        _V9_STATE.update({
            "temporary_immunity_key": None,
            "temporary_immunity_turn": None,
        })

    pending = _V9_STATE.get("temporary_immunity_pending")
    selection = getattr(obs, "select", None)
    selection_context = getattr(selection, "context", None)
    expires_turn = turn + int(
        selection_context == getattr(SelectContext, "TO_ACTIVE", object())
    ) if turn is not None else turn
    for log in (getattr(obs, "logs", None) or []):
        log_type = getattr(log, "type", None)
        player_index = getattr(log, "playerIndex", None)
        attack_id = getattr(log, "attackId", None)
        mode = TEMPORARY_ATTACK_IMMUNITY_ATTACKS.get(attack_id)
        if log_type == 15 and player_index == opponent_index and mode is not None:
            pending = (
                turn,
                player_index,
                getattr(log, "serial", None),
                getattr(log, "cardId", None),
                attack_id,
            )
            _V9_STATE["temporary_immunity_pending"] = pending
            if mode == "always":
                candidate_key = (
                    pending[2],
                    pending[3] if pending[3] is not None
                    else getattr(opponent_active, "id", None),
                )
                if candidate_key == active_key:
                    if pending != _V9_STATE.get("temporary_immunity_event"):
                        _DIAG["temporary_immunity_heads"] += 1
                    _V9_STATE.update({
                        "temporary_immunity_key": candidate_key,
                        "temporary_immunity_turn": expires_turn,
                        "temporary_immunity_event": pending,
                        "temporary_immunity_pending": None,
                    })
                pending = None
            continue
        if log_type != 22 or pending is None or player_index != pending[1]:
            continue
        event_is_new = pending != _V9_STATE.get("temporary_immunity_event")
        head_value = getattr(log, "head", False)
        is_head = head_value is True or str(head_value).lower() == "true"
        candidate_key = (
            pending[2],
            pending[3] if pending[3] is not None
            else getattr(opponent_active, "id", None),
        )
        if is_head and candidate_key == active_key:
            if event_is_new:
                _DIAG["temporary_immunity_heads"] += 1
            _V9_STATE.update({
                "temporary_immunity_key": candidate_key,
                "temporary_immunity_turn": expires_turn,
                "temporary_immunity_event": pending,
            })
        else:
            if event_is_new:
                _DIAG["temporary_immunity_tails"] += 1
            _V9_STATE.update({
                "temporary_immunity_key": None,
                "temporary_immunity_turn": None,
                "temporary_immunity_event": pending,
            })
        pending = None
        _V9_STATE["temporary_immunity_pending"] = None


def _update_v9_state(obs):
    _remember_visible_mist(obs)
    _remember_attack_disable(obs)
    _remember_temporary_attack_immunity(obs)
    state = getattr(obs, "current", None)
    if state is None or _V9_STATE["turn"] == getattr(state, "turn", None):
        return
    _V9_STATE["turn"] = getattr(state, "turn", None)
    me = state.players[state.yourIndex]
    active = me.active[0] if me.active else None
    if active is None or active.id != C.FEZANDIPITI_EX:
        _V9_STATE.update({"fez_active_serial": None, "fez_active_turns": 0})
        return
    serial = getattr(active, "serial", None)
    if serial == _V9_STATE["fez_active_serial"]:
        _V9_STATE["fez_active_turns"] += 1
    else:
        _V9_STATE.update({"fez_active_serial": serial, "fez_active_turns": 1})
    if _V9_STATE["fez_active_turns"] > 2:
        _DIAG["fez_active_stall_turns"] += 1


def agent(obs_dict):
    global pre_turn
    try:
        if isinstance(obs_dict, dict) and obs_dict.get("select") is None:
            _V9_STATE.update({
                "turn": None,
                "fez_active_serial": None,
                "fez_active_turns": 0,
                "attack_disabled_key": None,
                "attack_disabled_turn": None,
                "attack_disable_event": None,
                "temporary_immunity_key": None,
                "temporary_immunity_turn": None,
                "temporary_immunity_event": None,
                "temporary_immunity_pending": None,
                "articuno_breaker_armed": False,
                "planned_hammer_target_key": None,
                "own_ko_turn": None,
                "mist_seen_serials": set(),
            })
            _DIAG["deck_returns"] += 1
            return my_deck
    except Exception:
        pass
    _DIAG["decisions"] += 1
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            _DIAG["deck_returns"] += 1; _DIAG["decisions"] -= 1
            return my_deck
        _update_v9_state(obs)
        if obs.current is not None and pre_turn != obs.current.turn:
            pre_turn = obs.current.turn
        try:
            sel = AlakazamPolicy(obs).choose()
            _DIAG["policy_ok"] += 1
            return sel
        except Exception as exc:
            _diag_record_error(exc); _DIAG["policy_fallback"] += 1
            return _legal_fallback(obs.select)
    except Exception as exc:
        _diag_record_error(exc); _DIAG["obs_fallback"] += 1
        return _legal_fallback_from_dict(obs_dict if isinstance(obs_dict, dict) else {})
