# =============================================================
# main.py  マリィのオーロンゲex＋ユキメノコ＋マシマシラ エージェント v5（1000ログ行動模倣改善）
# -------------------------------------------------------------
# 方針:
#  1. 高ランクAgent(jneums)の3対戦ログから抽出した60枚と行動順を再現
#  2. 先攻を選び、序盤はベロバー3系統＋悪エネ付きマシマシラ2体を優先
#  3. パンクアップは原則5枚選び、現アタッカー2・後続2・第3系統1へ分散
#  4. フーディン対面はユキメノコ2体、ミラーはオーロンゲ系統を優先
#  5. マシマシラのダメカン移動後、シャドーバレット180＋ベンチ30でKOを作る
#  6. スパイクタウンジム、ポケパッド、ペトレルを攻撃前に使い盤面を継続補充
#
# 参考提出物と同じインターフェース:
#  - select がない呼び出しでは60枚のカードIDを返す
#  - select がある呼び出しでは合法選択肢のindex配列を返す
#  - AGENT_LOG=1 でJSONLログを出力
# =============================================================
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

# ============ [1] カード定義 ============
# Destined Rivals のカードID（競技カードリスト準拠）
MARNIES_IMPIDIMP = 646
MARNIES_MORGREM = 647
MARNIES_GRIMMSNARL_EX = 648
MARNIES_MORPEKO = 649

BUDEW = 235
MUNKIDORI = 112
SNORUNT = 860
FROSLASS = 104
TATSUGIRI = 122
YVELTAL = 689

POFFIN = 1086
POKEPAD = 1152
RARE_CANDY = 1079
NIGHT_STRETCHER = 1097
UNFAIR_STAMP = 1080
WONDROUS_PATCH = 1146
AIR_BALLOON = 1174
HANDHELD_FAN = 1161
POKEGEAR = 1122
TOOL_SCRAPPER = 1137

LILLIE = 1227
ROCKET_PETREL = 1219
DAWN = 1231
BOSS_ORDERS = 1182
SPIKEMUTH_GYM = 1259

DARK_E = 7

# 高ランクAgent(jneums)が3ログすべてで使用した同一の60枚。
DECK = (
    [MARNIES_IMPIDIMP] * 4
    + [MARNIES_MORGREM] * 3
    + [MARNIES_GRIMMSNARL_EX] * 3
    + [SNORUNT] * 2
    + [FROSLASS] * 2
    + [MUNKIDORI] * 4
    + [POFFIN] * 4
    + [POKEPAD] * 4
    + [NIGHT_STRETCHER] * 3
    + [RARE_CANDY] * 3
    + [UNFAIR_STAMP] * 1
    + [POKEGEAR] * 1
    + [TOOL_SCRAPPER] * 1
    + [LILLIE] * 4
    + [ROCKET_PETREL] * 4
    + [DAWN] * 1
    + [BOSS_ORDERS] * 2
    + [SPIKEMUTH_GYM] * 4
    + [DARK_E] * 10
)
assert len(DECK) == 60

POKEMON_IDS = {
    MARNIES_IMPIDIMP,
    MARNIES_MORGREM,
    MARNIES_GRIMMSNARL_EX,
    MUNKIDORI,
    SNORUNT,
    FROSLASS,
}
BASIC_IDS = {MARNIES_IMPIDIMP, MUNKIDORI, SNORUNT}
ITEM_IDS = {
    POFFIN,
    POKEPAD,
    RARE_CANDY,
    NIGHT_STRETCHER,
    UNFAIR_STAMP,
    POKEGEAR,
    TOOL_SCRAPPER,
}
TOOL_IDS: set[int] = set()
SUPPORTER_IDS = {LILLIE, ROCKET_PETREL, DAWN, BOSS_ORDERS}
STADIUM_IDS = {SPIKEMUTH_GYM}
EX_IDS = {MARNIES_GRIMMSNARL_EX}
MARNIE_LINE = {MARNIES_IMPIDIMP, MARNIES_MORGREM, MARNIES_GRIMMSNARL_EX}
MARNIE_POKEMON = MARNIE_LINE
ENERGY_IDS = {DARK_E}

BASE_HP = {
    MARNIES_IMPIDIMP: 70,
    MARNIES_MORGREM: 100,
    MARNIES_GRIMMSNARL_EX: 320,
    MUNKIDORI: 110,
    SNORUNT: 70,
    FROSLASS: 90,
}

# Logs: Impidimp is preferred; Munkidori is the fallback active and is often
# benched twice during setup. Snorunt is not an early active unless forced.
SETUP_ACTIVE_PRIORITY = {
    MARNIES_IMPIDIMP: 12000,
    MUNKIDORI: 8500,
    SNORUNT: 2500,
}

ABRA, KADABRA, ALAKAZAM = 741, 742, 743
ALAKAZAM_IDS = {ABRA, KADABRA, ALAKAZAM}

ALAKAZAM_NAMES = (
    'Alakazam', 'Kadabra', 'Abra',
    'フーディン', 'ユンゲラー', 'ケーシィ',
)

GRIMMSNARL_MIRROR_IDS = {
    MARNIES_IMPIDIMP,
    MARNIES_MORGREM,
    MARNIES_GRIMMSNARL_EX,
}

# Episode-local memory. The Kaggle process can play multiple matches, so these
# values are reset whenever the deck is requested.
_SEEN_OPPONENT_IDS: set[int] = set()
_OWN_KO_TURN = -999


def _reset_episode_memory() -> None:
    global _PLAN, _LAST_TURN, _LAST_PLAYER, _DC_TARGET_HP
    global _SEEN_OPPONENT_IDS, _OWN_KO_TURN
    _PLAN = None
    _LAST_TURN = -1
    _LAST_PLAYER = -1
    _DC_TARGET_HP = None
    _SEEN_OPPONENT_IDS = set()
    _OWN_KO_TURN = -999


def _update_opponent_memory(state: dict[str, Any]) -> None:
    opponent = _opp(state)
    for _, pokemon in _slots(opponent):
        cid = _cid(pokemon)
        if cid is not None:
            _SEEN_OPPONENT_IDS.add(cid)
    for card in _discard_cards(opponent):
        cid = _cid(card)
        if cid is not None:
            _SEEN_OPPONENT_IDS.add(cid)


def _opponent_archetype(state: dict[str, Any]) -> str:
    visible = set(_field_ids(_opp(state))) | set(_discard_ids(_opp(state))) | _SEEN_OPPONENT_IDS
    if visible & ALAKAZAM_IDS:
        return 'alakazam'
    if visible & GRIMMSNARL_MIRROR_IDS:
        return 'mirror'
    return 'unknown'


def _target_marnie_line_count(state: dict[str, Any]) -> int:
    # Against Alakazam, the winning log used two Grimmsnarl lines and spent
    # the remaining slots on two Froslass and two Munkidori. Mirrors used
    # three Grimmsnarl lines and delayed Froslass.
    return 2 if _opponent_archetype(state) == 'alakazam' else 3


def _target_munkidori_count(state: dict[str, Any]) -> int:
    # The winner deployed two early in every game and a third in the mirror
    # when bench space remained. Keep two as the stable minimum.
    return 3 if _opponent_archetype(state) == 'mirror' else 2


def _target_snorunt_count(state: dict[str, Any]) -> int:
    archetype = _opponent_archetype(state)
    if archetype == 'alakazam':
        return 2
    if archetype == 'mirror':
        # Froslass was delayed heavily in the mirror logs because it also
        # damages our own ability Pokémon.
        return 0 if _num(state.get('turn'), 0) < 8 else 1
    return 1

LOG_ENABLED = os.environ.get('AGENT_LOG', '0') == '1'
LOG_PATH = os.environ.get('AGENT_LOG_PATH', 'agent_log.jsonl')


def _log(record: dict[str, Any]) -> None:
    if not LOG_ENABLED:
        return
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        pass


# ---------- Enum定数（参考提出物/cabt準拠） ----------
ST_MAIN, ST_CARD, ST_ATTACHED, ST_CARD_OR_ATT, ST_ENERGY = 0, 1, 2, 3, 4
ST_SKILL, ST_ATTACK, ST_EVOLVE, ST_COUNT, ST_YESNO, ST_COND = 5, 6, 7, 8, 9, 10
OT_NUMBER, OT_YES, OT_NO, OT_CARD = 0, 1, 2, 3
OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY_OPT = 4, 5, 6
OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_ABILITY = 7, 8, 9, 10
OT_DISCARD, OT_RETREAT, OT_ATTACK, OT_END = 11, 12, 13, 14
AREA_DECK, AREA_HAND, AREA_DISCARD, AREA_ACTIVE, AREA_BENCH = 1, 2, 3, 4, 5
AREA_LOOKING = 12
CTX_SETUP_ACTIVE, CTX_SETUP_BENCH, CTX_SWITCH, CTX_TO_ACTIVE = 1, 2, 3, 4
CTX_TO_BENCH, CTX_TO_FIELD, CTX_TO_HAND, CTX_DISCARD, CTX_TO_DECK = 5, 6, 7, 8, 9
CTX_DMG_COUNTER, CTX_DMG_COUNTER_ANY, CTX_DAMAGE = 13, 14, 15
CTX_DAMAGE_SOURCE = 16
CTX_EVOLVES_FROM, CTX_EVOLVES_TO = 18, 19
CTX_ATTACH_FROM, CTX_ATTACH_TO, CTX_LOOK, CTX_EFFECT = 21, 22, 24, 25
CTX_ATTACK, CTX_EVOLVE_CTX = 35, 37
CTX_DC_COUNT = 39
CTX_IS_FIRST, CTX_MULLIGAN, CTX_ACTIVATE = 41, 42, 43


# ============ [2] 安全な盤面ヘルパ ============
def _num(value: Any, default: int = 0) -> int:
    try:
        if value is None or isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value)
        sign = -1 if text.strip().startswith('-') else 1
        digits = ''.join(ch for ch in text if ch.isdigit())
        return sign * int(digits) if digits else default
    except Exception:
        return default


def _card_name(card: Any) -> str:
    if not isinstance(card, dict):
        return ''
    for key in ('name', 'cardName', 'displayName', 'label'):
        value = card.get(key)
        if value:
            return str(value)
    return ''


def _cid(card: Any) -> int | None:
    return card.get('id') if isinstance(card, dict) else None


def _me(state: dict[str, Any]) -> dict[str, Any]:
    players = state.get('players') or []
    if not players:
        return {}
    idx = _num(state.get('yourIndex'), 0)
    return players[idx] if 0 <= idx < len(players) else players[0]


def _opp(state: dict[str, Any]) -> dict[str, Any]:
    players = state.get('players') or []
    if len(players) < 2:
        return {}
    idx = _num(state.get('yourIndex'), 0)
    return players[1 - idx]


def _slots(player: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    result: list[tuple[int, dict[str, Any]]] = []
    active = player.get('active') or []
    if isinstance(active, dict):
        active = [active]
    if active and active[0] is not None:
        result.append((0, active[0]))
    for index, pokemon in enumerate(player.get('bench') or []):
        if pokemon is not None:
            result.append((index + 1, pokemon))
    return result


def _active(player: dict[str, Any]) -> dict[str, Any] | None:
    slots = _slots(player)
    return slots[0][1] if slots and slots[0][0] == 0 else None


def _bench(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [pk for slot, pk in _slots(player) if slot > 0]


def _hand_cards(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in (player.get('hand') or []) if isinstance(c, dict)]


def _hand_ids(player: dict[str, Any]) -> list[int]:
    return [c['id'] for c in _hand_cards(player) if c.get('id') is not None]


def _discard_cards(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in (player.get('discard') or []) if isinstance(c, dict)]


def _discard_ids(player: dict[str, Any]) -> list[int]:
    return [c['id'] for c in _discard_cards(player) if c.get('id') is not None]


def _prizes_left(player: dict[str, Any]) -> int:
    prize = player.get('prize')
    if isinstance(prize, list):
        return len(prize)
    return _num(player.get('prizeCount'), 6)


def _energy_ids(pokemon: Any) -> list[int]:
    if not isinstance(pokemon, dict):
        return []
    for key in ('energies', 'energyCards', 'attachedEnergy'):
        values = pokemon.get(key) or []
        result: list[int] = []
        for value in values:
            if isinstance(value, dict) and value.get('id') is not None:
                result.append(value['id'])
            elif isinstance(value, int):
                result.append(value)
        if result:
            return result
    return []


def _dark_count(pokemon: Any) -> int:
    return sum(1 for energy in _energy_ids(pokemon) if energy == DARK_E)


def _damage_of(pokemon: Any) -> int:
    if not isinstance(pokemon, dict):
        return 0
    hp = _num(pokemon.get('hp'), 0)
    max_hp = _num(pokemon.get('maxHp'), 0) or BASE_HP.get(_cid(pokemon), hp)
    return max(0, max_hp - hp)


def _remaining_hp(pokemon: Any) -> int:
    if not isinstance(pokemon, dict):
        return 999
    hp = _num(pokemon.get('hp'), 0)
    return hp if hp > 0 else max(1, BASE_HP.get(_cid(pokemon), 999) - _damage_of(pokemon))


def _is_ex(pokemon: Any) -> bool:
    if not isinstance(pokemon, dict):
        return False
    if _cid(pokemon) in EX_IDS or pokemon.get('ex') is True:
        return True
    name = _card_name(pokemon).lower().replace('ｅｘ', 'ex')
    return name.endswith(' ex') or name.endswith('ex')


def _prize_worth(pokemon: Any) -> int:
    return 2 if _is_ex(pokemon) else 1


def _target_value(pokemon: Any) -> float:
    return (
        _prize_worth(pokemon) * 1500.0
        + len(_energy_ids(pokemon)) * 180.0
        + _remaining_hp(pokemon) * 1.5
    )


def _contains_any(text: str, names: Iterable[str]) -> bool:
    low = text.lower()
    return any(name.lower() in low for name in names)


def _is_alakazam_line(pokemon: Any) -> bool:
    return _cid(pokemon) in ALAKAZAM_IDS or _contains_any(_card_name(pokemon), ALAKAZAM_NAMES)


def _weak_to_dark(pokemon: Any) -> bool:
    if not isinstance(pokemon, dict):
        return False
    if _is_alakazam_line(pokemon):
        return True
    values: list[str] = []
    for key in ('weakness', 'weaknessType'):
        value = pokemon.get(key)
        if isinstance(value, dict):
            values.extend(str(v) for v in value.values())
        elif isinstance(value, list):
            values.extend(str(v) for v in value)
        elif value is not None:
            values.append(str(value))
    merged = ' '.join(values).lower()
    return any(token in merged for token in ('dark', 'darkness', '悪'))


def _field_ids(player: dict[str, Any]) -> list[int]:
    return [_cid(pk) for _, pk in _slots(player) if _cid(pk) is not None]


def _field_count(player: dict[str, Any], card_id: int) -> int:
    return _field_ids(player).count(card_id)


def _marnie_line_count(player: dict[str, Any]) -> int:
    """Number of occupied Impidimp/Morgrem/Grimmsnarl evolution lines."""
    return sum(1 for _, pokemon in _slots(player) if _cid(pokemon) in MARNIE_LINE)


def _grim_count(player: dict[str, Any]) -> int:
    return _field_count(player, MARNIES_GRIMMSNARL_EX)


def _bench_space(player: dict[str, Any]) -> int:
    max_bench = _num(player.get('maxBench'), 5) or 5
    return max(0, max_bench - len(_bench(player)))


def _has_attacker_ready(player: dict[str, Any], exclude_active: bool = False) -> bool:
    for slot, pokemon in _slots(player):
        if exclude_active and slot == 0:
            continue
        if _cid(pokemon) == MARNIES_GRIMMSNARL_EX and _dark_count(pokemon) >= 2:
            return True
    return False

def _card_at(state: dict[str, Any], selection: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    area = option.get('area')
    option_type = option.get('type')
    # MAIN options encode hand cards only by index. v3 treated these as None,
    # so every PLAY/EVOLVE card received a generic score.
    if area is None and option_type in (OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_DISCARD):
        area = AREA_HAND
    index = _num(option.get('index'), 0)
    player_index = _num(option.get('playerIndex'), _num(state.get('yourIndex'), 0))
    try:
        if area == AREA_DECK and selection.get('deck'):
            return selection['deck'][index]
        players = state.get('players') or []
        player = players[player_index]
        if area == AREA_HAND:
            return (player.get('hand') or [])[index]
        if area == AREA_DISCARD:
            return (player.get('discard') or [])[index]
        if area == AREA_ACTIVE:
            active = player.get('active') or []
            if isinstance(active, dict):
                active = [active]
            return active[index]
        if area == AREA_BENCH:
            return (player.get('bench') or [])[index]
        if area == AREA_LOOKING:
            return (state.get('looking') or [])[index]
        if area == 7:
            stadium = state.get('stadium') or []
            if isinstance(stadium, dict):
                stadium = [stadium]
            return stadium[index] if 0 <= index < len(stadium) else None
    except Exception:
        return None
    return None

def _option_card_id(state: dict[str, Any], selection: dict[str, Any], option: dict[str, Any]) -> int | None:
    for key in ('cardId', 'id'):
        value = option.get(key)
        if isinstance(value, int):
            return value
    return _cid(_card_at(state, selection, option))

def _effect_card_id(selection: dict[str, Any]) -> int | None:
    """Card responsible for the current follow-up selection."""
    return _cid(selection.get('effect')) or _cid(selection.get('contextCard'))


def _in_play_target(state: dict[str, Any], option: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
    """Resolve a MAIN destination encoded by inPlayArea/inPlayIndex."""
    me = _me(state)
    area = option.get('inPlayArea')
    index = _num(option.get('inPlayIndex'), 0)
    if area == AREA_ACTIVE:
        return _active(me), 0
    if area == AREA_BENCH:
        bench = _bench(me)
        if 0 <= index < len(bench):
            return bench[index], index + 1
    return None, -1


def _option_text(option: dict[str, Any]) -> str:
    values = [option.get(k, '') for k in ('name', 'label', 'text', 'description')]
    for key in ('attack', 'move', 'skill'):
        obj = option.get(key)
        if isinstance(obj, dict):
            values.extend(obj.get(k, '') for k in ('name', 'label', 'text', 'description'))
    return ' '.join(str(v) for v in values if v)


def _option_damage(option: dict[str, Any]) -> int:
    for key in ('damage', 'baseDamage', 'attackDamage', 'dmg'):
        damage = _num(option.get(key), 0)
        if damage > 0:
            return damage
    for key in ('attack', 'move', 'skill'):
        obj = option.get(key)
        if isinstance(obj, dict):
            for subkey in ('damage', 'baseDamage', 'attackDamage', 'dmg'):
                damage = _num(obj.get(subkey), 0)
                if damage > 0:
                    return damage
    text = _option_text(option)
    if 'Shadow Bullet' in text or 'シャドーバレット' in text:
        return 180
    if 'Spiky Wheel' in text or 'Spike Wheel' in text or 'トゲトゲぐるま' in text:
        return 20
    if 'Dark Feather' in text:
        return 110
    if 'Clutch' in text:
        return 20
    if 'Itchy Pollen' in text or 'むずむずかふん' in text:
        return 10
    return 0


# ============ [3] 攻撃計画 ============
@dataclass
class Plan:
    attacker_slot: int = -1
    target_slot: int = -1
    attacker_id: int | None = None
    expected_damage: int = 0
    needs_energy: bool = False
    boss_needed: bool = False
    score: float = -1e18


_PLAN: Plan | None = None
_LAST_TURN = -1
_LAST_PLAYER = -1
_DC_TARGET_HP: int | None = None


def _damage_for_attacker(pokemon: dict[str, Any], target: dict[str, Any] | None = None, include_manual: bool = False) -> int:
    cid = _cid(pokemon)
    dark = _dark_count(pokemon) + (1 if include_manual else 0)
    if cid == MARNIES_GRIMMSNARL_EX:
        damage = 180 if dark >= 2 else 0
    elif cid == MARNIES_MORGREM:
        damage = 60 if dark >= 2 else 0
    elif cid == MARNIES_IMPIDIMP:
        damage = 10 if dark >= 1 else 0
    else:
        damage = 0
    if damage > 0 and target is not None and cid in MARNIE_POKEMON and _weak_to_dark(target):
        damage *= 2
    return damage

def _build_plan(state: dict[str, Any], allow_boss: bool = True) -> Plan:
    me, opponent = _me(state), _opp(state)
    plan = Plan()
    hand = _hand_ids(me)
    can_manual = DARK_E in hand and not bool(state.get('energyAttached'))
    boss_in_hand = BOSS_ORDERS in hand and not bool(state.get('supporterPlayed'))
    opponent_slots = _slots(opponent)
    if not opponent_slots:
        return plan

    for attacker_slot, attacker in _slots(me):
        for target_slot, target in opponent_slots:
            is_active_target = target_slot == 0
            boss_needed = not is_active_target
            if boss_needed and (not allow_boss or not boss_in_hand):
                continue
            damage_now = _damage_for_attacker(attacker, target, include_manual=False)
            damage_after_attach = _damage_for_attacker(attacker, target, include_manual=can_manual)
            damage = max(damage_now, damage_after_attach)
            needs_energy = damage_now == 0 and damage_after_attach > 0
            if damage <= 0:
                continue
            hp = _remaining_hp(target)
            prizes = _prize_worth(target)
            ko = damage >= hp
            score = damage * 18 + _target_value(target)
            if ko:
                score += 11000 + prizes * 5000
                if _prizes_left(opponent) <= prizes:
                    score += 100000
            else:
                score += min(damage, hp) * 8
            if _is_alakazam_line(target):
                score += 9000
            if boss_needed:
                score -= 900
            if attacker_slot == 0:
                score += 900
            if _cid(attacker) == YVELTAL:
                score += 600  # 非exの後続アタッカーを活用
            if score > plan.score:
                plan = Plan(
                    attacker_slot=attacker_slot,
                    target_slot=target_slot,
                    attacker_id=_cid(attacker),
                    expected_damage=damage,
                    needs_energy=needs_energy,
                    boss_needed=boss_needed,
                    score=score,
                )
    return plan


def _attack_score(state: dict[str, Any], plan: Plan) -> float:
    me, opponent = _me(state), _opp(state)
    active = _active(me)
    target = _active(opponent)
    if active is None or target is None:
        return -20000
    damage = _damage_for_attacker(active, target)
    if damage <= 0:
        return -18000
    hp = _remaining_hp(target)
    score = 25000 + damage * 25
    if damage >= hp:
        score += 16000 + _prize_worth(target) * 6000
        if _prizes_left(opponent) <= _prize_worth(target):
            score += 120000
    if _is_alakazam_line(target):
        score += 10000
    if _cid(active) == MARNIES_GRIMMSNARL_EX:
        # ベンチ30点の追加価値
        score += 3500
        for bench_target in _bench(opponent):
            if _remaining_hp(bench_target) <= 30:
                score += 10000 + _prize_worth(bench_target) * 4000
            elif _is_alakazam_line(bench_target):
                score += 2500
    if plan.attacker_slot == 0:
        score += 1000
    return score


# ============ [4] MAIN行動採点 ============
def _boss_score(state: dict[str, Any], plan: Plan) -> float:
    me, opponent = _me(state), _opp(state)
    if bool(state.get('supporterPlayed')) or not _bench(opponent):
        return -16000
    active = _active(me)
    current_target = _active(opponent)
    if active is None:
        return -12000
    current_damage = _damage_for_attacker(active, current_target) if current_target else 0
    current_ko = current_target is not None and current_damage >= _remaining_hp(current_target)
    best = -12000.0
    for target in _bench(opponent):
        damage = _damage_for_attacker(active, target)
        if damage <= 0 or damage < _remaining_hp(target):
            continue
        score = _target_value(target)
        if damage > 0 and damage >= _remaining_hp(target):
            score += 14000 + _prize_worth(target) * 5500
            if _prizes_left(opponent) <= _prize_worth(target):
                score += 100000
        if _is_alakazam_line(target):
            score += 9500
        if _dark_count(target) >= 2:
            score += 900
        best = max(best, score)
    if current_ko:
        best -= 9000
    if plan.boss_needed:
        best += 3500
    return best


def _need_poffin(state: dict[str, Any]) -> bool:
    me = _me(state)
    if _bench_space(me) <= 0:
        return False
    line_need = _marnie_line_count(me) < _target_marnie_line_count(state)
    snorunt_need = _field_count(me, SNORUNT) + _field_count(me, FROSLASS) < _target_snorunt_count(state)
    return line_need or snorunt_need

def _has_rare_candy_target(me: dict[str, Any]) -> bool:
    return _field_count(me, MARNIES_IMPIDIMP) > 0


def _useful_discard(me: dict[str, Any]) -> bool:
    wanted = {
        MARNIES_IMPIDIMP,
        MARNIES_MORGREM,
        MARNIES_GRIMMSNARL_EX,
        MUNKIDORI,
        SNORUNT,
        FROSLASS,
        DARK_E,
    }
    return any(cid in wanted for cid in _discard_ids(me))

def _supporter_score(card_id: int, state: dict[str, Any], plan: Plan) -> float:
    if bool(state.get('supporterPlayed')):
        return -20000
    me = _me(state)
    hand_n = len(_hand_ids(me))
    turn = _num(state.get('turn'), 0)
    if card_id == BOSS_ORDERS:
        boss = _boss_score(state, plan)
        return 95000 if boss > 12000 else (72000 if boss > 0 else -12000)
    if card_id == ROCKET_PETREL:
        if _grim_count(me) == 0 and _marnie_line_count(me) > 0:
            return 88000
        if DARK_E not in _hand_ids(me) and _useful_discard(me):
            return 82000
        return 73000
    if card_id == LILLIE:
        score = 76000 + max(0, 7 - hand_n) * 1200
        if turn <= 4:
            score += 5000
        return score
    if card_id == DAWN:
        return 85000 if _grim_count(me) < 2 and _marnie_line_count(me) > 0 else 70000
    return 1000

def _basic_play_score(card_id: int, state: dict[str, Any]) -> float:
    me = _me(state)
    if _bench_space(me) <= 0:
        return -20000
    field = _field_ids(me)
    if card_id == MARNIES_IMPIDIMP:
        count = _marnie_line_count(me)
        target = _target_marnie_line_count(state)
        if count < target:
            return 72000 - count * 2500
        return -10000
    if card_id == MUNKIDORI:
        count = field.count(MUNKIDORI)
        target = _target_munkidori_count(state)
        if count < target:
            return 75000 - count * 1800
        return -9000
    if card_id == SNORUNT:
        count = field.count(SNORUNT) + field.count(FROSLASS)
        target = _target_snorunt_count(state)
        core_ready = (
            _marnie_line_count(me) >= _target_marnie_line_count(state)
            and _field_count(me, MUNKIDORI) >= 2
        )
        if count < target and core_ready:
            return 73000 if _opponent_archetype(state) == 'alakazam' else 66000
        return -11000
    return 0

def _attachment_target_score(pokemon: dict[str, Any], slot: int, state: dict[str, Any], punk_up: bool = False) -> float:
    cid = _cid(pokemon)
    dark = _dark_count(pokemon)
    me = _me(state)
    if punk_up and cid not in MARNIE_POKEMON:
        return -50000
    if punk_up:
        if cid == MARNIES_GRIMMSNARL_EX and dark < 2:
            return 100000 - dark * 3000 + (2500 if slot == 0 else 0)
        if cid == MARNIES_MORGREM and dark < 2:
            return 93000 - dark * 2500
        if cid == MARNIES_IMPIDIMP and dark < 2:
            return 88000 - dark * 2500
        if cid == MARNIES_GRIMMSNARL_EX:
            return 30000 - dark * 1000
        if cid == MARNIES_MORGREM:
            return 28000 - dark * 900
        if cid == MARNIES_IMPIDIMP:
            return 26000 - dark * 900
        return -50000
    if cid == MUNKIDORI:
        if dark == 0:
            powered = sum(1 for _, pk in _slots(me) if _cid(pk) == MUNKIDORI and _dark_count(pk) > 0)
            return 90000 - powered * 2200
        return -9000
    if cid == MARNIES_GRIMMSNARL_EX and dark < 2:
        return 70000 - dark * 2500
    if cid == MARNIES_MORGREM and dark < 2:
        return 66000 - dark * 2200
    if cid == MARNIES_IMPIDIMP and dark < 2:
        return 64000 - dark * 2200
    return -15000

def _retreat_score(state: dict[str, Any], plan: Plan) -> float:
    me = _me(state)
    active = _active(me)
    if active is None or not _bench(me):
        return -20000
    ready_bench = any(
        _cid(pk) == MARNIES_GRIMMSNARL_EX and _dark_count(pk) >= 2
        for slot, pk in _slots(me) if slot > 0
    )
    if ready_bench and _cid(active) != MARNIES_GRIMMSNARL_EX:
        return 98000
    if plan.attacker_slot > 0:
        return 82000
    return -6000

def _main_score(option: dict[str, Any], state: dict[str, Any], selection: dict[str, Any], plan: Plan) -> float:
    option_type = option.get('type')
    me = _me(state)
    opponent = _opp(state)
    card = _card_at(state, selection, option)
    card_id = _cid(card) or option.get('cardId')
    archetype = _opponent_archetype(state)

    if option_type == OT_END:
        return -1000
    if option_type == OT_ATTACK:
        score = _attack_score(state, plan)
        active = _active(me)
        target = _active(opponent)
        damage = _damage_for_attacker(active or {}, target)
        final_ko = (
            target is not None
            and damage >= _remaining_hp(target)
            and _prizes_left(opponent) <= _prize_worth(target)
        )
        if final_ko:
            return 250000
        # The logged agent used all abilities/searches before attacking.
        return min(score + (9000 if option.get('attackId') == 937 else 0), 62000)
    if option_type == OT_RETREAT:
        return _retreat_score(state, plan)
    if option_type == OT_ATTACH:
        target, slot = _in_play_target(state, option)
        return _attachment_target_score(target or {}, slot, state)
    if option_type == OT_EVOLVE:
        evolves_to = option.get('evolvesTo') or card_id
        if evolves_to == MARNIES_GRIMMSNARL_EX:
            return 102000 if _grim_count(me) == 0 else (93000 if _grim_count(me) == 1 else 76000)
        if evolves_to == MARNIES_MORGREM:
            return 86000 if _grim_count(me) < 2 else 74000
        if evolves_to == FROSLASS:
            if archetype == 'alakazam':
                return 90000
            if archetype == 'mirror' and _num(state.get('turn'), 0) < 8:
                return 5000
            return 70000
        return 5000
    if option_type == OT_ABILITY:
        context_id = card_id or _cid(selection.get('contextCard'))
        if option.get('area') == 7:
            target_lines = _target_marnie_line_count(state)
            missing = _marnie_line_count(me) < target_lines or _grim_count(me) < min(target_lines, _marnie_line_count(me))
            return 69000 if missing else 64000
        if context_id == MARNIES_GRIMMSNARL_EX:
            return 108000 if _num(me.get('deckCount'), 60) > 0 else -6000
        if context_id == MUNKIDORI or (card is not None and _cid(card) == MUNKIDORI):
            return 120000 if any(_damage_of(pk) > 0 for _, pk in _slots(me)) else -5000
        return 8000
    if option_type == OT_PLAY:
        if card_id in BASIC_IDS:
            return _basic_play_score(card_id, state)
        if card_id == POFFIN:
            return 91000 if _need_poffin(state) and _num(me.get('deckCount'), 60) > 4 else -6500
        if card_id == POKEPAD:
            return 84000 if _bench_space(me) > 0 and _num(me.get('deckCount'), 60) > 6 else 66000
        if card_id == RARE_CANDY:
            return 100000 if _has_rare_candy_target(me) else -10000
        if card_id == NIGHT_STRETCHER:
            return 82000 if _useful_discard(me) else -7500
        if card_id == UNFAIR_STAMP:
            opponent_hand = max(len(_hand_ids(opponent)), _num(opponent.get('handCount'), 0))
            recent_ko = _num(state.get('turn'), 0) - _OWN_KO_TURN <= 2
            return 94000 if recent_ko and opponent_hand >= 4 else -10000
        if card_id == POKEGEAR:
            return 80000 if not bool(state.get('supporterPlayed')) and _num(me.get('deckCount'), 60) > 6 else -5000
        if card_id == TOOL_SCRAPPER:
            opponent_has_tool = any(bool(pk.get('tools')) for _, pk in _slots(opponent))
            return 87000 if opponent_has_tool else -9000
        if card_id == SPIKEMUTH_GYM:
            target_lines = _target_marnie_line_count(state)
            missing = _marnie_line_count(me) < target_lines or _grim_count(me) < min(target_lines, _marnie_line_count(me))
            return 88000 if missing else 70000
        if card_id in SUPPORTER_IDS:
            return _supporter_score(card_id, state, plan)
    if option_type == OT_DISCARD:
        return -2000
    return 0

# ============ [5] 対象選択採点 ============
def _setup_active_score(card_id: int | None) -> float:
    return SETUP_ACTIVE_PRIORITY.get(card_id, 0)


def _search_priority(context_card_id: int | None, candidate_id: int | None, state: dict[str, Any]) -> float:
    me = _me(state)
    hand = _hand_ids(me)
    field = _field_ids(me)
    archetype = _opponent_archetype(state)
    line_target = _target_marnie_line_count(state)
    if candidate_id is None:
        return -1000

    if context_card_id == POFFIN:
        if candidate_id == MARNIES_IMPIDIMP:
            count = _marnie_line_count(me) + hand.count(MARNIES_IMPIDIMP)
            return 50000 if count < line_target else -9000
        if candidate_id == SNORUNT:
            count = field.count(SNORUNT) + field.count(FROSLASS) + hand.count(SNORUNT)
            target = _target_snorunt_count(state)
            core_ready = _marnie_line_count(me) >= line_target
            return 54000 if core_ready and count < target else -8000
        return -7000

    if context_card_id == NIGHT_STRETCHER:
        powered_munki = sum(1 for _, pk in _slots(me) if _cid(pk) == MUNKIDORI and _dark_count(pk) > 0)
        if candidate_id == DARK_E:
            return 56000 if powered_munki < _target_munkidori_count(state) or DARK_E not in hand else 38000
        priorities = {
            MARNIES_GRIMMSNARL_EX: 50000,
            MARNIES_MORGREM: 47000,
            MARNIES_IMPIDIMP: 45000,
            MUNKIDORI: 44000,
            FROSLASS: 41000 if archetype == 'alakazam' else 26000,
            SNORUNT: 36000 if archetype == 'alakazam' else 20000,
        }
        return priorities.get(candidate_id, -3000)

    if context_card_id == POKEPAD:
        munk_count = field.count(MUNKIDORI)
        if candidate_id == MUNKIDORI and munk_count < _target_munkidori_count(state):
            return 58000 - munk_count * 1000
        if candidate_id == MARNIES_MORGREM:
            basic_lines = field.count(MARNIES_IMPIDIMP)
            return 55000 if basic_lines > 0 else 16000
        if candidate_id == MARNIES_IMPIDIMP:
            return 53000 if _marnie_line_count(me) < line_target else -7000
        if candidate_id == FROSLASS:
            snorunt_ready = SNORUNT in field
            return 56000 if snorunt_ready and archetype == 'alakazam' else 30000
        if candidate_id == SNORUNT:
            target = _target_snorunt_count(state)
            current = field.count(SNORUNT) + field.count(FROSLASS)
            return 48000 if current < target and _marnie_line_count(me) >= line_target else -8000
        return 0

    if context_card_id == RARE_CANDY:
        return 60000 if candidate_id == MARNIES_GRIMMSNARL_EX else -8000

    if context_card_id == ROCKET_PETREL:
        if candidate_id == RARE_CANDY and MARNIES_IMPIDIMP in field and _grim_count(me) == 0:
            return 62000
        if candidate_id == POFFIN and _marnie_line_count(me) < line_target:
            return 59000
        if candidate_id == POKEPAD and _field_count(me, MUNKIDORI) < _target_munkidori_count(state):
            return 58000
        if candidate_id == NIGHT_STRETCHER and _useful_discard(me):
            return 57000
        if candidate_id == BOSS_ORDERS and _prizes_left(_opp(state)) <= 2:
            return 56000
        priorities = {
            RARE_CANDY: 50000,
            POKEPAD: 49000,
            NIGHT_STRETCHER: 48000,
            POFFIN: 45000,
            TOOL_SCRAPPER: 27000,
            POKEGEAR: 24000,
        }
        return priorities.get(candidate_id, 100)

    if context_card_id == DAWN:
        priorities = {
            MARNIES_GRIMMSNARL_EX: 60000,
            MARNIES_MORGREM: 57000,
            MARNIES_IMPIDIMP: 54000,
            FROSLASS: 52000 if archetype == 'alakazam' else 30000,
            SNORUNT: 30000,
        }
        return priorities.get(candidate_id, 100)

    if context_card_id == SPIKEMUTH_GYM:
        line_count = _marnie_line_count(me)
        imp_count = field.count(MARNIES_IMPIDIMP)
        morg_count = field.count(MARNIES_MORGREM)
        if candidate_id == MARNIES_IMPIDIMP:
            return 64000 if line_count < line_target else -7000
        if candidate_id == MARNIES_MORGREM:
            return 62000 if imp_count > 0 else 18000
        if candidate_id == MARNIES_GRIMMSNARL_EX:
            candy_ready = RARE_CANDY in hand and imp_count > 0
            return 61000 if morg_count > 0 or candy_ready else 28000
        return -5000

    if context_card_id == MARNIES_GRIMMSNARL_EX:
        return 60000 if candidate_id == DARK_E else -12000

    generic = {
        MARNIES_GRIMMSNARL_EX: 55000,
        MARNIES_MORGREM: 52000,
        MARNIES_IMPIDIMP: 50000,
        MUNKIDORI: 49000,
        FROSLASS: 47000,
        SNORUNT: 40000,
        RARE_CANDY: 54000,
        DARK_E: 51000,
        BOSS_ORDERS: 43000,
    }
    return generic.get(candidate_id, 100)

def _discard_priority(card_id: int | None, state: dict[str, Any]) -> float:
    """1,000 replay behavior-cloned discard preference.

    Selection rates were learned from Luca's 384 discard decisions. Duplicate
    setup pieces are released first; the only route to Grimmsnarl ex and the
    ACE SPEC are protected.
    """
    me = _me(state)
    hand = _hand_ids(me)
    count = hand.count(card_id) if card_id is not None else 0
    learned = {
        POKEPAD: 9200, POFFIN: 9000, MARNIES_IMPIDIMP: 8200,
        MARNIES_MORGREM: 7600, DAWN: 7200, RARE_CANDY: 6900,
        SPIKEMUTH_GYM: 6500, SNORUNT: 6100, NIGHT_STRETCHER: 5600,
        TOOL_SCRAPPER: 5200, ROCKET_PETREL: 4300, BOSS_ORDERS: 3900,
        LILLIE: 3300, DARK_E: 2500, FROSLASS: 1300,
        MARNIES_GRIMMSNARL_EX: -3500, MUNKIDORI: -4200,
        UNFAIR_STAMP: -10000, POKEGEAR: -7000,
    }
    score = learned.get(card_id, 1000)
    if count >= 2:
        score += 5000
    if count >= 3:
        score += 2500
    if _grim_count(me) == 0 and card_id in (MARNIES_GRIMMSNARL_EX, RARE_CANDY, MARNIES_IMPIDIMP):
        score -= 10000
    if card_id == DARK_E and count <= 1:
        score -= 7000
    return score

def _switch_target_score(pokemon: dict[str, Any], slot: int, state: dict[str, Any]) -> float:
    opponent_active = _active(_opp(state))
    damage = _damage_for_attacker(pokemon, opponent_active)
    score = damage * 50
    if opponent_active is not None and damage >= _remaining_hp(opponent_active):
        score += 18000
    cid = _cid(pokemon)
    # Learned from 2,142 replacement-active decisions: Luca preserved the
    # Munkidori engine and promoted the Grimmsnarl line whenever possible.
    if cid == MARNIES_GRIMMSNARL_EX:
        score += 36000 + _dark_count(pokemon) * 2500
    elif cid == MARNIES_MORGREM:
        score += 21000 + _dark_count(pokemon) * 1800
    elif cid == MARNIES_IMPIDIMP:
        score += 16000 + _dark_count(pokemon) * 1500
    elif cid == MUNKIDORI and _dark_count(pokemon) > 0:
        score += 2500
    if _remaining_hp(pokemon) <= 40:
        score -= 9000
    return score

def _opponent_target_score(pokemon: dict[str, Any], direct_damage: int = 0) -> float:
    hp = _remaining_hp(pokemon)
    cid = _cid(pokemon)
    score = _target_value(pokemon)
    if direct_damage > 0 and hp <= direct_damage:
        score += 24000 + _prize_worth(pokemon) * 7000
    if _is_alakazam_line(pokemon):
        score += 13000
    if cid == MARNIES_IMPIDIMP:
        score += 9000
    elif cid == MARNIES_MORGREM:
        score += 8000
    elif cid == MARNIES_GRIMMSNARL_EX:
        score += 4500
    if hp <= 30:
        score += 8500
    elif hp <= 60:
        score += 4000
    return score

def _ctx_score(option: dict[str, Any], selection: dict[str, Any], state: dict[str, Any], plan: Plan) -> float:
    context = selection.get('context')
    context_card_id = _effect_card_id(selection)
    card = _card_at(state, selection, option)
    candidate_id = _cid(card) or _option_card_id(state, selection, option)
    player_index = _num(option.get('playerIndex'), _num(state.get('yourIndex'), 0))
    own_index = _num(state.get('yourIndex'), 0)
    slot = 0 if option.get('area') == AREA_ACTIVE else _num(option.get('index'), 0) + 1

    if context == CTX_SETUP_ACTIVE:
        return _setup_active_score(candidate_id)
    if context in (CTX_SETUP_BENCH, CTX_TO_BENCH, CTX_TO_FIELD):
        # Poffin's follow-up must use search priorities rather than normal
        # hand-play priorities. This is what enables the Alakazam line of
        # two Impidimp + two Snorunt seen in the winning log.
        if context_card_id == POFFIN:
            return _search_priority(context_card_id, candidate_id, state)
        if candidate_id in BASIC_IDS:
            return _basic_play_score(candidate_id, state)
    if context in (CTX_SWITCH, CTX_TO_ACTIVE):
        return _switch_target_score(card or {}, slot, state)
    if context == CTX_DISCARD:
        return _discard_priority(candidate_id, state)
    if context in (CTX_ATTACH_TO, CTX_ATTACH_FROM):
        punk_up = context_card_id == MARNIES_GRIMMSNARL_EX
        # Punk Up first presents Basic Darkness Energy from the deck as
        # context=22, then presents Marnie's Pokémon destinations as context=21.
        if punk_up and candidate_id == DARK_E and context == CTX_ATTACH_TO:
            return 30000
        return _attachment_target_score(card or {}, slot, state, punk_up=punk_up)
    if context in (CTX_EVOLVES_FROM, CTX_EVOLVE_CTX):
        if candidate_id == MARNIES_IMPIDIMP:
            # エネ付き・バトル場を優先して即攻撃へつなぐ。
            return 26000 + _dark_count(card) * 2500 + (3000 if option.get('area') == AREA_ACTIVE else 0)
        if candidate_id == MARNIES_MORGREM:
            return 24000 + _dark_count(card) * 2000
        if candidate_id == SNORUNT:
            return 14500
    if context == CTX_EVOLVES_TO:
        if candidate_id == MARNIES_GRIMMSNARL_EX:
            return 32000
        if candidate_id == MARNIES_MORGREM:
            return 21000
        if candidate_id == FROSLASS:
            return 16500
    if context in (CTX_TO_HAND, CTX_LOOK, CTX_EFFECT):
        return _search_priority(context_card_id, candidate_id, state)
    if context == CTX_DAMAGE_SOURCE:
        if card is None or player_index != own_index or _damage_of(card) <= 0:
            return -12000
        cid = _cid(card)
        # Logged policy heals fragile engine pieces before the 320-HP active.
        role_bonus = 22000 if cid == MUNKIDORI else (18000 if cid in (MARNIES_IMPIDIMP, MARNIES_MORGREM, SNORUNT) else 5000)
        return role_bonus + min(_damage_of(card), 30) * 80
    if context in (CTX_DMG_COUNTER, CTX_DMG_COUNTER_ANY):
        if player_index == own_index:
            if card is None:
                return -10000
            damage = _damage_of(card)
            hp = _remaining_hp(card)
            if damage <= 0 or hp <= 30:
                return -12000
            score = damage * 90
            if _cid(card) == MARNIES_GRIMMSNARL_EX:
                score += 7000
            if _cid(card) == MUNKIDORI:
                score += 2500
            return score
        return _opponent_target_score(card or {}, direct_damage=30)
    if context == CTX_DAMAGE:
        # シャドーバレットのベンチ30点を、KO/フーディン系へ優先。
        return _opponent_target_score(card or {}, direct_damage=30)
    if context == CTX_ATTACK:
        damage = _option_damage(option)
        if damage > 0:
            return damage * 1000
        return max(1, plan.expected_damage) * 10

    if candidate_id is not None:
        return _search_priority(context_card_id, candidate_id, state)
    return 0


# ---------- 選択数の適正化 ----------
def _finalize(selection: dict[str, Any], scored: list[tuple[float, int]]) -> list[int]:
    n = len(selection.get('option') or [])
    if n == 0:
        return []
    min_count = max(0, _num(selection.get('minCount'), 0))
    max_count = min(n, max(0, _num(selection.get('maxCount'), 0)))
    if max_count == 0:
        return []
    scored.sort(key=lambda pair: (pair[0], -pair[1]), reverse=True)
    if max_count == 1:
        best_score, best_index = scored[0]
        if min_count == 0 and best_score <= 0:
            return []
        return [best_index]
    chosen = [index for score, index in scored if score > 0][:max_count]
    if len(chosen) < min_count:
        for _, index in scored:
            if index not in chosen:
                chosen.append(index)
            if len(chosen) >= min_count:
                break
    return chosen[:max_count]


# ============ [6] 実行 ============
def _yes_no_choice(selection: dict[str, Any], state: dict[str, Any]) -> list[int]:
    context = selection.get('context')
    context_card_id = _effect_card_id(selection)
    me = _me(state)
    want_yes = True
    if context == CTX_IS_FIRST:
        # The high-rank agent selected first when it received the choice.
        want_yes = True
    elif context == CTX_MULLIGAN:
        want_yes = False
    elif context_card_id == MARNIES_GRIMMSNARL_EX:
        want_yes = _num(me.get('deckCount'), 60) > 0
    elif context_card_id == MUNKIDORI:
        want_yes = any(_damage_of(pk) > 0 for _, pk in _slots(me))
    for index, option in enumerate(selection.get('option') or []):
        if (option.get('type') == OT_YES) == want_yes:
            return [index]
    return [0]

def _count_choice(selection: dict[str, Any]) -> list[int]:
    global _DC_TARGET_HP
    values = [(_num(option.get('number'), 0), index) for index, option in enumerate(selection.get('option') or [])]
    if not values:
        return []
    if selection.get('context') == CTX_DC_COUNT and _DC_TARGET_HP:
        need = (_DC_TARGET_HP + 9) // 10
        fitting = [(number, index) for number, index in values if number >= need]
        if fitting:
            result = [min(fitting)[1]]
            _DC_TARGET_HP = None
            return result
    result = [max(values)[1]]
    if selection.get('context') == CTX_DC_COUNT:
        _DC_TARGET_HP = None
    return result


def _attack_choice(selection: dict[str, Any], state: dict[str, Any]) -> list[int]:
    options = selection.get('option') or []
    if not options:
        return []
    active = _active(_me(state))
    active_id = _cid(active)

    def score(index: int) -> float:
        option = options[index]
        attack_id = option.get('attackId')
        damage = _option_damage(option)
        text = _option_text(option)
        if attack_id == 937:
            base = 180000
        elif attack_id == 998:
            base = 110000
        elif attack_id == 997:
            base = 20000
        elif attack_id == 935:
            base = 10000 if _dark_count(active) >= 1 else -10000
        elif attack_id == 934:
            base = 1000
        elif attack_id == 323:
            base = 10000
        elif damage > 0:
            base = damage * 1000
        elif active_id == MARNIES_GRIMMSNARL_EX:
            base = 180000
        elif active_id == YVELTAL:
            base = _damage_for_attacker(active or {}) * 1000
        elif active_id == BUDEW:
            base = 10000
        else:
            base = index
        if 'Shadow Bullet' in text or 'シャドーバレット' in text:
            base += 5000
        if 'Spiky Wheel' in text or 'Spike Wheel' in text or 'トゲトゲぐるま' in text:
            base += 4500
        if 'Itchy Pollen' in text or 'むずむずかふん' in text:
            base += 2500 if _num(state.get('turn'), 0) <= 2 else 0
        return base

    return [max(range(len(options)), key=score)]


def _ideal_setup_bench_choice(state: dict[str, Any], selection: dict[str, Any]) -> list[int]:
    """Use the logged setup: keep Impidimp active and bench available Munkidori."""
    options = selection.get('option') or []
    me = _me(state)
    min_count = max(0, _num(selection.get('minCount'), 0))
    max_count = min(len(options), max(0, _num(selection.get('maxCount'), 0)))
    desired = {
        MARNIES_IMPIDIMP: max(0, 3 - _marnie_line_count(me)),
        MUNKIDORI: max(0, 2 - _field_count(me, MUNKIDORI)),
        SNORUNT: 0,
    }
    chosen: list[int] = []
    for card_id in (MUNKIDORI, MARNIES_IMPIDIMP, SNORUNT):
        candidates = [
            index for index, option in enumerate(options)
            if _option_card_id(state, selection, option) == card_id
        ]
        chosen.extend(candidates[: desired[card_id]])
        if len(chosen) >= max_count:
            return chosen[:max_count]
    if len(chosen) < min_count:
        ranked = sorted(
            ((_basic_play_score(_option_card_id(state, selection, option), state), index)
             for index, option in enumerate(options) if index not in chosen),
            reverse=True,
        )
        chosen.extend(index for _, index in ranked[: min_count - len(chosen)])
    return chosen[:max_count]

def _punk_up_energy_choice(state: dict[str, Any], selection: dict[str, Any]) -> list[int] | None:
    """Select up to five Darkness Energy, matching all three high-rank logs."""
    if selection.get('context') != CTX_ATTACH_TO or _effect_card_id(selection) != MARNIES_GRIMMSNARL_EX:
        return None
    options = selection.get('option') or []
    dark_indices = [
        index for index, option in enumerate(options)
        if _option_card_id(state, selection, option) == DARK_E
    ]
    max_count = min(len(options), max(0, _num(selection.get('maxCount'), 0)), 5)
    min_count = max(0, _num(selection.get('minCount'), 0))
    target_count = max(min_count, min(5, len(dark_indices), max_count))
    return dark_indices[:target_count]

def decide(obs: dict[str, Any], selection: dict[str, Any]) -> list[int]:
    global _PLAN, _LAST_TURN, _LAST_PLAYER, _DC_TARGET_HP
    state = obs.get('current') or {}
    selection_type = selection.get('type')
    options = selection.get('option') or []
    _update_opponent_memory(state)
    global _OWN_KO_TURN
    if selection.get('context') == CTX_TO_ACTIVE and _active(_me(state)) is None and _num(state.get('turn'), 0) > 0:
        _OWN_KO_TURN = _num(state.get('turn'), 0)

    turn = _num(state.get('turn'), -1)
    player = _num(state.get('yourIndex'), 0)
    if turn != _LAST_TURN or player != _LAST_PLAYER:
        _PLAN = None
        _DC_TARGET_HP = None
    _LAST_TURN, _LAST_PLAYER = turn, player

    if selection_type == ST_YESNO:
        return _yes_no_choice(selection, state)
    if selection_type == ST_COUNT:
        return _count_choice(selection)
    if selection_type == ST_ATTACK:
        return _attack_choice(selection, state)
    if not state.get('players'):
        min_count = max(1, _num(selection.get('minCount'), 1))
        return list(range(min(min_count, len(options))))

    punk_energy = _punk_up_energy_choice(state, selection)
    if punk_energy is not None:
        return punk_energy

    if selection.get('context') == CTX_SETUP_BENCH:
        return _ideal_setup_bench_choice(state, selection)

    if selection_type == ST_MAIN:
        _PLAN = _build_plan(state)
        scored = [(_main_score(option, state, selection, _PLAN), index) for index, option in enumerate(options)]
        result = _finalize(selection, scored)
        if result:
            return result
        for index, option in enumerate(options):
            if option.get('type') == OT_END:
                return [index]
        return [0]

    plan = _PLAN or _build_plan(state)
    scored = [(_ctx_score(option, selection, state, plan), index) for index, option in enumerate(options)]
    result = _finalize(selection, scored)

    if selection.get('context') in (CTX_DMG_COUNTER, CTX_DMG_COUNTER_ANY, CTX_DAMAGE):
        _DC_TARGET_HP = None
        if result:
            selected = _card_at(state, selection, options[result[0]])
            selected_player = _num(options[result[0]].get('playerIndex'), player)
            if selected is not None and selected_player != player:
                _DC_TARGET_HP = _remaining_hp(selected)

    if result:
        return result
    return [0] if _num(selection.get('minCount'), 0) > 0 else []


def agent(obs_dict: dict[str, Any]) -> list[int]:
    try:
        selection = obs_dict.get('select')
        if selection is None:
            _reset_episode_memory()
            _log({'t': time.time(), 'phase': 'deck', 'version': 'grimmsnarl_v5'})
            return list(DECK)
        if not selection.get('option'):
            return []
        choice = decide(obs_dict, selection)
        if not choice and _num(selection.get('minCount'), 0) > 0:
            choice = [0]
        state = obs_dict.get('current') or {}
        _log({
            't': time.time(),
            'version': 'grimmsnarl_v5',
            'st': selection.get('type'),
            'ctx': selection.get('context'),
            'n': len(selection.get('option') or []),
            'choice': choice,
            'turn': state.get('turn'),
            'supporterPlayed': state.get('supporterPlayed'),
            'energyAttached': state.get('energyAttached'),
            'opponentArchetype': _opponent_archetype(state),
            'seenOpponentIds': sorted(_SEEN_OPPONENT_IDS),
        })
        return choice
    except Exception as exc:
        _log({'t': time.time(), 'version': 'grimmsnarl_v5', 'error': str(exc)})
        try:
            selection = obs_dict.get('select') or {}
            option_count = len(selection.get('option') or [])
            if option_count == 0:
                return []
            required = max(1, _num(selection.get('minCount'), 1))
            return list(range(min(required, option_count)))
        except Exception:
            return [0]
