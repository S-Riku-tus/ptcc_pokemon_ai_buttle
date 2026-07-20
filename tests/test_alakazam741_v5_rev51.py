"""alakazam741_v5 rev5.1 (ユーザーのv5ログ確認に基づく修正) の再現テスト。

実ゲーム状態を最小限モックし、修正前の悪手を選ばないこと・望ましい代替行動を
選ぶこと・正常な局面では従来の挙動を壊さないことを確認する。
"""
from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "vendor"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AGENT_DIR = ROOT / "agents" / "alakazam" / "alakazam741_v5"


def _load_v5():
    spec = importlib.util.spec_from_file_location(
        "agent_alakazam741_v5_under_test", AGENT_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_v5()
C = M.C

from cg.api import (  # noqa: E402
    AreaType, Card, Observation, Option, OptionType, Player, Pokemon,
    Select, SelectContext, State,
)

MIST_ENERGY = 11
TR_ARTICUNO = 414          # 全体保護特性(たねロケット団ポケモンへの効果を防ぐ)
PLAIN_140 = 66             # 1サイド・140HPの汎用ターゲットとして流用


def mk_poke(cid, hp=None, energies=(), energy_card_ids=None):
    data = M.card_table[cid]
    if energy_card_ids is None:
        energy_card_ids = [C.PSYCHIC_ENERGY] * len(energies)
    return Pokemon(
        id=cid, hp=data.hp if hp is None else hp, maxHp=data.hp,
        energies=list(energies),
        energyCards=[Card(id=e) for e in energy_card_ids], tools=[])


def mk_me(active, bench=(), hand=(), deck_count=30, prizes=4, hand_count=None):
    hand_cards = [Card(id=c) for c in hand]
    return Player(
        active=[active], bench=list(bench), hand=hand_cards,
        handCount=len(hand_cards) if hand_count is None else hand_count,
        deckCount=deck_count, prize=[Card() for _ in range(prizes)], discard=[])


def mk_opp(active, bench=(), hand_count=4, deck_count=30, prizes=4):
    return Player(
        active=[active], bench=list(bench), hand=None, handCount=hand_count,
        deckCount=deck_count, prize=[Card() for _ in range(prizes)], discard=[])


def mk_obs(me, opp, options, context=SelectContext.MAIN, turn=6):
    select = Select(context=context, minCount=1, maxCount=1, option=list(options))
    state = State(turn=turn, yourIndex=0, players=[me, opp], stadium=[])
    return Observation(select=select, current=state)


def policy(obs):
    return M.AlakazamPolicy(obs)


# ── デッキ構成 (R2) ──────────────────────────────────────────────────────────
def test_deck_is_60_and_rev51_composition():
    deck = [int(x) for x in (AGENT_DIR / "deck.csv").read_text(
        encoding="utf-8-sig").split()]
    counts = Counter(deck)
    assert len(deck) == 60
    assert counts[C.BOSS_ORDERS] == 0          # ボスの指令は全撤去
    assert counts[C.DUDUNSPARCE_EX] == 1       # 効果無効対策のノココッチex
    assert counts[C.POKE_PAD] == 3             # ポケパッド 2→3
    # 勝ち筋のライン/エネは不変
    assert counts[C.ALAKAZAM] == 4 and counts[C.KADABRA] == 4 and counts[C.ABRA] == 4
    assert counts[C.DUNSPARCE] == 4 and counts[C.DUDUNSPARCE] == 3
    assert counts[C.PSYCHIC_ENERGY] + counts[C.TELEPATH_ENERGY] == 7
    assert counts[C.HYPER_AROMA] == 1          # ACE SPECは1枚のみ
    assert max(c for cid, c in counts.items()) <= 4
    # 進化元が進化先より少なくない
    assert counts[C.DUNSPARCE] >= counts[C.DUDUNSPARCE] + counts[C.DUDUNSPARCE_EX]


def test_static_validation_passes():
    from scripts.validate_agent import validate_agent
    result = validate_agent(AGENT_DIR)
    assert result["deck_size"] == 60
    assert not result["warnings"]


# ── R1: ふしぎなアメ — 今ターン攻撃が立つならユンゲラー橋より優先 ─────────────
def _candy_setup(abra_energies, opp_hp):
    me = mk_me(
        active=mk_poke(C.ABRA, energies=abra_energies),
        hand=[C.KADABRA, C.ALAKAZAM, C.RARE_CANDY,
              C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY, C.POKE_PAD, C.HILDA])
    opp = mk_opp(mk_poke(PLAIN_140, hp=opp_hp))
    options = [
        Option(type=OptionType.PLAY, area=AreaType.HAND, index=2),          # アメ
        Option(type=OptionType.EVOLVE, area=AreaType.HAND, index=0,         # ユンゲラー
               inPlayArea=AreaType.ACTIVE, inPlayIndex=0),
        Option(type=OptionType.END),
    ]
    return mk_obs(me, opp, options)


def test_candy_preferred_when_it_enables_ko_this_turn():
    # エネ付きケーシィ+手札7枚: アメ後の手札5枚 → PH 100 ≥ 相手60HP = KOが立つ
    obs = _candy_setup(abra_energies=[5], opp_hp=60)
    assert policy(obs).choose()[0] == 0        # アメ(21000) > ユンゲラー橋(20000)


def test_bridge_preferred_when_abra_cannot_attack():
    # エネ無しケーシィ: アメで進化しても今ターン殴れない → 従来通り橋を優先
    obs = _candy_setup(abra_energies=[], opp_hp=60)
    assert policy(obs).choose()[0] == 1        # ユンゲラー進化(20000) > アメ(8000)


# ── R3: フーディンライン退避 ─────────────────────────────────────────────────
def _danger_setup(my_hand_count, opp_hp, bench, opp_hand_count=8):
    """相手はミラーのフーディン(PH=20×手札)で、こちらのActiveフーディンを
    次ターンKOできる盤面。"""
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]), bench=bench,
        hand=[C.PSYCHIC_ENERGY] * my_hand_count)
    opp = mk_opp(mk_poke(C.ALAKAZAM, hp=opp_hp, energies=[5]),
                 hand_count=opp_hand_count)
    options = [
        Option(type=OptionType.RETREAT),
        Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
        Option(type=OptionType.END),
    ]
    return mk_obs(me, opp, options)


def test_retreat_saves_alakazam_when_doomed_and_no_ko():
    # 手札3枚(PH60) vs 相手140HP: KO不可。相手PH=160でこちらは次ターン確実に死ぬ。
    obs = _danger_setup(my_hand_count=3, opp_hp=140,
                        bench=[mk_poke(C.DUDUNSPARCE)])
    assert policy(obs).choose()[0] == 0        # 退避(5600) > 非KO攻撃(~1060)


def test_lethal_attack_beats_retreat():
    # 同じ危険盤面でもKOが立つなら攻撃(ユーザー例外: 後続で継続 or 今取る)
    obs = _danger_setup(my_hand_count=3, opp_hp=60,
                        bench=[mk_poke(C.DUDUNSPARCE)])
    assert policy(obs).choose()[0] == 1        # KO攻撃(30000) > 退避(-1)


def test_no_retreat_without_shield():
    # ベンチがフーディンラインだけなら退避しない(ラインをラインの盾にしない)
    obs = _danger_setup(my_hand_count=3, opp_hp=140,
                        bench=[mk_poke(C.KADABRA)])
    assert policy(obs).choose()[0] == 1        # 退避-1 → 攻撃が最善


def test_no_retreat_when_safe():
    # 相手の打点が足りない(相手手札2枚=PH40 < 140HP)なら退避しない
    obs = _danger_setup(my_hand_count=3, opp_hp=140,
                        bench=[mk_poke(C.DUDUNSPARCE)], opp_hand_count=2)
    assert policy(obs).choose()[0] == 1


def test_ex_shield_not_used_when_it_loses_the_game():
    # 相手の残りサイド2 → ノココッチex(2サイド)を盾に出すと負け筋 → 退避しない
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               bench=[mk_poke(C.DUDUNSPARCE_EX)],
               hand=[C.PSYCHIC_ENERGY] * 3)
    opp = mk_opp(mk_poke(C.ALAKAZAM, hp=140, energies=[5]),
                 hand_count=8, prizes=2)
    options = [Option(type=OptionType.RETREAT),
               Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
               Option(type=OptionType.END)]
    obs = mk_obs(me, opp, options)
    assert policy(obs).choose()[0] == 1


def test_switch_promotes_dudunsparce_shield_over_second_alakazam():
    # 退避後のSWITCHで、+300のフーディンではなく のこっち系の盾を前に出す
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               bench=[mk_poke(C.DUDUNSPARCE), mk_poke(C.ALAKAZAM, energies=[5])],
               hand=[C.PSYCHIC_ENERGY] * 3)
    opp = mk_opp(mk_poke(C.ALAKAZAM, hp=140, energies=[5]), hand_count=8)
    options = [
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=0),
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=1, playerIndex=0),
    ]
    obs = mk_obs(me, opp, options, context=SelectContext.SWITCH)
    assert policy(obs).choose()[0] == 0        # 66の盾 > 2体目のフーディン


def test_promote_after_ko_still_prefers_alakazam():
    # 通常の昇格(前が倒れた直後=退避モードでない)は従来通りフーディン優先
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               bench=[mk_poke(C.DUDUNSPARCE), mk_poke(C.ALAKAZAM, energies=[5])],
               hand=[C.PSYCHIC_ENERGY] * 3)
    opp = mk_opp(mk_poke(C.ALAKAZAM, hp=140, energies=[5]), hand_count=2)  # 危険なし
    options = [
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=0, playerIndex=0),
        Option(type=OptionType.CARD, area=AreaType.BENCH, index=1, playerIndex=0),
    ]
    obs = mk_obs(me, opp, options, context=SelectContext.SWITCH)
    assert policy(obs).choose()[0] == 1


# ── R2: ノココッチex — 効果ロック対面で進化・給エネ・貫通150 ──────────────────
def _lock_board_obs(options, hand=(), bench=(), context=SelectContext.MAIN):
    """相手: ロケット団のフリーザー(全体保護特性)が前 → PHは0ダメ。"""
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               bench=list(bench), hand=list(hand))
    opp = mk_opp(mk_poke(TR_ARTICUNO))
    return mk_obs(me, opp, options, context=context)


def test_evolve_dudun_ex_under_effect_lock():
    obs = _lock_board_obs(
        options=[
            Option(type=OptionType.EVOLVE, area=AreaType.HAND, index=0,   # 66
                   inPlayArea=AreaType.BENCH, inPlayIndex=0),
            Option(type=OptionType.EVOLVE, area=AreaType.HAND, index=1,   # 306
                   inPlayArea=AreaType.BENCH, inPlayIndex=0),
        ],
        hand=[C.DUDUNSPARCE, C.DUDUNSPARCE_EX],
        bench=[mk_poke(C.DUNSPARCE)])
    assert policy(obs).choose()[0] == 1        # 306(21200) > 66(19000)


def test_evolve_engine_66_in_normal_matchup():
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               bench=[mk_poke(C.DUNSPARCE)],
               hand=[C.DUDUNSPARCE, C.DUDUNSPARCE_EX])
    opp = mk_opp(mk_poke(PLAIN_140))
    options = [
        Option(type=OptionType.EVOLVE, area=AreaType.HAND, index=0,
               inPlayArea=AreaType.BENCH, inPlayIndex=0),
        Option(type=OptionType.EVOLVE, area=AreaType.HAND, index=1,
               inPlayArea=AreaType.BENCH, inPlayIndex=0),
    ]
    obs = mk_obs(me, opp, options)
    assert policy(obs).choose()[0] == 0        # 通常対面は66を温存進化(306は900)


def test_dudun_ex_attach_until_drill_then_stop():
    obs = _lock_board_obs(options=[Option(type=OptionType.END)])
    pol = policy(obs)
    fueling = mk_poke(C.DUDUNSPARCE_EX, energies=[5])          # Tailは払えるが…
    assert pol._dudun_ex_attach_score(fueling) == 8100          # Drillまで貼り続ける
    full = mk_poke(C.DUDUNSPARCE_EX, energies=[5, 5, 5])
    assert pol._dudun_ex_attach_score(full) == -1               # 3エネで打ち止め


def test_destructive_drill_pierces_effect_lock():
    obs = _lock_board_obs(options=[Option(type=OptionType.END)])
    pol = policy(obs)
    target = pol.opponent.active[0]
    assert pol._alakazam_damage(M.POWERFUL_HAND, target) == 0   # PHは0(従来通り)
    assert pol._alakazam_damage(M.DESTRUCTIVE_DRILL, target) == 150  # 貫通150


def test_no_damage_path_false_when_dudun_ex_available():
    # 306が場にいる/山札に眠っている間は「詰み」扱いにしない(掘って探す)
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               bench=[mk_poke(C.DUDUNSPARCE_EX, energies=[5, 5, 5])],
               hand=[C.PSYCHIC_ENERGY], deck_count=30)
    opp = mk_opp(mk_poke(TR_ARTICUNO))
    obs = mk_obs(me, opp, [Option(type=OptionType.END)])
    assert policy(obs)._no_damage_path() is False


# ── R4: 逃げ足ドロー — ベンチ2体以上なら高手札×低山札でも使う ─────────────────
def _run_away_obs(bench, hand_count=12, deck_count=14):
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), bench=bench,
               hand=[C.PSYCHIC_ENERGY] * hand_count, deck_count=deck_count, prizes=4)
    opp = mk_opp(mk_poke(PLAIN_140))
    dudun_i = next(i for i, p in enumerate(bench) if p.id == C.DUDUNSPARCE)
    options = [
        Option(type=OptionType.ABILITY, area=AreaType.BENCH, index=dudun_i),
        Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
        Option(type=OptionType.END),
    ]
    return mk_obs(me, opp, options)


def test_run_away_draw_used_with_two_plus_bench():
    obs = _run_away_obs(
        bench=[mk_poke(C.DUDUNSPARCE), mk_poke(C.DUNSPARCE), mk_poke(C.ABRA)])
    pol = policy(obs)
    assert pol._score(pol.select.option[0]) == 15000   # 旧: 12/14ガードで-1だった


def test_run_away_draw_still_guarded_when_bench_thin():
    # ベンチ1体(=盤面2体)で手札に補充が無ければ従来通り使わない
    obs = _run_away_obs(bench=[mk_poke(C.DUDUNSPARCE)])
    pol = policy(obs)
    assert pol._score(pol.select.option[0]) < 0


def test_run_away_draw_respects_deck_floor():
    # 山札フロア(=max(8,サイド+3))を割る消費は今も禁止(山札切れ対策の維持)
    obs = _run_away_obs(
        bench=[mk_poke(C.DUDUNSPARCE), mk_poke(C.DUNSPARCE), mk_poke(C.ABRA)],
        hand_count=5, deck_count=9)
    pol = policy(obs)
    assert pol._score(pol.select.option[0]) < 0


# ── 形式面 ───────────────────────────────────────────────────────────────────
def test_agent_returns_deck_on_setup():
    deck = M.agent({"select": None})
    assert len(deck) == 60 and deck.count(C.DUDUNSPARCE_EX) == 1


def test_agent_returns_legal_selection_on_mock_dict():
    # 想定外のdictでも合法なフォールバックを返す(クラッシュしない)
    out = M.agent({"select": {"minCount": 1, "maxCount": 1,
                              "option": [{"type": 14}]}})
    assert out == [0]
