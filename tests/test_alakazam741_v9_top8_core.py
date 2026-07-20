"""Golden-state tests for alakazam741_v9_top8_core.

These lock the top-8 general core behaviours the analysis requires: attack reservation, the
pre-attack gate, the dynamic (non-fixed-floor) deck model, the RECOVER / LOCKED states, and the
safety invariants (never end while attackable, never retreat off an attacker, never self-KO the
last Dudunsparce, never place a 0-damage Powerful Hand, never lose a current KO to a hand spend).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.agent_loader import load_dir_agent_module

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agents" / "alakazam" / "alakazam741_v9_top8_core"

for path in (ROOT / "vendor", ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _load_agent(name: str):
    return load_dir_agent_module(ROOT / "agents" / "alakazam" / name)


M = _load_agent("alakazam741_v9_top8_core")
V8 = _load_agent("alakazam741_v8")
C = M.C

from cg.api import (  # noqa: E402
    AreaType, Card, Observation, Option, OptionType, Player, Pokemon, Select, SelectContext, State,
)

MIST_ENERGY = 11
JAMMING_TOWER = 1246  # any opposing stadium id (used only as "some enemy stadium")


def mk_poke(cid, hp=None, energies=(), energy_card_ids=None):
    data = M.card_table[cid]
    if energy_card_ids is None:
        energy_card_ids = [C.PSYCHIC_ENERGY] * len(energies)
    return Pokemon(
        id=cid,
        hp=data.hp if hp is None else hp,
        maxHp=data.hp,
        energies=list(energies),
        energyCards=[Card(id=e) for e in energy_card_ids],
        tools=[],
    )


def mk_me(active, bench=(), hand=(), deck_count=30, prizes=4, hand_count=None, discard=()):
    cards = [Card(id=c) for c in hand]
    return Player(
        active=[active],
        bench=list(bench),
        hand=cards,
        handCount=len(cards) if hand_count is None else hand_count,
        deckCount=deck_count,
        prize=[Card() for _ in range(prizes)],
        discard=[Card(id=c) for c in discard],
    )


def mk_opp(active, bench=(), hand_count=4, deck_count=30, prizes=4):
    return Player(
        active=[active],
        bench=list(bench),
        hand=None,
        handCount=hand_count,
        deckCount=deck_count,
        prize=[Card() for _ in range(prizes)],
        discard=[],
    )


def mk_obs(me, opp, options, context=SelectContext.MAIN, turn=6, looking=None,
           context_card=None, stadium=()):
    select = Select(context=context, minCount=1, maxCount=1,
                    option=list(options), contextCard=context_card)
    state = State(turn=turn, yourIndex=0, players=[me, opp],
                  stadium=list(stadium), looking=list(looking or []))
    return Observation(select=select, current=state)


def policy(obs):
    return M.AlakazamPolicy(obs)


# ── infrastructure invariants (items 19-22) ──────────────────────────────────
def test_loader_uses_bundled_policy_base():  # item 19
    bundled = Path(sys.modules[M.BasePolicy.__module__].__file__).resolve()
    assert bundled == (AGENT_DIR / "policy_base.py").resolve()
    # A different agent loaded in the same process must not swap our BasePolicy source.
    assert Path(sys.modules[V8.__name__].__file__).resolve().name == "main.py"


def test_deck_is_60_cards_identical_to_v8():  # item 20
    v9 = (AGENT_DIR / "deck.csv").read_text(encoding="utf-8-sig").split()
    v8 = (ROOT / "agents" / "alakazam" / "alakazam741_v8" / "deck.csv").read_text(encoding="utf-8-sig").split()
    assert len(v9) == 60
    assert v9 == v8  # first version keeps v8's exact 60 cards


def test_metadata_name_and_version_are_v9():  # item 21
    metadata = json.loads((AGENT_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "alakazam741_v9_top8_core"
    assert metadata["version"].startswith("9.")


def test_agent_returns_legal_fallback_on_bad_observation():  # item 22
    assert M.agent({"select": {"minCount": 1, "maxCount": 1, "option": [{"type": 14}]}}) == [0]


# ── attack reservation / pressure (items 1, 2, 3, 11, 12, 14) ─────────────────
def test_attackable_alakazam_does_not_end():  # item 1
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), hand=[C.PSYCHIC_ENERGY] * 6)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(me, opp, [Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
                           Option(type=OptionType.END)])
    assert policy(obs).choose()[0] == 0


def test_attackable_alakazam_does_not_retreat_to_a_shield():  # item 2
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               bench=[mk_poke(C.ALAKAZAM, energies=[5])], hand=[C.PSYCHIC_ENERGY] * 6)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(me, opp, [Option(type=OptionType.RETREAT),
                           Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
                           Option(type=OptionType.END)])
    assert policy(obs).choose()[0] == 1


def test_current_ko_is_not_lost_to_hand_spend():  # item 3
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               hand=[C.BUDDY_POFFIN, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY,
                     C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY])
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=100))  # 20*5 == 100 exactly KOs
    obs = mk_obs(me, opp, [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
                           Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
                           Option(type=OptionType.END)])
    assert policy(obs).choose()[0] == 1


def test_pressure_attacks_after_a_safe_draw():  # item 11
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), bench=[mk_poke(C.DUDUNSPARCE)],
               hand=[C.PSYCHIC_ENERGY] * 4)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    # With a safe benched Dudunsparce draw offered, it runs BEFORE the (non-lethal) attack...
    with_ability = mk_obs(me, opp, [
        Option(type=OptionType.ABILITY, area=AreaType.BENCH, index=0),
        Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
        Option(type=OptionType.END)])
    assert policy(with_ability).choose()[0] == 0
    # ...and once no pre-attack action remains, the attack is taken (never END).
    attack_only = mk_obs(me, opp, [Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
                                    Option(type=OptionType.END)])
    assert policy(attack_only).choose()[0] == 0


def test_pressure_rejects_pre_attack_action_that_loses_the_attack():  # item 12
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               hand=[C.POKE_PAD, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY,
                     C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY])
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=100))  # 20*5 == 100 KOs; Poké Pad would drop it to 80
    obs = mk_obs(me, opp, [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
                           Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
                           Option(type=OptionType.END)])
    assert policy(obs).choose()[0] == 1


def test_setup_completes_attacker_before_extra_development():  # item 10 (spec numbering)
    # SETUP: evolving the energised Kadabra into Alakazam (this-turn attacker) beats benching
    # another Abra.
    me = mk_me(active=mk_poke(C.KADABRA, energies=[5]), bench=[mk_poke(C.ABRA)],
               hand=[C.ALAKAZAM, C.ABRA])
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(me, opp, [
        Option(type=OptionType.EVOLVE, area=AreaType.HAND, index=0,
               inPlayArea=AreaType.ACTIVE, inPlayIndex=0),
        Option(type=OptionType.PLAY, area=AreaType.HAND, index=1)])
    pol = policy(obs)
    assert pol._state == M.TurnState.SETUP
    assert pol.choose()[0] == 0


# ── Dudunsparce / dynamic deck model (items 4, 5, 6, 7, 8, 9) ─────────────────
def test_safe_dudunsparce_then_attack():  # item 4
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), bench=[mk_poke(C.DUDUNSPARCE)],
               hand=[C.PSYCHIC_ENERGY] * 4, deck_count=30)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(me, opp, [Option(type=OptionType.ABILITY, area=AreaType.BENCH, index=0),
                           Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
                           Option(type=OptionType.END)])
    pol = policy(obs)
    assert pol._state == M.TurnState.PRESSURE
    assert pol._attack_reserved is True
    assert pol.choose()[0] == 0  # the safe draw runs first, above the attack


def test_last_active_dudunsparce_does_not_run_away_draw():  # item 5
    me = mk_me(active=mk_poke(C.DUDUNSPARCE), bench=[], hand=[C.PSYCHIC_ENERGY] * 3)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(me, opp, [Option(type=OptionType.ABILITY, area=AreaType.ACTIVE, index=0),
                           Option(type=OptionType.END)])
    assert policy(obs).choose()[0] == 1


def test_dudunsparce_returns_raise_effective_deck():  # item 6
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    # deck 3, 2 prizes, a backup Abra so secures_backup is not what enables the draw.
    with_dudun = mk_obs(
        mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
              bench=[mk_poke(C.DUDUNSPARCE), mk_poke(C.ABRA)],
              hand=[C.PSYCHIC_ENERGY] * 3, deck_count=3, prizes=2), opp, [])
    pol = policy(with_dudun)
    assert pol._deck_returns_available() == 2
    assert pol._effective_deck() == pol.me.deckCount + 2
    assert pol._optional_spend_ok(cost=1) is True
    # Same low deck WITHOUT the Dudunsparce return -> the optional draw is refused.
    without_dudun = mk_obs(
        mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), bench=[mk_poke(C.ABRA)],
              hand=[C.PSYCHIC_ENERGY] * 3, deck_count=3, prizes=2), opp, [])
    assert policy(without_dudun)._optional_spend_ok(cost=1) is False


def test_fixed_threshold_alone_does_not_reject_draw():  # item 7
    # v8 floor = max(8, prizes+3) = 8, so deckCount 8 would block; the dynamic model allows it
    # because we still win (2 turns) long before decking out (7).
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), bench=[mk_poke(C.ABRA)],
               hand=[C.PSYCHIC_ENERGY] * 3, deck_count=8, prizes=2)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    pol = policy(mk_obs(me, opp, []))
    assert 8 <= max(8, len(me.prize) + 3)  # v8 would have hit its floor here
    assert pol._optional_spend_ok(cost=1) is True


def test_deckout_before_win_rejects_optional_draw():  # item 8
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), bench=[mk_poke(C.ABRA)],
               hand=[C.PSYCHIC_ENERGY] * 3, deck_count=3, prizes=6)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    pol = policy(mk_obs(me, opp, []))
    assert pol._turns_to_deckout() < pol._turns_to_win()
    assert pol._optional_spend_ok(cost=1) is False


def test_draw_to_reach_ko_allowed_even_at_low_deck():  # item 9
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               hand=[C.PSYCHIC_ENERGY] * 6, deck_count=3, prizes=6)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))  # 20*6=120, +3 draw -> 180 >= 140
    pol = policy(mk_obs(me, opp, []))
    assert pol._optional_spend_ok(cost=1, makes_lethal=False) is False   # not by the race
    assert pol._optional_spend_ok(cost=1, makes_lethal=True) is True     # but a KO rescues it


# ── RECOVER (item 13) ─────────────────────────────────────────────────────────
def test_recover_prioritises_rebuild_over_stadium():  # item 13
    me = mk_me(active=mk_poke(C.ABRA), hand=[C.RARE_CANDY, C.ALAKAZAM, C.BATTLE_CAGE],
               discard=[C.ALAKAZAM])  # a lost Alakazam -> RECOVER
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(me, opp,
                 [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),   # Rare Candy
                  Option(type=OptionType.PLAY, area=AreaType.HAND, index=2)],  # Battle Cage
                 stadium=[Card(id=JAMMING_TOWER, playerIndex=1)])
    pol = policy(obs)
    assert pol._state == M.TurnState.RECOVER
    assert pol.choose()[0] == 0  # rebuild (Rare Candy) beats the stadium


# ── LOCKED / Enhanced Hammer (items 14, 15, 16) ───────────────────────────────
def test_zero_damage_powerful_hand_is_not_used():  # item 14
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), hand=[C.PSYCHIC_ENERGY] * 6)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140, energies=[5], energy_card_ids=[MIST_ENERGY]))
    obs = mk_obs(me, opp, [Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
                           Option(type=OptionType.END)])
    pol = policy(obs)
    assert pol._alakazam_damage(M.POWERFUL_HAND, opp.active[0]) == 0
    assert pol.choose()[0] == 1  # END rather than a 0-damage attack


def test_enhanced_hammer_unlocks_and_attacks_same_turn():  # item 15
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               hand=[C.ENHANCED_HAMMER] + [C.PSYCHIC_ENERGY] * 5)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140, energies=[5], energy_card_ids=[MIST_ENERGY]))
    obs = mk_obs(me, opp, [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
                           Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
                           Option(type=OptionType.END)])
    pol = policy(obs)
    assert pol._state == M.TurnState.LOCKED
    assert pol._enhanced_hammer_worthwhile() is True
    assert pol.choose()[0] == 0  # play the Hammer to unlock; the attack lands after


def test_enhanced_hammer_not_wasted_when_attack_still_impossible():  # item 16
    # Alakazam has no energy: even after removing Mist it cannot attack, so the Hammer is held.
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[]),
               hand=[C.ENHANCED_HAMMER, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY])
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140, energies=[5], energy_card_ids=[MIST_ENERGY]))
    obs = mk_obs(me, opp, [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
                           Option(type=OptionType.END)])
    pol = policy(obs)
    assert pol._enhanced_hammer_worthwhile() is False
    assert pol.choose()[0] == 1  # END, not a wasted Hammer


# ── Xerosic (items 17, 18) ────────────────────────────────────────────────────
def test_non_mirror_prefers_meaningful_attack_over_xerosic():  # item 17
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               hand=[C.XEROSIC] + [C.PSYCHIC_ENERGY] * 5)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140), hand_count=8)  # big hand but NOT a mirror
    obs = mk_obs(me, opp, [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
                           Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
                           Option(type=OptionType.END)])
    assert policy(obs).choose()[0] == 1


def test_xerosic_rejected_when_it_would_lose_current_ko():  # item 18
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               hand=[C.XEROSIC] + [C.PSYCHIC_ENERGY] * 4)
    opp = mk_opp(mk_poke(C.ALAKAZAM, hp=100, energies=[5]), hand_count=6)  # mirror, big hand
    obs = mk_obs(me, opp, [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
                           Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
                           Option(type=OptionType.END)])
    assert policy(obs).choose()[0] == 1  # 20*5==100 KOs now; Xerosic (-1 card) is refused


def test_mirror_xerosic_runs_before_a_non_winning_attack():  # supporting item 17/mirror
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]),
               hand=[C.XEROSIC] + [C.PSYCHIC_ENERGY] * 5)
    opp = mk_opp(mk_poke(C.ALAKAZAM, hp=140, energies=[5]), hand_count=6)  # mirror, non-lethal
    obs = mk_obs(me, opp, [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
                           Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
                           Option(type=OptionType.END)])
    assert policy(obs).choose()[0] == 0  # 20*6=120 < 140, so disrupt the mirror hand first
