from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / "main.py").exists():
    AGENT_DIR = HERE
    ROOT = HERE.parent
else:
    ROOT = Path(__file__).resolve().parents[1]
    AGENT_DIR = ROOT / "agents" / "alakazam741_v7"

for p in (ROOT / "vendor", ROOT, AGENT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.modules.pop("policy_base", None)


def _load_v7():
    spec = importlib.util.spec_from_file_location(
        "agent_alakazam741_v7_under_test", AGENT_DIR / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = _load_v7()
C = M.C

from cg.api import (  # noqa: E402
    AreaType,
    Card,
    Observation,
    Option,
    OptionType,
    Player,
    Pokemon,
    Select,
    SelectContext,
    State,
)


MIST_ENERGY = 11
TEAM_ROCKET_ARTICUNO = 414


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


def mk_me(active, bench=(), hand=(), discard=(), deck_count=30, prizes=4, hand_count=None):
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


def mk_obs(
    me,
    opp,
    options,
    context=SelectContext.MAIN,
    turn=6,
    looking=None,
    context_card=None,
    min_count=1,
    max_count=1,
    stadium=(),
    stadium_played=False,
):
    select = Select(
        context=context,
        minCount=min_count,
        maxCount=max_count,
        option=list(options),
        contextCard=context_card,
    )
    state = State(
        turn=turn,
        yourIndex=0,
        players=[me, opp],
        stadium=list(stadium),
        looking=list(looking or []),
        stadiumPlayed=stadium_played,
    )
    return Observation(select=select, current=state)


def policy(obs):
    return M.AlakazamPolicy(obs)


def test_deck_is_60_and_v7_composition():
    deck = [int(x) for x in (AGENT_DIR / "deck.csv").read_text(encoding="utf-8-sig").split()]
    counts = Counter(deck)
    assert len(deck) == 60
    assert counts[C.HYPER_AROMA] == 1
    assert counts[C.LILLIE] == 1
    assert counts[C.DUNSPARCE] == 3
    assert counts[C.BATTLE_CAGE] == 3
    assert counts[C.NIGHT_STRETCHER] == 3
    assert counts[1120] == 0  # Crushing Hammer was removed after the matchup review.
    assert counts[1182] == 0  # Boss's Orders was intentionally removed.
    assert counts[C.ALAKAZAM] == counts[C.KADABRA] == counts[C.ABRA] == 4
    assert max(c for cid, c in counts.items() if cid not in range(1, 9)) <= 4


def test_static_validation_passes():
    from scripts.validate_agent import validate_agent

    result = validate_agent(AGENT_DIR)
    assert result["deck_size"] == 60
    assert result["metadata"]["name"] == "alakazam741_v7"
    assert not result["warnings"]


def test_margin_ko_allows_safe_draw_before_attack():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUDUNSPARCE)],
        hand=[C.PSYCHIC_ENERGY] * 8,
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [
            Option(type=OptionType.ABILITY, area=AreaType.BENCH, index=0),
            Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
            Option(type=OptionType.END),
        ],
    )
    pol = policy(obs)
    assert pol._phase() == M.Phase.PRESSURE
    assert pol.choose()[0] == 0


def test_winning_ko_attacks_immediately():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUDUNSPARCE)],
        hand=[C.PSYCHIC_ENERGY] * 8,
        prizes=1,
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [
            Option(type=OptionType.ABILITY, area=AreaType.BENCH, index=0),
            Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
            Option(type=OptionType.END),
        ],
    )
    assert policy(obs).choose()[0] == 1


def test_current_ko_is_not_lost_to_hand_spend():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        hand=[C.BUDDY_POFFIN, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY,
              C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=100))
    obs = mk_obs(
        me,
        opp,
        [
            Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
            Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
            Option(type=OptionType.END),
        ],
    )
    assert policy(obs).choose()[0] == 1


def test_last_active_dudunsparce_with_unplayable_item_does_not_run_away():
    me = mk_me(active=mk_poke(C.DUDUNSPARCE), bench=[], hand=[C.RARE_CANDY])
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [
            Option(type=OptionType.ABILITY, area=AreaType.ACTIVE, index=0),
            Option(type=OptionType.END),
        ],
    )
    assert policy(obs).choose()[0] == 1


def test_active_dudunsparce_escapes_only_to_ready_bench_attacker():
    me = mk_me(
        active=mk_poke(C.DUDUNSPARCE),
        bench=[mk_poke(C.ALAKAZAM, energies=[5])],
        hand=[C.PSYCHIC_ENERGY] * 3,
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [
            Option(type=OptionType.ABILITY, area=AreaType.ACTIVE, index=0),
            Option(type=OptionType.END),
        ],
    )
    assert policy(obs).choose()[0] == 0


def test_low_deck_declines_optional_activation():
    me = mk_me(active=mk_poke(C.KADABRA), hand=[], deck_count=8, prizes=4)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.YES), Option(type=OptionType.NO)],
        context=SelectContext.ACTIVATE,
    )
    assert policy(obs).choose()[0] == 1


def test_effect_prevented_powerful_hand_is_not_used():
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), hand=[C.PSYCHIC_ENERGY] * 6)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140, energies=[5], energy_card_ids=[MIST_ENERGY]))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND), Option(type=OptionType.END)],
    )
    assert policy(obs)._phase() == M.Phase.LOCKED
    assert policy(obs).choose()[0] == 1


def test_board_wide_articuno_protection_blocks_powerful_hand():
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), hand=[C.PSYCHIC_ENERGY] * 8)
    opp = mk_opp(mk_poke(TEAM_ROCKET_ARTICUNO, hp=120))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND), Option(type=OptionType.END)],
    )
    assert policy(obs)._effect_prevented(opp.active[0])
    assert policy(obs).choose()[0] == 1


def test_locked_board_does_not_take_optional_draw():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUDUNSPARCE)],
        hand=[C.PSYCHIC_ENERGY] * 6,
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140, energies=[5], energy_card_ids=[MIST_ENERGY]))
    obs = mk_obs(
        me,
        opp,
        [
            Option(type=OptionType.ABILITY, area=AreaType.BENCH, index=0),
            Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
            Option(type=OptionType.END),
        ],
    )
    assert policy(obs).choose()[0] == 2


def test_enhanced_hammer_unlocks_only_an_immediate_ko():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        hand=[C.ENHANCED_HAMMER] + [C.PSYCHIC_ENERGY] * 7,
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140, energies=[5], energy_card_ids=[MIST_ENERGY]))
    obs = mk_obs(
        me,
        opp,
        [
            Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
            Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
            Option(type=OptionType.END),
        ],
    )
    assert policy(obs).choose()[0] == 0

    thin = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        hand=[C.ENHANCED_HAMMER, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
    )
    thin_obs = mk_obs(
        thin,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0), Option(type=OptionType.END)],
    )
    assert policy(thin_obs).choose()[0] == 1


def test_battle_cage_precedes_non_ko_attack_against_bench_threat():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUNSPARCE)],
        hand=[C.BATTLE_CAGE, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
         Option(type=OptionType.END)],
    )
    pol = policy(obs)
    pol.opponent_threatens_bench = lambda: True
    assert pol.choose()[0] == 0


def test_battle_cage_does_not_break_exact_ko():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUNSPARCE)],
        hand=[C.BATTLE_CAGE, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY,
              C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=100))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
         Option(type=OptionType.END)],
    )
    pol = policy(obs)
    pol.opponent_threatens_bench = lambda: True
    assert pol.choose()[0] == 1


def test_night_stretcher_recovers_first_backup_before_attack():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUNSPARCE)],
        hand=[C.NIGHT_STRETCHER, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
        discard=[C.ABRA],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
         Option(type=OptionType.END)],
    )
    assert policy(obs).choose()[0] == 0


def test_night_stretcher_is_hand_neutral_for_exact_ko():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUNSPARCE)],
        hand=[C.NIGHT_STRETCHER, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY,
              C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
        discard=[C.ABRA],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=100))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
         Option(type=OptionType.END)],
    )
    assert policy(obs).choose()[0] == 0


def test_night_stretcher_selects_route_completing_target():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUNSPARCE)],
        hand=[],
        discard=[C.ABRA, C.DUDUNSPARCE, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    options = [
        Option(type=OptionType.CARD, area=AreaType.DISCARD, index=i, playerIndex=0)
        for i in range(3)
    ]
    obs = mk_obs(
        me,
        opp,
        options,
        context=SelectContext.TO_HAND,
        context_card=Card(id=C.NIGHT_STRETCHER),
    )
    assert policy(obs).choose()[0] == 0


def test_first_backup_abra_precedes_non_ko_attack():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUNSPARCE)],
        hand=[C.ABRA, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
         Option(type=OptionType.END)],
    )
    assert policy(obs).choose()[0] == 0


def test_first_backup_abra_does_not_break_exact_ko():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUNSPARCE)],
        hand=[C.ABRA, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY,
              C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=100))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
         Option(type=OptionType.END)],
    )
    assert policy(obs).choose()[0] == 1


def test_battle_cage_replaces_opposing_stadium_before_non_ko_attack():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUNSPARCE)],
        hand=[C.BATTLE_CAGE, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
         Option(type=OptionType.END)],
        stadium=[Card(id=9999)],
    )
    assert policy(obs).choose()[0] == 0


def test_battle_cage_is_held_without_threat_or_opposing_stadium():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUNSPARCE)],
        hand=[C.BATTLE_CAGE, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
         Option(type=OptionType.END)],
    )
    assert policy(obs).choose()[0] == 1


def test_night_stretcher_is_held_without_a_route_completing_target():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.DUNSPARCE)],
        hand=[C.NIGHT_STRETCHER, C.PSYCHIC_ENERGY],
        discard=[C.ALAKAZAM],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
         Option(type=OptionType.END)],
    )
    assert policy(obs).choose()[0] == 1


def test_night_stretcher_prefers_basic_psychic_when_energy_starved():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM),
        hand=[],
        discard=[C.ABRA, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    options = [
        Option(type=OptionType.CARD, area=AreaType.DISCARD, index=i, playerIndex=0)
        for i in range(2)
    ]
    obs = mk_obs(
        me,
        opp,
        options,
        context=SelectContext.TO_HAND,
        context_card=Card(id=C.NIGHT_STRETCHER),
    )
    assert policy(obs).choose()[0] == 1


def test_phase_tiers_build_attacker_before_weak_attack():
    me = mk_me(active=mk_poke(C.KADABRA, energies=[5]), hand=[C.ALAKAZAM])
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [
            Option(type=OptionType.EVOLVE, area=AreaType.HAND, index=0,
                   inPlayArea=AreaType.ACTIVE, inPlayIndex=0),
            Option(type=OptionType.ATTACK, attackId=M.SUPER_PSY_BOLT),
            Option(type=OptionType.END),
        ],
    )
    assert policy(obs)._phase() in (M.Phase.SETUP, M.Phase.RECOVER)
    assert policy(obs).choose()[0] == 0


def test_attackable_alakazam_does_not_retreat():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.ALAKAZAM, energies=[5])],
        hand=[C.PSYCHIC_ENERGY] * 4,
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.RETREAT), Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND), Option(type=OptionType.END)],
    )
    assert policy(obs).choose()[0] == 1


def test_retreat_when_active_cannot_attack_and_bench_alakazam_ready():
    me = mk_me(
        active=mk_poke(C.DUNSPARCE),
        bench=[mk_poke(C.ALAKAZAM, energies=[5])],
        hand=[C.PSYCHIC_ENERGY] * 3,
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(me, opp, [Option(type=OptionType.RETREAT), Option(type=OptionType.END)])
    assert policy(obs).choose()[0] == 0


def test_lillie_does_not_break_complete_route():
    me = mk_me(
        active=mk_poke(C.ABRA, energies=[5]),
        hand=[C.LILLIE, C.RARE_CANDY, C.ALAKAZAM, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.PLAY, area=AreaType.HAND, index=1),
         Option(type=OptionType.END)],
    )
    assert policy(obs).choose()[0] == 1


def test_poffin_is_used_before_lillie():
    me = mk_me(
        active=mk_poke(C.DUNSPARCE),
        hand=[C.LILLIE, C.BUDDY_POFFIN, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.PLAY, area=AreaType.HAND, index=1),
         Option(type=OptionType.END)],
    )
    assert policy(obs).choose()[0] == 1


def test_useful_energy_attach_precedes_lillie():
    me = mk_me(active=mk_poke(C.ABRA), hand=[C.LILLIE, C.PSYCHIC_ENERGY])
    opp = mk_opp(mk_poke(C.DUDUNSPARCE))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
         Option(type=OptionType.ATTACH, area=AreaType.HAND, index=1,
                inPlayArea=AreaType.ACTIVE, inPlayIndex=0),
         Option(type=OptionType.END)],
    )
    assert policy(obs).choose()[0] == 1


def test_lillie_used_with_thin_no_attacker_hand():
    me = mk_me(active=mk_poke(C.DUNSPARCE), hand=[C.LILLIE, C.PSYCHIC_ENERGY])
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(me, opp, [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0), Option(type=OptionType.END)])
    assert policy(obs).choose()[0] == 0


def test_hyper_aroma_selects_balanced_three_card_package():
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    looking = [
        Card(id=C.KADABRA), Card(id=C.KADABRA), Card(id=C.KADABRA),
        Card(id=C.DUDUNSPARCE), Card(id=C.DUDUNSPARCE), Card(id=C.DUDUNSPARCE),
    ]
    options = [
        Option(type=OptionType.CARD, area=AreaType.LOOKING, index=i, playerIndex=0)
        for i in range(len(looking))
    ]
    obs = mk_obs(
        mk_me(active=mk_poke(C.ABRA), bench=[mk_poke(C.DUNSPARCE)], hand=[]),
        opp,
        options,
        context=SelectContext.TO_HAND,
        looking=looking,
        context_card=Card(id=C.HYPER_AROMA),
        min_count=3,
        max_count=3,
    )
    selected = policy(obs).choose()
    selected_ids = [looking[i].id for i in selected]
    assert len(selected) == 3
    assert C.KADABRA in selected_ids
    assert C.DUDUNSPARCE in selected_ids
    assert max(Counter(selected_ids).values()) <= 2


def test_agent_returns_legal_fallback_on_bad_dict():
    out = M.agent({"select": {"minCount": 1, "maxCount": 1, "option": [{"type": 14}]}})
    assert out == [0]
