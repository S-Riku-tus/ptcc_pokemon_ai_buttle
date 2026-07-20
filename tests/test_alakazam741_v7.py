from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agents" / "alakazam" / "alakazam741_v7"

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


def mk_me(active, bench=(), hand=(), deck_count=30, prizes=4, hand_count=None):
    cards = [Card(id=c) for c in hand]
    return Player(
        active=[active],
        bench=list(bench),
        hand=cards,
        handCount=len(cards) if hand_count is None else hand_count,
        deckCount=deck_count,
        prize=[Card() for _ in range(prizes)],
        discard=[],
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


def mk_obs(me, opp, options, context=SelectContext.MAIN, turn=6, looking=None, context_card=None):
    select = Select(
        context=context,
        minCount=1,
        maxCount=1,
        option=list(options),
        contextCard=context_card,
    )
    state = State(
        turn=turn,
        yourIndex=0,
        players=[me, opp],
        stadium=[],
        looking=list(looking or []),
    )
    return Observation(select=select, current=state)


def policy(obs):
    return M.AlakazamPolicy(obs)


def test_deck_is_60_and_v7_composition():
    deck = [int(x) for x in (AGENT_DIR / "deck.csv").read_text(encoding="utf-8-sig").split()]
    counts = Counter(deck)
    assert len(deck) == 60
    assert counts[C.HYPER_AROMA] == 1
    assert counts[C.ENRICHING_ENERGY] == 0
    assert counts[C.LILLIE] == 1
    assert counts[C.DUNSPARCE] == 3
    assert counts[C.ALAKAZAM] == 4 and counts[C.KADABRA] == 4 and counts[C.ABRA] == 4
    assert max(c for cid, c in counts.items() if cid not in range(1, 9)) <= 4


def test_static_validation_passes():
    from scripts.validate_agent import validate_agent

    result = validate_agent(AGENT_DIR)
    assert result["deck_size"] == 60
    assert result["metadata"]["name"] == "alakazam741_v7"
    assert not result["warnings"]


def test_pressure_phase_attacks_over_optional_draw_and_end():
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
    assert pol.choose()[0] == 1


def test_last_active_dudunsparce_does_not_run_away_draw():
    me = mk_me(active=mk_poke(C.DUDUNSPARCE), bench=[], hand=[C.PSYCHIC_ENERGY] * 3)
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


def test_effect_prevented_powerful_hand_is_not_used():
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), hand=[C.PSYCHIC_ENERGY] * 6)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140, energies=[5], energy_card_ids=[MIST_ENERGY]))
    obs = mk_obs(
        me,
        opp,
        [
            Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
            Option(type=OptionType.END),
        ],
    )
    pol = policy(obs)
    assert pol._phase() == M.Phase.LOCKED
    assert pol.choose()[0] == 1


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


def test_hammer_unlocks_effect_prevention_before_attack():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        hand=[C.ENHANCED_HAMMER, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
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


def test_current_ko_is_not_lost_to_hand_spend():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        hand=[C.BUDDY_POFFIN, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
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
        [
            Option(type=OptionType.RETREAT),
            Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND),
            Option(type=OptionType.END),
        ],
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
        [
            Option(type=OptionType.PLAY, area=AreaType.HAND, index=0),
            Option(type=OptionType.PLAY, area=AreaType.HAND, index=1),
            Option(type=OptionType.END),
        ],
    )
    assert policy(obs).choose()[0] == 1


def test_lillie_used_with_thin_no_attacker_hand():
    me = mk_me(active=mk_poke(C.DUNSPARCE), hand=[C.LILLIE, C.PSYCHIC_ENERGY])
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(me, opp, [Option(type=OptionType.PLAY, area=AreaType.HAND, index=0), Option(type=OptionType.END)])
    assert policy(obs).choose()[0] == 0


def test_hyper_aroma_selects_kadabra_then_dudunsparce():
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    options = [
        Option(type=OptionType.CARD, area=AreaType.LOOKING, index=0, playerIndex=0),
        Option(type=OptionType.CARD, area=AreaType.LOOKING, index=1, playerIndex=0),
    ]
    looking = [Card(id=C.DUDUNSPARCE), Card(id=C.KADABRA)]
    obs = mk_obs(
        mk_me(active=mk_poke(C.ABRA), hand=[]),
        opp,
        options,
        context=SelectContext.TO_HAND,
        looking=looking,
        context_card=Card(id=C.HYPER_AROMA),
    )
    assert policy(obs).choose()[0] == 1

    obs = mk_obs(
        mk_me(active=mk_poke(C.KADABRA), hand=[]),
        opp,
        options,
        context=SelectContext.TO_HAND,
        looking=looking,
        context_card=Card(id=C.HYPER_AROMA),
    )
    assert policy(obs).choose()[0] == 0


def test_agent_returns_legal_fallback_on_bad_dict():
    out = M.agent({"select": {"minCount": 1, "maxCount": 1, "option": [{"type": 14}]}})
    assert out == [0]
