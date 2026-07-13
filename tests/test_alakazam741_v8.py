from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from scripts.agent_loader import load_dir_agent_module

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agents" / "alakazam741_v8"

for path in (ROOT / "vendor", ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _load_agent(name: str):
    return load_dir_agent_module(ROOT / "agents" / name)


M = _load_agent("alakazam741_v8")
V2 = _load_agent("alakazam741_v2")
V3 = _load_agent("alakazam741_v3")
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


def test_loader_uses_bundled_policy_base_even_after_other_agents_load():
    bundled = Path(sys.modules[M.BasePolicy.__module__].__file__).resolve()
    assert bundled == (AGENT_DIR / "policy_base.py").resolve()
    assert Path(sys.modules[V2.__name__].__file__).resolve().name == "main.py"
    assert Path(sys.modules[V3.__name__].__file__).resolve().name == "main.py"


def test_deck_is_60_cards():
    deck = [int(x) for x in (AGENT_DIR / "deck.csv").read_text(encoding="utf-8-sig").split()]
    assert len(deck) == 60


def test_metadata_name_and_version_are_v8():
    metadata = json.loads((AGENT_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "alakazam741_v8"
    assert metadata["version"] == "8.0.0"


def test_phase_tier_route_runs_through_score_main():
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), hand=[C.PSYCHIC_ENERGY] * 6)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND), Option(type=OptionType.END)],
    )
    pol = policy(obs)
    called = {}

    def fake_score_main(self, option, raw):
        called["option_type"] = option.type
        called["raw"] = raw
        return 424242

    pol._score_main = fake_score_main.__get__(pol, M.AlakazamPolicy)
    assert pol.score(obs.select.option[0]) == 424242
    assert called["option_type"] == OptionType.ATTACK
    assert isinstance(called["raw"], int)


def test_attackable_alakazam_does_not_end():
    me = mk_me(active=mk_poke(C.ALAKAZAM, energies=[5]), hand=[C.PSYCHIC_ENERGY] * 6)
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140))
    obs = mk_obs(
        me,
        opp,
        [Option(type=OptionType.ATTACK, attackId=M.POWERFUL_HAND), Option(type=OptionType.END)],
    )
    assert policy(obs).choose()[0] == 0


def test_attackable_alakazam_does_not_retreat():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        bench=[mk_poke(C.ALAKAZAM, energies=[5])],
        hand=[C.PSYCHIC_ENERGY] * 6,
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


def test_zero_damage_powerful_hand_is_not_used():
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
    assert policy(obs).choose()[0] == 1


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


def test_mirror_prefers_xerosic_over_non_winning_attack_when_opp_hand_is_six_plus():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        hand=[C.XEROSIC, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.ALAKAZAM, hp=140, energies=[5]), hand_count=6)
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


def test_non_mirror_prefers_meaningful_attack_over_xerosic():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        hand=[C.XEROSIC, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.DUDUNSPARCE, hp=140), hand_count=8)
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


def test_winning_ko_beats_xerosic_in_mirror():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        hand=[
            C.XEROSIC,
            C.PSYCHIC_ENERGY,
            C.PSYCHIC_ENERGY,
            C.PSYCHIC_ENERGY,
            C.PSYCHIC_ENERGY,
            C.PSYCHIC_ENERGY,
            C.PSYCHIC_ENERGY,
        ],
    )
    opp = mk_opp(mk_poke(C.ALAKAZAM, hp=140, energies=[5]), hand_count=6)
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


def test_xerosic_is_rejected_when_it_would_lose_current_ko():
    me = mk_me(
        active=mk_poke(C.ALAKAZAM, energies=[5]),
        hand=[C.XEROSIC, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY, C.PSYCHIC_ENERGY],
    )
    opp = mk_opp(mk_poke(C.ALAKAZAM, hp=100, energies=[5]), hand_count=6)
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


def test_hyper_aroma_multi_pick_uses_custom_selection():
    looking = [Card(id=C.KADABRA), Card(id=C.KADABRA), Card(id=C.DUDUNSPARCE)]
    options = [
        Option(type=OptionType.CARD, area=AreaType.LOOKING, index=0, playerIndex=0),
        Option(type=OptionType.CARD, area=AreaType.LOOKING, index=1, playerIndex=0),
        Option(type=OptionType.CARD, area=AreaType.LOOKING, index=2, playerIndex=0),
    ]
    select = Select(
        context=SelectContext.TO_HAND,
        minCount=3,
        maxCount=3,
        option=options,
        contextCard=Card(id=C.HYPER_AROMA),
    )
    state = State(
        turn=6,
        yourIndex=0,
        players=[
            mk_me(active=mk_poke(C.ABRA), hand=[]),
            mk_opp(mk_poke(C.DUDUNSPARCE, hp=140)),
        ],
        stadium=[],
        looking=looking,
    )
    pol = policy(Observation(select=select, current=state))

    def fail_rank():
        raise AssertionError("rank should not be called for Hyper Aroma set selection")

    pol.rank = fail_rank
    assert pol.choose() == [0, 2, 1]


def test_agent_returns_legal_fallback_on_bad_observation():
    assert M.agent({"select": {"minCount": 1, "maxCount": 1, "option": [{"type": 14}]}}) == [0]


def test_static_validation_reports_v8_metadata():
    from scripts.validate_agent import validate_agent

    result = validate_agent(AGENT_DIR)
    assert result["deck_size"] == 60
    assert result["metadata"]["name"] == "alakazam741_v8"
    assert result["metadata"]["version"] == "8.0.0"
    assert not result["warnings"]


def test_v8_deck_still_has_four_piece_core():
    deck = [int(x) for x in (AGENT_DIR / "deck.csv").read_text(encoding="utf-8-sig").split()]
    counts = Counter(deck)
    assert counts[C.ALAKAZAM] == 4
    assert counts[C.KADABRA] == 4
    assert counts[C.ABRA] == 4
