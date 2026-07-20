from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


AGENT = Path(__file__).resolve().parent
REPO = AGENT.parents[1]
for path in (REPO / "vendor", AGENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

for module_name in ("fallback_v12", "policy_base"):
    sys.modules.pop(module_name, None)

from cg.api import (  # noqa: E402
    AreaType,
    Card,
    EnergyType,
    Observation,
    Option,
    OptionType,
    Player,
    Pokemon,
    Select,
    SelectContext,
    State,
)
import fallback_v12 as policy  # noqa: E402


def card(cid: int, player: int = 0) -> Card:
    return Card(id=cid, serial=cid * 100, playerIndex=player)


def pokemon(cid: int, *, hp: int = 100, energies=(), player: int = 0) -> Pokemon:
    return Pokemon(
        id=cid,
        serial=cid * 100,
        playerIndex=player,
        hp=hp,
        maxHp=hp,
        energies=list(energies),
        energyCards=[],
        tools=[],
        preEvolution=[],
    )


def player(*, active, bench=(), hand=(), discard=(), deck_count=40, prize_count=6) -> Player:
    bench_cards = list(bench)
    while len(bench_cards) < 5:
        bench_cards.append(None)
    return Player(
        active=[active] if active is not None else [],
        bench=bench_cards,
        benchMax=5,
        hand=list(hand),
        handCount=len(hand),
        discard=list(discard),
        deckCount=deck_count,
        prize=[card(9000 + i) for i in range(prize_count)],
    )


def option(kind, *, index=0, area=AreaType.HAND, in_area=None, in_index=0,
           attack_id=None, player_index=0) -> Option:
    return Option(
        type=kind,
        index=index,
        area=area,
        inPlayArea=in_area,
        inPlayIndex=in_index,
        attackId=attack_id,
        playerIndex=player_index,
    )


def make_policy(mine: Player, opponent: Player, options=(), *, turn=3,
                context=SelectContext.MAIN, context_card=None, min_count=1,
                max_count=1) -> policy.AlakazamPolicy:
    policy._TURN_STATE.update({
        "turn": None,
        "boss_committed": False,
        "last_opp_prizes": None,
        "ko_last_opponent_turn": False,
    })
    state = State(
        yourIndex=0,
        players=[mine, opponent],
        turn=turn,
        firstPlayer=0,
        supporterPlayed=False,
        stadiumPlayed=False,
        energyAttached=False,
        retreated=False,
        stadium=[],
    )
    select = Select(
        context=context,
        option=list(options),
        minCount=min_count,
        maxCount=max_count,
        contextCard=context_card,
        deck=None,
    )
    result = policy.AlakazamPolicy(Observation(current=state, select=select, logs=[]))
    return result


def ordinary_opponent(*, discard=()) -> Player:
    return player(active=pokemon(24, hp=230, player=1), discard=discard)


def test_deck_matches_requested_sixty_cards():
    counts = Counter(policy.my_deck)
    assert len(policy.my_deck) == 60
    assert counts[policy.C.SHAYMIN] == 0
    assert counts[policy.C.BOSS_ORDERS] == 0
    assert counts[policy.C.ENRICHING_ENERGY] == 0
    assert counts[policy.C.DUNSPARCE] == 3
    assert counts[policy.C.GENESECT] == 1
    assert counts[policy.C.PSYCHIC_ENERGY] == 3
    assert counts[policy.C.LUCKY_HELMET] == 1
    assert counts[policy.C.MAX_ROD] == 1
    assert [cid for cid in policy.my_deck if policy._is_ace_spec(cid)] == [policy.C.MAX_ROD]


def test_psychic_attachment_that_enables_active_alakazam_beats_end():
    mine = player(
        active=pokemon(policy.C.ALAKAZAM, hp=140),
        bench=[pokemon(policy.C.DUNSPARCE, hp=70)],
        hand=[card(policy.C.PSYCHIC_ENERGY)],
    )
    attach = option(
        OptionType.ATTACH,
        index=0,
        in_area=AreaType.ACTIVE,
        in_index=0,
    )
    end = option(OptionType.END)
    current = make_policy(mine, ordinary_opponent(), [attach, end])
    assert current._attachment_enables_active_alakazam(attach)
    assert current.choose() == [0]


def test_attack_is_mandatory_once_active_alakazam_is_fuelled():
    mine = player(
        active=pokemon(policy.C.ALAKAZAM, hp=140, energies=[EnergyType.PSYCHIC]),
        bench=[pokemon(policy.C.DUNSPARCE, hp=70)],
        hand=[card(1081), card(1086), card(1152)],
    )
    attack = option(OptionType.ATTACK, attack_id=policy.POWERFUL_HAND)
    end = option(OptionType.END)
    current = make_policy(mine, ordinary_opponent(), [end, attack])
    assert current.choose() == [1]


def test_last_body_survival_benches_a_basic_before_an_ordinary_attack():
    mine = player(
        active=pokemon(policy.C.ABRA, hp=50, energies=[EnergyType.PSYCHIC]),
        hand=[card(policy.C.DUNSPARCE)],
    )
    bench = option(OptionType.PLAY, index=0)
    attack = option(OptionType.ATTACK, attack_id=policy.ABRA_TELEPORT)
    current = make_policy(mine, ordinary_opponent(), [attack, bench])
    assert current._survival_bench_needed()
    assert current.choose() == [1]


def test_fezandipiti_is_benched_from_turn_one_and_not_promoted_or_fuelled():
    mine = player(
        active=pokemon(policy.C.ABRA, hp=50),
        bench=[pokemon(policy.C.DUNSPARCE, hp=70)],
        hand=[card(policy.C.FEZANDIPITI_EX), card(policy.C.PSYCHIC_ENERGY)],
    )
    current = make_policy(mine, ordinary_opponent(), turn=1)
    assert current._fezandipiti_worthwhile()
    fez = pokemon(policy.C.FEZANDIPITI_EX, hp=210)
    abra = pokemon(policy.C.ABRA, hp=50)
    own_choice = option(OptionType.CARD, player_index=0)
    assert current._score_active_choice(own_choice, fez) < current._score_active_choice(own_choice, abra)

    mine_with_fez = player(
        active=pokemon(policy.C.ABRA, hp=50),
        bench=[fez],
        hand=[card(policy.C.PSYCHIC_ENERGY)],
    )
    attach = option(OptionType.ATTACH, index=0, in_area=AreaType.BENCH, in_index=0)
    assert make_policy(mine_with_fez, ordinary_opponent(), [attach])._score_attach(attach) < 0


def test_genesect_is_held_after_opposing_ace_has_been_seen():
    mine = player(
        active=pokemon(policy.C.ABRA, hp=50),
        bench=[pokemon(policy.C.DUNSPARCE, hp=70)],
        hand=[card(policy.C.GENESECT), card(policy.C.LUCKY_HELMET)],
    )
    assert make_policy(mine, ordinary_opponent())._genesect_worthwhile()
    no_helmet = player(
        active=pokemon(policy.C.ABRA, hp=50),
        bench=[pokemon(policy.C.DUNSPARCE, hp=70)],
        hand=[card(policy.C.GENESECT)],
    )
    assert not make_policy(no_helmet, ordinary_opponent())._genesect_worthwhile()
    used_ace = ordinary_opponent(discard=[card(policy.C.MAX_ROD, player=1)])
    assert not make_policy(mine, used_ace)._genesect_worthwhile()


def test_lucky_helmet_targets_genesect_before_active_alakazam():
    genesect = pokemon(policy.C.GENESECT, hp=110)
    alakazam = pokemon(policy.C.ALAKAZAM, hp=140, energies=[EnergyType.PSYCHIC])
    mine = player(active=alakazam, bench=[genesect], hand=[])
    current = make_policy(
        mine,
        ordinary_opponent(),
        context=SelectContext.ATTACH_TO,
        context_card=card(policy.C.LUCKY_HELMET),
    )
    assert current._score_attach_target(genesect, False) > current._score_attach_target(
        alakazam, True)


def test_search_is_blocked_when_every_role_target_is_out_of_deck():
    active_abra = pokemon(policy.C.ABRA, hp=50)
    visible = (
        [card(policy.C.ABRA) for _ in range(3)]
        + [card(policy.C.KADABRA) for _ in range(4)]
        + [card(policy.C.ALAKAZAM) for _ in range(4)]
        + [card(policy.C.DUNSPARCE) for _ in range(3)]
        + [card(policy.C.DUDUNSPARCE) for _ in range(3)]
        + [card(policy.C.GENESECT)]
    )
    mine = player(active=active_abra, hand=visible)
    current = make_policy(mine, ordinary_opponent(), turn=1)
    assert not current._search_card_has_goal(policy.C.POKE_PAD)


def test_max_rod_waits_for_critical_or_multi_card_recovery():
    mine = player(
        active=pokemon(policy.C.KADABRA, hp=80),
        bench=[pokemon(policy.C.DUNSPARCE, hp=70)],
        discard=[card(policy.C.ALAKAZAM), card(policy.C.PSYCHIC_ENERGY)],
    )
    assert make_policy(mine, ordinary_opponent())._max_rod_worthwhile()

    low_value = player(
        active=pokemon(policy.C.ABRA, hp=50),
        bench=[pokemon(policy.C.DUNSPARCE, hp=70)],
        discard=[card(policy.C.FEZANDIPITI_EX)],
    )
    assert not make_policy(low_value, ordinary_opponent(), turn=1)._max_rod_worthwhile()


def test_evolution_and_search_hand_deltas_include_card_effects():
    mine = player(
        active=pokemon(policy.C.ALAKAZAM, hp=140, energies=[EnergyType.PSYCHIC]),
        bench=[pokemon(policy.C.ABRA, hp=50), pokemon(policy.C.KADABRA, hp=80)],
        hand=[card(policy.C.KADABRA), card(policy.C.ALAKAZAM), card(policy.C.RARE_CANDY),
              card(policy.C.POKE_PAD), card(policy.C.DAWN)],
    )
    kadabra = option(OptionType.EVOLVE, index=0, in_area=AreaType.BENCH, in_index=0)
    alakazam = option(OptionType.EVOLVE, index=1, in_area=AreaType.BENCH, in_index=1)
    candy = option(OptionType.PLAY, index=2)
    pad = option(OptionType.PLAY, index=3)
    dawn = option(OptionType.PLAY, index=4)
    current = make_policy(mine, ordinary_opponent(), [kadabra, alakazam, candy, pad, dawn])
    assert current._hand_delta(kadabra) == 1
    assert current._hand_delta(alakazam) == 2
    assert current._hand_delta(candy) == 1
    assert current._hand_delta(pad) == 0
    assert current._hand_delta(dawn) == 2


def test_search_backup_bypass_requires_eta_one_result():
    mine = player(
        active=pokemon(policy.C.ALAKAZAM, hp=140, energies=[EnergyType.PSYCHIC]),
        bench=[pokemon(policy.C.ABRA, hp=50)],
        hand=[card(policy.C.RARE_CANDY), card(policy.C.PSYCHIC_ENERGY)],
    )
    current = make_policy(mine, ordinary_opponent())
    assert current._backup_eta() > 1
    assert current._search_secures_backup(policy.C.POKE_PAD)
    assert not current._search_secures_backup(policy.C.BUDDY_POFFIN)
    assert current._search_deck_cost(policy.C.POKE_PAD) == 1
    assert current._search_deck_cost(policy.C.DAWN) == 3


def test_fezandipiti_does_not_take_last_essential_bench_slot():
    mine = player(
        active=pokemon(policy.C.GENESECT, hp=110),
        bench=[pokemon(24, hp=100), pokemon(24, hp=100), pokemon(24, hp=100), pokemon(24, hp=100)],
        hand=[card(policy.C.FEZANDIPITI_EX)],
    )
    current = make_policy(mine, ordinary_opponent(), turn=1)
    assert current._open_bench_slots() == 1
    assert current._essential_bench_reserve() >= 1
    assert not current._fezandipiti_worthwhile()


def test_fezandipiti_is_not_proactively_fetched_by_telepath_energy():
    mine = player(
        active=pokemon(policy.C.ABRA, hp=50),
        bench=[pokemon(policy.C.DUNSPARCE, hp=70)],
    )
    options = [
        option(OptionType.CARD, area=AreaType.DECK, index=0),
        option(OptionType.CARD, area=AreaType.DECK, index=1),
    ]
    current = make_policy(
        mine,
        ordinary_opponent(),
        options,
        turn=1,
        context=SelectContext.TO_BENCH,
        context_card=card(policy.C.TELEPATH_ENERGY),
        min_count=1,
        max_count=1,
    )
    current.select.deck = [card(policy.C.FEZANDIPITI_EX), card(policy.C.ABRA)]
    assert current.custom_selection() == [1]


def test_fezandipiti_draw_stops_when_hand_attack_and_backup_are_already_secure():
    mine = player(
        active=pokemon(policy.C.ALAKAZAM, hp=140, energies=[EnergyType.PSYCHIC]),
        bench=[pokemon(policy.C.ALAKAZAM, hp=140, energies=[EnergyType.PSYCHIC]),
               pokemon(policy.C.FEZANDIPITI_EX, hp=210)],
        hand=[card(1081 + i) for i in range(15)],
        deck_count=30,
    )
    current = make_policy(mine, ordinary_opponent())
    policy._TURN_STATE["ko_last_opponent_turn"] = True
    assert not current._fez_draw_needed()


def test_post_ko_fezandipiti_clock_has_no_recursive_dependency():
    mine = player(
        active=pokemon(policy.C.ALAKAZAM, hp=140, energies=[EnergyType.PSYCHIC]),
        bench=[pokemon(policy.C.FEZANDIPITI_EX, hp=210)],
        hand=[card(1081), card(1082), card(1083), card(1084)],
        deck_count=30,
    )
    current = make_policy(mine, ordinary_opponent())
    policy._TURN_STATE["ko_last_opponent_turn"] = True
    assert current._turns_to_win() >= 1
    assert isinstance(current._fez_draw_needed(), bool)


def test_nighttime_mine_requires_immediate_active_tax_effect():
    mine = player(active=pokemon(policy.C.ALAKAZAM, hp=140, energies=[EnergyType.PSYCHIC]))
    current = make_policy(mine, ordinary_opponent())
    current._nighttime_mine_tax_stops_active = lambda: False
    assert not current._nighttime_mine_worthwhile()
    current._nighttime_mine_tax_stops_active = lambda: True
    assert current._nighttime_mine_worthwhile()


def test_survival_bench_prefers_abra_over_fezandipiti():
    mine = player(
        active=pokemon(policy.C.GENESECT, hp=110),
        hand=[card(policy.C.FEZANDIPITI_EX), card(policy.C.ABRA)],
    )
    fez_play = option(OptionType.PLAY, index=0)
    abra_play = option(OptionType.PLAY, index=1)
    current = make_policy(mine, ordinary_opponent(), [fez_play, abra_play], turn=1)
    assert current._survival_bench_needed()
    assert current.choose() == [1]


def test_optional_fez_spend_cannot_turn_two_hit_ko_into_three_hit_ko():
    mine = player(
        active=pokemon(policy.C.ALAKAZAM, hp=140, energies=[EnergyType.PSYCHIC]),
        bench=[pokemon(policy.C.DUNSPARCE, hp=70)],
        hand=[card(policy.C.FEZANDIPITI_EX), card(1081), card(1082), card(1083), card(1084)],
    )
    fez_play = option(OptionType.PLAY, index=0)
    attack = option(OptionType.ATTACK, attack_id=policy.POWERFUL_HAND)
    opponent = player(active=pokemon(24, hp=200, player=1))
    current = make_policy(mine, opponent, [fez_play, attack], turn=2)
    assert current._plan["damage"] == 100
    assert current._fezandipiti_worthwhile()
    assert not current._preserves_attack(fez_play)
    assert current.choose() == [1]


def test_achievable_hand_does_not_count_unusable_supporter_search():
    mine = player(
        active=pokemon(policy.C.ALAKAZAM, hp=140, energies=[EnergyType.PSYCHIC]),
        hand=[card(policy.C.DAWN), card(1081), card(1082)],
    )
    current = make_policy(mine, ordinary_opponent())
    current._search_card_has_goal = lambda cid: False
    assert current._achievable_hand() == mine.handCount
