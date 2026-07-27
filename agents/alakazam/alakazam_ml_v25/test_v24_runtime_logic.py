from __future__ import annotations

import test_v11_runtime_logic as h
from test_v18_runtime_logic import _main_board, _option


def _attacking_alakazam(policy, *, serial=10):
    return h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=serial,
    )


def _attack(policy):
    return _option(h.OptionType.ATTACK, attack_id=policy.POWERFUL_HAND)


def test_v24_bosses_damaged_bench_copy_when_active_same_ex_is_not_koable():
    policy = h.load_policy()
    active_ex = h.Pokemon(
        666, hp=260, maxHp=260, playerIndex=1, serial=20
    )
    damaged_bench_copy = h.Pokemon(
        666, hp=200, maxHp=260, playerIndex=1, serial=21
    )
    boss = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=11,
        active=_attacking_alakazam(policy),
        opponent_active=active_ex,
        opponent_bench=[damaged_bench_copy],
        options=[_attack(policy), boss],
    )

    # Powerful Hand is 220 before Boss and 200 after spending it.
    assert obj._boss_prize_upgrade_targets() == [damaged_bench_copy]
    assert obj.choose() == [1]


def test_v24_keeps_pressure_on_two_hit_grimmsnarl_instead_of_early_munk_gust():
    policy = h.load_policy()
    grimmsnarl = h.Pokemon(
        policy.GRIMMSNARL_EX_ID,
        hp=420,
        maxHp=420,
        playerIndex=1,
        serial=30,
    )
    munkidori = h.Pokemon(
        112,
        hp=110,
        maxHp=110,
        playerIndex=1,
        serial=31,
        energies=[h.EnergyType.DARKNESS],
    )
    morgrem = h.Pokemon(
        647,
        hp=100,
        maxHp=100,
        playerIndex=1,
        serial=32,
        energies=[h.EnergyType.DARKNESS, h.EnergyType.DARKNESS],
    )
    boss = _option(h.OptionType.PLAY, index=0)
    attack = _attack(policy)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=18,
        active=_attacking_alakazam(policy),
        opponent_active=grimmsnarl,
        opponent_bench=[munkidori, morgrem],
        options=[boss, attack],
    )

    assert obj._active_offered_attack_damage(grimmsnarl) == 360
    assert obj._munk_gust_abandons_ex_pressure(munkidori)
    assert obj._boss_target_score(munkidori) < 0
    # Do not replace the blocked Munkidori route with another one-prizer.
    assert obj._boss_target_score(morgrem) < 0
    assert obj.choose() == [1]

    # Spending cards on setup must not reopen the rejected one-prize route
    # later in the same turn.
    obj.me.handCount = 8
    assert obj._grim_ex_two_hit_pressure()
    assert obj._boss_target_score(morgrem) < 0


def test_v24_munk_gust_remains_a_fallback_when_grimmsnarl_is_not_two_hit():
    policy = h.load_policy()
    # The compact test card table omits Trainer metadata; the production table
    # contains it and the normal (non-hard-gated) Boss path needs that metadata.
    policy.card_table[policy.C.BOSS_ORDERS] = h.CardData(
        policy.C.BOSS_ORDERS,
        cardType=h.CardType.SUPPORTER,
    )
    grimmsnarl = h.Pokemon(
        policy.GRIMMSNARL_EX_ID,
        hp=310,
        maxHp=420,
        playerIndex=1,
        serial=40,
    )
    munkidori = h.Pokemon(
        112,
        hp=90,
        maxHp=110,
        playerIndex=1,
        serial=41,
        energies=[h.EnergyType.DARKNESS],
    )
    boss = _option(h.OptionType.PLAY, index=0)
    obj = _main_board(
        policy,
        hand_ids=[policy.C.BOSS_ORDERS],
        hand_count=6,
        active=_attacking_alakazam(policy),
        opponent_active=grimmsnarl,
        opponent_bench=[munkidori],
        options=[_attack(policy), boss],
    )

    assert not obj._munk_gust_abandons_ex_pressure(munkidori)
    assert obj._boss_target_score(munkidori) > 0
    assert obj.choose() == [1]


def test_v24_active_dudunsparce_cycles_last_two_cards_into_alakazam_ko():
    policy = h.load_policy()
    dudunsparce = h.Pokemon(policy.C.DUDUNSPARCE, serial=50)
    alakazam = _attacking_alakazam(policy, serial=51)
    opponent = h.Pokemon(
        9000, hp=180, maxHp=180, playerIndex=1, serial=52
    )
    ability = _option(
        h.OptionType.ABILITY,
        area=h.AreaType.ACTIVE,
        index=0,
    )
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=7,
        active=dudunsparce,
        bench=[alakazam],
        opponent_active=opponent,
        options=[ability, _option(h.OptionType.END)],
    )
    obj.me.deckCount = 2

    route = obj._active_dudun_low_deck_cycle_route()
    assert route is not None
    assert route["ko"]
    assert route["damage"] == 180
    assert obj.choose() == [0]

    # The follow-up ACTIVATE prompt must not reapply the old flat cost=3 guard.
    obj.select.effect = h.Card(policy.C.DUDUNSPARCE)
    obj.select.context = h.SelectContext.ACTIVATE
    obj.context = h.SelectContext.ACTIVATE
    assert obj._activate_draw_ok()


def test_v25_active_dudunsparce_refills_an_empty_deck():
    policy = h.load_policy()
    ability = _option(
        h.OptionType.ABILITY,
        area=h.AreaType.ACTIVE,
        index=0,
    )
    obj = _main_board(
        policy,
        hand_ids=[],
        hand_count=9,
        active=h.Pokemon(policy.C.DUDUNSPARCE, serial=60),
        bench=[_attacking_alakazam(policy, serial=61)],
        opponent_active=h.Pokemon(
            9000, hp=100, maxHp=100, playerIndex=1, serial=62
        ),
        options=[ability, _option(h.OptionType.END)],
    )
    obj.me.deckCount = 0

    assert obj._active_dudun_low_deck_cycle_route() is None
    assert obj._dudun_cycle_post_deck(obj.me.active[0]) == 2
    assert obj._score_ability(ability) > 0
    assert obj.choose() == [0]


def test_v24_can_attach_third_energy_then_retreat_dudunsparce_for_ko():
    policy = h.load_policy()
    dudunsparce = h.Pokemon(
        policy.C.DUDUNSPARCE,
        energies=[h.EnergyType.PSYCHIC, h.EnergyType.PSYCHIC],
        serial=70,
    )
    attach = _option(
        h.OptionType.ATTACH,
        index=0,
        target_area=h.AreaType.ACTIVE,
        target_index=0,
    )
    obj = _main_board(
        policy,
        hand_ids=[policy.C.PSYCHIC_ENERGY],
        hand_count=8,
        active=dudunsparce,
        bench=[_attacking_alakazam(policy, serial=71)],
        opponent_active=h.Pokemon(
            9000, hp=140, maxHp=140, playerIndex=1, serial=72
        ),
        options=[attach, _option(h.OptionType.END)],
    )

    source = obj.me.hand[0]
    assert obj._dudun_retreat_attack_route(source) is not None
    assert obj.choose() == [0]

    dudunsparce.energies.append(h.EnergyType.PSYCHIC)
    obj.me.handCount = 7
    obj.state.energyAttached = True
    retreat = _option(h.OptionType.RETREAT)
    obj.select.option = [retreat, _option(h.OptionType.END)]

    assert obj._dudun_retreat_attack_route() is not None
    assert obj.choose() == [0]
