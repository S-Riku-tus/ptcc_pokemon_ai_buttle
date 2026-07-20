from types import SimpleNamespace

from test_v11_runtime_logic import (
    AreaType,
    Card,
    EnergyType,
    Pokemon,
    SelectContext,
    bare_policy,
    load_policy,
)


def own_choice():
    return SimpleNamespace(playerIndex=0)


def expose_real_effect_logic(obj):
    # The inherited v11 helper replaces this method for older isolated tests.
    # v12 tests intentionally exercise the real scoped board-protection method.
    if "_effect_prevented" in obj.__dict__:
        del obj.__dict__["_effect_prevented"]


def rocket_board(policy, *, evolved=False):
    active = Pokemon(432, hp=110, maxHp=110, playerIndex=1)
    articuno = Pokemon(policy.ROCKET_ARTICUNO_ID, hp=120, maxHp=120, playerIndex=1)
    bench = [articuno]
    if evolved:
        bench.append(Pokemon(401, hp=130, maxHp=130, playerIndex=1))
    return active, bench


def test_v12_teleport_uses_low_value_pivots_and_never_exposes_the_line():
    policy = load_policy()
    opponent = Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    abra = Pokemon(policy.C.ABRA, hp=50, maxHp=50,
                   energies=[EnergyType.PSYCHIC])
    obj = bare_policy(policy, active=abra, opp_active=opponent)
    obj.context = SelectContext.SWITCH
    obj.select.effect = Card(policy.C.ABRA)

    dunsparce = Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70)
    dudunsparce = Pokemon(policy.C.DUDUNSPARCE, hp=140, maxHp=140)
    spare_abra = Pokemon(policy.C.ABRA, hp=50, maxHp=50)
    fez = Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210)
    kadabra = Pokemon(policy.C.KADABRA, hp=80, maxHp=80)
    alakazam = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140)

    scores = {
        card.id: obj._score_active_choice(own_choice(), card)
        for card in (dunsparce, dudunsparce, spare_abra, fez, kadabra, alakazam)
    }
    assert scores[policy.C.DUNSPARCE] > scores[policy.C.DUDUNSPARCE]
    assert scores[policy.C.DUDUNSPARCE] > scores[policy.C.ABRA]
    assert scores[policy.C.ABRA] > scores[policy.C.FEZANDIPITI_EX]
    assert scores[policy.C.FEZANDIPITI_EX] > scores[policy.C.KADABRA]
    assert scores[policy.C.KADABRA] > scores[policy.C.ALAKAZAM]


def test_v12_teleport_only_treats_fez_as_abra_value_when_it_is_likely_safe():
    policy = load_policy()
    powered_ex = Pokemon(666, hp=230, maxHp=230, playerIndex=1,
                         energies=[EnergyType.COLORLESS, EnergyType.COLORLESS])
    obj = bare_policy(policy, active=Pokemon(policy.C.ABRA), opp_active=powered_ex)
    fez = Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210)
    spare_abra = Pokemon(policy.C.ABRA, hp=50, maxHp=50)

    assert obj._opponent_can_ko_fez_next_turn(fez)
    assert obj._score_teleport_choice(spare_abra) > obj._score_teleport_choice(fez)


def test_v12_mandatory_single_teleport_target_remains_legal_even_if_alakazam():
    policy = load_policy()
    select = SimpleNamespace(minCount=1, maxCount=1, option=[object()])
    score = -20000
    assert policy.normalize_selection([0], [score], select) == [0]


def test_v12_articuno_scope_blocks_only_basic_team_rocket_pokemon():
    policy = load_policy()
    active, bench = rocket_board(policy, evolved=True)
    obj = bare_policy(policy, opp_active=active, opp_bench=bench)
    expose_real_effect_logic(obj)

    articuno, evolved_spidops = bench
    non_rocket_abra = Pokemon(policy.C.ABRA, hp=50, maxHp=50, playerIndex=1)
    assert obj._effect_prevented(active)
    assert obj._effect_prevented(articuno)
    assert not obj._effect_prevented(evolved_spidops)
    assert not obj._effect_prevented(non_rocket_abra)


def test_v12_boss_escapes_to_koable_unprotected_evolution_not_articuno():
    policy = load_policy()
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    active, bench = rocket_board(policy, evolved=True)
    obj = bare_policy(policy, hand_count=8, active=mine,
                      opp_active=active, opp_bench=bench)
    expose_real_effect_logic(obj)

    articuno, evolved_spidops = bench
    assert obj._boss_damage_after_spend(articuno) == 0
    assert obj._boss_damage_after_spend(evolved_spidops) == 140
    assert obj._boss_target_score(evolved_spidops) > 0
    assert obj._boss_target_score(articuno) < 0


def test_v12_all_protected_rocket_basics_unlock_gradual_fez_investment():
    policy = load_policy()
    active, bench = rocket_board(policy)
    fez = Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210)
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    obj = bare_policy(policy, active=mine, bench=[fez],
                      opp_active=active, opp_bench=bench)
    expose_real_effect_logic(obj)

    assert obj._articuno_breaker_required()
    assert obj._fez_mode(fez) == "ALTERNATE_ATTACKER"
    assert obj._fez_attach_score(fez, False) >= 16000

    fez.energies = [EnergyType.PSYCHIC]
    assert obj._fez_attach_score(fez, False) > 0
    fez.energies = [EnergyType.PSYCHIC, EnergyType.PSYCHIC]
    assert obj._fez_attach_score(fez, False) > 0


def test_v12_ready_fez_is_promoted_and_targets_articuno_first():
    policy = load_policy()
    active, bench = rocket_board(policy)
    other_basic = Pokemon(400, hp=50, maxHp=50, playerIndex=1)
    bench.append(other_basic)
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    fez = Pokemon(policy.C.FEZANDIPITI_EX, hp=210, maxHp=210,
                  energies=[EnergyType.PSYCHIC] * 3)
    obj = bare_policy(policy, active=mine, bench=[fez],
                      opp_active=active, opp_bench=bench)
    expose_real_effect_logic(obj)

    assert obj._direct_damage_ready(fez) == 100
    assert obj._score_retreat() >= 17000
    assert obj._score_active_choice(own_choice(), fez) > obj._score_active_choice(
        own_choice(), mine
    )
    articuno = bench[0]
    assert obj._fez_target_score(articuno) > obj._fez_target_score(other_basic)


def test_v12_powerful_hand_is_not_selected_into_articuno_protection():
    policy = load_policy()
    active, bench = rocket_board(policy)
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    obj = bare_policy(policy, hand_count=10, active=mine,
                      opp_active=active, opp_bench=bench)
    expose_real_effect_logic(obj)
    option = SimpleNamespace(attackId=policy.POWERFUL_HAND)

    assert obj._alakazam_damage(policy.POWERFUL_HAND, active) == 0
    assert obj._score_attack(option) < 0


def test_v12_dudunsparce_is_the_single_prize_breaker_when_fez_is_absent():
    policy = load_policy()
    active, bench = rocket_board(policy)
    mine = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                   energies=[EnergyType.PSYCHIC])
    dudunsparce = Pokemon(policy.C.DUDUNSPARCE, hp=140, maxHp=140,
                          energies=[EnergyType.PSYCHIC])
    obj = bare_policy(policy, active=mine, bench=[dudunsparce],
                      opp_active=active, opp_bench=bench)
    expose_real_effect_logic(obj)

    score = obj._articuno_breaker_attach_score(dudunsparce, False)
    assert score > obj._score_attach_target(mine, True)
    dudunsparce.energies = [EnergyType.PSYCHIC] * 3
    assert obj._direct_damage_ready(dudunsparce) == 90


def test_v12_invested_dunsparce_does_not_trade_back_to_blocked_alakazam():
    policy = load_policy()
    active, bench = rocket_board(policy)
    dunsparce = Pokemon(policy.C.DUNSPARCE, hp=70, maxHp=70,
                        energies=[EnergyType.PSYCHIC])
    blocked_alakazam = Pokemon(policy.C.ALAKAZAM, hp=140, maxHp=140,
                               energies=[EnergyType.PSYCHIC])
    obj = bare_policy(policy, active=dunsparce, bench=[blocked_alakazam],
                      opp_active=active, opp_bench=bench)
    expose_real_effect_logic(obj)

    option = SimpleNamespace(attackId=policy.DUNSPARCE_TRADE)
    assert obj._score_attack(option) < 0
