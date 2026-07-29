from __future__ import annotations

import ast
from types import SimpleNamespace as NS

import test_v11_runtime_logic as h
from test_v13_runtime_logic import _real_effect_method


def _main(obj, options=()):
    obj.context = h.SelectContext.MAIN
    obj.select = NS(
        context=h.SelectContext.MAIN,
        contextCard=None,
        effect=None,
        option=list(options),
        minCount=1,
        maxCount=1,
    )
    obj.state.energyAttached = getattr(obj.state, "energyAttached", False)
    obj.state.retreated = getattr(obj.state, "retreated", False)
    return obj


def _memory_board(policy, *, head):
    mine = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=25,
    )
    phantump = h.Pokemon(878, hp=70, maxHp=70, playerIndex=1, serial=77)
    attack = NS(type=h.OptionType.ATTACK, attackId=policy.POWERFUL_HAND)
    end = NS(type=h.OptionType.END)
    obj = _main(
        h.bare_policy(policy, active=mine, opp_active=phantump),
        [attack, end],
    )
    _real_effect_method(policy, obj)
    obj.state.turn = 16
    obj.state.yourIndex = 0
    obj.state.players = [obj.me, obj.opponent]
    obs = NS(
        current=obj.state,
        logs=[
            NS(type=15, playerIndex=1, attackId=1266, cardId=878, serial=77),
            NS(type=22, playerIndex=1, head=head),
        ],
    )
    return obj, phantump, attack, end, obs


def test_v31_deck_is_preserved_and_ensemble_provenance_is_frozen():
    import hashlib
    import json

    here = h.Path(__file__).resolve().parent
    assert hashlib.sha256((here / "deck.csv").read_bytes()).hexdigest() == (
        "57c7d4800cfc0f36581077a40b24912d33056cafcc14cca3783094ce6c122bfe"
    )
    assert hashlib.sha256((here / "legacy_ranker_model.json").read_bytes()).hexdigest() == (
        "22f41bfa04b4224c566d74d2642f4d8703fa36448dd815cc9b45c61c759e0bbb"
    )
    v29 = json.loads(
        (here / "v29_ranker_model.json").read_text(encoding="utf-8")
    )
    assert v29["runtime_scope"] == "v29_residual_main_policy"
    assert v29["training_decisions"] == 18336
    assert len(v29["feature_names"]) == 422

    model = json.loads(
        (here / "ranker_model.json").read_text(encoding="utf-8")
    )
    numeric = json.loads(
        (here / "ranker_numeric_model.json").read_text(encoding="utf-8")
    )
    assert model["runtime_scope"] == (
        "v31_majkel_two_ranker_ensemble"
    )
    assert model["teacher_trajectories"] == 999
    assert numeric["runtime_scope"] == model["runtime_scope"]
    assert numeric["ensemble_weight"] == 1.3
    assert model["baseline"] == (
        "v29_runtime_choice_and_raw_ranker_score"
    )
    assert len(model["feature_names"]) > 600


def test_v22_preserves_unmodified_v20_deck_safety_heads():
    import hashlib

    here = h.Path(__file__).resolve().parent

    tree = ast.parse((here / "fallback_policy.py").read_text(encoding="utf-8"))
    methods = {
        node.name: hashlib.sha256(
            ast.dump(node, include_attributes=False).encode("utf-8")
        ).hexdigest()
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    expected = {
        "_deck_preserve": "b7e1e2462b5ca5fc595b837e0b546a736e9c26d330758e8797d8cd9f3c5ae0b7",
        "_deck_spend_ok": "e011528b88134bcd2f28d90a61d4b40c5da67c2fc057fd6607e5397e7489572f",
        "_score_to_bench": "07c1e57f18d6c3328dd336683f3ded09f728c46f775251abb361086d0e0b46b0",
    }
    assert {name: methods[name] for name in expected} == expected

    policy = h.load_policy()
    assert not hasattr(policy.AlakazamPolicy, "_grimmsnarl_bench_allowed")
    assert not hasattr(policy.AlakazamPolicy, "_ko_and_continuity_locked")
    assert not hasattr(policy.AlakazamPolicy, "_abra_switch_score")


def test_v17_cubchoo_lock_retreats_to_ready_alakazam():
    policy = h.load_policy()
    policy.diag_reset()
    active = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=25,
    )
    backup = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=27,
    )
    cubchoo = h.Pokemon(
        506,
        energies=[h.EnergyType.COLORLESS],
        playerIndex=1,
        serial=77,
    )
    retreat = NS(type=h.OptionType.RETREAT)
    end = NS(type=h.OptionType.END)
    obj = _main(
        h.bare_policy(policy, active=active, bench=[backup], opp_active=cubchoo),
        [retreat, end],
    )
    obj.state.turn = 5
    obj.state.yourIndex = 0
    obj.state.players = [obj.me, obj.opponent]
    obs = NS(
        current=obj.state,
        logs=[NS(type=15, playerIndex=1, attackId=716)],
    )

    policy._remember_attack_disable(obs)

    assert obj._attack_disabled_on_active()
    assert obj._score_retreat() == 45000
    assert obj.choose() == [0]


def test_v17_attack_lock_memory_clears_after_active_changes():
    policy = h.load_policy()
    policy.diag_reset()
    first = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=25,
    )
    second = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=27,
    )
    obj = _main(
        h.bare_policy(
            policy,
            active=first,
            bench=[second],
            opp_active=h.Pokemon(506, playerIndex=1),
        )
    )
    obj.state.turn = 7
    obj.state.yourIndex = 0
    obj.state.players = [obj.me, obj.opponent]
    policy._V9_STATE.update(
        {
            "attack_disabled_key": (25, policy.C.ALAKAZAM),
            "attack_disabled_turn": 7,
        }
    )
    obj.me.active = [second]

    policy._remember_attack_disable(NS(current=obj.state, logs=[]))

    assert policy._V9_STATE["attack_disabled_key"] is None


def test_v17_splashing_dodge_head_blocks_zero_progress_attack():
    policy = h.load_policy()
    policy.diag_reset()
    assert 1266 in policy.TEMPORARY_ATTACK_IMMUNITY_ATTACKS
    obj, phantump, attack, _, obs = _memory_board(policy, head=True)

    policy._remember_temporary_attack_immunity(obs)

    assert obj._temporary_attack_immunity_applies(phantump)
    assert obj._effect_prevented(phantump)
    assert obj._alakazam_damage(policy.POWERFUL_HAND, phantump) == 0
    assert obj._score_attack(attack) < 0
    assert obj.choose() == [1]


def test_v17_splashing_dodge_tail_allows_normal_attack():
    policy = h.load_policy()
    policy.diag_reset()
    obj, phantump, attack, _, obs = _memory_board(policy, head=False)

    policy._remember_temporary_attack_immunity(obs)

    assert not obj._temporary_attack_immunity_applies(phantump)
    assert not obj._effect_prevented(phantump)
    assert obj._alakazam_damage(policy.POWERFUL_HAND, phantump) > 0
    assert obj._score_attack(attack) > 0


def test_v17_temporary_immunity_clears_after_target_leaves_active():
    policy = h.load_policy()
    policy.diag_reset()
    obj, phantump, _, _, obs = _memory_board(policy, head=True)
    policy._remember_temporary_attack_immunity(obs)
    assert obj._temporary_attack_immunity_applies(phantump)

    replacement = h.Pokemon(879, hp=140, maxHp=140, playerIndex=1, serial=88)
    obj.opponent.active = [replacement]
    obj.opponent.bench = [phantump, None, None, None, None]
    policy._remember_temporary_attack_immunity(NS(current=obj.state, logs=[]))

    assert policy._V9_STATE["temporary_immunity_key"] is None
    assert not obj._temporary_attack_immunity_applies(phantump)


def test_v17_ko_promotion_memory_survives_into_following_main_turn():
    policy = h.load_policy()
    policy.diag_reset()
    obj, phantump, _, _, obs = _memory_board(policy, head=True)
    obs.select = NS(context=h.SelectContext.TO_ACTIVE)
    policy._remember_temporary_attack_immunity(obs)
    assert policy._V9_STATE["temporary_immunity_turn"] == 17

    obj.state.turn = 17
    obj.select.context = h.SelectContext.MAIN
    policy._remember_temporary_attack_immunity(NS(current=obj.state, logs=[], select=obj.select))
    assert obj._temporary_attack_immunity_applies(phantump)

    obj.state.turn = 18
    policy._remember_temporary_attack_immunity(NS(current=obj.state, logs=[], select=obj.select))
    assert not obj._temporary_attack_immunity_applies(phantump)


def test_v17_temporary_immunity_active_enables_boss_ko_escape():
    policy = h.load_policy()
    policy.diag_reset()
    obj, _, _, _, obs = _memory_board(policy, head=True)
    target = h.Pokemon(9000, hp=100, maxHp=100, playerIndex=1, serial=91)
    obj.opponent.bench = [target, None, None, None, None]
    policy._remember_temporary_attack_immunity(obs)

    assert obj._boss_effect_lock_escape_ko(target)
    assert obj._boss_target_score(target) > 0


def test_v17_attach_retreat_attack_route_includes_active_alakazam():
    policy = h.load_policy()
    policy.diag_reset()
    active = h.Pokemon(policy.C.ALAKAZAM, energies=[], serial=25)
    backup = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=27,
    )
    target = h.Pokemon(9000, hp=100, maxHp=100, playerIndex=1)
    obj = _main(
        h.bare_policy(
            policy,
            hand_count=6,
            active=active,
            bench=[backup],
            opp_active=target,
        )
    )
    obj.state.turn = 7
    obj.state.yourIndex = 0
    obj.state.players = [obj.me, obj.opponent]
    policy._V9_STATE.update(
        {"attack_disabled_key": (25, policy.C.ALAKAZAM), "attack_disabled_turn": 7}
    )
    policy.card_table[policy.C.ALAKAZAM].retreatCost = 1
    source = h.Card(policy.C.PSYCHIC_ENERGY)

    route = obj._best_pivot_attack_route(assume_attach=True, source=source)
    assert route is not None
    assert route["attacker"] is backup
    assert route["ko"]
    assert obj._pivot_attach_score(active, h.AreaType.ACTIVE, source) > 28000

    active.energies = [h.EnergyType.PSYCHIC]
    obj.state.energyAttached = True
    obj.select.option = [NS(type=h.OptionType.RETREAT), NS(type=h.OptionType.END)]
    assert obj._score_retreat() >= 45000


def test_v17_completed_fez_pivots_out_of_articuno_lock():
    policy = h.load_policy()
    policy.diag_reset()
    active = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=25,
    )
    fez = h.Pokemon(
        policy.C.FEZANDIPITI_EX,
        hp=210,
        maxHp=210,
        energies=[h.EnergyType.COLORLESS] * 3,
        serial=27,
    )
    backup = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=28,
    )
    rocket_basic = h.Pokemon(400, hp=60, maxHp=60, playerIndex=1)
    articuno = h.Pokemon(
        policy.ROCKET_ARTICUNO_ID,
        hp=120,
        maxHp=120,
        playerIndex=1,
    )
    obj = _main(
        h.bare_policy(
            policy,
            active=active,
            bench=[fez, backup],
            opp_active=rocket_basic,
            opp_bench=[articuno],
        )
    )
    _real_effect_method(policy, obj)

    route = obj._best_pivot_attack_route()
    assert obj._articuno_breaker_mode()
    assert route is not None and route["attacker"] is fez
    assert obj._score_retreat() > 0

    obj.context = h.SelectContext.ACTIVE
    obj.state.retreated = True
    assert obj._score_active_choice(NS(playerIndex=0), fez) > obj._score_active_choice(
        NS(playerIndex=0), backup
    )


def test_v17_incomplete_fez_is_not_promoted_into_articuno_lock():
    policy = h.load_policy()
    policy.diag_reset()
    active = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=25,
    )
    fez = h.Pokemon(
        policy.C.FEZANDIPITI_EX,
        energies=[h.EnergyType.COLORLESS] * 2,
        serial=27,
    )
    backup = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
        serial=28,
    )
    rocket_basic = h.Pokemon(400, hp=60, maxHp=60, playerIndex=1)
    articuno = h.Pokemon(policy.ROCKET_ARTICUNO_ID, playerIndex=1)
    obj = _main(
        h.bare_policy(
            policy,
            active=active,
            bench=[fez, backup],
            opp_active=rocket_basic,
            opp_bench=[articuno],
        )
    )
    _real_effect_method(policy, obj)
    obj.context = h.SelectContext.ACTIVE
    obj.state.retreated = True

    assert obj._candidate_attack_route(fez) is None
    assert obj._score_active_choice(NS(playerIndex=0), fez) < obj._score_active_choice(
        NS(playerIndex=0), backup
    )


def test_v17_articuno_breaker_persists_while_spidops_is_boss_escape():
    policy = h.load_policy()
    policy.diag_reset()
    active = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    fez = h.Pokemon(
        policy.C.FEZANDIPITI_EX,
        energies=[h.EnergyType.COLORLESS],
    )
    rocket_basic = h.Pokemon(400, hp=60, maxHp=60, playerIndex=1)
    articuno = h.Pokemon(
        policy.ROCKET_ARTICUNO_ID,
        hp=120,
        maxHp=120,
        playerIndex=1,
    )
    obj = _main(
        h.bare_policy(
            policy,
            hand_count=9,
            active=active,
            bench=[fez],
            opp_active=rocket_basic,
            opp_bench=[articuno],
        )
    )
    _real_effect_method(policy, obj)
    assert obj._articuno_breaker_mode()

    spidops = h.Pokemon(401, hp=140, maxHp=140, playerIndex=1)
    obj.opponent.bench = [articuno, spidops, None, None, None]
    assert not obj._articuno_breaker_required()
    assert obj._articuno_breaker_mode()
    assert obj._rocket_evolution_escape_target(spidops)
    assert obj._boss_target_score(spidops) > 0
    assert obj._fez_attach_score(fez, False, h.Card(policy.C.PSYCHIC_ENERGY)) > 0


def test_v17_two_articuno_board_concentrates_energy_on_fez():
    policy = h.load_policy()
    policy.diag_reset()
    active = h.Pokemon(
        policy.C.ALAKAZAM,
        energies=[h.EnergyType.PSYCHIC],
    )
    fez = h.Pokemon(policy.C.FEZANDIPITI_EX, energies=[])
    abra = h.Pokemon(policy.C.ABRA)
    rocket_basic = h.Pokemon(400, playerIndex=1)
    articuno_1 = h.Pokemon(policy.ROCKET_ARTICUNO_ID, playerIndex=1, serial=80)
    articuno_2 = h.Pokemon(policy.ROCKET_ARTICUNO_ID, playerIndex=1, serial=81)
    obj = _main(
        h.bare_policy(
            policy,
            active=active,
            bench=[fez, abra],
            opp_active=rocket_basic,
            opp_bench=[articuno_1, articuno_2],
        )
    )
    _real_effect_method(policy, obj)
    obj.select.effect = h.Card(policy.C.PSYCHIC_ENERGY)

    assert obj._articuno_breaker_mode()
    assert obj._score_attach_target(abra, is_active=False) < 0
    assert obj._score_attach_target(fez, is_active=False) > 0
