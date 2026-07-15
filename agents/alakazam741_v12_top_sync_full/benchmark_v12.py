"""Alternating-seat v12/v11 behavioral comparison for the official local CG engine.

The native engine does not expose an RNG seed setter.  ``--seed`` therefore fixes Python-side
selection only; the report records this limitation instead of claiming identical shuffled games.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))

from cg.api import AreaType, CardType, OptionType, SelectContext, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from scripts.agent_loader import load_dir_agent_module  # noqa: E402

SEARCH_CARDS = {1086, 1152, 1225, 1231}
SEARCH_ABILITIES = {66, 140}
ALAKAZAM_LINE = {741, 742, 743}
ENGINE_LINE = {305, 66}
SHAYMIN = 343
FEZANDIPITI_EX = 140
BOSS_ORDERS = 1182
ENHANCED_HAMMER = 1081
POWERFUL_HAND = 1072


def _load(name: str):
    path = ROOT / "agents" / name
    module = load_dir_agent_module(path.resolve())
    agent = module.agent
    return agent, module, agent({"select": None})


def _blank_game():
    return {
        "turns": {},
        "seen_turns": [],
        "searches": 0,
        "attacks": 0,
        "alakazam_attacks": 0,
        "alakazam_attack_hands": [],
        "overkill": [],
        "boss_uses": 0,
        "boss_attack_turns": set(),
        "hammer_uses": 0,
        "hammer_next_attacks": 0,
        "fez_funded": False,
        "fez_attacked": False,
        "shaymin_deployed": False,
        "shaymin_prevented_bench_kos": 0,
        "attackable_ends": 0,
        "zero_damage_powerful_hands": 0,
        "min_board": None,
        "min_deck": None,
    }


class Recorder:
    def __init__(self, labels, modules):
        self.labels = labels
        self.modules = modules
        self.games = {label: _blank_game() for label in labels}
        self.hammer_waiting = {0: None, 1: None}

    def _own_turn(self, game, engine_turn):
        if engine_turn not in game["seen_turns"]:
            game["seen_turns"].append(engine_turn)
        return game["seen_turns"].index(engine_turn) + 1

    def observe(self, obs_dict, action):
        if not action or not obs_dict.get("select"):
            return
        obs = to_observation_class(obs_dict)
        if obs.current is None or obs.select is None:
            return
        seat = obs.current.yourIndex
        label = self.labels[seat]
        module = self.modules[seat]
        game = self.games[label]
        pending_turn = self.hammer_waiting[seat]
        if pending_turn is not None and pending_turn != obs.current.turn:
            self.hammer_waiting[seat] = None
        me = obs.current.players[seat]
        opponent = obs.current.players[1 - seat]
        board = [p for p in me.active + me.bench if p is not None]
        game["min_board"] = len(board) if game["min_board"] is None else min(
            game["min_board"], len(board))
        game["min_deck"] = me.deckCount if game["min_deck"] is None else min(
            game["min_deck"], me.deckCount)
        if obs.select.context != SelectContext.MAIN:
            return
        idx = action[0]
        options = obs.select.option or []
        if not 0 <= idx < len(options):
            return
        option = options[idx]
        policy = module.AlakazamPolicy(obs)
        own_turn = self._own_turn(game, obs.current.turn)
        turn_rec = game["turns"].setdefault(own_turn, {
            "attacked": False,
            "board": 0,
            "abra_bodies": 0,
            "alakazam": 0,
            "boss": False,
        })
        turn_rec.update({
            "board": len(board),
            "abra_bodies": sum(p.id in ALAKAZAM_LINE for p in board),
            "alakazam": sum(p.id == 743 for p in board),
        })

        if option.type == OptionType.PLAY:
            cid = policy._play_card_id(option)
            if cid in SEARCH_CARDS:
                game["searches"] += 1
            if cid == BOSS_ORDERS:
                game["boss_uses"] += 1
                turn_rec["boss"] = True
            elif cid == ENHANCED_HAMMER:
                game["hammer_uses"] += 1
                self.hammer_waiting[seat] = obs.current.turn
            elif cid == SHAYMIN:
                game["shaymin_deployed"] = True
        elif option.type == OptionType.ABILITY:
            card = module.get_card(obs, option.area, option.index, seat)
            if card is not None and card.id in SEARCH_ABILITIES:
                game["searches"] += 1
        elif option.type in (OptionType.ATTACH, OptionType.ENERGY):
            target = module.get_card(obs, option.inPlayArea, option.inPlayIndex, seat)
            if target is not None and target.id == FEZANDIPITI_EX:
                game["fez_funded"] = True
        elif option.type == OptionType.ATTACK:
            damage = policy._attack_damage_for_option(option)
            if damage > 0:
                game["attacks"] += 1
                turn_rec["attacked"] = True
                target = opponent.active[0] if opponent.active else None
                if target is not None:
                    game["overkill"].append(max(0, damage - target.hp))
                active = me.active[0] if me.active else None
                if active is not None and active.id == 743:
                    game["alakazam_attacks"] += 1
                    game["alakazam_attack_hands"].append(me.handCount)
                elif active is not None and active.id == FEZANDIPITI_EX:
                    game["fez_attacked"] = True
                if turn_rec["boss"]:
                    game["boss_attack_turns"].add(own_turn)
                owner = 1 - seat
                if self.hammer_waiting[owner] is not None:
                    self.games[self.labels[owner]]["hammer_next_attacks"] += 1
                    self.hammer_waiting[owner] = None
                self._record_shaymin_prevention(obs, option, seat, module)
            if option.attackId == POWERFUL_HAND and damage <= 0:
                game["zero_damage_powerful_hands"] += 1
        elif option.type == OptionType.END:
            if policy._attack_reserved and policy._has_meaningful_attack_option():
                game["attackable_ends"] += 1

    def _record_shaymin_prevention(self, obs, option, attacker_seat, module):
        defender_seat = 1 - attacker_seat
        defender_label = self.labels[defender_seat]
        defender_game = self.games[defender_label]
        defender = obs.current.players[defender_seat]
        defender_board = [p for p in defender.active + defender.bench if p is not None]
        if not defender_game["shaymin_deployed"] or not any(p.id == SHAYMIN for p in defender_board):
            return
        amount = module.AlakazamPolicy._bench_damage_amount(option.attackId)
        if amount <= 0:
            return
        prevented = 0
        for pokemon in defender.bench:
            if pokemon is None or pokemon.id not in ALAKAZAM_LINE | ENGINE_LINE:
                continue
            data = module.card_table.get(pokemon.id)
            rule_box = bool(data and (getattr(data, "ex", False)
                                      or getattr(data, "megaEx", False)))
            if not rule_box and pokemon.hp <= amount:
                prevented += 1
        defender_game["shaymin_prevented_bench_kos"] += prevented


def _play(labels, agents, modules, decks, max_steps=8000):
    recorder = Recorder(labels, modules)
    obs, start = battle_start(decks[0], decks[1])
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorPlayer={start.errorPlayer}")
    crash = illegal = 0
    try:
        for _ in range(max_steps):
            if obs["current"]["result"] >= 0:
                result = obs["current"]["result"]
                winner = result if result in (0, 1) else -1
                return winner, obs, recorder, crash, illegal
            seat = obs["current"]["yourIndex"]
            try:
                action = agents[seat](obs)
                recorder.observe(obs, action)
            except Exception:
                crash += 1
                return 1 - seat, obs, recorder, crash, illegal
            try:
                obs = battle_select(list(action))
            except Exception:
                illegal += 1
                return 1 - seat, obs, recorder, crash, illegal
        return -1, obs, recorder, crash, illegal
    finally:
        battle_finish()


def _finish_game(game, won, final_player):
    turns = game["turns"]
    first_attack = next((turn for turn in sorted(turns) if turns[turn]["attacked"]), None)
    game["first_attack_turn"] = first_attack
    game["own_turns"] = len(turns)
    game["won"] = won
    game["deckout_loss"] = bool(not won and getattr(final_player, "deckCount", 1) == 0)
    final_board = sum(p is not None for p in final_player.active + final_player.bench)
    game["board_wipe_loss"] = bool(not won and final_board == 0)
    game["t1_board"] = turns.get(1, {}).get("board", 0)
    game["t1_abra"] = turns.get(1, {}).get("abra_bodies", 0)
    game["t2_alakazam"] = turns.get(2, {}).get("alakazam", 0)
    game["boss_same_turn_attacks"] = len(game["boss_attack_turns"])
    game["attack_turns"] = sum(rec["attacked"] for rec in turns.values())
    return game


def _mean(values):
    return sum(values) / len(values) if values else None


def _aggregate(games):
    attack_count = sum(g["attacks"] for g in games)
    own_turns = sum(g["own_turns"] for g in games)
    search_count = sum(g["searches"] for g in games)
    boss_uses = sum(g["boss_uses"] for g in games)
    hammer_uses = sum(g["hammer_uses"] for g in games)
    funded = [g for g in games if g["fez_funded"]]
    first_attacks = [g["first_attack_turn"] for g in games if g["first_attack_turn"]]
    hands = [value for g in games for value in g["alakazam_attack_hands"]]
    overkill = [value for g in games for value in g["overkill"]]
    return {
        "games": len(games),
        "win_rate": _mean([int(g["won"]) for g in games]),
        "first_turn_abra_avg": _mean([g["t1_abra"] for g in games]),
        "first_turn_board_avg": _mean([g["t1_board"] for g in games]),
        "second_turn_alakazam_avg": _mean([g["t2_alakazam"] for g in games]),
        "avg_first_attack_own_turn": _mean(first_attacks),
        "games_reaching_attack_rate": len(first_attacks) / len(games),
        "all_own_turn_attack_rate": attack_count / own_turns if own_turns else 0,
        "attacks_per_game": attack_count / len(games),
        "alakazam_attacks_per_game": sum(g["alakazam_attacks"] for g in games) / len(games),
        "searches_per_attack": search_count / attack_count if attack_count else None,
        "alakazam_attack_avg_hand": _mean(hands),
        "avg_overkill": _mean(overkill),
        "deckout_loss_rate": sum(g["deckout_loss"] for g in games) / len(games),
        "board_wipe_loss_rate": sum(g["board_wipe_loss"] for g in games) / len(games),
        "boss_same_turn_attack_rate": (
            sum(g["boss_same_turn_attacks"] for g in games) / boss_uses if boss_uses else None),
        "boss_uses": boss_uses,
        "hammer_opponent_next_turn_attack_rate": (
            sum(g["hammer_next_attacks"] for g in games) / hammer_uses if hammer_uses else None),
        "hammer_uses": hammer_uses,
        "fez_funded_games": len(funded),
        "fez_funded_to_attack_rate": (
            sum(g["fez_attacked"] for g in funded) / len(funded) if funded else None),
        "shaymin_deployed_games": sum(g["shaymin_deployed"] for g in games),
        "shaymin_prevented_bench_kos": sum(
            g["shaymin_prevented_bench_kos"] for g in games),
        "attackable_ends": sum(g["attackable_ends"] for g in games),
        "zero_damage_powerful_hands": sum(g["zero_damage_powerful_hands"] for g in games),
    }


def run(games, seed):
    if games % 2:
        raise ValueError("--games must be even for exact seat swapping")
    random.seed(seed)
    names = ["alakazam741_v12_top_sync", "alakazam741_v11_board_depth"]
    results = defaultdict(list)
    crashes = illegal = 0
    for game_index in range(games):
        # PrizeTracker and per-turn scratch are intentionally process-persistent during one game.
        # Load fresh modules per game so no hidden state leaks into the next shuffled deck.
        loaded = {name: _load(name) for name in names}
        labels = names if game_index % 2 == 0 else list(reversed(names))
        agents = [loaded[label][0] for label in labels]
        modules = [loaded[label][1] for label in labels]
        decks = [loaded[label][2] for label in labels]
        winner, final_obs, recorder, game_crash, game_illegal = _play(
            labels, agents, modules, decks)
        crashes += game_crash
        illegal += game_illegal
        final = to_observation_class(final_obs)
        for seat, label in enumerate(labels):
            results[label].append(_finish_game(
                recorder.games[label], winner == seat, final.current.players[seat]))
    return {
        "engine": "official local cg API",
        "games": games,
        "seat_swapped": True,
        "python_seed": seed,
        "same_shuffle_seed": False,
        "seed_limitation": "The native battle API exposes no RNG seed setter.",
        "crashes": crashes,
        "illegal_selects": illegal,
        "metrics": {label: _aggregate(results[label]) for label in names},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=741)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args.games, args.seed)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
