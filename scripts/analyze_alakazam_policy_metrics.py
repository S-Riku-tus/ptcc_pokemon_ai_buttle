"""Behavioral-metric harness for the Alakazam policies.

Runs balanced (alternating-seat) local games between a PRIMARY agent and an opponent, and — for
the primary agent's own MAIN decisions only — reconstructs the top-8 analysis metrics directly
from the decision stream: first-attack turn, per-own-turn attack rate, Dudunsparce cycling and
same-turn attack, retreats and same-turn attack, 0-damage Powerful Hands, attackable-but-END,
deckout losses, fallbacks and crashes.

The primary's own policy object is rebuilt on each of its decisions to reuse its exact damage /
state / attack-reservation logic (no metric re-implementation drift).

Usage:
  python scripts/analyze_alakazam_policy_metrics.py alakazam741_v9_top8_core alakazam741_v8 \
      --games 200 --seed 0 --out experiments/alakazam741_v9_top8_core
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "agents" / "_base"))
sys.path.insert(0, str(ROOT))

from scripts.agent_loader import diag_snapshot, load_dir_agent_module  # noqa: E402
from cg.api import OptionType, SelectContext, to_observation_class  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def _load(name_or_path):
    p = Path(name_or_path)
    if not p.is_dir():
        p = ROOT / "agents" / name_or_path
    module = load_dir_agent_module(p.resolve())
    diag = getattr(module, "_DIAG", None)
    return module.agent, module, diag


def _load_opponent(spec, fallback_deck):
    if spec in ("random", "first"):
        deck = fallback_deck

        def first_agent(obs):
            if obs["select"] is None:
                return list(deck)
            return list(range(min(obs["select"]["maxCount"], len(obs["select"]["option"]))))

        def random_agent(obs):
            if obs["select"] is None:
                return list(deck)
            sel = obs["select"]
            n = len(sel["option"])
            return random.sample(range(n), min(sel["maxCount"], n))

        return (random_agent if spec == "random" else first_agent), None, None
    if spec.startswith("generic:"):
        from generic_policy import make_generic_agent
        deck_dir = ROOT / "agents" / "_opponents" / spec.split(":", 1)[1]
        deck = [int(x) for x in (deck_dir / "deck.csv").read_text(encoding="utf-8-sig").split()]
        return make_generic_agent(deck), None, None
    return _load(spec)


class GameRecorder:
    """Per-game aggregation of the primary agent's own MAIN decisions."""

    def __init__(self, module):
        self.M = module
        self.reset()

    def reset(self):
        self.turns = {}                 # own_turn_index -> dict(attacked, dudun, retreat, alakazam)
        self._seen_turn_numbers = []     # engine turn numbers this seat has acted on
        self.min_deck = None
        self.zero_damage_attacks = 0
        self.attackable_ends = 0
        self.exceptions = 0

    def _own_turn_index(self, turn_number):
        if turn_number not in self._seen_turn_numbers:
            self._seen_turn_numbers.append(turn_number)
        return self._seen_turn_numbers.index(turn_number) + 1  # 1-based

    def observe(self, obs_dict, action):
        sel = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        if sel is None or not action:
            return
        try:
            obs = to_observation_class(obs_dict)
        except Exception:
            self.exceptions += 1
            return
        if obs.current is None or obs.select is None:
            return
        me = obs.current.players[obs.current.yourIndex]
        if me is not None:
            dc = getattr(me, "deckCount", None)
            if dc is not None:
                self.min_deck = dc if self.min_deck is None else min(self.min_deck, dc)
        if obs.select.context != SelectContext.MAIN:
            return
        opts = obs.select.option or []
        idx = action[0] if isinstance(action, (list, tuple)) and action else None
        if idx is None or not (0 <= idx < len(opts)):
            return
        opt = opts[idx]
        ti = self._own_turn_index(obs.current.turn)
        rec = self.turns.setdefault(ti, {"attacked": False, "dudun": False,
                                         "retreat": False, "alakazam_attack": False})
        try:
            pol = self.M.AlakazamPolicy(obs)
        except Exception:
            pol = None
        t = opt.type
        if t == OptionType.ATTACK:
            dmg = pol._attack_damage_for_option(opt) if pol is not None else 1
            if dmg > 0:
                rec["attacked"] = True
                active = me.active[0] if me and me.active else None
                if active is not None and active.id == self.M.C.ALAKAZAM:
                    rec["alakazam_attack"] = True
            if opt.attackId == self.M.POWERFUL_HAND and dmg <= 0:
                self.zero_damage_attacks += 1
        elif t == OptionType.ABILITY:
            card = pol and self.M.get_card(obs, opt.area, opt.index, obs.current.yourIndex)
            if card is not None and card.id == self.M.C.DUDUNSPARCE:
                rec["dudun"] = True
        elif t == OptionType.RETREAT:
            rec["retreat"] = True
        elif t == OptionType.END:
            if pol is not None and pol._attack_reserved and pol._has_meaningful_attack_option():
                self.attackable_ends += 1

    def summarize(self, won, lost):
        own_turns = len(self.turns)
        attack_turns = sum(1 for v in self.turns.values() if v["attacked"])
        alakazam_attacks = sum(1 for v in self.turns.values() if v["alakazam_attack"])
        first_attack = None
        for ti in sorted(self.turns):
            if self.turns[ti]["attacked"]:
                first_attack = ti
                break
        attacked_by_turn2 = any(self.turns[ti]["attacked"] for ti in self.turns if ti <= 2)
        dudun_turns = sum(1 for v in self.turns.values() if v["dudun"])
        dudun_then_attack = sum(1 for v in self.turns.values() if v["dudun"] and v["attacked"])
        retreat_turns = sum(1 for v in self.turns.values() if v["retreat"])
        retreat_then_attack = sum(1 for v in self.turns.values() if v["retreat"] and v["attacked"])
        return {
            "own_turns": own_turns,
            "attack_turns": attack_turns,
            "alakazam_attacks": alakazam_attacks,
            "first_attack_turn": first_attack,
            "attacked_by_turn2": attacked_by_turn2,
            "dudun_turns": dudun_turns,
            "dudun_then_attack": dudun_then_attack,
            "retreat_turns": retreat_turns,
            "retreat_then_attack": retreat_then_attack,
            "zero_damage_attacks": self.zero_damage_attacks,
            "attackable_ends": self.attackable_ends,
            "exceptions": self.exceptions,
            "attack_rate_by_turn": {ti: self.turns[ti]["attacked"] for ti in self.turns},
            "won": won,
            "lost": lost,
            "deckout_suspected": bool(lost and self.min_deck == 0),
        }


def play_game(agents, decks, primary_seat, recorder, stats, max_steps=8000):
    obs, start = battle_start(decks[0], decks[1])
    if obs is None:
        raise RuntimeError(f"battle_start failed (errorPlayer={start.errorPlayer})")
    try:
        for _ in range(max_steps):
            cur = obs["current"]
            if cur["result"] >= 0:
                return 0 if cur["result"] == 0 else 1 if cur["result"] == 1 else -1
            seat = cur["yourIndex"]
            try:
                action = agents[seat](obs)
            except Exception:
                stats["crash"][seat] += 1
                return 1 - seat
            if seat == primary_seat:
                try:
                    recorder.observe(obs, action)
                except Exception:
                    recorder.exceptions += 1
            try:
                obs = battle_select(list(action))
            except Exception:
                stats["illegal"][seat] += 1
                return 1 - seat
        return -1
    finally:
        battle_finish()


def aggregate(per_game):
    n = len(per_game)
    if n == 0:
        return {}

    def mean(key):
        return sum(g[key] for g in per_game) / n

    first_attacks = [g["first_attack_turn"] for g in per_game if g["first_attack_turn"]]
    total_own = sum(g["own_turns"] for g in per_game)
    total_attack_turns = sum(g["attack_turns"] for g in per_game)
    wins = sum(1 for g in per_game if g["won"])
    dudun_turns = sum(g["dudun_turns"] for g in per_game)
    dudun_then = sum(g["dudun_then_attack"] for g in per_game)
    retreat_turns = sum(g["retreat_turns"] for g in per_game)
    retreat_then = sum(g["retreat_then_attack"] for g in per_game)
    return {
        "games": n,
        "win_rate": wins / n,
        "avg_first_attack_turn": (sum(first_attacks) / len(first_attacks)) if first_attacks else None,
        "games_with_attack_rate": len(first_attacks) / n,
        "attack_by_2nd_own_turn_rate": sum(1 for g in per_game if g["attacked_by_turn2"]) / n,
        "overall_own_turn_attack_rate": (total_attack_turns / total_own) if total_own else 0.0,
        "attacks_per_game": mean("attack_turns"),
        "alakazam_attacks_per_game": mean("alakazam_attacks"),
        "dudun_abilities_per_game": mean("dudun_turns"),
        "dudun_same_turn_attack_rate": (dudun_then / dudun_turns) if dudun_turns else None,
        "retreats_per_game": mean("retreat_turns"),
        "retreat_same_turn_attack_rate": (retreat_then / retreat_turns) if retreat_turns else None,
        "attackable_ends_total": sum(g["attackable_ends"] for g in per_game),
        "zero_damage_attacks_total": sum(g["zero_damage_attacks"] for g in per_game),
        "deckout_suspected_total": sum(1 for g in per_game if g["deckout_suspected"]),
        "recorder_exceptions_total": sum(g["exceptions"] for g in per_game),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("primary")
    parser.add_argument("opponent")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="directory to write metrics.json")
    args = parser.parse_args()

    random.seed(args.seed)
    primary_agent, primary_module, _ = _load(args.primary)
    primary_deck = primary_agent({"select": None})
    opp_agent, opp_module, _ = _load_opponent(args.opponent, primary_deck)
    opp_deck = opp_agent({"select": None})

    diag_before = diag_snapshot(getattr(primary_module, "_DIAG", None))

    per_game = []
    first_wins = second_wins = first_games = second_games = 0
    stats = {"crash": [0, 0], "illegal": [0, 0]}
    recorder = GameRecorder(primary_module)

    for g in range(args.games):
        primary_first = (g % 2 == 0)
        agents = [primary_agent, opp_agent] if primary_first else [opp_agent, primary_agent]
        decks = [primary_deck, opp_deck] if primary_first else [opp_deck, primary_deck]
        primary_seat = 0 if primary_first else 1
        recorder.reset()
        result = play_game(agents, decks, primary_seat, recorder, stats)
        won = (result == primary_seat)
        lost = (result != -1 and result != primary_seat)
        if primary_first:
            first_games += 1
            first_wins += int(won)
        else:
            second_games += 1
            second_wins += int(won)
        per_game.append(recorder.summarize(won, lost))

    agg = aggregate(per_game)
    diag_after = diag_snapshot(getattr(primary_module, "_DIAG", None))
    metrics = {
        "primary": args.primary,
        "opponent": args.opponent,
        "games": args.games,
        "seed": args.seed,
        "win_rate": agg.get("win_rate"),
        "first_player_win_rate": (first_wins / first_games) if first_games else None,
        "second_player_win_rate": (second_wins / second_games) if second_games else None,
        "behavior": agg,
        "crashes": sum(stats["crash"]),
        "illegal_selects": sum(stats["illegal"]),
        "diag_before": diag_before,
        "diag_after": diag_after,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.out:
        out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"metrics_{args.primary}_vs_{args.opponent}.json"
        (out_dir / fname).write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"\nwrote {out_dir / fname}")


if __name__ == "__main__":
    main()
