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
import re
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


def _deck_override(path, fallback):
    if not path:
        return list(fallback)
    values = [int(value) for value in Path(path).read_text(encoding="utf-8-sig").split()]
    if len(values) != 60:
        raise ValueError(f"{path}: expected 60 card ids, got {len(values)}")
    return values


class GameRecorder:
    """Per-game aggregation of the primary agent's own MAIN decisions."""

    def __init__(self, module):
        self.M = module
        # Alakazam submissions keep the policy implementation in
        # ``fallback_policy.py`` and import that module from ``main.py``.
        # Older versions of this harness incorrectly looked for the policy
        # class on main.py itself, silently turning every reconstructed
        # decision into a recorder exception.
        self.P = getattr(module, "fallback_policy", module)
        self.reset()

    def reset(self):
        self.turns = {}                 # own_turn_index -> dict(attacked, dudun, retreat, alakazam)
        self._seen_turn_numbers = []     # engine turn numbers this seat has acted on
        self.min_deck = None
        self.terminal = {}
        self.zero_damage_attacks = 0
        self.attackable_ends = 0
        self.enriching_attachments = 0
        self.enriching_cycle_attachments = 0
        self.enriching_with_backup_fuel_available = 0
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
                                         "retreat": False, "alakazam_attack": False,
                                         "enriching": False})
        try:
            pol = self.P.AlakazamPolicy(obs)
        except Exception:
            pol = None
        t = opt.type
        if t == OptionType.ATTACK:
            # Selecting an attack ends the turn even when protection reduces
            # its damage to zero, so keep action frequency separate from
            # effectiveness.  Powerful Hand damage can be reconstructed with
            # the policy's stable cross-version helper.
            rec["attacked"] = True
            active = me.active[0] if me and me.active else None
            if active is not None and active.id == self.P.C.ALAKAZAM:
                rec["alakazam_attack"] = True
            dmg = 1
            if pol is not None and opt.attackId == self.P.POWERFUL_HAND:
                opponent = obs.current.players[1 - obs.current.yourIndex]
                target = opponent.active[0] if opponent and opponent.active else None
                dmg = pol._alakazam_damage(opt.attackId, target)
            if opt.attackId == self.P.POWERFUL_HAND and dmg <= 0:
                self.zero_damage_attacks += 1
        elif t == OptionType.ABILITY:
            card = pol and self.P.get_card(obs, opt.area, opt.index, obs.current.yourIndex)
            if card is not None and card.id == self.P.C.DUDUNSPARCE:
                rec["dudun"] = True
        elif t == OptionType.RETREAT:
            rec["retreat"] = True
        elif t in (OptionType.ATTACH, OptionType.ENERGY):
            source = self.P.get_card(obs, self.P.AreaType.HAND, opt.index, obs.current.yourIndex)
            if source is not None and source.id == self.P.C.ENRICHING_ENERGY:
                rec["enriching"] = True
                self.enriching_attachments += 1
                target = self.P.get_card(
                    obs, opt.inPlayArea, opt.inPlayIndex, obs.current.yourIndex
                )
                if target is not None and target.id in (self.P.C.DUNSPARCE,
                                                        self.P.C.DUDUNSPARCE):
                    self.enriching_cycle_attachments += 1
                if pol is not None:
                    psychic = next(
                        (card for card in (me.hand or [])
                         if self.P.ENERGY_PROVIDES.get(card.id) == self.P.EnergyType.PSYCHIC),
                        None,
                    )
                    active = me.active[0] if me.active else None
                    backup_needs_fuel = any(
                        pokemon is not None
                        and pokemon.id in self.P.ALAKAZAM_IDS
                        and pol._should_fuel(pokemon)
                        and pol._attach_helps(pokemon, psychic)
                        for pokemon in (me.bench or [])
                    )
                    if (psychic is not None and active is not None
                            and active.id in self.P.ALAKAZAM_IDS
                            and pol._can_attack(active) and backup_needs_fuel):
                        self.enriching_with_backup_fuel_available += 1
        elif t == OptionType.END:
            if pol is not None:
                attack_options = [candidate for candidate in opts if candidate.type == OptionType.ATTACK]
                if any(pol._score_attack(candidate) > 0 for candidate in attack_options):
                    self.attackable_ends += 1

    def observe_terminal(self, obs_dict, primary_seat):
        try:
            current = (obs_dict or {}).get("current") or {}
            players = current.get("players") or []
            if not 0 <= primary_seat < len(players):
                return
            me = players[primary_seat] or {}
            opponent = players[1 - primary_seat] or {}
            self.terminal = {
                "deck_count": int(me.get("deckCount") or 0),
                "prize_count": len(me.get("prize") or []),
                "board_count": sum(
                    card is not None
                    for card in (me.get("active") or []) + (me.get("bench") or [])
                ),
                "opponent_prize_count": len(opponent.get("prize") or []),
            }
        except Exception:
            self.exceptions += 1

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
        post_first_attack_idle_turns = 0
        if first_attack is not None:
            post_first_attack_idle_turns = sum(
                not self.turns[ti]["attacked"] for ti in self.turns if ti > first_attack
            )
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
            "post_first_attack_idle_turns": post_first_attack_idle_turns,
            "enriching_attachments": self.enriching_attachments,
            "enriching_cycle_attachments": self.enriching_cycle_attachments,
            "enriching_with_backup_fuel_available": self.enriching_with_backup_fuel_available,
            "enriching_same_turn_attack": sum(
                value["enriching"] and value["attacked"] for value in self.turns.values()
            ),
            "exceptions": self.exceptions,
            "attack_rate_by_turn": {ti: self.turns[ti]["attacked"] for ti in self.turns},
            "won": won,
            "lost": lost,
            "deckout_suspected": bool(lost and self.min_deck == 0),
            "terminal_deckout": bool(lost and self.terminal.get("deck_count") == 0),
            "terminal_boardout": bool(lost and self.terminal.get("board_count") == 0),
            "terminal_prize_loss": bool(
                lost and self.terminal.get("opponent_prize_count") == 0
            ),
        }


def play_game(agents, decks, primary_seat, recorder, stats, max_steps=8000):
    obs, start = battle_start(decks[0], decks[1])
    if obs is None:
        raise RuntimeError(f"battle_start failed (errorPlayer={start.errorPlayer})")
    try:
        for _ in range(max_steps):
            cur = obs["current"]
            if cur["result"] >= 0:
                recorder.observe_terminal(obs, primary_seat)
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
    rich_games = [g for g in per_game if g["enriching_attachments"] > 0]
    no_rich_games = [g for g in per_game if g["enriching_attachments"] == 0]
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
        "post_first_attack_idle_turns_per_game": mean("post_first_attack_idle_turns"),
        "enriching_attachments_per_game": mean("enriching_attachments"),
        "enriching_cycle_attachments_per_game": mean("enriching_cycle_attachments"),
        "enriching_same_turn_attack_rate": (
            sum(g["enriching_same_turn_attack"] for g in per_game)
            / max(1, sum(g["enriching_attachments"] for g in per_game))
        ),
        "enriching_with_backup_fuel_available_total": sum(
            g["enriching_with_backup_fuel_available"] for g in per_game
        ),
        "games_with_enriching": len(rich_games),
        "win_rate_with_enriching": (
            sum(g["won"] for g in rich_games) / len(rich_games) if rich_games else None
        ),
        "win_rate_without_enriching": (
            sum(g["won"] for g in no_rich_games) / len(no_rich_games) if no_rich_games else None
        ),
        "deckouts_with_enriching": sum(g["terminal_deckout"] for g in rich_games),
        "deckouts_without_enriching": sum(g["terminal_deckout"] for g in no_rich_games),
        "deckout_suspected_total": sum(1 for g in per_game if g["deckout_suspected"]),
        "terminal_deckout_total": sum(1 for g in per_game if g["terminal_deckout"]),
        "terminal_boardout_total": sum(1 for g in per_game if g["terminal_boardout"]),
        "terminal_prize_loss_total": sum(1 for g in per_game if g["terminal_prize_loss"]),
        "recorder_exceptions_total": sum(g["exceptions"] for g in per_game),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("primary")
    parser.add_argument("opponent")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="directory to write metrics.json")
    parser.add_argument("--primary-deck", default=None, help="optional 60-card deck override")
    parser.add_argument("--opponent-deck", default=None, help="optional 60-card deck override")
    args = parser.parse_args()

    random.seed(args.seed)
    primary_agent, primary_module, _ = _load(args.primary)
    primary_deck = _deck_override(args.primary_deck, primary_agent({"select": None}))
    opp_agent, opp_module, _ = _load_opponent(args.opponent, primary_deck)
    opp_deck = _deck_override(args.opponent_deck, opp_agent({"select": None}))

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
        "primary_deck_override": args.primary_deck,
        "opponent_deck_override": args.opponent_deck,
        "win_rate": agg.get("win_rate"),
        "first_player_win_rate": (first_wins / first_games) if first_games else None,
        "second_player_win_rate": (second_wins / second_games) if second_games else None,
        "behavior": agg,
        "crashes": sum(stats["crash"]),
        "illegal_selects": sum(stats["illegal"]),
        "diag_before": diag_before,
        "diag_after": diag_after,
    }
    runtime = getattr(primary_module, "_RUNTIME", None)
    if runtime is not None and callable(getattr(runtime, "snapshot", None)):
        metrics["ml_runtime"] = runtime.snapshot()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.out:
        out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        def safe_name(spec):
            leaf = Path(spec).name or spec
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", leaf)

        fname = f"metrics_{safe_name(args.primary)}_vs_{safe_name(args.opponent)}.json"
        (out_dir / fname).write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"\nwrote {out_dir / fname}")


if __name__ == "__main__":
    main()
