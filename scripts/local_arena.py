"""Local benchmark arena using the bundled cabt engine (vendor/cg).

Runs head-to-head matches between two agents with alternating seats and
reports wins/losses/draws, error losses, and per-move timing.

Agent specs:
  <dir-name>            an agent under agents/<dir-name>/ with main.py + deck.csv
  random | first        the official baseline agents (fall back to the active
                        Alakazam deck unless overridden with --deck-a/--deck-b)

Examples:
  python scripts/local_arena.py alakazam_ml_v31 alakazam_ml_v30 --games 40
  python scripts/local_arena.py marnies_grimmsnarl_ex_v2 alakazam_ml_v11 --games 80
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))          # -> import cg.api / cg.game

from agent_loader import diag_snapshot, load_dir_agent as load_agent_dir  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def load_dir_agent(agent_dir: Path):
    agent, diag, _module = load_agent_dir(agent_dir)
    return agent, diag


def load_baseline(kind: str, deck: list[int]):
    def random_agent(obs):
        if obs["select"] is None:
            return list(deck)
        sel = obs["select"]
        n = len(sel["option"])
        return random.sample(range(n), min(sel["maxCount"], n))

    def first_agent(obs):
        if obs["select"] is None:
            return list(deck)
        return list(range(min(obs["select"]["maxCount"], len(obs["select"]["option"]))))

    return (random_agent if kind == "random" else first_agent), None


def load_deck_override(path: str | None, fallback: list[int]) -> list[int]:
    if not path:
        return list(fallback)
    values = [int(value) for value in Path(path).read_text(encoding="utf-8-sig").split()]
    if len(values) != 60:
        raise ValueError(f"{path}: expected 60 ids, got {len(values)}")
    return values


def resolve(spec: str, fallback_deck: list[int]):
    if spec in ("random", "first"):
        return load_baseline(spec, fallback_deck)

    direct = Path(spec)
    if direct.is_dir():
        return load_dir_agent(direct.resolve())

    for base in (ROOT / "agents", ROOT / "archive" / "agents"):
        candidate = base / spec
        if candidate.is_dir():
            return load_dir_agent(candidate)
        # Agents are grouped one level deep by main Pokemon
        # (agents/<pokemon>/<agent>); match bare names there too.
        if base.is_dir():
            for group in sorted(base.iterdir(), key=lambda p: p.name):
                nested = group / spec
                if group.is_dir() and nested.is_dir():
                    return load_dir_agent(nested)

    raise FileNotFoundError(spec)


def play_game(agents, decks, stats, logical_sides=None, max_steps=8000):
    """agents/decks are seat-ordered [p0, p1]. Returns 0, 1 (winner) or -1 (draw)."""
    obs, start_data = battle_start(decks[0], decks[1])
    if obs is None:
        raise RuntimeError(f"battle_start failed (errorPlayer={start_data.errorPlayer})")
    try:
        for _ in range(max_steps):
            cur = obs["current"]
            if cur["result"] >= 0:
                return 0 if cur["result"] == 0 else 1 if cur["result"] == 1 else -1
            seat = cur["yourIndex"]
            t0 = time.perf_counter()
            try:
                action = agents[seat](obs)
            except Exception:
                stats["agent_error"][seat] += 1
                return 1 - seat
            elapsed = time.perf_counter() - t0
            stats["time"][seat] += elapsed
            stats["moves"][seat] += 1
            if logical_sides is not None:
                side = logical_sides[seat]
                stats["agent_time"][side] += elapsed
                stats["agent_moves"][side] += 1
            try:
                obs = battle_select(list(action))
            except Exception:
                stats["illegal"][seat] += 1
                return 1 - seat
        return -1
    finally:
        battle_finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_a")
    parser.add_argument("agent_b")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="suppress per-game result lines")
    parser.add_argument("--diag-json", action="store_true", help="print complete agent diagnostics as JSON")
    parser.add_argument("--report", type=Path,
                        help="write a machine-readable arena summary")
    parser.add_argument("--deck-a", help="override A's engine deck with a 60-id CSV/text file")
    parser.add_argument("--deck-b", help="override B's engine deck with a 60-id CSV/text file")
    args = parser.parse_args()

    random.seed(args.seed)

    fallback_deck = [
        int(x) for x in
        (ROOT / "agents" / "alakazam" / "alakazam_ml_v31" / "deck.csv")
        .read_text(encoding="utf-8-sig").split()
    ]
    agent_a, diag_a = resolve(args.agent_a, fallback_deck)
    agent_b, diag_b = resolve(args.agent_b, fallback_deck)

    deck_a = load_deck_override(args.deck_a, agent_a({"select": None}))
    deck_b = load_deck_override(args.deck_b, agent_b({"select": None}))

    # Keyed by side, not by spec: running an agent against itself as a
    # seat-bias control collided both sides into one counter and printed
    # "A: 60 B: 60" over 60 games.
    wins = {"A": 0, "B": 0, "draw": 0}
    first_wins = {"A": 0, "B": 0}
    stats = {
        "agent_error": [0, 0],
        "illegal": [0, 0],
        "time": [0.0, 0.0],
        "moves": [0, 0],
        "agent_time": {"A": 0.0, "B": 0.0},
        "agent_moves": {"A": 0, "B": 0},
    }

    for g in range(args.games):
        a_first = (g % 2 == 0)
        agents = [agent_a, agent_b] if a_first else [agent_b, agent_a]
        decks = [deck_a, deck_b] if a_first else [deck_b, deck_a]
        logical_sides = ["A", "B"] if a_first else ["B", "A"]
        result = play_game(agents, decks, stats, logical_sides)
        if result == -1:
            wins["draw"] += 1
            label = "draw"
        else:
            winner_is_a = (result == 0) == a_first
            side = "A" if winner_is_a else "B"
            label = args.agent_a if winner_is_a else args.agent_b
            wins[side] += 1
            if result == 0:
                first_wins[side] += 1
        if not args.quiet:
            print(f"game {g + 1:>3}: seat0={'A' if a_first else 'B'} -> {label}")

    total = args.games
    played = total - wins["draw"]
    print("\n== RESULT ==")
    print(f"A {args.agent_a}: {wins['A']}  "
          f"B {args.agent_b}: {wins['B']}  draw: {wins['draw']}")
    if played:
        print(f"A win rate (excl. draws): {wins['A'] / played:.1%}")
    print(f"first-seat wins: A={first_wins['A']} B={first_wins['B']}")
    print(f"errors (crash): {stats['agent_error']}  illegal selects: {stats['illegal']}")
    for i, name in enumerate(("seat-pool-0", "seat-pool-1")):
        if stats["moves"][i]:
            print(f"{name}: {stats['moves'][i]} moves, "
                  f"avg {stats['time'][i] / stats['moves'][i] * 1000:.2f} ms/move")
    for side, spec in (("A", args.agent_a), ("B", args.agent_b)):
        if stats["agent_moves"][side]:
            print(
                f"agent {side} {spec}: {stats['agent_moves'][side]} moves, "
                f"avg {stats['agent_time'][side] / stats['agent_moves'][side] * 1000:.2f} ms/move"
            )
    for tag, diag in (("A", diag_a), ("B", diag_b)):
        snap = diag_snapshot(diag)
        if snap:
            print(
                f"diag {tag}: decisions={snap['decisions']} "
                f"policy_ok={snap['policy_ok']} "
                f"policy_fallback={snap['policy_fallback']} "
                f"obs_fallback={snap['obs_fallback']} "
                f"deck_returns={snap['deck_returns']} "
                f"errors={snap['errors']} "
                f"fallback_rate={snap['fallback_rate']:.1%}"
            )
        if args.diag_json and isinstance(diag, dict):
            print(f"diag-json {tag}: {json.dumps(diag, ensure_ascii=False, sort_keys=True)}")

    if args.report:
        report = {
            "agent_a": args.agent_a,
            "agent_b": args.agent_b,
            "games": args.games,
            "seed": args.seed,
            "wins": wins,
            "win_rate_a_excluding_draws": (
                wins["A"] / played if played else None
            ),
            "first_wins": first_wins,
            "agent_error_by_seat": stats["agent_error"],
            "illegal_by_seat": stats["illegal"],
            "seat_moves": stats["moves"],
            "seat_avg_ms": [
                (
                    stats["time"][index] / stats["moves"][index] * 1000
                    if stats["moves"][index] else None
                )
                for index in range(2)
            ],
            "agent_moves": stats["agent_moves"],
            "agent_avg_ms": {
                side: (
                    stats["agent_time"][side]
                    / stats["agent_moves"][side] * 1000
                    if stats["agent_moves"][side] else None
                )
                for side in ("A", "B")
            },
            "diagnostics": {
                "A": diag_snapshot(diag_a),
                "B": diag_snapshot(diag_b),
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
