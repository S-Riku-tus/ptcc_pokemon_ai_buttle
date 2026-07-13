"""Local benchmark arena using the bundled cabt engine (vendor/cg).

Runs head-to-head matches between two agents with alternating seats and
reports wins/losses/draws, error losses, and per-move timing.

Agent specs:
  <dir-name>            an agent under agents/<dir-name>/ with main.py + deck.csv
  generic:<dir-name>    a deck.csv under agents/_opponents/<dir-name>/ piloted by
                        the shared GenericPolicy (fair, non-crashing opponent)
  random | first        the official baseline agents (need a 60-card deck via
                        --random-deck; defaults to the active Alakazam deck)

Examples:
  python scripts/local_arena.py alakazam741_v2 alakazam741_v1 --games 40
  python scripts/local_arena.py alakazam741_v2 generic:grimmsnarl --games 80
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))          # -> import cg.api / cg.game
sys.path.insert(0, str(ROOT / "agents" / "_base"))  # -> shared generic_policy imports

from agent_loader import diag_snapshot, load_dir_agent as load_agent_dir  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def load_dir_agent(agent_dir: Path):
    agent, diag, _module = load_agent_dir(agent_dir)
    return agent, diag


def load_generic_agent(deck_dir: Path):
    from generic_policy import make_generic_agent
    deck = [int(x) for x in (deck_dir / "deck.csv").read_text(encoding="utf-8-sig").split()]
    if len(deck) != 60:
        raise ValueError(f"{deck_dir}: expected 60 ids, got {len(deck)}")
    return make_generic_agent(deck), None


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


def resolve(spec: str, fallback_deck: list[int]):
    if spec in ("random", "first"):
        return load_baseline(spec, fallback_deck)
    if spec.startswith("generic:"):
        return load_generic_agent(ROOT / "agents" / "_opponents" / spec.split(":", 1)[1])

    direct = Path(spec)
    if direct.is_dir():
        return load_dir_agent(direct.resolve())

    for base in (ROOT / "agents", ROOT / "archive" / "agents"):
        candidate = base / spec
        if candidate.is_dir():
            return load_dir_agent(candidate)

    raise FileNotFoundError(spec)


def play_game(agents, decks, stats, max_steps=8000):
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
            stats["time"][seat] += time.perf_counter() - t0
            stats["moves"][seat] += 1
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
    args = parser.parse_args()

    random.seed(args.seed)

    fallback_deck = [
        int(x) for x in
        (ROOT / "agents" / "alakazam741_v2" / "deck.csv")
        .read_text(encoding="utf-8-sig").split()
    ]
    agent_a, diag_a = resolve(args.agent_a, fallback_deck)
    agent_b, diag_b = resolve(args.agent_b, fallback_deck)

    deck_a = agent_a({"select": None})
    deck_b = agent_b({"select": None})

    wins = {args.agent_a: 0, args.agent_b: 0, "draw": 0}
    first_wins = {args.agent_a: 0, args.agent_b: 0}
    stats = {"agent_error": [0, 0], "illegal": [0, 0], "time": [0.0, 0.0], "moves": [0, 0]}

    for g in range(args.games):
        a_first = (g % 2 == 0)
        agents = [agent_a, agent_b] if a_first else [agent_b, agent_a]
        decks = [deck_a, deck_b] if a_first else [deck_b, deck_a]
        result = play_game(agents, decks, stats)
        if result == -1:
            wins["draw"] += 1
            label = "draw"
        else:
            winner_is_a = (result == 0) == a_first
            label = args.agent_a if winner_is_a else args.agent_b
            wins[label] += 1
            if result == 0:
                first_wins[label] += 1
        print(f"game {g + 1:>3}: seat0={'A' if a_first else 'B'} -> {label}")

    total = args.games
    played = total - wins["draw"]
    print("\n== RESULT ==")
    print(f"A {args.agent_a}: {wins[args.agent_a]}  "
          f"B {args.agent_b}: {wins[args.agent_b]}  draw: {wins['draw']}")
    if played:
        print(f"A win rate (excl. draws): {wins[args.agent_a] / played:.1%}")
    print(f"first-seat wins: A={first_wins[args.agent_a]} B={first_wins[args.agent_b]}")
    print(f"errors (crash): {stats['agent_error']}  illegal selects: {stats['illegal']}")
    for i, name in enumerate(("seat-pool-0", "seat-pool-1")):
        if stats["moves"][i]:
            print(f"{name}: {stats['moves'][i]} moves, "
                  f"avg {stats['time'][i] / stats['moves'][i] * 1000:.2f} ms/move")
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


if __name__ == "__main__":
    main()
