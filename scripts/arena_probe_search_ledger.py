"""Head-to-head runner that reports the arithmetic-search layer's own ledger.

``local_arena`` reports wins and ms/move; what a budget change needs is
searches per game, override rate, branch errors and seconds actually drawn.
Seats alternate. Arena win rates are NOT evidence of strength at this n - see
the arena-determinism note - so this prints them only as a crash check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_loader import load_dir_agent  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def find(name: str) -> Path:
    for base in (ROOT / "agents", ROOT / "archive" / "agents"):
        if (base / name).is_dir():
            return base / name
        for group in sorted(p for p in base.iterdir() if p.is_dir()):
            if (group / name).is_dir():
                return group / name
    raise SystemExit(f"agent not found: {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_a")
    ap.add_argument("agent_b")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    agent_a, _, module_a = load_dir_agent(find(args.agent_a))
    agent_b, _, module_b = load_dir_agent(find(args.agent_b))
    deck_a = agent_a({"select": None})
    deck_b = agent_b({"select": None})

    wins = {"A": 0, "B": 0, "draw": 0}
    per_game = []
    illegal = [0, 0]
    crashes = [0, 0]

    for g in range(args.games):
        a_first = g % 2 == 0
        agents = [agent_a, agent_b] if a_first else [agent_b, agent_a]
        decks = [deck_a, deck_b] if a_first else [deck_b, deck_a]
        for module in (module_a, module_b):
            if hasattr(module, "diag_reset"):
                module.diag_reset()
        seconds = [0.0, 0.0]
        moves = [0, 0]
        obs, start = battle_start(decks[0], decks[1])
        if obs is None:
            raise SystemExit("battle_start failed")
        result = -1
        try:
            for _ in range(8000):
                cur = obs["current"]
                if cur["result"] >= 0:
                    result = cur["result"]
                    break
                seat = cur["yourIndex"]
                t0 = time.perf_counter()
                try:
                    action = agents[seat](obs)
                except Exception:
                    crashes[seat] += 1
                    result = 1 - seat
                    break
                seconds[seat] += time.perf_counter() - t0
                moves[seat] += 1
                try:
                    obs = battle_select(list(action))
                except Exception:
                    illegal[seat] += 1
                    result = 1 - seat
                    break
        finally:
            battle_finish()

        if result < 0:
            wins["draw"] += 1
        else:
            winner_is_a = (result == 0) == a_first
            wins["A" if winner_is_a else "B"] += 1
        a_seat = 0 if a_first else 1
        snap_a = module_a.diag_snapshot().get("arithmetic_search", {})
        per_game.append({
            "game": g + 1,
            "a_seat": a_seat,
            "a_won": (result == 0) == a_first,
            "a_seconds": round(seconds[a_seat], 2),
            "b_seconds": round(seconds[1 - a_seat], 2),
            "a_moves": moves[a_seat],
            "search": {
                k: snap_a.get(k) for k in (
                    "considered", "searched", "overrides", "branches",
                    "determinizations", "branch_errors",
                    "incomplete_branches", "skip_budget",
                    "skip_already_searched_turn", "turns_searched",
                    "skip_nonrobust", "skip_no_candidates",
                    "search_seconds", "search_seconds_mean",
                    "budget_stops", "budget_degraded",
                )
            },
        })
        print(json.dumps(per_game[-1]))

    searched = sum(r["search"].get("searched") or 0 for r in per_game)
    considered = sum(r["search"].get("considered") or 0 for r in per_game)
    overrides = sum(r["search"].get("overrides") or 0 for r in per_game)
    errors = sum(r["search"].get("branch_errors") or 0 for r in per_game)
    incomplete = sum(r["search"].get("incomplete_branches") or 0 for r in per_game)
    a_secs = sum(r["a_seconds"] for r in per_game)
    n = len(per_game)
    summary = {
        "agent_a": args.agent_a,
        "agent_b": args.agent_b,
        "games": n,
        "wins": wins,
        "illegal_by_seat": illegal,
        "crashes_by_seat": crashes,
        "a_seconds_per_game": round(a_secs / n, 2),
        "a_seconds_max": max(r["a_seconds"] for r in per_game),
        "searches_per_game": round(searched / n, 2),
        "considered_per_game": round(considered / n, 2),
        "search_coverage": round(searched / considered, 4) if considered else None,
        "overrides_per_game": round(overrides / n, 2),
        "override_rate_of_searches": round(overrides / searched, 4) if searched else None,
        "branch_errors": errors,
        "incomplete_branches": incomplete,
    }
    print("\n== SUMMARY ==")
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"summary": summary, "games": per_game}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
