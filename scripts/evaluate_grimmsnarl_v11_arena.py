"""Head-to-head arena with per-game v11 arithmetic-search diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))

from agent_loader import load_dir_agent  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


def resolve(name: str) -> Path:
    path = Path(name)
    if path.is_dir():
        return path.resolve()
    for group in (ROOT / "agents").iterdir():
        candidate = group / name
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(name)


def deck(agent) -> list[int]:
    return list(agent({"select": None}))


def numeric(snapshot: dict[str, Any]) -> dict[str, int | float | bool]:
    return {
        key: value for key, value in snapshot.items()
        if isinstance(value, (int, float, bool))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("challenger")
    parser.add_argument("control")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=8000)
    args = parser.parse_args()

    challenger, _, challenger_module = load_dir_agent(resolve(args.challenger))
    control, _, control_module = load_dir_agent(resolve(args.control))
    challenger_deck = deck(challenger)
    control_deck = deck(control)
    wins = Counter()
    search_totals: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    games: list[dict[str, Any]] = []
    elapsed_total = {"challenger": 0.0, "control": 0.0}
    moves_total = {"challenger": 0, "control": 0}

    for game_index in range(args.games):
        challenger_first = game_index % 2 == 0
        agents = (
            [challenger, control]
            if challenger_first else [control, challenger]
        )
        decks = (
            [challenger_deck, control_deck]
            if challenger_first else [control_deck, challenger_deck]
        )
        labels = (
            ["challenger", "control"]
            if challenger_first else ["control", "challenger"]
        )
        for module in (challenger_module, control_module):
            reset = getattr(module, "diag_reset", None)
            if reset is not None:
                reset()
        observation, start_data = battle_start(decks[0], decks[1])
        if observation is None:
            raise RuntimeError(
                f"battle_start failed: {start_data.errorPlayer}",
            )
        winner = "draw"
        illegal = None
        try:
            for _ in range(args.max_steps):
                current = observation["current"]
                result = int(current["result"])
                if result >= 0:
                    winner = labels[result] if result in (0, 1) else "draw"
                    break
                seat = int(current["yourIndex"])
                started = time.perf_counter()
                try:
                    action = agents[seat](observation)
                    observation = battle_select(list(action))
                except Exception as error:  # noqa: BLE001
                    winner = labels[1 - seat]
                    illegal = f"{type(error).__name__}: {error}"
                    break
                elapsed = time.perf_counter() - started
                elapsed_total[labels[seat]] += elapsed
                moves_total[labels[seat]] += 1
        finally:
            battle_finish()

        wins[winner] += 1
        snapshot_fn = getattr(challenger_module, "diag_snapshot", None)
        snapshot = snapshot_fn() if snapshot_fn is not None else {}
        search = snapshot.get("arithmetic_search") or {}
        for key, value in numeric(search).items():
            if key not in {
                "enabled", "min_turn", "top_k",
                "determinizations_per_search", "max_rank_margin",
                "min_mean_utility_gain", "default_searches_per_turn",
                "alakazam_second_searches_per_turn",
            }:
                search_totals[key] += value
        game_records = list(search.get("override_records") or [])
        records.extend({"game": game_index + 1, **row} for row in game_records)
        games.append({
            "game": game_index + 1,
            "challenger_first": challenger_first,
            "winner": winner,
            "illegal": illegal,
            "search": numeric(search),
            "override_records": game_records,
        })
        print(
            f"game={game_index + 1} first={challenger_first} winner={winner} "
            f"searched={search.get('searched', 0)} "
            f"overrides={search.get('overrides', 0)}",
            flush=True,
        )

    report = {
        "challenger": args.challenger,
        "control": args.control,
        "games_count": args.games,
        "wins": dict(wins),
        "search_totals": dict(search_totals),
        "latency_ms": {
            label: round(
                1000 * elapsed_total[label] / max(1, moves_total[label]), 2,
            )
            for label in ("challenger", "control")
        },
        "moves": moves_total,
        "override_records": records,
        "games": games,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "wins": report["wins"],
        "search_totals": report["search_totals"],
        "latency_ms": report["latency_ms"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
