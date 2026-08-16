"""Head-to-head local arena for the Grimmsnarl line, with worker parallelism.

The native shuffle has no seed entry point, so a single 40-game arena is worth
nothing as evidence ([[arena-cannot-be-paired]]: the same binary against the
same opponent scored 77.5% and 47.5%).  The answer is not to distrust the arena
but to run it at a sample size where the noise is smaller than the effect: at
n=400 the standard error on a win rate is 2.5 points, and the whole point of a
challenger is to move it by more than that.

Seats alternate every game so first-player advantage is shared, and each worker
process runs its own copy of the native library so games can be run in
parallel.

Usage:
  python scripts/arena_grimmsnarl.py --a agents/grimmsnarl/grimmsnarl_ml_vfinal \
      --b agents/grimmsnarl/grimmsnarl_ml_v22 --games 200 --workers 6 \
      --report experiments/grimmsnarl_vfinal/arena_vfinal_v22.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OPT_ATTACK = 13
MAX_STEPS = 8000
GRIMMSNARL_EX = 648
SHADOW_BULLET = 937


def load_deck(module: Any, agent_dir: Path) -> list[int]:
    exported = getattr(module, "MY_DECK", None)
    if exported is not None:
        deck = [int(card_id) for card_id in exported]
    else:
        deck = [
            int(line.strip())
            for line in (agent_dir / "deck.csv")
            .read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    if len(deck) != 60:
        raise ValueError(f"{agent_dir} has {len(deck)} cards, expected 60")
    return deck


def play_one(modules, decks) -> dict[str, Any] | None:
    from cg.game import battle_finish, battle_select, battle_start

    for module in modules:
        try:
            module.diag_reset()
        except Exception:  # noqa: BLE001
            pass
    observation, _ = battle_start(list(decks[0]), list(decks[1]))
    if observation is None:
        return None
    own_turns = [set(), set()]
    shadow = [0, 0]
    attacks = [0, 0]
    exceptions: Counter = Counter()
    result = -1
    prizes_left = [6, 6]
    started = time.monotonic()
    try:
        for _ in range(MAX_STEPS):
            select = observation.get("select")
            current = observation.get("current") or {}
            if select is None or int(current.get("result", -1)) >= 0:
                break
            seat = int(current.get("yourIndex", 0))
            players = current.get("players") or [{}, {}]
            for side in (0, 1):
                prize = (players[side] or {}).get("prize")
                if isinstance(prize, list):
                    prizes_left[side] = len(prize)
            if int(select.get("context", -1)) == 0:
                own_turns[seat].add(int(current.get("turn") or 0))
            try:
                action = modules[seat].agent(observation)
            except Exception as error:  # noqa: BLE001
                exceptions[f"seat{seat}:{type(error).__name__}"] += 1
                action = list(range(int(select.get("minCount") or 0)))
            if not isinstance(action, list):
                action = list(range(int(select.get("minCount") or 0)))
            options = select.get("option") or []
            for index in action:
                if not isinstance(index, int) or not 0 <= index < len(options):
                    continue
                option = options[index]
                if int(option.get("type", -1)) == OPT_ATTACK:
                    attacks[seat] += 1
                    if int(option.get("attackId", -1)) == SHADOW_BULLET:
                        shadow[seat] += 1
            try:
                observation = battle_select(action)
            except Exception as error:  # noqa: BLE001
                exceptions[f"seat{seat}:select:{type(error).__name__}"] += 1
                observation = battle_select(
                    list(range(max(1, int(select.get("minCount") or 0))))
                )
        result = int((observation.get("current") or {}).get("result", -1))
    finally:
        battle_finish()
    diags = []
    for module in modules:
        try:
            diags.append(module.diag_snapshot())
        except Exception:  # noqa: BLE001
            diags.append({})
    return {
        "result": result,
        "own_turns": [len(values) for values in own_turns],
        "shadow": shadow,
        "attacks": attacks,
        "prizes_left": prizes_left,
        "seconds": round(time.monotonic() - started, 2),
        "search": [
            (diag.get("search") or {}) if isinstance(diag, dict) else {}
            for diag in diags
        ],
        "exceptions": dict(exceptions),
    }


def worker(task) -> list[dict[str, Any]]:
    a_dir, b_dir, indices = task
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from agent_loader import load_dir_agent_module

    module_a = load_dir_agent_module(Path(a_dir))
    module_b = load_dir_agent_module(Path(b_dir))
    deck_a = load_deck(module_a, Path(a_dir))
    deck_b = load_deck(module_b, Path(b_dir))
    out = []
    for index in indices:
        a_seat = index % 2
        modules = [module_a, module_b] if a_seat == 0 else [module_b, module_a]
        decks = [deck_a, deck_b] if a_seat == 0 else [deck_b, deck_a]
        try:
            outcome = play_one(modules, decks)
        except Exception as error:  # noqa: BLE001
            out.append({"index": index, "error": f"{type(error).__name__}: {error}"})
            continue
        if outcome is None:
            continue
        outcome["index"] = index
        outcome["a_seat"] = a_seat
        outcome["a_won"] = int(outcome["result"] == a_seat)
        out.append(outcome)
    return out


def wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return (0.0, 0.0)
    z = 1.959963985
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return (
        round((centre - margin) / denominator, 4),
        round((centre + margin) / denominator, 4),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    a_dir = str(args.a.resolve())
    b_dir = str(args.b.resolve())
    chunks: list[list[int]] = [[] for _ in range(args.workers)]
    for index in range(args.games):
        chunks[index % args.workers].append(index)
    tasks = [(a_dir, b_dir, chunk) for chunk in chunks if chunk]

    started = time.monotonic()
    if args.workers == 1:
        results = [worker(tasks[0])]
    else:
        with mp.Pool(processes=len(tasks)) as pool:
            results = pool.map(worker, tasks)
    games = [game for batch in results for game in batch if "error" not in game]
    errors = [game for batch in results for game in batch if "error" in game]

    wins = sum(game["a_won"] for game in games)
    total = len(games)
    low, high = wilson(wins, total)
    first_games = [g for g in games if g["a_seat"] == 0]
    second_games = [g for g in games if g["a_seat"] == 1]

    def mean(key, is_a: bool):
        values = [
            game[key][game["a_seat"] if is_a else 1 - game["a_seat"]]
            for game in games if key in game
        ]
        return round(statistics.mean(values), 3) if values else None

    search_totals: Counter = Counter()
    for game in games:
        for side, payload in enumerate(game.get("search") or []):
            if not isinstance(payload, dict):
                continue
            label = "a" if side == game["a_seat"] else "b"
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    search_totals[f"{label}.{key}"] += value
                elif isinstance(value, dict):
                    for sub, count in value.items():
                        search_totals[f"{label}.{key}.{sub}"] += count

    report = {
        "a": str(args.a), "b": str(args.b),
        "games": total,
        "a_record": f"{wins}-{total - wins}",
        "a_win_rate": round(wins / total, 4) if total else 0.0,
        "a_win_rate_wilson95": [low, high],
        "a_as_first": {
            "games": len(first_games),
            "wins": sum(g["a_won"] for g in first_games),
        },
        "a_as_second": {
            "games": len(second_games),
            "wins": sum(g["a_won"] for g in second_games),
        },
        "a_shadow_mean": mean("shadow", True),
        "b_shadow_mean": mean("shadow", False),
        "a_attacks_mean": mean("attacks", True),
        "b_attacks_mean": mean("attacks", False),
        "a_prizes_left_mean": mean("prizes_left", True),
        "b_prizes_left_mean": mean("prizes_left", False),
        "seconds_per_game": round(
            statistics.mean([g["seconds"] for g in games]), 2) if games else None,
        "wall_clock_seconds": round(time.monotonic() - started, 1),
        "search_totals": {
            key: round(value, 2) for key, value in sorted(search_totals.items())
        },
        "errors": [game["error"] for game in errors][:20],
        "exceptions": dict(sum(
            (Counter(game.get("exceptions") or {}) for game in games), Counter()
        )),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report["detail"] = games
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
