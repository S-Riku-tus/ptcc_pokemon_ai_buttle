"""Run alternating-seat cabt battles between two versioned agents."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from kaggle_environments import make  # type: ignore

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Result:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    errors: int = 0


def read_deck(path: Path) -> list[int]:
    deck = [
        int(value)
        for value in path.read_text(encoding="utf-8-sig").split()
    ]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 IDs, got {len(deck)}")
    return deck


def add_cg_to_path() -> None:
    candidates = [ROOT / "vendor", ROOT]
    for parent in candidates:
        if (parent / "cg" / "api.py").exists():
            sys.path.insert(0, str(parent))
            return
    raise FileNotFoundError(
        "cg/api.py was not found. Copy the official cg directory to vendor/cg."
    )


def load_agent(agent_dir: Path, module_name: str) -> tuple[Callable, list[int]]:
    main_path = agent_dir / "main.py"
    deck_path = agent_dir / "deck.csv"
    if not main_path.exists() or not deck_path.exists():
        raise FileNotFoundError(f"{agent_dir} must contain main.py and deck.csv")

    sys.path.insert(0, str(agent_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, main_path)
        if spec is None or spec.loader is None:
            raise ImportError(main_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    return module.agent, read_deck(deck_path)


def play(
    agent_a: Callable,
    deck_a: list[int],
    agent_b: Callable,
    deck_b: list[int],
) -> int | None:
    env = make("cabt", configuration={"decks": [deck_a, deck_b]}, debug=True)
    try:
        env.run([agent_a, agent_b])
        rewards = [step.reward for step in env.steps[-1]]
        if rewards[0] > rewards[1]:
            return 0
        if rewards[1] > rewards[0]:
            return 1
        return -1
    except Exception as exc:
        print(f"battle error: {type(exc).__name__}: {exc}")
        return None


def resolve_agent(value: str) -> Path:
    direct = Path(value)
    if direct.is_dir():
        return direct.resolve()
    candidate = ROOT / "agents" / value
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="alakazam_ml_v31")
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "benchmark_result.json",
    )
    args = parser.parse_args()

    add_cg_to_path()
    our_dir = resolve_agent(args.agent)
    opponent_dir = resolve_agent(args.opponent)

    our_agent, our_deck = load_agent(our_dir, "our_agent")
    opp_agent, opp_deck = load_agent(opponent_dir, "opponent_agent")

    result = Result()
    for game in range(args.games):
        swapped = game % 2 == 1
        winner = (
            play(opp_agent, opp_deck, our_agent, our_deck)
            if swapped
            else play(our_agent, our_deck, opp_agent, opp_deck)
        )

        result.games += 1
        if winner is None:
            result.errors += 1
        elif winner == -1:
            result.draws += 1
        else:
            our_won = (winner == 1) if swapped else (winner == 0)
            if our_won:
                result.wins += 1
            else:
                result.losses += 1

        if (game + 1) % 10 == 0:
            print(asdict(result))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(asdict(result), indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
