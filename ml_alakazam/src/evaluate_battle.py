from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from agents.alakazam741_v12_top_sync_full import benchmark_v12 as detailed  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from scripts.agent_loader import load_dir_agent_module  # noqa: E402
from scripts.local_arena import resolve  # noqa: E402


HYBRID = ROOT / "ml_alakazam" / "agents" / "alakazam_ml_v1"


def _agent_path(spec: str) -> Path:
    direct = Path(spec)
    if direct.is_dir():
        return direct.resolve()
    candidate = ROOT / "agents" / spec
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(spec)


def _load_fresh(spec: str):
    module = load_dir_agent_module(_agent_path(spec))
    return module.agent, module, module.agent({"select": None})


def wilson(wins: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = wins / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def run_detailed_pair(agent_a: str, agent_b: str, games: int, seed: int) -> dict[str, Any]:
    if games % 2:
        raise ValueError("games must be even")
    random.seed(seed)
    results = defaultdict(list)
    crashes = illegal = 0
    runtime_totals = defaultdict(float)
    for game_index in range(games):
        loaded = {name: _load_fresh(name) for name in (agent_a, agent_b)}
        labels = [agent_a, agent_b] if game_index % 2 == 0 else [agent_b, agent_a]
        agents = [loaded[label][0] for label in labels]
        modules = [loaded[label][1] for label in labels]
        decks = [loaded[label][2] for label in labels]
        winner, final_obs, recorder, game_crash, game_illegal = detailed._play(
            labels, agents, modules, decks
        )
        crashes += game_crash
        illegal += game_illegal
        final = detailed.to_observation_class(final_obs)
        for seat, label in enumerate(labels):
            results[label].append(detailed._finish_game(
                recorder.games[label], winner == seat, final.current.players[seat]
            ))
            runtime = getattr(modules[seat], "_RUNTIME", None)
            if runtime is not None:
                for key, value in runtime.snapshot().items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        runtime_totals[key] += value
    metrics = {label: detailed._aggregate(results[label]) for label in (agent_a, agent_b)}
    for label in (agent_a, agent_b):
        wins = sum(game["won"] for game in results[label])
        metrics[label]["win_rate_95ci"] = wilson(wins, games)
    if runtime_totals.get("decisions"):
        decisions = runtime_totals["decisions"]
        metrics[agent_a]["ml_runtime"] = {
            "decisions": int(decisions),
            "model_selection_rate": runtime_totals["model_selected"] / decisions,
            "model_override_rate": runtime_totals["model_override"] / decisions,
            "fallback_rate": runtime_totals["fallback"] / decisions,
            "low_confidence_rate": runtime_totals["low_confidence"] / decisions,
            "average_inference_ms": runtime_totals["inference_us"] / decisions / 1000.0,
        }
    return {"games": games, "crashes": crashes, "illegal_selects": illegal, "metrics": metrics}


def run_pool_match(agent_spec: str, opponent_spec: str, games: int, seed: int) -> dict[str, Any]:
    fallback_deck = [int(value) for value in (HYBRID / "deck.csv").read_text().split()]
    agent, _ = resolve(agent_spec, fallback_deck)
    opponent, _ = resolve(opponent_spec, fallback_deck)
    decks = [agent({"select": None}), opponent({"select": None})]
    random.seed(seed)
    wins = losses = draws = crashes = illegal = moves = 0
    elapsed = 0.0
    for game_index in range(games):
        a_first = game_index % 2 == 0
        agents = [agent, opponent] if a_first else [opponent, agent]
        ordered_decks = decks if a_first else list(reversed(decks))
        obs, start = battle_start(ordered_decks[0], ordered_decks[1])
        if obs is None:
            crashes += 1
            continue
        winner = -1
        try:
            for _ in range(8000):
                result = obs["current"]["result"]
                if result >= 0:
                    winner = result if result in (0, 1) else -1
                    break
                seat = obs["current"]["yourIndex"]
                started = time.perf_counter()
                try:
                    action = agents[seat](obs)
                except Exception:
                    crashes += 1
                    winner = 1 - seat
                    break
                if (seat == 0) == a_first:
                    elapsed += time.perf_counter() - started
                    moves += 1
                try:
                    obs = battle_select(list(action))
                except Exception:
                    illegal += 1
                    winner = 1 - seat
                    break
        finally:
            battle_finish()
        if winner < 0:
            draws += 1
        elif (winner == 0) == a_first:
            wins += 1
        else:
            losses += 1
    decisive = wins + losses
    return {
        "games": games, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": wins / decisive if decisive else 0.0,
        "win_rate_95ci": wilson(wins, decisive),
        "crashes": crashes, "illegal_selects": illegal,
        "average_inference_ms": elapsed / max(moves, 1) * 1000.0,
    }


def evaluate(games: int, pool_games: int, seed: int) -> dict[str, Any]:
    hybrid_spec = str(HYBRID)
    detailed_pairs = {}
    for offset, opponent in enumerate((
        "alakazam741_v9_top8_core", "alakazam741_v11_board_depth", "alakazam741_v12_top_sync_full"
    )):
        detailed_pairs[opponent] = run_detailed_pair(hybrid_spec, opponent, games, seed + offset)
    pool_specs = {
        "Alakazam generic": "generic:alakazam741",
        "Crustle": "generic:crustle",
        "Grimmsnarl": "generic:grimmsnarl",
        "Mega Kangaskhan": "generic:kangaskhan",
        "Mega Starmie": "generic:megastarmie",
        "Team Rocket Spidops": str(ROOT / "agents" / "kashiwashira_spidops_reconstruction_v2"),
    }
    pool = {
        label: run_pool_match(hybrid_spec, spec, pool_games, seed + 100 + index)
        for index, (label, spec) in enumerate(pool_specs.items())
    }
    return {
        "engine": "bundled local cg compatibility engine",
        "python_seed": seed,
        "seat_swapped": True,
        "same_shuffle_seed": False,
        "seed_limitation": "The native battle API exposes no RNG seed setter.",
        "unavailable_opponents": ["Cinderace deck", "Mega Lucario deck", "Dragapult deck"],
        "detailed_pairs": detailed_pairs,
        "opponent_pool": pool,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--pool-games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=741)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "reports" / "battle_evaluation.json")
    args = parser.parse_args()
    report = evaluate(args.games, args.pool_games, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
