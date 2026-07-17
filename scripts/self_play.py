"""Run local self-play benchmarks across repo agents.

This is a thin orchestration layer over scripts/local_arena.py.  It discovers
agent directories, runs alternating-seat matches for every pair, and writes
machine-readable CSV/JSON summaries under data/runs/local_self_play/.

Examples:
  python scripts/self_play.py alakazam741_v4 alakazam741_v3 --games 40
  python scripts/self_play.py --games 20
  python scripts/self_play.py alakazam741_v4 alakazam741_v3 alakazam741_v2 --games 80
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from agent_loader import diag_delta, diag_snapshot
import local_arena

ROOT = local_arena.ROOT


@dataclass
class AgentRuntime:
    spec: str
    agent: Callable[[dict[str, Any]], list[int]]
    deck: list[int]
    diag: dict[str, Any] | None = None


@dataclass
class GameResult:
    matchup: str
    game: int
    seat0: str
    seat1: str
    winner: str
    result: str
    error_agent: str = ""
    error_type: str = ""
    illegal_agent: str = ""
    elapsed_ms: float = 0.0
    moves_seat0: int = 0
    moves_seat1: int = 0
    time_seat0_ms: float = 0.0
    time_seat1_ms: float = 0.0


@dataclass
class MatchupSummary:
    matchup: str
    agent_a: str
    agent_b: str
    games: int = 0
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    errors_a: int = 0
    errors_b: int = 0
    illegal_a: int = 0
    illegal_b: int = 0
    first_seat_wins_a: int = 0
    first_seat_wins_b: int = 0
    moves_a: int = 0
    moves_b: int = 0
    time_a_ms: float = 0.0
    time_b_ms: float = 0.0
    decisions_a: int = 0
    decisions_b: int = 0
    policy_ok_a: int = 0
    policy_ok_b: int = 0
    policy_fallback_a: int = 0
    policy_fallback_b: int = 0
    obs_fallback_a: int = 0
    obs_fallback_b: int = 0
    deck_returns_a: int = 0
    deck_returns_b: int = 0
    fallback_rate_a: float = 0.0
    fallback_rate_b: float = 0.0
    errors_a_detail: dict[str, int] = field(default_factory=dict)
    errors_b_detail: dict[str, int] = field(default_factory=dict)
    games_detail: list[GameResult] = field(default_factory=list)

    @property
    def win_rate_a_ex_draws(self) -> float:
        decided = self.wins_a + self.wins_b
        return self.wins_a / decided if decided else 0.0


TRAJECTORY_SCHEMA = {
    "game_id": "stable matchup/game identifier",
    "agent_version": "agent spec that made the decision",
    "deck_hash": "hash of the acting agent initial deck",
    "seat": "0 or 1 in the current battle",
    "step": "decision index within the battle loop",
    "observation": "raw public observation passed to the agent",
    "legal_options": "select.option candidates from the observation",
    "selected_action": "action returned by the agent",
    "fallback_action": "reserved for agents exposing per-decision fallback traces",
    "ml_selected_action": "reserved for agents exposing per-decision ML traces",
    "ml_probability": "reserved for agents exposing per-decision ML confidence",
    "ml_margin": "reserved for agents exposing per-decision ML margin",
    "ml_adopted": "reserved for agents exposing whether the ML action was used",
    "final_result": "filled after the game completes",
    "termination_reason": "battle result or error type",
    "execution_time_ms": "agent decision time for this step",
    "exception": "agent exception type, when any",
    "illegal_action": "battle_select exception type, when any",
}


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except TypeError:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _deck_hash(deck: list[int]) -> str:
    import hashlib

    text = ",".join(str(card_id) for card_id in sorted(deck))
    return hashlib.sha256(text.encode("ascii")).hexdigest()[:16]


def read_deck(path: Path) -> list[int]:
    deck = [int(x) for x in path.read_text(encoding="utf-8-sig").split()]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 ids, got {len(deck)}")
    return deck


def fallback_deck() -> list[int]:
    for name in ("alakazam741_v4", "alakazam741_v3", "alakazam741_v2", "alakazam741_v1"):
        path = ROOT / "agents" / name / "deck.csv"
        if path.exists():
            return read_deck(path)
    raise FileNotFoundError("No fallback deck found under agents/.")


def discover_agents(include_archive: bool = False) -> list[str]:
    roots = [ROOT / "agents"]
    if include_archive:
        roots.append(ROOT / "archive" / "agents")

    specs: list[str] = []
    for base in roots:
        if not base.exists():
            continue
        for child in sorted(base.iterdir(), key=lambda p: p.name):
            if child.name.startswith("_"):
                continue
            if (child / "main.py").exists() and (child / "deck.csv").exists():
                specs.append(child.name if base.name == "agents" else str(child))
    return specs


def load_runtime(spec: str, fallback: list[int]) -> AgentRuntime:
    agent, diag = local_arena.resolve(spec, fallback)
    deck = list(agent({"select": None}))
    if len(deck) != 60:
        raise ValueError(f"{spec}: agent returned {len(deck)} deck ids, expected 60")
    return AgentRuntime(spec=spec, agent=agent, deck=deck, diag=diag)


def play_one(
    seat_specs: list[str],
    seat_agents: list[Callable[[dict[str, Any]], list[int]]],
    seat_decks: list[list[int]],
    max_steps: int,
    trajectory_records: list[dict[str, Any]] | None = None,
    game_id: str = "",
) -> GameResult:
    from cg.game import battle_finish, battle_select, battle_start

    started = time.perf_counter()
    obs, start_data = battle_start(seat_decks[0], seat_decks[1])
    if obs is None:
        return GameResult(
            matchup=f"{seat_specs[0]}__vs__{seat_specs[1]}",
            game=0,
            seat0=seat_specs[0],
            seat1=seat_specs[1],
            winner=seat_specs[1 - int(start_data.errorPlayer == 1)],
            result="start_error",
            error_agent=seat_specs[start_data.errorPlayer]
            if start_data.errorPlayer in (0, 1)
            else "",
            error_type=f"battle_start:{start_data.errorType}",
        )

    moves = [0, 0]
    try:
        time_by_seat = [0.0, 0.0]
        for step in range(max_steps):
            cur = obs["current"]
            if cur["result"] >= 0:
                winner = (
                    seat_specs[0]
                    if cur["result"] == 0
                    else seat_specs[1]
                    if cur["result"] == 1
                    else "draw"
                )
                return GameResult(
                    matchup=f"{seat_specs[0]}__vs__{seat_specs[1]}",
                    game=0,
                    seat0=seat_specs[0],
                    seat1=seat_specs[1],
                    winner=winner,
                    result="win" if winner != "draw" else "draw",
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    moves_seat0=moves[0],
                    moves_seat1=moves[1],
                    time_seat0_ms=time_by_seat[0],
                    time_seat1_ms=time_by_seat[1],
                )

            seat = cur["yourIndex"]
            record: dict[str, Any] | None = None
            select = obs.get("select") or {}
            if trajectory_records is not None:
                record = {
                    "game_id": game_id,
                    "agent_version": seat_specs[seat],
                    "deck_hash": _deck_hash(seat_decks[seat]),
                    "seat": seat,
                    "step": step,
                    "observation": _json_safe(obs),
                    "legal_options": _json_safe(select.get("option") or []),
                    "selected_action": None,
                    "fallback_action": None,
                    "ml_selected_action": None,
                    "ml_probability": None,
                    "ml_margin": None,
                    "ml_adopted": None,
                    "final_result": None,
                    "termination_reason": None,
                    "execution_time_ms": None,
                    "exception": None,
                    "illegal_action": None,
                }
            try:
                decision_started = time.perf_counter()
                action = seat_agents[seat](obs)
                decision_ms = (time.perf_counter() - decision_started) * 1000
                time_by_seat[seat] += decision_ms
                moves[seat] += 1
                if record is not None:
                    record["selected_action"] = _json_safe(list(action))
                    record["execution_time_ms"] = decision_ms
            except Exception as exc:  # noqa: BLE001 - benchmark should record crashes
                if record is not None:
                    record["exception"] = type(exc).__name__
                    trajectory_records.append(record)
                return GameResult(
                    matchup=f"{seat_specs[0]}__vs__{seat_specs[1]}",
                    game=0,
                    seat0=seat_specs[0],
                    seat1=seat_specs[1],
                    winner=seat_specs[1 - seat],
                    result="agent_error",
                    error_agent=seat_specs[seat],
                    error_type=type(exc).__name__,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    moves_seat0=moves[0],
                    moves_seat1=moves[1],
                    time_seat0_ms=time_by_seat[0],
                    time_seat1_ms=time_by_seat[1],
                )

            try:
                obs = battle_select(list(action))
            except Exception as exc:  # noqa: BLE001 - benchmark should record illegal actions
                if record is not None:
                    record["illegal_action"] = type(exc).__name__
                    trajectory_records.append(record)
                return GameResult(
                    matchup=f"{seat_specs[0]}__vs__{seat_specs[1]}",
                    game=0,
                    seat0=seat_specs[0],
                    seat1=seat_specs[1],
                    winner=seat_specs[1 - seat],
                    result="illegal_select",
                    illegal_agent=seat_specs[seat],
                    error_type=type(exc).__name__,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    moves_seat0=moves[0],
                    moves_seat1=moves[1],
                    time_seat0_ms=time_by_seat[0],
                    time_seat1_ms=time_by_seat[1],
                )
            if record is not None:
                trajectory_records.append(record)

        return GameResult(
            matchup=f"{seat_specs[0]}__vs__{seat_specs[1]}",
            game=0,
            seat0=seat_specs[0],
            seat1=seat_specs[1],
            winner="draw",
            result="max_steps_draw",
            elapsed_ms=(time.perf_counter() - started) * 1000,
            moves_seat0=moves[0],
            moves_seat1=moves[1],
            time_seat0_ms=time_by_seat[0],
            time_seat1_ms=time_by_seat[1],
        )
    finally:
        battle_finish()


def run_matchup(
    agent_a: AgentRuntime,
    agent_b: AgentRuntime,
    games: int,
    max_steps: int,
    quiet: bool,
    trajectory_records: list[dict[str, Any]] | None = None,
    game_seed_base: int | None = None,
) -> MatchupSummary:
    diag_before_a = diag_snapshot(agent_a.diag)
    diag_before_b = diag_snapshot(agent_b.diag)
    summary = MatchupSummary(
        matchup=f"{agent_a.spec}__vs__{agent_b.spec}",
        agent_a=agent_a.spec,
        agent_b=agent_b.spec,
    )

    for game in range(1, games + 1):
        # Optional paired-evaluation mode: make every game independent of how
        # many random effects the previous game's policies happened to invoke.
        # Separate variants can then be compared on identical starting RNG
        # streams by using the same base seed.
        if game_seed_base is not None:
            random.seed(game_seed_base + game - 1)
        game_id = f"{summary.matchup}:{game}"
        a_first = game % 2 == 1
        if a_first:
            seat_specs = [agent_a.spec, agent_b.spec]
            seat_agents = [agent_a.agent, agent_b.agent]
            seat_decks = [agent_a.deck, agent_b.deck]
        else:
            seat_specs = [agent_b.spec, agent_a.spec]
            seat_agents = [agent_b.agent, agent_a.agent]
            seat_decks = [agent_b.deck, agent_a.deck]

        trajectory_start = len(trajectory_records) if trajectory_records is not None else 0
        result = play_one(
            seat_specs,
            seat_agents,
            seat_decks,
            max_steps,
            trajectory_records=trajectory_records,
            game_id=game_id,
        )
        if trajectory_records is not None:
            for record in trajectory_records[trajectory_start:]:
                record["final_result"] = result.result
                record["termination_reason"] = result.error_type or result.result
        result.matchup = summary.matchup
        result.game = game
        summary.games += 1
        summary.games_detail.append(result)

        if result.winner == agent_a.spec:
            summary.wins_a += 1
            if a_first:
                summary.first_seat_wins_a += 1
        elif result.winner == agent_b.spec:
            summary.wins_b += 1
            if not a_first:
                summary.first_seat_wins_b += 1
        else:
            summary.draws += 1

        if result.error_agent == agent_a.spec:
            summary.errors_a += 1
        elif result.error_agent == agent_b.spec:
            summary.errors_b += 1
        if result.illegal_agent == agent_a.spec:
            summary.illegal_a += 1
        elif result.illegal_agent == agent_b.spec:
            summary.illegal_b += 1

        if result.seat0 == agent_a.spec:
            summary.moves_a += result.moves_seat0
            summary.moves_b += result.moves_seat1
            summary.time_a_ms += result.time_seat0_ms
            summary.time_b_ms += result.time_seat1_ms
        else:
            summary.moves_a += result.moves_seat1
            summary.moves_b += result.moves_seat0
            summary.time_a_ms += result.time_seat1_ms
            summary.time_b_ms += result.time_seat0_ms

        if not quiet:
            print(
                f"{summary.matchup} game {game:>3}: "
                f"seat0={result.seat0} winner={result.winner} "
                f"result={result.result}"
            )

    diag_after_a = diag_snapshot(agent_a.diag)
    diag_after_b = diag_snapshot(agent_b.diag)
    diag_used_a = diag_delta(diag_before_a, diag_after_a)
    diag_used_b = diag_delta(diag_before_b, diag_after_b)
    if diag_used_a:
        summary.decisions_a = diag_used_a["decisions"]
        summary.policy_ok_a = diag_used_a["policy_ok"]
        summary.policy_fallback_a = diag_used_a["policy_fallback"]
        summary.obs_fallback_a = diag_used_a["obs_fallback"]
        summary.deck_returns_a = diag_used_a["deck_returns"]
        summary.fallback_rate_a = diag_used_a["fallback_rate"]
        summary.errors_a_detail = diag_used_a["errors"]
    if diag_used_b:
        summary.decisions_b = diag_used_b["decisions"]
        summary.policy_ok_b = diag_used_b["policy_ok"]
        summary.policy_fallback_b = diag_used_b["policy_fallback"]
        summary.obs_fallback_b = diag_used_b["obs_fallback"]
        summary.deck_returns_b = diag_used_b["deck_returns"]
        summary.fallback_rate_b = diag_used_b["fallback_rate"]
        summary.errors_b_detail = diag_used_b["errors"]
    return summary


def flatten_summary(summary: MatchupSummary) -> dict[str, Any]:
    row = asdict(summary)
    row.pop("games_detail", None)
    row["win_rate_a_ex_draws"] = summary.win_rate_a_ex_draws
    row["avg_ms_a_per_move"] = summary.time_a_ms / summary.moves_a if summary.moves_a else 0.0
    row["avg_ms_b_per_move"] = summary.time_b_ms / summary.moves_b if summary.moves_b else 0.0
    return row


def write_outputs(
    output_dir: Path,
    summaries: list[MatchupSummary],
    args: argparse.Namespace,
    trajectory_records: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": args_dict,
        "matchups": [flatten_summary(s) for s in summaries],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_rows = [flatten_summary(s) for s in summaries]
    if summary_rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    game_rows = [
        asdict(game)
        for summary in summaries
        for game in summary.games_detail
    ]
    if game_rows:
        with (output_dir / "games.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(game_rows[0].keys()))
            writer.writeheader()
            writer.writerows(game_rows)
    if trajectory_records is not None:
        (output_dir / "trajectory_schema.json").write_text(
            json.dumps(TRAJECTORY_SCHEMA, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (output_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
            for record in trajectory_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "agents",
        nargs="*",
        help="Agent specs. Defaults to all active agents under agents/.",
    )
    parser.add_argument("--games", type=int, default=20, help="Games per matchup.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reseed-each-game",
        action="store_true",
        help="Reset RNG to seed + game index before every game for paired A/B evaluation.",
    )
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--include-archive", action="store_true")
    parser.add_argument("--include-mirror", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: data/runs/local_self_play/<timestamp>",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Write full per-decision trajectory JSONL for later human-reviewed training candidates.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    agent_specs = args.agents or discover_agents(args.include_archive)
    if len(agent_specs) < 2 and not args.include_mirror:
        parser.error("Need at least two agents, or pass --include-mirror.")

    pairs = list(combinations(agent_specs, 2))
    if args.include_mirror:
        pairs.extend((spec, spec) for spec in agent_specs)

    fallback = fallback_deck()
    runtime_cache: dict[str, AgentRuntime] = {}

    def runtime(spec: str) -> AgentRuntime:
        if spec not in runtime_cache:
            runtime_cache[spec] = load_runtime(spec, fallback)
        return runtime_cache[spec]

    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROOT / "data" / "runs" / "local_self_play" / started

    summaries: list[MatchupSummary] = []
    trajectory_records: list[dict[str, Any]] | None = [] if args.save_trajectories else None
    for a, b in pairs:
        print(f"\n== {a} vs {b} ({args.games} games) ==")
        agent_a = runtime(a)
        agent_b = load_runtime(b, fallback) if a == b else runtime(b)
        summaries.append(
            run_matchup(
                agent_a,
                agent_b,
                games=args.games,
                max_steps=args.max_steps,
                quiet=args.quiet,
                trajectory_records=trajectory_records,
                game_seed_base=args.seed if args.reseed_each_game else None,
            )
        )

    write_outputs(output_dir, summaries, args, trajectory_records)

    print("\n== SUMMARY ==")
    for summary in summaries:
        row = flatten_summary(summary)
        print(
            f"{summary.agent_a} vs {summary.agent_b}: "
            f"{summary.wins_a}-{summary.wins_b}-{summary.draws} "
            f"A win rate={row['win_rate_a_ex_draws']:.1%} "
            f"errors A/B={summary.errors_a}/{summary.errors_b} "
            f"illegal A/B={summary.illegal_a}/{summary.illegal_b} "
            f"policy_fallback A/B={summary.policy_fallback_a}/{summary.policy_fallback_b} "
            f"obs_fallback A/B={summary.obs_fallback_a}/{summary.obs_fallback_b}"
        )
    print(f"Saved: {output_dir}")


if __name__ == "__main__":
    main()
