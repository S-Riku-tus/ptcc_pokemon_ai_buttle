"""Common-random-number Champion/Challenger gauntlet for cabt agents.

For each opponent, seed, and evaluated seat this runner plays two games:
Champion-vs-opponent and Challenger-vs-opponent.  The native engine is reset to
the same verified seed immediately before both games, so the evaluated agents
receive the same deal, coin flip, prize placement, and subsequent chance stream
until their actions make the games diverge.  Playing both evaluated seats
removes the large first/second-player imbalance.

Unlike a direct head-to-head, the paired outcome estimates the change in win
rate against a configurable ladder-like opponent mixture.  Confidence
intervals use seed clusters (the mean of the two seats) rather than pretending
the mirrored games are independent.

Examples:
  python scripts/paired_gauntlet.py --config configs/paired_gauntlet/grimmsnarl.json
  python scripts/paired_gauntlet.py --champion grimmsnarl_ml_v20 \
      --challenger grimmsnarl_ml_v21 --opponent mirror=grimmsnarl_ml_v8:1 \
      --blocks 8 --workers 4 --out artifacts/paired_gauntlet/smoke.json

``--blocks`` is the number of unique opponent/seed blocks.  With both seats it
costs four games per block (two policies x two seats), plus calibration repeats.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
_PATHS = [str(ROOT / "vendor"), str(ROOT / "scripts"), str(ROOT)]
sys.path[:] = _PATHS + [entry for entry in sys.path if entry not in _PATHS]

from agent_loader import load_dir_agent, load_shared_policy_base  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from cg_seed import EngineSeedController  # noqa: E402


@dataclass(frozen=True)
class Opponent:
    label: str
    agent: str
    weight: float


@dataclass(frozen=True)
class Job:
    block_id: int
    opponent: str
    seed: int
    evaluated_seat: int
    treatment: str
    repeat: int = 0


_STATE: dict[str, Any] = {}


def _resolve_agent(spec: str) -> Path:
    direct = Path(spec)
    if direct.is_dir():
        return direct.resolve()
    for base in (ROOT / "agents", ROOT / "archive" / "agents"):
        if not base.is_dir():
            continue
        candidate = base / spec
        if candidate.is_dir():
            return candidate.resolve()
        for group in base.iterdir():
            nested = group / spec
            if group.is_dir() and nested.is_dir():
                return nested.resolve()
    raise FileNotFoundError(spec)


def _read_deck(path: Path) -> list[int]:
    values = [int(value) for value in path.read_text(encoding="utf-8-sig").split()]
    if len(values) != 60:
        raise ValueError(f"{path}: expected 60 card ids, got {len(values)}")
    return values


def _load_generic(deck_name: str) -> dict[str, Any]:
    deck_path = ROOT / "agents" / "_opponents" / deck_name / "deck.csv"
    if not deck_path.exists():
        raise FileNotFoundError(
            f"generic opponent {deck_name!r} has no deck at {deck_path}"
        )
    deck = _read_deck(deck_path)
    shared = load_shared_policy_base()
    previous = sys.modules.get("policy_base")
    sys.modules["policy_base"] = shared
    try:
        module_name = f"paired_generic_{deck_name}"
        spec = importlib.util.spec_from_file_location(
            module_name, ROOT / "agents" / "_base" / "generic_policy.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not load agents/_base/generic_policy.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("policy_base", None)
        else:
            sys.modules["policy_base"] = previous

    return {
        "deck": deck,
        # GenericPolicy's PrizeTracker is stateful and has no reset hook, so a
        # fresh closure is intentionally created for each game.
        "factory": lambda: module.make_generic_agent(deck),
        "module": None,
    }


_TEACHER_VARIANT = re.compile(r"^(?P<base>.+)@teacher_code=(?P<code>-?\d+)$")


def _parse_agent_spec(spec: str) -> tuple[str, int | None]:
    """Split an optional inference-only teacher pin from an agent spec.

    This does not alter the agent directory or model artifact.  It is intended
    for controlled opponent proxies such as the v8 model conditioned on the
    1220-rated pilot's already-trained categorical code.
    """
    match = _TEACHER_VARIANT.fullmatch(spec)
    if match is None:
        return spec, None
    return match.group("base"), int(match.group("code"))


def _load_runtime(spec: str) -> dict[str, Any]:
    if spec.startswith("generic:"):
        return _load_generic(spec.split(":", 1)[1])
    base_spec, teacher_code = _parse_agent_spec(spec)
    agent, _diag, module = load_dir_agent(_resolve_agent(base_spec))
    if teacher_code is not None:
        ranker = getattr(module, "_RANKER", None)
        if ranker is None or getattr(ranker, "teacher_index", -1) < 0:
            raise ValueError(
                f"{base_spec}: @teacher_code requires a loaded ranker with "
                "teacher_team_id in its feature schema"
            )
        ranker.teacher_code = teacher_code
        # A whole-policy pin supersedes the class-specific alternate teacher.
        ranker.escalation_code = None
    deck = list(agent({"select": None}))
    if len(deck) != 60:
        raise ValueError(f"{spec}: agent returned a {len(deck)}-card deck")
    return {"agent": agent, "deck": deck, "module": module, "factory": None}


def _reset_runtime(runtime: dict[str, Any]) -> Callable[[dict[str, Any]], list[int]]:
    factory = runtime.get("factory")
    if callable(factory):
        return factory()
    module = runtime.get("module")
    reset = getattr(module, "diag_reset", None)
    if callable(reset):
        reset()
    return runtime["agent"]


def _worker_init(champion: str, challenger: str, opponents: list[dict[str, Any]]) -> None:
    """Never raise: Pool endlessly respawns workers whose initializer fails."""
    try:
        controller = EngineSeedController()
        runtimes = {
            "champion": _load_runtime(champion),
            "challenger": _load_runtime(challenger),
        }
        opponent_runtimes = {
            item["label"]: _load_runtime(item["agent"]) for item in opponents
        }
        _STATE.update(
            controller=controller,
            runtimes=runtimes,
            opponents=opponent_runtimes,
            init_error=None,
        )
    except Exception as exc:  # noqa: BLE001
        _STATE["init_error"] = f"{type(exc).__name__}: {exc}"


def _canonical_observation(observation: dict[str, Any]) -> bytes:
    visible = {key: value for key, value in observation.items() if key != "search_begin_input"}
    return json.dumps(
        visible, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")


def _play(job: Job, max_steps: int) -> dict[str, Any]:
    if _STATE.get("init_error"):
        return {**job.__dict__, "error": f"worker_init: {_STATE['init_error']}"}

    evaluated_runtime = _STATE["runtimes"][job.treatment]
    opponent_runtime = _STATE["opponents"][job.opponent]
    evaluated_agent = _reset_runtime(evaluated_runtime)
    opponent_agent = _reset_runtime(opponent_runtime)
    if job.evaluated_seat == 0:
        agents = [evaluated_agent, opponent_agent]
        decks = [evaluated_runtime["deck"], opponent_runtime["deck"]]
    else:
        agents = [opponent_agent, evaluated_agent]
        decks = [opponent_runtime["deck"], evaluated_runtime["deck"]]

    _STATE["controller"].set_seed(job.seed)
    started = time.perf_counter()
    observation, start_data = battle_start(decks[0], decks[1])
    if observation is None:
        return {
            **job.__dict__,
            "error": f"battle_start:{start_data.errorPlayer}:{start_data.errorType}",
            "seconds": time.perf_counter() - started,
        }

    visible_hash = hashlib.sha256()
    action_hash = hashlib.sha256()
    moves = 0
    error = None
    result: int | None = None
    first_player: int | None = None
    try:
        for _ in range(max_steps):
            current = observation["current"]
            observed_first = current.get("firstPlayer", -1)
            if observed_first in (0, 1) and first_player is None:
                first_player = int(observed_first)
            visible_hash.update(_canonical_observation(observation))
            if current["result"] >= 0:
                result = int(current["result"])
                break
            seat = int(current["yourIndex"])
            try:
                action = list(agents[seat](observation))
                action_hash.update(json.dumps(action, separators=(",", ":")).encode("ascii"))
                observation = battle_select(action)
            except Exception as exc:  # noqa: BLE001
                error = f"seat{seat}:{type(exc).__name__}:{exc}"
                # An acting agent error is a loss for that seat, but the whole
                # evaluation remains invalid because errors are reported.
                result = 1 - seat
                break
            moves += 1
        if result is None:
            error = error or "max_steps"
        evaluated_win = (
            None if result not in (0, 1) else int(result == job.evaluated_seat)
        )
        return {
            **job.__dict__,
            "result": result,
            "evaluated_win": evaluated_win,
            "first_player": first_player,
            "evaluated_went_first": (
                None if first_player is None else int(first_player == job.evaluated_seat)
            ),
            "moves": moves,
            "seconds": time.perf_counter() - started,
            "observable_hash": visible_hash.hexdigest(),
            "action_hash": action_hash.hexdigest(),
            "error": error,
        }
    finally:
        battle_finish()


def _run_job(payload: tuple[Job, int]) -> dict[str, Any]:
    job, max_steps = payload
    return _play(job, max_steps)


def _allocate_blocks(opponents: list[Opponent], blocks: int) -> dict[str, int]:
    total_weight = sum(item.weight for item in opponents)
    if total_weight <= 0:
        raise ValueError("opponent weights must sum to a positive value")
    if blocks < len(opponents):
        raise ValueError("blocks must be at least the number of opponents")
    # Every configured matchup receives at least one seed.  Allocate the
    # remainder proportionally; otherwise small smoke runs silently omit the
    # low-weight matchups they are intended to validate.
    allocated = {item.label: 1 for item in opponents}
    targets = {
        item.label: blocks * item.weight / total_weight for item in opponents
    }
    remaining = blocks - len(opponents)
    for _ in range(remaining):
        item = max(
            opponents,
            key=lambda candidate: (
                targets[candidate.label] - allocated[candidate.label],
                candidate.weight,
                candidate.label,
            ),
        )
        allocated[item.label] += 1
    return allocated


def build_schedule(
    opponents: list[Opponent],
    blocks: int,
    base_seed: int,
    both_seats: bool,
    calibration_blocks: int,
) -> list[Job]:
    allocated = _allocate_blocks(opponents, blocks)
    jobs: list[Job] = []
    block_id = 0
    for opponent in opponents:
        for local_index in range(allocated[opponent.label]):
            seed = (base_seed + (block_id + 1) * 7919) & 0xFFFFFFFF
            if seed == 0:
                seed = 1
            repeats = (0, 1) if local_index < calibration_blocks else (0,)
            for repeat in repeats:
                for seat in ((0, 1) if both_seats else (block_id % 2,)):
                    for treatment in ("champion", "challenger"):
                        jobs.append(
                            Job(
                                block_id=block_id,
                                opponent=opponent.label,
                                seed=seed,
                                evaluated_seat=seat,
                                treatment=treatment,
                                repeat=repeat,
                            )
                        )
            block_id += 1
    return jobs


def _wilson(wins: float, games: int, z: float = 1.96) -> list[float]:
    if games <= 0:
        return [0.0, 0.0]
    p = wins / games
    denom = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / denom
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _paired_exact_p(challenger_only: int, champion_only: int) -> float:
    """Two-sided exact sign/McNemar p-value for discordant paired outcomes."""
    discordant = challenger_only + champion_only
    if discordant == 0:
        return 1.0
    tail = min(challenger_only, champion_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if row.get("repeat") == 0 and row.get("error") is None]
    paired: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for row in primary:
        win = row.get("evaluated_win")
        if win is None:
            continue
        key = (row["opponent"], row["seed"], row["evaluated_seat"])
        paired[key][row["treatment"]] = float(win)

    pair_rows = []
    for (opponent, seed, seat), values in paired.items():
        if "champion" not in values or "challenger" not in values:
            continue
        pair_rows.append(
            {
                "opponent": opponent,
                "seed": seed,
                "seat": seat,
                "champion": values["champion"],
                "challenger": values["challenger"],
                "difference": values["challenger"] - values["champion"],
            }
        )

    clusters: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in pair_rows:
        clusters[(row["opponent"], row["seed"])].append(row["difference"])
    cluster_values = [statistics.fmean(values) for values in clusters.values()]
    effect = statistics.fmean(cluster_values) if cluster_values else 0.0
    if len(cluster_values) >= 2:
        se = statistics.stdev(cluster_values) / math.sqrt(len(cluster_values))
        ci = [max(-1.0, effect - 1.96 * se), min(1.0, effect + 1.96 * se)]
        mde80 = 2.8016 * statistics.stdev(cluster_values) / math.sqrt(len(cluster_values))
    else:
        se = None
        ci = [None, None]
        mde80 = None

    treatment_rates = {}
    for treatment in ("champion", "challenger"):
        games = len(pair_rows)
        wins = sum(row[treatment] for row in pair_rows)
        treatment_rates[treatment] = {
            "games": games,
            "wins": wins,
            "win_rate": wins / games if games else None,
            "wilson95": _wilson(wins, games),
        }

    challenger_only = sum(row["difference"] > 0 for row in pair_rows)
    champion_only = sum(row["difference"] < 0 for row in pair_rows)
    return {
        "complete_pairs": len(pair_rows),
        "seed_clusters": len(cluster_values),
        "challenger_minus_champion": effect,
        "cluster_standard_error": se,
        "cluster_95ci": ci,
        "mde_80pct_approx": mde80,
        "paired_exact_p": _paired_exact_p(challenger_only, champion_only),
        "discordant": {
            "challenger_only_win": challenger_only,
            "champion_only_win": champion_only,
            "same_outcome": sum(row["difference"] == 0 for row in pair_rows),
        },
        "treatments": treatment_rates,
        "pair_rows": pair_rows,
    }


def _calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("repeat") not in (0, 1):
            continue
        key = (
            row.get("block_id"), row.get("opponent"), row.get("seed"),
            row.get("evaluated_seat"), row.get("treatment"),
        )
        grouped[key][int(row["repeat"])] = row
    checked = mismatches = 0
    details = []
    fields = (
        "result", "evaluated_win", "first_player", "evaluated_went_first",
        "moves", "observable_hash", "action_hash", "error",
    )
    for key, repeats in grouped.items():
        if 0 not in repeats or 1 not in repeats:
            continue
        checked += 1
        changed = [field for field in fields if repeats[0].get(field) != repeats[1].get(field)]
        if changed:
            mismatches += 1
            details.append({"key": key, "different_fields": changed})
    return {
        "duplicate_games_checked": checked,
        "mismatches": mismatches,
        "passed": checked > 0 and mismatches == 0,
        "details": details[:50],
    }


def _parse_opponent(text: str) -> Opponent:
    try:
        label, rest = text.split("=", 1)
        agent, weight = rest.rsplit(":", 1)
        return Opponent(label=label, agent=agent, weight=float(weight))
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            "opponent must be LABEL=AGENT:WEIGHT (generic decks use generic:NAME as AGENT)"
        ) from exc


def _load_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    if args.config:
        settings.update(json.loads(args.config.read_text(encoding="utf-8")))
    for key in (
        "champion", "challenger", "blocks", "workers", "base_seed",
        "max_steps", "calibration_blocks", "out",
    ):
        value = getattr(args, key)
        if value is not None:
            settings[key] = str(value) if key == "out" else value
    if args.opponent:
        settings["opponents"] = [item.__dict__ for item in args.opponent]
    if args.one_seat:
        settings["both_seats"] = False
    return settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--champion")
    parser.add_argument("--challenger")
    parser.add_argument("--opponent", action="append", type=_parse_opponent)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--calibration-blocks", type=int)
    parser.add_argument("--one-seat", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    settings = _load_settings(args)

    champion = settings.get("champion")
    challenger = settings.get("challenger")
    if not champion or not challenger:
        parser.error("champion and challenger are required")
    raw_opponents = settings.get("opponents") or []
    opponents = [Opponent(**item) for item in raw_opponents]
    if not opponents:
        parser.error("at least one opponent is required")
    blocks = int(settings.get("blocks", 20))
    workers = max(1, int(settings.get("workers", 1)))
    base_seed = int(settings.get("base_seed", 20260813))
    max_steps = int(settings.get("max_steps", 8000))
    both_seats = bool(settings.get("both_seats", True))
    calibration_blocks = max(1, int(settings.get("calibration_blocks", 1)))
    out_path = Path(settings.get("out", "artifacts/paired_gauntlet/report.json"))

    if blocks < len(opponents):
        parser.error("blocks must be at least the number of opponents")
    if any(item.weight <= 0 for item in opponents):
        parser.error("all opponent weights must be positive")

    # Fail in the parent before spawning.  This catches the wrong ROOT/cg DLL
    # and unknown binaries without entering a worker-respawn loop.
    parent_controller = EngineSeedController()
    engine_status = parent_controller.status()
    parent_controller.restore()

    schedule = build_schedule(
        opponents, blocks, base_seed, both_seats, calibration_blocks
    )
    opponent_payload = [item.__dict__ for item in opponents]
    started = time.perf_counter()
    if workers == 1:
        _worker_init(champion, challenger, opponent_payload)
        rows = [_run_job((job, max_steps)) for job in schedule]
        controller = _STATE.get("controller")
        if controller is not None:
            controller.restore()
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            workers,
            initializer=_worker_init,
            initargs=(champion, challenger, opponent_payload),
        ) as pool:
            rows = []
            progress_every = max(1, len(schedule) // 20)
            for completed, row in enumerate(
                pool.imap_unordered(
                    _run_job, ((job, max_steps) for job in schedule), chunksize=1
                ),
                start=1,
            ):
                rows.append(row)
                if completed % progress_every == 0 or completed == len(schedule):
                    elapsed = time.perf_counter() - started
                    print(
                        f"progress {completed}/{len(schedule)} "
                        f"({completed / elapsed * 3600:.0f} games/hour)",
                        file=sys.stderr,
                        flush=True,
                    )
    rows.sort(
        key=lambda row: (
            row.get("block_id", -1), row.get("repeat", -1),
            row.get("evaluated_seat", -1), row.get("treatment", ""),
        )
    )
    wall = time.perf_counter() - started

    calibration = _calibration(rows)
    errors = [row for row in rows if row.get("error")]
    overall = paired_summary(rows)
    by_opponent = {
        item.label: paired_summary([row for row in rows if row.get("opponent") == item.label])
        for item in opponents
    }
    by_evaluated_order = {
        label: paired_summary(
            [row for row in rows if row.get("evaluated_went_first") == value]
        )
        for label, value in (("first", 1), ("second", 0))
        if any(row.get("evaluated_went_first") == value for row in rows)
    }
    report = {
        "valid": calibration["passed"] and not errors,
        "config": {
            "champion": champion,
            "challenger": challenger,
            "opponents": opponent_payload,
            "blocks": blocks,
            "both_seats": both_seats,
            "workers": workers,
            "base_seed": base_seed,
            "max_steps": max_steps,
            "calibration_blocks": calibration_blocks,
            "represented_ladder_weight": settings.get("represented_ladder_weight"),
            "unrepresented_matchups": settings.get("unrepresented_matchups", []),
            "limitations": settings.get("limitations", []),
        },
        "engine": engine_status,
        "throughput": {
            "games": len(rows),
            "primary_games": sum(row.get("repeat") == 0 for row in rows),
            "wall_seconds": wall,
            "games_per_hour": len(rows) / wall * 3600 if wall else None,
            "mean_game_seconds": statistics.fmean(
                row.get("seconds", 0.0) for row in rows
            ) if rows else None,
        },
        "calibration": calibration,
        "errors": errors[:100],
        "overall": overall,
        "by_opponent": by_opponent,
        "by_evaluated_order": by_evaluated_order,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        key: value
        for key, value in report.items()
        if key not in ("rows", "by_opponent", "by_evaluated_order")
    }
    summary["overall"] = {
        key: value for key, value in overall.items() if key != "pair_rows"
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {out_path.resolve()}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
