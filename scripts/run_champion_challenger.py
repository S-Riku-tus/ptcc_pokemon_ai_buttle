"""One-command Champion-Challenger evaluation.

Runs a fair (seat-swapped) match between the current Champion and a new
Challenger, optionally against a Baseline, aggregates result/safety/tactical/ML
metrics, computes a statistical confidence interval, judges promotion, and
writes a timestamped artifact bundle with a Markdown report.

It NEVER promotes: it does not overwrite the Champion, swap models, edit config,
commit, tag, push, or submit to Kaggle. Formal promotion is a separate,
human-invoked step (scripts/promote_challenger.py).

Examples:
  python scripts/run_champion_challenger.py \
    --config configs/champion_challenger/alakazam.json \
    --challenger alakazam_ml_v4_candidate

  python scripts/run_champion_challenger.py \
    --champion alakazam_ml_v3 --challenger alakazam_ml_v4_candidate --games 200
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = SCRIPTS_DIR.parent
# Mirror scripts/local_arena.py path setup so `import cg` and shared agent base
# modules resolve for both agent loading and the engine game loop.
for _extra in (SCRIPTS_DIR, _REPO_ROOT / "vendor", _REPO_ROOT / "agents" / "_base"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import cc_core  # noqa: E402
from cc_core import CHALLENGER_ROLE, CHAMPION_ROLE, ROOT  # noqa: E402

# Module names that agent main.py files import by bare name; they must be
# reloaded per agent so a Champion and a similar Challenger do not share a
# cached copy of fallback_v12 / ml_runtime / etc.
_AGENT_LOCAL_MODULES = (
    "fallback_v12",
    "ml_runtime",
    "ml_features",
    "policy_base",
    "common_runtime",
    "generic_policy",
)

ATTACK_OPTION_TYPE = 13


# ---------------------------------------------------------------------------
# Agent loading (engine-touching)
# ---------------------------------------------------------------------------


def _purge_agent_local_modules() -> None:
    for name in _AGENT_LOCAL_MODULES:
        sys.modules.pop(name, None)


def load_agent(agent_dir: Path) -> dict[str, Any]:
    """Load an agent directory into a runtime bundle.

    Returns dict with: agent (callable), module, deck (list[int]),
    diag_snapshot (callable|None), diag_reset (callable|None).
    """
    import agent_loader

    _purge_agent_local_modules()
    agent, _diag, module = agent_loader.load_dir_agent(agent_dir)
    deck = list(agent({"select": None}))
    if len(deck) != 60:
        raise ValueError(f"{agent_dir.name}: agent returned {len(deck)} deck ids, expected 60")
    return {
        "agent": agent,
        "module": module,
        "deck": deck,
        "diag_snapshot": getattr(module, "diag_snapshot", None),
        "diag_reset": getattr(module, "diag_reset", None),
    }


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def preflight(config: dict[str, Any], run_baseline: bool, log: Callable[[str], None]) -> dict[str, Any]:
    """Validate agents statically and confirm they load. Fatal errors raise."""
    import validate_agent

    result: dict[str, Any] = {
        "champion_import_failed": False,
        "challenger_import_failed": False,
        "champion_model_failed": False,
        "challenger_model_failed": False,
        "ok": True,
    }

    roles = [(CHAMPION_ROLE, config["champion_agent"]), (CHALLENGER_ROLE, config["challenger_agent"])]
    if run_baseline and config.get("baseline_agent"):
        roles.append(("baseline", config["baseline_agent"]))

    runtimes: dict[str, dict[str, Any]] = {}
    for role, spec in roles:
        agent_dir = cc_core.resolve_agent_dir(spec)  # raises FileNotFoundError -> fatal
        log(f"[preflight] {role}={spec} dir={agent_dir}")

        static = validate_agent.validate_agent(agent_dir)  # deck 60 + static import check
        if static["deck_size"] != 60:
            raise ValueError(f"{spec}: deck.csv is not 60 cards ({static['deck_size']})")
        for warning in static.get("warnings", []):
            log(f"[preflight] WARN {spec}: {warning}")

        try:
            runtime = load_agent(agent_dir)
        except Exception as exc:  # import failure is fatal for that role
            log(f"[preflight] FATAL {role} import failed: {type(exc).__name__}: {exc}")
            result[f"{role}_import_failed"] = True
            result["ok"] = False
            raise RuntimeError(f"{role} agent {spec!r} failed to import: {exc}") from exc

        # model load check (guarded ML agents expose diag_snapshot with model_loaded)
        snap = runtime["diag_snapshot"]
        if callable(snap):
            try:
                info = snap()
                ml_info = info.get("ml") if isinstance(info, dict) else None
                if isinstance(ml_info, dict) and ml_info.get("model_loaded") is False:
                    log(f"[preflight] WARN {role} model not loaded: {ml_info.get('model_error')}")
                    result[f"{role}_model_failed"] = True
            except Exception as exc:  # diagnostics are optional; never fatal
                log(f"[preflight] {role} diag_snapshot unavailable: {exc}")

        runtime["dir"] = agent_dir
        runtime["spec"] = spec
        runtimes[role] = runtime

    result["runtimes"] = runtimes
    return result


# ---------------------------------------------------------------------------
# Instrumented single game
# ---------------------------------------------------------------------------


def _in_play_count(player: dict[str, Any]) -> int:
    active = player.get("active") or []
    bench = player.get("bench") or []
    return len([c for c in active if isinstance(c, dict)]) + len([c for c in bench if isinstance(c, dict)])


def _classify_loss(
    terminal_obs: dict[str, Any], loser_seat: int, winner_seat: int
) -> tuple[str, int | None, int | None]:
    """Best-effort loss reason from the terminal observation.

    Returns (reason, final_deck_count, in_play_count). reason is one of
    'deckout' | 'boardout' | 'prizes'. Priority: a true prize win (winner has
    no prizes left) is a normal loss even though the loser's board is usually
    empty too; only when the winner still holds prizes do we attribute the loss
    to the loser running out of Pokemon (boardout) or cards (deckout). Terminal
    board inspection is a heuristic and the report states this.
    """
    players = (terminal_obs.get("current") or {}).get("players") or [{}, {}]
    if not (0 <= loser_seat < len(players)):
        return ("prizes", None, None)
    loser = players[loser_seat] or {}
    deck_count = int(loser.get("deckCount", 0))
    in_play = _in_play_count(loser)
    winner_prizes_left = None
    if 0 <= winner_seat < len(players):
        winner_prizes_left = len((players[winner_seat] or {}).get("prize") or [])
    if winner_prizes_left == 0:
        return ("prizes", deck_count, in_play)
    if in_play == 0:
        return ("boardout", deck_count, in_play)
    if deck_count == 0:
        return ("deckout", deck_count, in_play)
    return ("prizes", deck_count, in_play)


def play_instrumented_game(
    seat_roles: list[str],
    seat_agents: list[Callable[[dict[str, Any]], list[int]]],
    seat_decks: list[list[int]],
    alakazam_card_id: int,
    max_steps: int,
    timeout_ms: float,
    trajectory_sink: list[dict[str, Any]] | None,
    game_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Play one game, returning (decision_events, terminal_meta).

    terminal_meta carries winner_role/result/termination/turns/first_player_role
    plus per-role {lost, loss_reason, crash, illegal, timeout, final_deck_count,
    in_play_count_final}.
    """
    from cg.game import battle_finish, battle_select, battle_start

    events: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        CHAMPION_ROLE: {"lost": False, "loss_reason": None, "crash": False, "illegal": False, "timeout": False},
        CHALLENGER_ROLE: {"lost": False, "loss_reason": None, "crash": False, "illegal": False, "timeout": False},
    }

    obs, start_data = battle_start(seat_decks[0], seat_decks[1])
    if obs is None:
        winner_seat = 1 - int(start_data.errorPlayer == 1)
        meta.update(
            {
                "winner_role": seat_roles[winner_seat],
                "result": "start_error",
                "termination": f"battle_start:{start_data.errorType}",
                "turns": 0,
                "first_player_role": None,
            }
        )
        loser_seat = 1 - winner_seat
        meta[seat_roles[loser_seat]]["lost"] = True
        return events, meta

    first_player_seat = int((obs.get("current") or {}).get("firstPlayer", -1))
    first_player_role = seat_roles[first_player_seat] if first_player_seat in (0, 1) else None

    last_turn = 0
    try:
        for step in range(max_steps):
            cur = obs["current"]
            last_turn = int(cur.get("turn", last_turn))
            if cur["result"] >= 0:
                result_code = cur["result"]
                if result_code == 2 or result_code not in (0, 1):
                    winner_role = "draw"
                    loser_seat = None
                else:
                    winner_role = seat_roles[result_code]
                    loser_seat = 1 - result_code
                meta.update(
                    {
                        "winner_role": winner_role,
                        "result": "win" if winner_role != "draw" else "draw",
                        "termination": "normal",
                        "turns": last_turn,
                        "first_player_role": first_player_role,
                    }
                )
                if loser_seat is not None:
                    reason, deck_count, in_play = _classify_loss(obs, loser_seat, result_code)
                    role = seat_roles[loser_seat]
                    meta[role].update(
                        {
                            "lost": True,
                            "loss_reason": reason,
                            "final_deck_count": deck_count,
                            "in_play_count_final": in_play,
                        }
                    )
                return events, meta

            seat = cur["yourIndex"]
            role = seat_roles[seat]
            select = obs.get("select") or {}
            options = select.get("option") or []
            player = (cur.get("players") or [{}, {}])[seat]
            hand_count = int(player.get("handCount", len(player.get("hand") or [])))
            opp = (cur.get("players") or [{}, {}])[1 - seat]
            opp_active = (opp.get("active") or [{}])
            opp_hp = float(opp_active[0].get("hp", 0)) if opp_active and isinstance(opp_active[0], dict) else 0.0
            my_active = (player.get("active") or [{}])
            my_active_id = int(my_active[0].get("id", -1)) if my_active and isinstance(my_active[0], dict) else -1

            traj_record = None
            if trajectory_sink is not None:
                traj_record = {
                    "game_id": game_id,
                    "role": role,
                    "seat": seat,
                    "turn": last_turn,
                    "step": step,
                    "select_type": int(select.get("type", -1)),
                    "select_context": int(select.get("context", -1)),
                    "num_options": len(options),
                    "selected_action": None,
                    "action_type": None,
                    "exception": None,
                    "illegal_action": None,
                }

            decision_started = time.perf_counter()
            try:
                action = list(seat_agents[seat](obs))
            except Exception as exc:  # crash
                decision_ms = (time.perf_counter() - decision_started) * 1000
                meta[role].update({"lost": True, "crash": True, "loss_reason": "crash"})
                meta.update(
                    {
                        "winner_role": seat_roles[1 - seat],
                        "result": "agent_error",
                        "termination": type(exc).__name__,
                        "turns": last_turn,
                        "first_player_role": first_player_role,
                    }
                )
                events.append(
                    {
                        "role": role, "seat": seat, "turn": last_turn,
                        "action_type": "crash", "is_attack": False,
                        "is_alakazam_attack": False, "is_search": False,
                        "hand_count": hand_count, "overkill": 0.0,
                        "decision_ms": decision_ms,
                    }
                )
                if traj_record is not None:
                    traj_record["exception"] = type(exc).__name__
                    trajectory_sink.append(traj_record)
                return events, meta
            decision_ms = (time.perf_counter() - decision_started) * 1000

            # classify selected action from the chosen option(s)
            action_type = "other"
            is_attack = False
            is_alakazam_attack = False
            overkill = 0.0
            if len(action) == 1 and 0 <= action[0] < len(options):
                option = options[action[0]]
                opt_type = int(option.get("type", -1))
                if opt_type == ATTACK_OPTION_TYPE:
                    is_attack = True
                    action_type = "attack"
                    if my_active_id == alakazam_card_id:
                        is_alakazam_attack = True
                        # Alakazam "powerful hand" attack scales with hand size.
                        est_damage = 20 * hand_count
                        overkill = max(0.0, est_damage - opp_hp) if opp_hp > 0 else 0.0
                elif opt_type == 14:
                    action_type = "end"
                elif opt_type == 12:
                    action_type = "retreat"
                elif opt_type == 9:
                    action_type = "evolve"
                elif opt_type == 10:
                    action_type = "ability"
                elif opt_type in (7, 8):
                    action_type = "play"
            # search sub-selection heuristic: a non-main select context
            is_search = int(select.get("context", 0)) != 0 and int(select.get("type", 0)) != 0

            soft_timeout = decision_ms > timeout_ms > 0
            if soft_timeout:
                meta[role]["timeout"] = True

            events.append(
                {
                    "role": role, "seat": seat, "turn": last_turn,
                    "action_type": action_type, "is_attack": is_attack,
                    "is_alakazam_attack": is_alakazam_attack, "is_search": is_search,
                    "hand_count": hand_count, "overkill": overkill,
                    "decision_ms": decision_ms,
                }
            )
            if traj_record is not None:
                traj_record["selected_action"] = action
                traj_record["action_type"] = action_type

            try:
                obs = battle_select(action)
            except Exception as exc:  # illegal action
                meta[role].update({"lost": True, "illegal": True, "loss_reason": "illegal"})
                meta.update(
                    {
                        "winner_role": seat_roles[1 - seat],
                        "result": "illegal_select",
                        "termination": type(exc).__name__,
                        "turns": last_turn,
                        "first_player_role": first_player_role,
                    }
                )
                if traj_record is not None:
                    traj_record["illegal_action"] = type(exc).__name__
                    trajectory_sink.append(traj_record)
                return events, meta
            if traj_record is not None:
                trajectory_sink.append(traj_record)

        # max steps reached -> draw
        meta.update(
            {
                "winner_role": "draw",
                "result": "max_steps_draw",
                "termination": "max_steps",
                "turns": last_turn,
                "first_player_role": first_player_role,
            }
        )
        return events, meta
    finally:
        battle_finish()


# ---------------------------------------------------------------------------
# Matchup runner
# ---------------------------------------------------------------------------


def _diag_ml_delta(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(after, dict):
        return {}
    before_ml = (before or {}).get("ml") if isinstance(before, dict) else {}
    after_ml = after.get("ml") if isinstance(after, dict) else {}
    if not isinstance(after_ml, dict):
        return {}
    before_ml = before_ml if isinstance(before_ml, dict) else {}
    delta: dict[str, Any] = {}
    for key, value in after_ml.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            delta[key] = value - float(before_ml.get(key, 0) or 0)
    return delta


def run_matchup(
    role_a: str,
    runtime_a: dict[str, Any],
    role_b: str,
    runtime_b: dict[str, Any],
    games: int,
    seat_swap: bool,
    base_seed: int,
    config: dict[str, Any],
    trajectory_sink: list[dict[str, Any]] | None,
    replay_sink: list[dict[str, Any]] | None,
    failures: list[str],
    log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], int, dict[str, dict[str, Any]]]:
    """Run role_a vs role_b. Returns (game_records, start_errors, ml_delta_by_role)."""
    import random

    assignments = cc_core.build_pairs(games, base_seed, seat_swap, role_a=role_a, role_b=role_b)
    alakazam_card_id = int(config.get("alakazam_card_id", cc_core.DEFAULT_ALAKAZAM_CARD_ID))
    max_steps = int(config.get("max_steps", 8000))
    timeout_ms = float(config.get("timeout_seconds_per_decision", 0.0)) * 1000

    runtime_by_role = {role_a: runtime_a, role_b: runtime_b}
    for runtime in (runtime_a, runtime_b):
        reset = runtime.get("diag_reset")
        if callable(reset):
            try:
                reset()
            except Exception:
                pass
    diag_before = {
        role: (rt["diag_snapshot"]() if callable(rt.get("diag_snapshot")) else None)
        for role, rt in runtime_by_role.items()
    }

    records: list[dict[str, Any]] = []
    start_errors = 0
    for assignment in assignments:
        random.seed(assignment.seed)  # seeds Python-level baselines; see cc_core note
        seat_roles = [assignment.seat0_role, assignment.seat1_role]
        seat_agents = [runtime_by_role[r]["agent"] for r in seat_roles]
        seat_decks = [runtime_by_role[r]["deck"] for r in seat_roles]
        game_id = f"{role_a}_vs_{role_b}:pair{assignment.pair_id}:g{assignment.game_index}"

        try:
            events, terminal_meta = play_instrumented_game(
                seat_roles, seat_agents, seat_decks,
                alakazam_card_id, max_steps, timeout_ms,
                trajectory_sink, game_id,
            )
        except Exception as exc:  # engine-level failure for this game; continue
            failures.append(f"{game_id}: engine error {type(exc).__name__}: {exc}")
            log(f"[warn] {game_id}: engine error {type(exc).__name__}: {exc}")
            continue

        if terminal_meta.get("result") == "start_error":
            start_errors += 1
            failures.append(f"{game_id}: battle_start error {terminal_meta.get('termination')}")

        champion_seat = seat_roles.index(CHAMPION_ROLE) if CHAMPION_ROLE in seat_roles else None
        challenger_seat = seat_roles.index(CHALLENGER_ROLE) if CHALLENGER_ROLE in seat_roles else None
        record_meta = {
            "pair_id": assignment.pair_id,
            "game_index": assignment.game_index,
            "seed": assignment.seed,
            "champion_seat": champion_seat,
            "challenger_seat": challenger_seat,
            "first_player_role": terminal_meta.get("first_player_role"),
            "winner_role": terminal_meta.get("winner_role"),
            "result": terminal_meta.get("result"),
            "termination": terminal_meta.get("termination"),
            "turns": terminal_meta.get("turns"),
            CHAMPION_ROLE: terminal_meta.get(CHAMPION_ROLE, {}),
            CHALLENGER_ROLE: terminal_meta.get(CHALLENGER_ROLE, {}),
        }
        record = cc_core.summarize_game(events, record_meta)
        records.append(record)
        if replay_sink is not None:
            replay_sink.append({"game_id": game_id, "events": events, "meta": record_meta})
        log(
            f"[game] {game_id} winner={record['winner_role']} "
            f"result={record['result']} turns={record['turns']}"
        )

    diag_after = {
        role: (rt["diag_snapshot"]() if callable(rt.get("diag_snapshot")) else None)
        for role, rt in runtime_by_role.items()
    }
    ml_delta = {
        role: _diag_ml_delta(diag_before.get(role), diag_after.get(role))
        for role in runtime_by_role
    }
    return records, start_errors, ml_delta


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _game_csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        rows.append(
            {
                "pair_id": r["pair_id"],
                "game_index": r["game_index"],
                "seed": r["seed"],
                "champion_seat": r["champion_seat"],
                "challenger_seat": r["challenger_seat"],
                "first_player_role": r["first_player_role"],
                "winner_role": r["winner_role"],
                "result": r["result"],
                "termination": r["termination"],
                "turns": r["turns"],
                "challenger_attacks": r[CHALLENGER_ROLE]["attacks"],
                "challenger_alakazam_attacks": r[CHALLENGER_ROLE]["alakazam_attacks"],
                "challenger_loss_reason": r[CHALLENGER_ROLE]["loss_reason"],
                "champion_loss_reason": r[CHAMPION_ROLE]["loss_reason"],
            }
        )
    return rows


def _seed_pair_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[Any, dict[str, Any]] = {}
    for r in records:
        p = pairs.setdefault(
            r["pair_id"],
            {"pair_id": r["pair_id"], "seed": r["seed"], "games": 0,
             "challenger_wins": 0, "champion_wins": 0, "draws": 0},
        )
        p["games"] += 1
        if r["winner_role"] == CHALLENGER_ROLE:
            p["challenger_wins"] += 1
        elif r["winner_role"] == CHAMPION_ROLE:
            p["champion_wins"] += 1
        else:
            p["draws"] += 1
    return list(pairs.values())


def build_report(
    config: dict[str, Any],
    preflight_result: dict[str, Any],
    records: list[dict[str, Any]],
    start_errors: int,
    ml_delta: dict[str, dict[str, Any]],
    baseline_results: dict[str, dict[str, Any]],
    failures: list[str],
    timestamp: str,
    output_dir: Path,
) -> dict[str, Any]:
    champion_dir = cc_core.resolve_agent_dir(config["champion_agent"])
    challenger_dir = cc_core.resolve_agent_dir(config["challenger_agent"])

    matchup = cc_core.aggregate_matchup(records, start_errors=start_errors)
    champion_metrics = cc_core.aggregate_role(records, CHAMPION_ROLE)
    challenger_metrics = cc_core.aggregate_role(records, CHALLENGER_ROLE)
    judgement = cc_core.judge_promotion(
        matchup, challenger_metrics, config, preflight_ok=preflight_result.get("ok", True)
    )

    report = {
        "meta": {
            "champion": config["champion_agent"],
            "challenger": config["challenger_agent"],
            "baseline": config.get("baseline_agent"),
            "champion_model_hash": cc_core.model_hash(champion_dir),
            "challenger_model_hash": cc_core.model_hash(challenger_dir),
            "champion_deck_hash": cc_core.deck_hash(champion_dir),
            "challenger_deck_hash": cc_core.deck_hash(challenger_dir),
            "timestamp": timestamp,
            "git_commit": _git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "seat_swap": config.get("seat_swap"),
            "seed": config.get("seed"),
        },
        "head_to_head": matchup,
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "ml_diagnostics": {
            "challenger": ml_delta.get(CHALLENGER_ROLE, {}),
            "champion": ml_delta.get(CHAMPION_ROLE, {}),
        },
        "baseline": baseline_results,
        "baseline_note": (
            "Baseline win rates are the agent's win rate vs the pure-rule baseline. "
            "A Challenger that beats the Champion but is weaker than the baseline is suspect."
        )
        if baseline_results
        else None,
        "judgement": judgement,
        "preflight": {
            "champion_import_failed": preflight_result.get("champion_import_failed"),
            "challenger_import_failed": preflight_result.get("challenger_import_failed"),
            "champion_model_failed": preflight_result.get("champion_model_failed"),
            "challenger_model_failed": preflight_result.get("challenger_model_failed"),
        },
        "failures": failures,
        "report_path": str(output_dir / "promotion_report.json"),
    }
    return report


def write_artifacts(
    output_dir: Path,
    config: dict[str, Any],
    report: dict[str, Any],
    records: list[dict[str, Any]],
    baseline_records: dict[str, list[dict[str, Any]]],
    trajectory_sink: list[dict[str, Any]] | None,
    replay_sink: list[dict[str, Any]] | None,
    environment: dict[str, Any],
    failures: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "config_resolved.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "promotion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "promotion_report.md").write_text(
        cc_core.render_markdown(report), encoding="utf-8"
    )

    _write_csv(output_dir / "game_results.csv", _game_csv_rows(records))
    _write_csv(output_dir / "seed_pair_results.csv", _seed_pair_rows(records))
    _write_csv(
        output_dir / "agent_metrics.csv",
        [report["champion_metrics"], report["challenger_metrics"]],
    )
    _write_csv(output_dir / "matchup_metrics.csv", [report["head_to_head"]])

    ml_rows = []
    for role in (CHALLENGER_ROLE, CHAMPION_ROLE):
        delta = report["ml_diagnostics"].get(role) or {}
        if delta:
            ml_rows.append({"role": role, **delta})
    _write_csv(output_dir / "ml_diagnostics.csv", ml_rows)

    _write_csv(output_dir / "failures.csv", [{"failure": f} for f in failures])

    if config.get("save_replays") and replay_sink is not None:
        replays_dir = output_dir / "replays"
        replays_dir.mkdir(exist_ok=True)
        (replays_dir / "matchup_replays.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in replay_sink) + "\n",
            encoding="utf-8",
        )
    if config.get("save_trajectories") and trajectory_sink is not None:
        traj_dir = output_dir / "trajectories"
        traj_dir.mkdir(exist_ok=True)
        with (traj_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
            for record in trajectory_sink:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--champion", default=None)
    parser.add_argument("--challenger", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-seat-swap", dest="seat_swap", action="store_const", const=False, default=None)
    parser.add_argument("--save-replays", dest="save_replays", action="store_const", const=True, default=None)
    parser.add_argument("--save-trajectories", dest="save_trajectories", action="store_const", const=True, default=None)
    parser.add_argument("--run-baseline", dest="run_baseline_comparison", action="store_const", const=True, default=None)
    parser.add_argument("--no-baseline", dest="run_baseline_comparison", action="store_const", const=False, default=None)
    parser.add_argument("--baseline-games", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--list-challengers", action="store_true", help="List detected challenger candidates and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = cc_core.load_config(args.config)

    overrides = {
        "champion": args.champion,
        "challenger": args.challenger,
        "baseline": args.baseline,
        "games": args.games,
        "seed": args.seed,
        "seat_swap": args.seat_swap,
        "save_replays": args.save_replays,
        "save_trajectories": args.save_trajectories,
        "run_baseline_comparison": args.run_baseline_comparison,
        "baseline_games": args.baseline_games,
        "output_root": args.output_root,
    }
    config = cc_core.apply_overrides(config, overrides)

    # --- challenger auto-detection ---
    if args.list_challengers or not config.get("challenger_agent"):
        candidates = cc_core.detect_challengers(
            config.get("champion_agent") or "", config.get("archetype")
        )
        if args.list_challengers:
            print(json.dumps(candidates, ensure_ascii=False, indent=2))
            return 0
        if not candidates:
            print(
                "No challenger specified and none auto-detected. Pass --challenger, or add a "
                "'*_candidate' / '*_challenger' agent (or metadata role='challenger').",
                file=sys.stderr,
            )
            return 2
        if len(candidates) == 1:
            config["challenger_agent"] = candidates[0]["name"]
            print(f"Auto-detected challenger: {candidates[0]['name']}")
        else:
            print("Multiple challenger candidates detected; specify one with --challenger:")
            for candidate in candidates:
                print(f"  - {candidate['name']} (created_at={candidate.get('created_at')})")
            return 2

    cc_core.validate_config(config)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{config['champion_agent']}_vs_{config['challenger_agent']}"
    output_dir = (ROOT / config["output_root"] / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run.log"
    log_lines: list[str] = []

    def log(message: str) -> None:
        log_lines.append(message)
        print(message)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if int(config.get("parallel_workers", 1)) > 1:
        log(
            "[note] parallel_workers>1 requested, but the bundled cg engine is a single "
            "global instance; running sequentially for correctness."
        )

    log(f"[start] champion={config['champion_agent']} challenger={config['challenger_agent']} "
        f"games={config['games']} seat_swap={config['seat_swap']}")

    run_baseline = bool(config.get("run_baseline_comparison")) and bool(config.get("baseline_agent"))
    failures: list[str] = []

    try:
        preflight_result = preflight(config, run_baseline, log)
    except Exception as exc:
        log(f"[fatal] preflight failed: {exc}")
        # Still emit an INVALID report so the run is auditable.
        report = {
            "meta": {
                "champion": config.get("champion_agent"),
                "challenger": config.get("challenger_agent"),
                "timestamp": timestamp,
            },
            "judgement": {"verdict": cc_core.VERDICT_INVALID, "checks": [], "reasons": [str(exc)]},
            "failures": [str(exc)],
        }
        (output_dir / "promotion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nVERDICT: {cc_core.VERDICT_INVALID}")
        print(f"Report: {output_dir}")
        return 3

    runtimes = preflight_result["runtimes"]
    trajectory_sink: list[dict[str, Any]] | None = [] if config.get("save_trajectories") else None
    replay_sink: list[dict[str, Any]] | None = [] if config.get("save_replays") else None

    log("[phase] champion vs challenger")
    records, start_errors, ml_delta = run_matchup(
        CHAMPION_ROLE, runtimes[CHAMPION_ROLE],
        CHALLENGER_ROLE, runtimes[CHALLENGER_ROLE],
        int(config["games"]), bool(config["seat_swap"]), int(config["seed"]),
        config, trajectory_sink, replay_sink, failures, log,
    )

    baseline_results: dict[str, dict[str, Any]] = {}
    baseline_records: dict[str, list[dict[str, Any]]] = {}
    if run_baseline:
        baseline_games = int(config.get("baseline_games", config["games"]))
        for role in (CHALLENGER_ROLE, CHAMPION_ROLE):
            log(f"[phase] {role} vs baseline")
            b_records, b_start_errors, _ = run_matchup(
                role, runtimes[role],
                "baseline", runtimes["baseline"],
                baseline_games, bool(config["seat_swap"]), int(config["seed"]),
                config, None, None, failures, log,
            )
            # aggregate role's win rate vs baseline
            wins = sum(1 for r in b_records if r["winner_role"] == role)
            decided = sum(1 for r in b_records if r["winner_role"] != "draw")
            baseline_results[f"{role}_vs_baseline"] = {
                "games": len(b_records),
                "wins": wins,
                "win_rate": (wins / decided) if decided else 0.0,
            }
            baseline_records[role] = b_records

    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "git_commit": _git_commit(),
        "argv": sys.argv,
    }

    report = build_report(
        config, preflight_result, records, start_errors, ml_delta,
        baseline_results, failures, timestamp, output_dir,
    )
    write_artifacts(
        output_dir, config, report, records, baseline_records,
        trajectory_sink, replay_sink, environment, failures,
    )

    verdict = report["judgement"]["verdict"]
    matchup = report["head_to_head"]
    log("")
    log("== SUMMARY ==")
    log(f"command: {' '.join(sys.argv)}")
    log(f"games: {matchup['games']} (champion {matchup['champion_wins']} - "
        f"challenger {matchup['challenger_wins']} - draws {matchup['draws']})")
    log(f"challenger win rate: {matchup['challenger_win_rate']*100:.1f}% "
        f"(95% CI {matchup['challenger_win_rate_ci_low']*100:.1f}%-"
        f"{matchup['challenger_win_rate_ci_high']*100:.1f}%)")
    log(f"challenger attack rate: {report['challenger_metrics']['attack_turn_rate']*100:.1f}% | "
        f"alakazam attacks/game: {report['challenger_metrics']['alakazam_attacks_per_game']:.2f}")
    log(f"safety: crashes={report['challenger_metrics']['crashes']} "
        f"illegal={report['challenger_metrics']['illegal_actions']} "
        f"timeouts={report['challenger_metrics']['timeouts']}")
    if failures:
        log(f"failures: {len(failures)} (see failures.csv)")
    log(f"VERDICT: {verdict}")
    log(f"report: {output_dir}")
    log("NOTE: Champion NOT changed. Formal promotion requires scripts/promote_challenger.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
