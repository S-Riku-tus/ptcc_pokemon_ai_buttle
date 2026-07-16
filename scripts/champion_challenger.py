from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_self_play(champion: str, challenger: str, games: int, output_dir: Path) -> Path:
    self_play_dir = output_dir / "self_play"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "self_play.py"),
        challenger,
        champion,
        "--games",
        str(games),
        "--output-dir",
        str(self_play_dir),
        "--quiet",
    ]
    subprocess.run(command, check=True)
    return self_play_dir / "summary.json"


def _summarize(summary: dict[str, Any], challenger: str, champion: str) -> dict[str, Any]:
    matchups = summary.get("matchups") or []
    if not matchups:
        raise ValueError("self-play summary has no matchups")
    matchup = matchups[0]
    agent_a = matchup["agent_a"]
    if agent_a != challenger:
        raise ValueError(
            f"Expected challenger as agent_a for promotion math; got {agent_a!r}. "
            "Run self_play as challenger champion."
        )
    games = int(matchup.get("games", 0))
    wins = int(matchup.get("wins_a", 0))
    losses = int(matchup.get("wins_b", 0))
    draws = int(matchup.get("draws", 0))
    decided = max(1, wins + losses)
    return {
        "champion": champion,
        "challenger": challenger,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate_ex_draws": wins / decided,
        "crashes": int(matchup.get("errors_a", 0)),
        "illegal_actions": int(matchup.get("illegal_a", 0)),
        "timeouts": 0,
        "fallback_rate": float(matchup.get("fallback_rate_a", 0.0)),
        "ml_override_rate": None,
        "first_attack_turn": None,
        "attack_count": None,
        "deck_outs": None,
        "board_wipes": None,
        "average_decision_time_ms": (
            float(matchup.get("time_a_ms", 0.0)) / max(1, int(matchup.get("moves_a", 0)))
        ),
    }


def evaluate(metrics: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "minimum_games": metrics["games"] >= int(gates["minimum_games"]),
        "minimum_win_rate": metrics["win_rate_ex_draws"] >= float(gates["minimum_win_rate"]),
        "maximum_illegal_actions": metrics["illegal_actions"] <= int(gates["maximum_illegal_actions"]),
        "maximum_crashes": metrics["crashes"] <= int(gates["maximum_crashes"]),
        "maximum_timeouts": metrics["timeouts"] <= int(gates.get("maximum_timeouts", 0)),
    }
    return {
        "promote": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "gates": gates,
        "note": "This script never promotes automatically; review the report before copying any challenger model.",
    }


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "promotion_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics = result["metrics"]
    lines = [
        "# Champion-Challenger Promotion Report",
        "",
        f"- champion: `{metrics['champion']}`",
        f"- challenger: `{metrics['challenger']}`",
        f"- promote: `{result['promote']}`",
        f"- games: `{metrics['games']}`",
        f"- win_rate_ex_draws: `{metrics['win_rate_ex_draws']:.3f}`",
        f"- crashes: `{metrics['crashes']}`",
        f"- illegal_actions: `{metrics['illegal_actions']}`",
        f"- average_decision_time_ms: `{metrics['average_decision_time_ms']:.3f}`",
        "",
        "## Gate Checks",
        "",
    ]
    lines.extend(f"- {name}: `{passed}`" for name, passed in result["checks"].items())
    (output_dir / "promotion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", required=True)
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "ml" / "configs" / "champion_challenger.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "ml" / "champion_challenger")
    args = parser.parse_args()

    gates = _load_json(args.config)
    summary_path = args.summary_json or _run_self_play(
        args.champion,
        args.challenger,
        int(gates["minimum_games"]),
        args.output_dir,
    )
    metrics = _summarize(_load_json(summary_path), args.challenger, args.champion)
    result = evaluate(metrics, gates)
    write_report(result, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

