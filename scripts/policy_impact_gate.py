"""Turn a teacher-forced policy footprint into a mandatory implementation gate.

The footprint is an upper-bound exposure measure: it counts stored decisions
where base and candidate answer differently while both are advanced with the
historical action.  It does not claim those changes improve win rate.  It does
answer the prior question that was repeatedly skipped: is this intervention
large enough to spend match-evaluation budget on at all?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def evaluate_impact(
    games: int,
    changed: int,
    games_touched: int | None = None,
    *,
    reject_below: float = 0.5,
    large_above: float = 2.0,
) -> dict[str, Any]:
    if games <= 0 or changed < 0:
        raise ValueError("games must be positive and changed must be non-negative")
    actions_per_game = changed / games
    if actions_per_game < reject_below:
        verdict = "REJECT_TOO_SMALL"
        required_paired_games = 0
    elif actions_per_game <= large_above:
        verdict = "MEASURE_WITH_2000_PAIRED_GAMES"
        required_paired_games = 2000
    else:
        verdict = "LARGE_ENOUGH_TO_IMPLEMENT"
        required_paired_games = 0
    return {
        "verdict": verdict,
        "games": games,
        "changed_actions": changed,
        "changed_actions_per_game": actions_per_game,
        "games_touched": games_touched,
        "games_touched_rate": (
            games_touched / games if games_touched is not None else None
        ),
        "thresholds": {
            "reject_below_actions_per_game": reject_below,
            "large_above_actions_per_game": large_above,
        },
        "required_paired_games": required_paired_games,
        "interpretation": (
            "Teacher-forced exposure only; passing this gate is necessary, not evidence of benefit."
        ),
    }


def _counts(payload: dict[str, Any], variant: str | None) -> tuple[int, int, int | None]:
    totals = payload.get("totals") or {}
    if "games" in totals and "changed" in totals:
        return int(totals["games"]), int(totals["changed"]), (
            int(totals["games_touched"]) if "games_touched" in totals else None
        )
    # Guard legal sweeps store per-version actual overrides and a top-level
    # game count.  This supports sizing a narrow guard independently from the
    # full candidate policy.
    if variant and variant in totals and "actual_overrides" in totals[variant]:
        games = payload.get("games")
        if isinstance(games, list):
            games = len(games)
        return int(games), int(totals[variant]["actual_overrides"]), None
    raise ValueError(
        "unsupported footprint: expected totals.games/changed or --variant with actual_overrides"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("footprint", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--reject-below", type=float, default=0.5)
    parser.add_argument("--large-above", type=float, default=2.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.footprint.read_text(encoding="utf-8"))
    games, changed, touched = _counts(payload, args.variant)
    result = evaluate_impact(
        games, changed, touched,
        reject_below=args.reject_below,
        large_above=args.large_above,
    )
    report = {
        "source": str(args.footprint),
        "variant": args.variant,
        **result,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if result["verdict"] == "REJECT_TOO_SMALL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

