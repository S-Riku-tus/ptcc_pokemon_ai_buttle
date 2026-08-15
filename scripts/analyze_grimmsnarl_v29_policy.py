"""Evidence table for Grimmsnarl v29's matchup-conditioned ranker switch.

Only historical actions/outcomes are used.  The controlled comparison holds
opponent rating and turn order fixed, but remains observational and small-n;
it supports a bounded challenger, not a guaranteed Elo claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_grimmsnarl_v27_vs_champions as champ  # noqa: E402


champ.GROUPS["v28"] = ("v28",)
TARGETS = ("Mega Lopunny / Froslass", "other: Hydrapple ex")


def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    wins = sum(int(row["won"]) for row in rows)
    ratings = [float(row["opponent_rating"]) for row in rows]
    rate = wins / n if n else None
    mean_opp = sum(ratings) / len(ratings) if ratings else None
    strength = None
    if rate is not None and mean_opp is not None and 0 < rate < 1:
        strength = mean_opp + 400 * math.log10(rate / (1 - rate))
    return {
        "games": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": None if rate is None else round(rate, 4),
        "opponent_mean": None if mean_opp is None else round(mean_opp, 1),
        "implied_strength": None if strength is None else round(strength, 1),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v28/version_games.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v29/policy_evidence.json",
    )
    args = parser.parse_args()
    rows = champ.load(args.games)

    cells: dict[str, dict[str, Any]] = {}
    for family in (*TARGETS, "Ogerpon"):
        cells[family] = {}
        for group in ("v22", "v25", "v28"):
            subset = [
                row for row in rows
                if row["group"] == group and row["opponent_family"] == family
            ]
            cells[family][group] = block(subset)

    target_pool = [
        row for row in rows
        if row["group"] in {"v22", "v25"}
        and row["opponent_family"] in TARGETS
    ]
    controlled = champ.fit_dummy(
        target_pool, lambda row: float(row["group"] == "v22")
    )
    per_family_controlled = {
        family: champ.fit_dummy(
            [
                row for row in rows
                if row["group"] in {"v22", "v25"}
                and row["opponent_family"] == family
            ],
            lambda row: float(row["group"] == "v22"),
        )
        for family in TARGETS
    }

    payload = {
        "source_games": str(args.games),
        "target_families": list(TARGETS),
        "cells": cells,
        "combined": {
            group: block([
                row for row in rows
                if row["group"] == group
                and row["opponent_family"] in TARGETS
            ])
            for group in ("v22", "v25", "v28")
        },
        "controlled_v22_dummy_vs_v25": controlled,
        "controlled_by_family": per_family_controlled,
        "interpretation": {
            "v22_source": "1220-rated elite-teacher ranker",
            "v25_source": "1095-rated current AlphaTCG ranker",
            "selection_rule": (
                "use v22 only in target families and existing wall routes; "
                "retain v25 elsewhere"
            ),
            "caveat": (
                "observational, small cells; teacher-forced footprint does "
                "not reveal counterfactual game outcomes"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
