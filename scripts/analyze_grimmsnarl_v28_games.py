"""Per-game detail and per-own-turn rates for one version's ladder run.

Two things the pooled tables cannot show:

* a loss autopsy - how each game ended (prizes on both sides, board-out,
  deck-out, whether we ever attacked), because "lost 9" is not a cause;
* per-own-turn rates - every count column divided by our own turns.  The
  wall cell shortens games, so raw per-game counts fall even when the
  per-turn behaviour is unchanged; the per-turn denominator is the only one
  that compares cells of different length.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_grimmsnarl_v27_vs_champions as champ  # noqa: E402

champ.GROUPS["v28"] = ("v28",)

PER_TURN = (
    "attacks", "shadow_attacks", "adrena_brains", "grim_evolutions",
    "froslass_true_evolutions", "stamps", "bosses", "lillies", "rare_candies",
    "our_decisions",
)


def rate(rows: Sequence[dict[str, Any]], column: str) -> float | None:
    total = sum(r[column] for r in rows if r[column] is not None)
    turns = sum(r["our_turns"] for r in rows if r["our_turns"])
    return total / turns if turns else None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v28/version_games.csv",
    )
    parser.add_argument("--target", default="v28")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v28/game_detail.json",
    )
    args = parser.parse_args()

    rows = champ.load(args.games)
    target = [r for r in rows if r["group"] == args.target]

    champ.section(f"{args.target}: every game")
    print(
        f"{'ep':>9} {'opp':>7} {'seat':<6} {'res':<3} {'family':<24} "
        f"{'turns':>5} {'ourT':>4} {'prizeL':>6} {'oppPL':>5} {'deck':>4} "
        f"{'body':>4} {'atk':>3} {'shd':>3} {'AB':>3} {'flags'}"
    )
    for row in target:
        flags = []
        if row["board_out"]:
            flags.append("board_out")
        if row["deck_out"]:
            flags.append("deck_out")
        if row["gate_violation"]:
            flags.append("gate")
        if row["exact_mirror"]:
            flags.append("mirror")
        if row["wall_family"]:
            flags.append("wall")
        print(
            f"{row['episode_id']:>9} {row['opponent_rating'] or 0:7.1f} "
            f"{row['went_first']:<6} {'W' if row['won'] else 'L':<3} "
            f"{row['opponent_family']:<24} {row['turns']:>5} {row['our_turns']:>4} "
            f"{row['our_prize_left']:>6} {row['opp_prize_left']:>5} "
            f"{row['our_deck_left']:>4} {row['our_bodies_left']:>4} "
            f"{row['attacks']:>3} {row['shadow_attacks']:>3} "
            f"{row['adrena_brains']:>3} {','.join(flags)}"
        )

    champ.section("losses only: how the game ended")
    losses = [r for r in target if not r["won"]]
    for row in losses:
        print(
            f"  ep {row['episode_id']} vs {row['opponent_family']:<24} "
            f"opp {row['opponent_rating']:.0f} {row['went_first']:<6} "
            f"our prizes left {row['our_prize_left']} / opp {row['opp_prize_left']}  "
            f"turns {row['turns']} (ours {row['our_turns']})  "
            f"first Shadow own turn {row['own_first_shadow_turn']}  "
            f"attacks {row['attacks']}"
        )
    print(f"\nlosses where the opponent took all 6 prizes: "
          f"{sum(1 for r in losses if r['opp_prize_left'] == 0)}/{len(losses)}")
    print(f"losses where we were 1 prize from winning: "
          f"{sum(1 for r in losses if r['our_prize_left'] == 1)}/{len(losses)}")
    print(f"losses where we never attacked: "
          f"{sum(1 for r in losses if not r['attacks'])}/{len(losses)}")

    champ.section("per own turn, all versions")
    groups = [g for g in ("v22", "v24", "v25", "v26", "v27", "v28") if any(r["group"] == g for r in rows)]
    header = f"{'per own turn':<28}" + "".join(f"{g:>10}" for g in groups)
    print(header)
    print("-" * len(header))
    per_turn: dict[str, dict[str, float | None]] = {}
    for column in PER_TURN:
        line = f"{column:<28}"
        cells = {}
        for name in groups:
            value = rate([r for r in rows if r["group"] == name], column)
            line += f"{champ.fmt(value, 3):>10}"
            cells[name] = None if value is None else round(value, 4)
        print(line)
        per_turn[column] = cells

    for label, predicate in (
        ("wall families", lambda r: r["wall_family"]),
        ("exact mirror", lambda r: r["exact_mirror"]),
        ("race, non-mirror", lambda r: not r["wall_family"] and not r["exact_mirror"]),
    ):
        print(f"\nper own turn, {label}:")
        print(header)
        for column in PER_TURN:
            line = f"{column:<28}"
            for name in groups:
                subset = [r for r in rows if r["group"] == name and predicate(r)]
                line += f"{champ.fmt(rate(subset, column), 3):>10}"
            print(line)

    champ.section("game length and end state by cell")
    for label, predicate in (
        ("all", lambda r: True),
        ("wall", lambda r: r["wall_family"]),
        ("mirror", lambda r: r["exact_mirror"]),
        ("race", lambda r: not r["wall_family"] and not r["exact_mirror"]),
    ):
        print(f"\n{label}:")
        for name in groups:
            subset = [r for r in rows if r["group"] == name and predicate(r)]
            if not subset:
                continue
            print(
                f"  {name:<5} n={len(subset):<4} our turns "
                f"{champ.fmt(champ.mean(r['our_turns'] for r in subset), 2)}  "
                f"our prizes left {champ.fmt(champ.mean(r['our_prize_left'] for r in subset), 2)}  "
                f"opp prizes left {champ.fmt(champ.mean(r['opp_prize_left'] for r in subset), 2)}  "
                f"first Shadow own turn {champ.fmt(champ.mean(r['own_first_shadow_turn'] for r in subset), 2)}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "target": args.target,
                "games": [
                    {k: v for k, v in row.items() if k != "group"} for row in target
                ],
                "per_turn": per_turn,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
