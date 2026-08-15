"""What separates a won game from a lost one, measured over every stored game.

v28 loses blowouts, not close games: in its 11 losses it holds 2-6 prizes while
the opponent sits on 1-2.  So the question is not "which endgame decision was
wrong" but "which measurable state precedes the blowout".  This script ranks
the candidate states by their controlled association with winning, over all
480 stored games, and then repeats each one inside v28 alone so a lever that
only exists in the pooled history is not read as a v28 lever.

Every fit controls for opponent rating and turn order.  Association is not
causation here - several past Grimmsnarl levers were confounds - so the output
is a ranked list of hypotheses with sample sizes, not a plan.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_grimmsnarl_v27_vs_champions as champ  # noqa: E402

champ.GROUPS["v28"] = ("v28",)

CANDIDATES: tuple[tuple[str, Callable[[dict[str, Any]], float | None]], ...] = (
    ("first Shadow by own turn 2", lambda r: _le(r["own_first_shadow_turn"], 2)),
    ("first Shadow by own turn 3", lambda r: _le(r["own_first_shadow_turn"], 3)),
    ("never landed a Shadow", lambda r: 1.0 if r["own_first_shadow_turn"] is None else 0.0),
    ("we attacked before they did", lambda r: _lt(r["first_attack_turn"], r["opp_first_attack_turn"])),
    ("opp attacked by shared turn 3", lambda r: _le(r["opp_first_attack_turn"], 3)),
    ("2+ Grimmsnarl ex evolved", lambda r: _ge(r["grim_evolutions"], 2)),
    ("3+ Grimmsnarl ex evolved", lambda r: _ge(r["grim_evolutions"], 3)),
    ("Rare Candy used", lambda r: _ge(r["rare_candies"], 1)),
    ("Froslass evolved", lambda r: _ge(r["froslass_true_evolutions"], 1)),
    ("Boss's Orders used", lambda r: _ge(r["bosses"], 1)),
    ("2+ Unfair Stamp", lambda r: _ge(r["stamps"], 2)),
    ("2+ Lillie's Determination", lambda r: _ge(r["lillies"], 2)),
    ("5+ Adrena-Brain", lambda r: _ge(r["adrena_brains"], 5)),
    ("zero Adrena-Brain", lambda r: 1.0 if r["adrena_brains"] == 0 else 0.0),
    ("gate violation", lambda r: float(r["gate_violation"])),
    ("went first", lambda r: 1.0 if r["went_first"] == "first" else 0.0),
)


def _le(value: Any, bound: int) -> float | None:
    return None if value is None else float(value <= bound)


def _lt(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left < right)


def _ge(value: Any, bound: int) -> float | None:
    return None if value is None else float(value >= bound)


def crosstab(
    rows: Sequence[dict[str, Any]], flag: Callable[[dict[str, Any]], float | None]
) -> tuple[int, int, int, int]:
    on_w = on_n = off_w = off_n = 0
    for row in rows:
        value = flag(row)
        if value is None:
            continue
        if value >= 0.5:
            on_n += 1
            on_w += row["won"]
        else:
            off_n += 1
            off_w += row["won"]
    return on_w, on_n, off_w, off_n


def line(label: str, rows: Sequence[dict[str, Any]], flag: Callable[..., float | None]) -> dict[str, Any]:
    on_w, on_n, off_w, off_n = crosstab(rows, flag)
    usable = [r for r in rows if flag(r) is not None]
    fit = champ.fit_dummy(usable, flag) if len(usable) >= 20 else {"n": len(usable)}
    on_rate = on_w / on_n if on_n else float("nan")
    off_rate = off_w / off_n if off_n else float("nan")
    print(
        f"  {label:<32} on {on_w:>3}/{on_n:<3} {on_rate:.3f}   "
        f"off {off_w:>3}/{off_n:<3} {off_rate:.3f}   "
        f"elo {str(fit.get('elo', '-')):>7}  p {str(fit.get('p', '-')):>6}"
    )
    return {
        "flag": label, "on": [on_w, on_n], "off": [off_w, off_n],
        "on_rate": None if not on_n else round(on_rate, 4),
        "off_rate": None if not off_n else round(off_rate, 4),
        **{k: v for k, v in fit.items() if k in ("elo", "p", "z", "n")},
    }


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
        default=ROOT / "experiments/grimmsnarl_ml_v28/levers.json",
    )
    args = parser.parse_args()

    rows = champ.load(args.games)
    target = [r for r in rows if r["group"] == args.target]
    report: dict[str, Any] = {}

    champ.section("1. Win/loss states, all 480 stored games")
    print("  elo = controlled effect of the flag on winning (opponent rating "
          "and turn order held fixed)")
    report["pooled"] = [line(label, rows, flag) for label, flag in CANDIDATES]

    champ.section(f"2. The same flags inside {args.target} alone")
    report["target"] = [line(label, target, flag) for label, flag in CANDIDATES]

    champ.section("3. How the games end")
    for label, subset in (
        ("all versions", rows),
        (args.target, target),
    ):
        wins = [r for r in subset if r["won"]]
        losses = [r for r in subset if not r["won"]]
        print(f"\n{label}: {len(wins)} wins / {len(losses)} losses")
        for name, group in (("wins", wins), ("losses", losses)):
            print(
                f"  {name:<7} our prizes left {champ.fmt(champ.mean(r['our_prize_left'] for r in group), 2)}"
                f"  opp prizes left {champ.fmt(champ.mean(r['opp_prize_left'] for r in group), 2)}"
                f"  our turns {champ.fmt(champ.mean(r['our_turns'] for r in group), 2)}"
                f"  first Shadow own turn {champ.fmt(champ.mean(r['own_first_shadow_turn'] for r in group), 2)}"
                f"  bodies left {champ.fmt(champ.mean(r['our_bodies_left'] for r in group), 2)}"
            )
        blowouts = [r for r in losses if (r["our_prize_left"] or 0) >= 4]
        close = [r for r in losses if (r["our_prize_left"] or 0) <= 2]
        print(f"  losses with 4+ of our prizes still unclaimed: "
              f"{len(blowouts)}/{len(losses)}"
              f"   losses within 2 prizes: {len(close)}/{len(losses)}")

    champ.section("4. Prize spread distribution")
    print("  spread = opponent prizes left - our prizes left (positive = we were ahead)")
    for label, subset in (("all versions", rows), (args.target, target)):
        spreads = [
            (r["opp_prize_left"] or 0) - (r["our_prize_left"] or 0) for r in subset
        ]
        spreads.sort()
        n = len(spreads)
        print(
            f"  {label:<14} n={n:<4} min {spreads[0]:>3}  p25 {spreads[n // 4]:>3}  "
            f"median {spreads[n // 2]:>3}  p75 {spreads[3 * n // 4]:>3}  "
            f"max {spreads[-1]:>3}  mean {sum(spreads) / n:+.2f}"
        )

    champ.section("5. First-Shadow turn against outcome, per family")
    families = sorted({r["opponent_family"] for r in rows if
                       len([x for x in rows if x["opponent_family"] == r["opponent_family"]]) >= 15})
    print(f"  {'family':<26}{'games':>6}{'win rate':>10}{'Shadow t (W)':>14}{'Shadow t (L)':>14}")
    family_report = []
    for family in families:
        subset = [r for r in rows if r["opponent_family"] == family]
        wins = [r for r in subset if r["won"]]
        losses = [r for r in subset if not r["won"]]
        shadow_w = champ.mean(r["own_first_shadow_turn"] for r in wins)
        shadow_l = champ.mean(r["own_first_shadow_turn"] for r in losses)
        print(
            f"  {family:<26}{len(subset):>6}{len(wins) / len(subset):>10.3f}"
            f"{champ.fmt(shadow_w, 2):>14}{champ.fmt(shadow_l, 2):>14}"
        )
        family_report.append({
            "family": family, "games": len(subset),
            "win_rate": round(len(wins) / len(subset), 4),
            "shadow_turn_wins": shadow_w, "shadow_turn_losses": shadow_l,
        })
    report["families"] = family_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
