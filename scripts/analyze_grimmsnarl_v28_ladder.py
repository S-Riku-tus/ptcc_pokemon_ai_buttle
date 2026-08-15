"""What is v28's 951.1 worth, and where did it come from?

Same evidence order as ``analyze_grimmsnarl_v27_vs_champions.py`` but with the
target version as an argument, because every future version needs this table:

1. pooled rate, Wilson interval and implied strength per version, so a lucky
   pairing draw cannot be read as a policy gain;
2. per-game rating trajectory, so a short run is not read as an equilibrium;
3. the >=950 opponent band on its own, which is the only band that moves the
   final number once the pairing slope is applied;
4. archetype cells, with the wall families called out because v28's whole
   thesis is the v22 wall ranker switch;
5. what the reference version would have scored on the target's exposure,
   stratified by family, band and seat;
6. behaviour columns, so a null result can be attributed to "the change never
   bound" rather than "the change bound and did nothing";
7. a logistic fit controlling for opponent rating and turn order;
8. a same-calendar-day restriction, because the field drifts about 100 Elo a
   day and every cross-day comparison is confounded by it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_grimmsnarl_v27_vs_champions as champ  # noqa: E402

champ.GROUPS["v28"] = ("v28",)

elo = champ.elo
fmt = champ.fmt
mean = champ.mean
rate_line = champ.rate_line
section = champ.section
wilson = champ.wilson

BEHAVIOUR = (
    ("own_first_shadow_turn", "first Shadow (own turn)"),
    ("own_first_ready_turn", "first ready (own turn)"),
    ("first_attack_turn", "first attack (shared turn)"),
    ("opp_first_attack_turn", "opp first attack"),
    ("our_turns", "our turns"),
    ("attacks", "attacks"),
    ("shadow_attacks", "Shadow Bullet attacks"),
    ("grim_evolutions", "Grimmsnarl ex evolutions"),
    ("rare_candies", "Rare Candy"),
    ("adrena_brains", "Adrena-Brain"),
    ("froslass_true_evolutions", "Froslass evolutions"),
    ("stamps", "Unfair Stamp"),
    ("bosses", "Boss's Orders"),
    ("lillies", "Lillie's Determination"),
    ("our_prize_left", "our prizes left"),
    ("opp_prize_left", "opp prizes left"),
    ("our_deck_left", "our deck left"),
    ("our_bodies_left", "our bodies left"),
    ("our_decisions", "decisions"),
    ("our_overage_used", "overage seconds used"),
)


def group_rows(rows: Sequence[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["group"] == name]


def day(row: dict[str, Any]) -> str:
    return str(row["create_time"])[:10]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v28/version_games.csv",
    )
    parser.add_argument("--target", default="v28")
    parser.add_argument("--reference", default="v22")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v28/ladder_verdict.json",
    )
    args = parser.parse_args()

    rows = champ.load(args.games)
    groups = [g for g in ("v22", "v24", "v25", "v26", "v27", "v28") if group_rows(rows, g)]
    target = group_rows(rows, args.target)
    reference = group_rows(rows, args.reference)
    report: dict[str, Any] = {"target": args.target, "reference": args.reference}

    section("1. Every version as a rate, not as a final number")
    print("strength = mean(opponent rating) + 400*log10(w/(1-w))\n")
    summary = []
    for name in groups:
        subset = group_rows(rows, name)
        print(rate_line(name, subset, width=8))
        wins = sum(r["won"] for r in subset)
        rated = [r for r in subset if r["opponent_rating"] is not None]
        opp = mean(r["opponent_rating"] for r in rated) or 0.0
        summary.append({
            "version": name, "games": len(subset), "wins": wins,
            "rate": round(wins / len(subset), 4),
            "opp_mean": round(opp, 1),
            "strength": round(opp + elo(wins / len(subset)), 1),
            "final_rating": subset[-1]["our_rating_after"],
            "days": sorted({day(r) for r in subset}),
        })
    report["versions"] = summary

    print("\nsame table, restricted to opponents rated >= 950:")
    for name in groups:
        subset = [
            r for r in group_rows(rows, name)
            if r["opponent_rating"] is not None and r["opponent_rating"] >= 950
        ]
        print(rate_line(name, subset, width=8))
    report["strong_band"] = {
        name: {
            "games": len([
                r for r in group_rows(rows, name)
                if r["opponent_rating"] is not None and r["opponent_rating"] >= 950
            ]),
            "wins": sum(
                r["won"] for r in group_rows(rows, name)
                if r["opponent_rating"] is not None and r["opponent_rating"] >= 950
            ),
        }
        for name in groups
    }

    section(f"2. Is {args.target} converged?")
    print("Kaggle starts every submission at 600, so early games are a climb.")
    print(f"{'#':>3} {'opp':>7} {'seat':<6} {'W/L':<4} {'rating after':>12}  opponent family")
    for index, row in enumerate(target, 1):
        print(
            f"{index:>3} {row['opponent_rating'] or 0:7.1f} {row['went_first']:<6} "
            f"{'W' if row['won'] else 'L':<4} {row['our_rating_after']:12.1f}  "
            f"{row['opponent_family']}"
        )
    half = len(target) // 2
    print("\nfirst half vs second half:")
    print(rate_line("  first half", target[:half], width=14))
    print(rate_line("  second half", target[half:], width=14))
    last10 = target[-10:]
    print(rate_line("  last 10", last10, width=14))
    report["trajectory"] = [
        {
            "n": i, "opponent_rating": r["opponent_rating"], "won": r["won"],
            "rating_after": r["our_rating_after"], "family": r["opponent_family"],
            "seat": r["went_first"],
        }
        for i, r in enumerate(target, 1)
    ]

    section("3. Opponent band")
    for name in groups:
        print(f"\n{name}:")
        subset = group_rows(rows, name)
        by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subset:
            if row["opponent_rating"] is not None:
                by_band[champ.band_name(row["opponent_rating"])].append(row)
        for band in ("0-700", "700-800", "800-900", "900-950", "950-1050", ">=1050"):
            if by_band.get(band):
                print(rate_line(f"  {band}", by_band[band], width=14))

    section("4. Archetype cells")
    families = sorted({row["opponent_family"] for row in rows})
    header = f"{'family':<26}" + "".join(f"{g:>14}" for g in groups)
    print(header)
    print("-" * len(header))
    family_table = {}
    for fam in families:
        line = f"{fam:<26}"
        cells = {}
        for name in groups:
            subset = [r for r in group_rows(rows, name) if r["opponent_family"] == fam]
            if subset:
                wins = sum(r["won"] for r in subset)
                line += f"{wins:>6}-{len(subset) - wins:<7}"
                cells[name] = [wins, len(subset)]
            else:
                line += f"{'-':>14}"
        print(line)
        family_table[fam] = cells
    report["families"] = family_table

    print("\nwall families (Grimmsnarl ex's Shadow Bullet can be blanked):")
    for name in groups:
        print(rate_line(f"  {name}", [r for r in group_rows(rows, name) if r["wall_family"]], width=10))
    print("\nnon-wall:")
    for name in groups:
        print(rate_line(f"  {name}", [r for r in group_rows(rows, name) if not r["wall_family"]], width=10))
    print("\nexact mirror (our own 60):")
    for name in groups:
        print(rate_line(f"  {name}", [r for r in group_rows(rows, name) if r["exact_mirror"]], width=10))

    section(f"5. What {args.reference} would have scored on {args.target}'s exposure")
    expectations = {}
    for key, name in (
        (lambda r: r["opponent_family"], "archetype family"),
        (lambda r: champ.band_name(r["opponent_rating"] or 0), "opponent band"),
        (lambda r: r["went_first"], "turn order"),
        (lambda r: "wall" if r["wall_family"] else "race", "wall vs race"),
    ):
        result = champ.stratified_expectation(reference, target, key, name)
        expectations[name] = result
        print(
            f"\nstratified by {name}: observed {result['observed_wins']} vs "
            f"expected {result['expected_wins']} "
            f"(residual {result['residual_wins']:+.2f}, z={result['z']}, "
            f"{result['games_in_uncovered_cells']} games in uncovered cells)"
        )
        for cell in result["cells"]:
            if cell["target_games"]:
                print(
                    f"  {cell['cell']:<28} {args.target} {cell['target_wins']}/"
                    f"{cell['target_games']:<3} vs {args.reference} rate "
                    f"{cell['reference_rate']:.3f} (n={cell['reference_games']:<3}) "
                    f"residual {cell['residual_wins']:+.2f}"
                )
    report["expectations"] = expectations

    section("6. Behaviour: did anything actually change on the board?")
    header = f"{'metric':<28}" + "".join(f"{g:>10}" for g in groups)
    print(header)
    print("-" * len(header))
    behaviour = {}
    for column, label in BEHAVIOUR:
        line = f"{label:<28}"
        cells = {}
        for name in groups:
            value = mean(r[column] for r in group_rows(rows, name))
            line += f"{fmt(value, 2):>10}"
            cells[name] = None if value is None else round(value, 3)
        print(line)
        behaviour[label] = cells
    report["behaviour"] = behaviour

    print("\nsame table restricted to wall families:")
    print(header)
    for column, label in BEHAVIOUR[:14]:
        line = f"{label:<28}"
        for name in groups:
            value = mean(
                r[column] for r in group_rows(rows, name) if r["wall_family"]
            )
            line += f"{fmt(value, 2):>10}"
        print(line)

    print("\ngate violations and board/deck-out counts:")
    for name in groups:
        subset = group_rows(rows, name)
        print(
            f"  {name:<5} gate_violation {sum(r['gate_violation'] for r in subset):>3}"
            f"  board_out {sum(r['board_out'] for r in subset):>3}"
            f"  deck_out {sum(r['deck_out'] for r in subset):>3}"
            f"  odd_status {sum(1 for r in subset if r['odd_status'] not in ('', 'None'))}"
        )

    section("7. Controlled effect vs the reference pool")
    for name in groups:
        if name == args.reference:
            continue
        pool = reference + group_rows(rows, name)
        result = champ.fit_dummy(pool, lambda r, n=name: 1.0 if r["group"] == n else 0.0)
        print(f"is_{name:<4} vs {args.reference} pool: {json.dumps(result)}")
        report.setdefault("logistic", {})[name] = result

    section("8. Turn order")
    for name in groups:
        subset = group_rows(rows, name)
        for seat in ("first", "second"):
            print(rate_line(
                f"  {name} {seat}", [r for r in subset if r["went_first"] == seat], width=16
            ))

    section("9. Calendar day: the field drifts, so only same-day rows compare")
    by_day: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_day[day(row)][row["group"]].append(row)
    for date in sorted(by_day):
        print(f"\n{date}:")
        for name in groups:
            subset = by_day[date].get(name)
            if subset:
                print(rate_line(f"  {name}", subset, width=10))

    target_days = sorted({day(r) for r in target})
    same_day = [r for r in rows if day(r) in target_days and r["group"] != args.target]
    print(f"\nall other versions on {target_days}:")
    print(rate_line("  others", same_day, width=10))
    print(rate_line(f"  {args.target}", target, width=10))

    section("10. Opponent field composition")
    for name in groups:
        subset = group_rows(rows, name)
        counts = Counter(r["opponent_family"] for r in subset)
        top = ", ".join(f"{k} {v}" for k, v in counts.most_common(6))
        distinct = len({r["opponent_submission"] for r in subset})
        print(f"  {name:<5} n={len(subset):<4} distinct opponents {distinct:<4} {top}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
