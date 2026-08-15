"""Why is v27 at 853 when v22 sat at 1000-1020 and v24 at 911-928?

Reads the per-game table built by ``build_grimmsnarl_version_games.py`` and
answers the question in the order the evidence has to be taken:

1. *Is the rating even comparable?*  A submission's final number is
   ``mean(opponent rating) + 400*log10(w/(1-w))``, so a weak pairing draw
   lowers the number without lowering the policy.  Every version is reported
   as pooled win rate with a Wilson interval and as an implied strength.
2. *Is the run converged?*  Kaggle starts every submission at 600, so a short
   run is still climbing.  The per-game rating trajectory says whether the
   final number is an equilibrium or a waypoint.
3. *Is there a controlled version effect?*  Logistic fit of the outcome on
   opponent rating, turn order and a version dummy, v22 as the reference.
4. *Where would the difference be?*  v22's cell rates applied to v27's own
   exposure give an expected win count; the residual is decomposed by
   archetype family, turn order, opponent band, and by the four cells v27
   actually changes (exact mirror, wall families, deck-out, clock).
5. *Did anything fail at runtime?*  v26/v27 are the first versions to spend a
   real search budget, so the Kaggle overage clock and any non-standard step
   status are reported per version.

Nothing here promotes a version.  Every number is either a count or a fit
over counts, and the sample sizes are printed next to all of them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

ELO = 400.0 / math.log(10.0)
GROUPS: dict[str, tuple[str, ...]] = {
    "v22": ("v22_a", "v22_b", "v22_c", "v22_d"),
    "v24": ("v24_a", "v24_b"),
    "v25": ("v25_a", "v25_b"),
    "v26": ("v26",),
    "v27": ("v27",),
}
BANDS = ((0, 700), (700, 800), (800, 900), (900, 950), (950, 1050), (1050, 9999))


def band_name(rating: float) -> str:
    for low, high in BANDS:
        if low <= rating < high:
            return f"{low}-{high}" if high < 9999 else f">={low}"
    return "?"


def elo(rate: float) -> float:
    rate = min(max(rate, 1e-4), 1 - 1e-4)
    return 400 * math.log10(rate / (1 - rate))


def load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        converted: dict[str, Any] = dict(row)
        for key in (
            "opponent_rating", "our_rating_before", "our_rating_after",
            "our_overage_used", "opp_overage_used", "our_overage_start",
            "our_overage_end",
        ):
            converted[key] = float(row[key]) if row[key] not in ("", "None") else None
        for key in (
            "won", "exact_mirror", "wall_family", "turns", "our_turns",
            "our_prize_left", "opp_prize_left", "our_deck_left",
            "opp_deck_left", "our_bodies_left", "board_out", "deck_out",
            "gate_violation", "shadow_attacks", "attacks", "grim_evolutions",
            "rare_candies", "adrena_brains", "froslass_actions",
            "froslass_true_evolutions", "stamps", "bosses", "lillies",
            "our_decisions", "our_multi_pick", "our_ends", "steps",
            "first_ready_turn", "first_shadow_turn", "first_attack_turn",
            "opp_first_attack_turn", "own_first_shadow_turn",
            "own_first_ready_turn", "episode_id", "seat",
        ):
            converted[key] = int(row[key]) if row[key] not in ("", "None") else None
        converted["group"] = next(
            (name for name, labels in GROUPS.items() if row["version"] in labels),
            row["version"],
        )
        rows.append(converted)
    rows.sort(key=lambda item: item["create_time"])
    return rows


def rate_line(label: str, rows: Sequence[dict[str, Any]], width: int = 26) -> str:
    if not rows:
        return f"{label:<{width}} n=  0"
    n = len(rows)
    wins = sum(row["won"] for row in rows)
    low, high = wilson(wins, n)
    rated = [row for row in rows if row["opponent_rating"] is not None]
    opp = sum(row["opponent_rating"] for row in rated) / len(rated) if rated else 0.0
    strength = opp + elo(wins / n) if rated else float("nan")
    return (
        f"{label:<{width}} n={n:>3}  {wins:>3}-{n - wins:<3} "
        f"{wins / n:.3f} [{low:.3f},{high:.3f}]  opp {opp:6.1f}  "
        f"strength {strength:7.1f}"
    )


def mean(values: Iterable[Any]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    return sum(numbers) / len(numbers) if numbers else None


def fmt(value: float | None, digits: int = 3) -> str:
    return "  n/a" if value is None else f"{value:.{digits}f}"


def fit_dummy(
    rows: Sequence[dict[str, Any]], flag: Callable[[dict[str, Any]], float]
) -> dict[str, Any]:
    """Controlled effect of one binary flag on winning.

    Controls: opponent rating (in 400-point units, the natural Elo scale) and
    turn order.  Reported as an Elo-equivalent so it can be read against the
    77-point byte-identical noise floor.
    """
    X, y = [], []
    for row in rows:
        if row["opponent_rating"] is None or row["went_first"] not in ("first", "second"):
            continue
        X.append([
            row["opponent_rating"] / 400.0,
            1.0 if row["went_first"] == "first" else 0.0,
            float(flag(row)),
        ])
        y.append(row["won"])
    matrix = np.asarray(X, float)
    target = np.asarray(y, int)
    if len(target) < 12 or len(set(target.tolist())) < 2 or len(set(matrix[:, 2].tolist())) < 2:
        return {"n": int(len(target)), "error": "insufficient variation"}
    model = LogisticRegression(penalty=None, max_iter=8000).fit(matrix, target)
    probabilities = model.predict_proba(matrix)[:, 1]
    design = np.hstack([matrix, np.ones((len(matrix), 1))])
    try:
        covariance = np.linalg.inv(
            design.T @ np.diag(probabilities * (1 - probabilities)) @ design
        )
    except np.linalg.LinAlgError:
        return {"n": int(len(target)), "error": "singular covariance"}
    se = float(np.sqrt(np.diag(covariance))[2])
    coefficient = float(model.coef_[0][2])
    z = coefficient / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {"n": int(len(target)), "elo": round(coefficient * ELO, 1),
            "z": round(z, 2), "p": round(p, 4)}


def stratified_expectation(
    reference: Sequence[dict[str, Any]],
    target: Sequence[dict[str, Any]],
    key: Callable[[dict[str, Any]], Any],
    name: str,
) -> dict[str, Any]:
    """Wins the reference policy would take on the target's own exposure.

    Cells with no reference coverage fall back to the reference's pooled rate
    and are counted, because an uncovered cell is exactly the kind of blind
    spot that produced the v25 wall verdict.
    """
    pooled = sum(row["won"] for row in reference) / len(reference)
    cells: dict[Any, list[int]] = defaultdict(list)
    for row in reference:
        cells[key(row)].append(row["won"])
    expected = 0.0
    uncovered = 0
    detail = []
    for cell, rows in sorted(
        defaultdict(list, {
            c: [r for r in target if key(r) == c] for c in {key(r) for r in target}
        }).items(),
        key=lambda item: -len(item[1]),
    ):
        source = cells.get(cell)
        rate = sum(source) / len(source) if source else pooled
        if not source:
            uncovered += len(rows)
        expected += rate * len(rows)
        observed = sum(r["won"] for r in rows)
        detail.append({
            "cell": str(cell),
            "target_games": len(rows),
            "target_wins": observed,
            "reference_games": len(source) if source else 0,
            "reference_rate": round(rate, 4),
            "expected_wins": round(rate * len(rows), 2),
            "residual_wins": round(observed - rate * len(rows), 2),
        })
    observed_total = sum(row["won"] for row in target)
    variance = sum(
        d["reference_rate"] * (1 - d["reference_rate"]) * d["target_games"]
        for d in detail
    )
    z = (observed_total - expected) / math.sqrt(variance) if variance > 0 else None
    return {
        "stratifier": name,
        "target_games": len(target),
        "observed_wins": observed_total,
        "expected_wins": round(expected, 2),
        "residual_wins": round(observed_total - expected, 2),
        "z": round(z, 2) if z is not None else None,
        "games_in_uncovered_cells": uncovered,
        "cells": sorted(detail, key=lambda d: d["residual_wins"]),
    }


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games",
        type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/version_games.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/v27_vs_champions.json",
    )
    args = parser.parse_args()

    rows = load(args.games)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_version: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
        by_version[row["version"]].append(row)
    payload: dict[str, Any] = {}

    # ------------------------------------------------------------------ 1
    section("1. Run inventory: rating, pairing draw and implied strength")
    print("A final rating is opponent mean + Elo(win rate); the pairing draw "
          "moves it as much as the policy does.\n")
    inventory = []
    for version, group in ((v, by_version[v]) for v in sorted(by_version)):
        final = group[-1]["our_rating_after"]
        print(rate_line(f"{version} (final {final:.1f})", group, width=26))
        inventory.append({
            "version": version, "games": len(group),
            "wins": sum(r["won"] for r in group), "final_rating": final,
            "opponent_mean": round(mean(r["opponent_rating"] for r in group), 1),
        })
    print()
    for name in ("v22", "v24", "v25", "v26", "v27"):
        print(rate_line(f"{name} pooled", by_group[name], width=26))
    payload["inventory"] = inventory

    # ------------------------------------------------------------------ 2
    section("2. Convergence: is the final number an equilibrium or a waypoint?")
    print("Kaggle starts every submission at 600.  If the last games still "
          "carry the rating upward, the final number understates the policy.\n")
    convergence = []
    for version in sorted(by_version):
        group = by_version[version]
        rated = [r for r in group if r["opponent_rating"] is not None]
        implied = (
            mean(r["opponent_rating"] for r in rated)
            + elo(sum(r["won"] for r in rated) / len(rated))
        )
        final = group[-1]["our_rating_after"]
        tail = group[-10:]
        drift = (
            tail[-1]["our_rating_after"] - tail[0]["our_rating_before"]
            if len(tail) > 1 else None
        )
        first_half = group[: len(group) // 2]
        second_half = group[len(group) // 2:]
        print(
            f"{version:<6} final {final:7.1f}  implied {implied:7.1f}  "
            f"gap {final - implied:+7.1f}  last10 drift {drift:+7.1f}  "
            f"opp mean 1st/2nd half "
            f"{mean(r['opponent_rating'] for r in first_half):6.1f} / "
            f"{mean(r['opponent_rating'] for r in second_half):6.1f}  "
            f"wr 1st/2nd "
            f"{mean(r['won'] for r in first_half):.3f} / "
            f"{mean(r['won'] for r in second_half):.3f}"
        )
        convergence.append({
            "version": version, "final": final, "implied_strength": round(implied, 1),
            "gap": round(final - implied, 1), "last10_drift": drift,
        })
    payload["convergence"] = convergence

    # ------------------------------------------------------------------ 3
    section("3. Controlled version effect vs the v22 champion pool")
    print("Logistic: won ~ opponent_rating/400 + went_first + is_<version>.\n")
    effects = {}
    for name in ("v24", "v25", "v26", "v27"):
        pool = by_group["v22"] + by_group[name]
        result = fit_dummy(pool, lambda r, n=name: r["group"] == n)
        effects[name] = result
        print(f"is_{name:<4} vs v22 pool: {json.dumps(result)}")
    combined = by_group["v22"] + by_group["v24"] + by_group["v27"]
    print("\nthree-way pool (v22 + v24 + v27):")
    for name in ("v24", "v27"):
        print(f"  is_{name}: "
              f"{json.dumps(fit_dummy(combined, lambda r, n=name: r['group'] == n))}")
    payload["controlled_effects"] = effects

    # ------------------------------------------------------------------ 4
    section("4. What v22 would have scored on v27's own exposure")
    reference = by_group["v22"]
    target = by_group["v27"]
    strata = {
        "opponent band": lambda r: band_name(r["opponent_rating"] or 0),
        "family": lambda r: r["opponent_family"],
        "turn order": lambda r: r["went_first"],
        "band x turn order": lambda r: (
            f"{band_name(r['opponent_rating'] or 0)} | {r['went_first']}"
        ),
        "family x turn order": lambda r: f"{r['opponent_family']} | {r['went_first']}",
    }
    expectations = {}
    for name, key in strata.items():
        result = stratified_expectation(reference, target, key, name)
        expectations[name] = result
        print(
            f"{name:<20} observed {result['observed_wins']:>3} / "
            f"{result['target_games']:<3}  expected "
            f"{result['expected_wins']:>6}  residual "
            f"{result['residual_wins']:>+6}  z={result['z']}  "
            f"uncovered={result['games_in_uncovered_cells']}"
        )
    print("\nper-family residual (family x turn order stratification, worst first):")
    for cell in expectations["family"]["cells"]:
        print(
            f"  {cell['cell']:<28} v27 {cell['target_wins']}/"
            f"{cell['target_games']:<3} vs v22 rate {cell['reference_rate']:.3f} "
            f"(n={cell['reference_games']:<3}) residual {cell['residual_wins']:+.2f}"
        )
    payload["expectations"] = expectations

    # Same exercise against v24, which is the version the user remembers as
    # the stable one.
    print("\nand what v24 would have scored on v27's exposure:")
    for name, key in (("family", strata["family"]), ("opponent band", strata["opponent band"])):
        result = stratified_expectation(by_group["v24"], target, key, name)
        print(
            f"  {name:<16} observed {result['observed_wins']} expected "
            f"{result['expected_wins']} residual {result['residual_wins']:+} "
            f"z={result['z']}"
        )
        payload.setdefault("expectations_vs_v24", {})[name] = result

    # ------------------------------------------------------------------ 5
    section("5. Cell tables: family, turn order, opponent band")
    families = sorted(
        {row["opponent_family"] for row in rows},
        key=lambda name: -sum(1 for r in rows if r["opponent_family"] == name),
    )
    header = f"{'cell':<30}" + "".join(f"{g:>16}" for g in ("v22", "v24", "v27"))
    print(header)
    print("-" * len(header))

    def cell_text(group: str, predicate: Callable[[dict[str, Any]], bool]) -> str:
        subset = [r for r in by_group[group] if predicate(r)]
        if not subset:
            return f"{'-':>16}"
        wins = sum(r["won"] for r in subset)
        return f"{wins:>3}/{len(subset):<3}={wins / len(subset):.3f}".rjust(16)

    for fam in families:
        line = f"{fam:<30}"
        for group in ("v22", "v24", "v27"):
            line += cell_text(group, lambda r, f=fam: r["opponent_family"] == f)
        print(line)
    print()
    for order in ("first", "second"):
        line = f"{'turn order: ' + order:<30}"
        for group in ("v22", "v24", "v27"):
            line += cell_text(group, lambda r, o=order: r["went_first"] == o)
        print(line)
    print()
    for low, high in BANDS:
        label = f"{low}-{high}" if high < 9999 else f">={low}"
        line = f"{'opponent ' + label:<30}"
        for group in ("v22", "v24", "v27"):
            line += cell_text(
                group,
                lambda r, lo=low, hi=high: (
                    r["opponent_rating"] is not None and lo <= r["opponent_rating"] < hi
                ),
            )
        print(line)
    print()
    for label, predicate in (
        ("exact mirror", lambda r: r["exact_mirror"] == 1),
        ("wall families", lambda r: r["wall_family"] == 1),
        ("everything else", lambda r: not r["exact_mirror"] and not r["wall_family"]),
    ):
        line = f"{label:<30}"
        for group in ("v22", "v24", "v27"):
            line += cell_text(group, predicate)
        print(line)

    # ------------------------------------------------------------------ 6
    section("6. Runtime: search budget and step status")
    print("v26/v27 are the first versions to spend a real per-episode search "
          "budget.  ``remainingOverageTime`` is the whole clock here because "
          "actTimeout is 0.\n")
    runtime = {}
    for version in sorted(by_version):
        group = by_version[version]
        used = [r["our_overage_used"] for r in group if r["our_overage_used"] is not None]
        opp_used = [r["opp_overage_used"] for r in group if r["opp_overage_used"] is not None]
        odd = [r for r in group if r["odd_status"]]
        starts = {r["our_overage_start"] for r in group if r["our_overage_start"]}
        print(
            f"{version:<6} bank {sorted(starts)[:2]}  our used mean "
            f"{fmt(mean(used), 1):>8}  max {max(used) if used else 0:8.1f}  "
            f"opp used mean {fmt(mean(opp_used), 1):>8}  "
            f"per-decision {fmt((mean(used) or 0) / (mean(r['our_decisions'] for r in group) or 1), 3)}  "
            f"odd-status games {len(odd)}"
        )
        runtime[version] = {
            "our_overage_used_mean": mean(used),
            "our_overage_used_max": max(used) if used else None,
            "opp_overage_used_mean": mean(opp_used),
            "odd_status_games": len(odd),
            "odd_status": [r["odd_status"] for r in odd][:5],
        }
    payload["runtime"] = runtime
    exhausted = [r for r in rows if (r["our_overage_end"] or 1e9) < 30]
    print(f"\ngames finishing with < 30s of overage bank left: {len(exhausted)}")
    for row in exhausted[:10]:
        print(
            f"  {row['version']} ep {row['episode_id']} won={row['won']} "
            f"end={row['our_overage_end']:.1f} turns={row['turns']}"
        )
    v27_used = [r["our_overage_used"] for r in by_group["v27"] if r["our_overage_used"] is not None]
    v22_used = [r["our_overage_used"] for r in by_group["v22"] if r["our_overage_used"] is not None]
    if v27_used and v22_used:
        u = mannwhitneyu(v27_used, v22_used, alternative="greater")
        print(f"\nv27 vs v22 overage used, Mann-Whitney one-sided greater: "
              f"U={u.statistic:.0f} p={u.pvalue:.4g}")
    print("\nv27 time spent, wins vs losses:")
    for won in (1, 0):
        subset = [r["our_overage_used"] for r in by_group["v27"] if r["won"] == won]
        print(f"  won={won}: n={len(subset)} mean {fmt(mean(subset), 1)}")

    # ------------------------------------------------------------------ 7
    section("7. Behaviour counts per game (the levers earlier verdicts argued about)")
    metrics = (
        "turns", "our_turns", "our_decisions", "attacks", "shadow_attacks",
        "grim_evolutions", "rare_candies", "adrena_brains", "stamps", "bosses",
        "lillies", "froslass_actions", "froslass_true_evolutions",
        "own_first_ready_turn", "own_first_shadow_turn", "gate_violation",
        "our_multi_pick", "our_ends",
    )
    header = f"{'metric':<28}" + "".join(f"{g:>10}" for g in ("v22", "v24", "v25", "v26", "v27"))
    print(header)
    print("-" * len(header))
    behaviour = {}
    for metric in metrics:
        line = f"{metric:<28}"
        behaviour[metric] = {}
        for group in ("v22", "v24", "v25", "v26", "v27"):
            value = mean(r[metric] for r in by_group[group])
            behaviour[metric][group] = value
            line += f"{fmt(value, 2):>10}"
        print(line)
    payload["behaviour"] = behaviour

    print("\nsame table restricted to the exact mirror (where v27's search binds):")
    print(header)
    print("-" * len(header))
    for metric in metrics:
        line = f"{metric:<28}"
        for group in ("v22", "v24", "v25", "v26", "v27"):
            subset = [r for r in by_group[group] if r["exact_mirror"]]
            line += f"{fmt(mean(r[metric] for r in subset), 2):>10}"
        print(line)
    for group in ("v22", "v24", "v25", "v26", "v27"):
        subset = [r for r in by_group[group] if r["exact_mirror"]]
        wins = sum(r["won"] for r in subset)
        print(f"  {group} mirror games {len(subset)} record {wins}-{len(subset) - wins}")

    # ------------------------------------------------------------------ 8
    section("8. Loss anatomy")
    header = f"{'measure':<34}" + "".join(f"{g:>10}" for g in ("v22", "v24", "v25", "v26", "v27"))
    print(header)
    print("-" * len(header))
    anatomy: dict[str, Any] = {}
    measures: dict[str, Callable[[list[dict[str, Any]]], float | None]] = {
        "losses": lambda g: float(sum(1 for r in g if not r["won"])),
        "mean prizes left on loss": lambda g: mean(
            r["our_prize_left"] for r in g if not r["won"]
        ),
        "losses with >=5 prizes left": lambda g: float(sum(
            1 for r in g if not r["won"] and (r["our_prize_left"] or 0) >= 5
        )),
        "losses with 1 prize left": lambda g: float(sum(
            1 for r in g if not r["won"] and r["our_prize_left"] == 1
        )),
        "board-out losses": lambda g: float(sum(
            1 for r in g if not r["won"] and r["board_out"]
        )),
        "deck-out losses": lambda g: float(sum(
            1 for r in g if not r["won"] and r["deck_out"]
        )),
        "mean turns on loss": lambda g: mean(r["turns"] for r in g if not r["won"]),
        "mean turns on win": lambda g: mean(r["turns"] for r in g if r["won"]),
        "mean our deck left": lambda g: mean(r["our_deck_left"] for r in g),
        "games over 30 turns": lambda g: float(sum(1 for r in g if r["turns"] > 30)),
    }
    for name, func in measures.items():
        line = f"{name:<34}"
        anatomy[name] = {}
        for group in ("v22", "v24", "v25", "v26", "v27"):
            value = func(by_group[group])
            anatomy[name][group] = value
            line += f"{fmt(value, 2):>10}"
        print(line)
    payload["loss_anatomy"] = anatomy

    # ------------------------------------------------------------------ 9
    section("9. Opponent field: who v27 actually played")
    for group in ("v22", "v24", "v27"):
        subset = by_group[group]
        counts: dict[str, int] = defaultdict(int)
        for row in subset:
            counts[row["opponent_family"]] += 1
        share = ", ".join(
            f"{name} {count}({count / len(subset):.0%})"
            for name, count in sorted(counts.items(), key=lambda i: -i[1])
        )
        print(f"{group}: {share}")
    print()
    for group in ("v22", "v24", "v27"):
        subset = by_group[group]
        repeats: dict[str, int] = defaultdict(int)
        for row in subset:
            repeats[row["opponent_submission"]] += 1
        top = sorted(repeats.items(), key=lambda i: -i[1])[:5]
        print(f"{group}: {len(repeats)} distinct opponents, most repeated {top}")

    # ------------------------------------------------------------------ 10
    section("10. Fisher tests on the cells that matter")
    tests = {}
    for label, predicate in (
        ("overall", lambda r: True),
        ("opponent >= 900", lambda r: (r["opponent_rating"] or 0) >= 900),
        ("opponent < 900", lambda r: (r["opponent_rating"] or 0) < 900),
        ("exact mirror", lambda r: r["exact_mirror"] == 1),
        ("wall families", lambda r: r["wall_family"] == 1),
        ("going second", lambda r: r["went_first"] == "second"),
        ("going first", lambda r: r["went_first"] == "first"),
    ):
        a = [r for r in by_group["v22"] if predicate(r)]
        b = [r for r in by_group["v27"] if predicate(r)]
        if not a or not b:
            continue
        table = [
            [sum(r["won"] for r in b), len(b) - sum(r["won"] for r in b)],
            [sum(r["won"] for r in a), len(a) - sum(r["won"] for r in a)],
        ]
        p = float(fisher_exact(table).pvalue)
        tests[label] = {
            "v27": f"{table[0][0]}-{table[0][1]}",
            "v22": f"{table[1][0]}-{table[1][1]}",
            "fisher_p": round(p, 4),
        }
        print(
            f"{label:<20} v27 {table[0][0]}-{table[0][1]} "
            f"({table[0][0] / len(b):.3f})  vs  v22 {table[1][0]}-{table[1][1]} "
            f"({table[1][0] / len(a):.3f})   Fisher p={p:.4f}"
        )
    payload["fisher"] = tests

    # ----------------------------------------------------------------- 11
    section("11. Is the whole post-v22 lineage down, or only v27?")
    print("v25, v26 and v27 all inherit the same added modules.  One version "
          "at n=35 cannot separate policy from draw; the pooled lineage can.\n")
    lineage = by_group["v25"] + by_group["v26"] + by_group["v27"]
    pool = by_group["v22"] + lineage
    print(rate_line("v22 pooled", by_group["v22"], width=26))
    print(rate_line("v25+v26+v27 pooled", lineage, width=26))
    print("controlled is_lineage: "
          + json.dumps(fit_dummy(pool, lambda r: r["group"] != "v22")))
    print("controlled is_v24:     "
          + json.dumps(fit_dummy(
              by_group["v22"] + by_group["v24"], lambda r: r["group"] == "v24"
          )))
    table = [
        [sum(r["won"] for r in lineage), len(lineage) - sum(r["won"] for r in lineage)],
        [
            sum(r["won"] for r in by_group["v22"]),
            len(by_group["v22"]) - sum(r["won"] for r in by_group["v22"]),
        ],
    ]
    print(f"raw Fisher lineage vs v22: p={float(fisher_exact(table).pvalue):.4f}")
    strat = stratified_expectation(
        by_group["v22"], lineage,
        lambda r: f"{band_name(r['opponent_rating'] or 0)} | {r['went_first']}",
        "band x turn order",
    )
    print(
        f"v22's rates on the lineage's own exposure: observed "
        f"{strat['observed_wins']} expected {strat['expected_wins']} "
        f"residual {strat['residual_wins']:+} z={strat['z']}"
    )
    payload["lineage"] = {
        "games": len(lineage),
        "wins": sum(r["won"] for r in lineage),
        "controlled": fit_dummy(pool, lambda r: r["group"] != "v22"),
        "stratified": strat,
    }

    # ----------------------------------------------------------------- 12
    section("12. Turn order inside each version")
    print("v22 closed the going-second gap at n=194.  v27 shows the widest "
          "split of any version in this corpus, on 34 games.\n")
    turn_order = {}
    for group in ("v22", "v24", "v25", "v26", "v27"):
        subset = by_group[group]
        first = [r for r in subset if r["went_first"] == "first"]
        second = [r for r in subset if r["went_first"] == "second"]
        if not first or not second:
            continue
        table = [
            [sum(r["won"] for r in first), len(first) - sum(r["won"] for r in first)],
            [sum(r["won"] for r in second), len(second) - sum(r["won"] for r in second)],
        ]
        p = float(fisher_exact(table).pvalue)
        result = fit_dummy(subset, lambda r: r["went_first"] == "first")
        turn_order[group] = {
            "first": f"{table[0][0]}-{table[0][1]}",
            "second": f"{table[1][0]}-{table[1][1]}",
            "fisher_p": round(p, 4), "controlled": result,
        }
        print(
            f"{group:<5} first {table[0][0]:>3}-{table[0][1]:<3}"
            f"({table[0][0] / len(first):.3f})  second {table[1][0]:>3}-"
            f"{table[1][1] and table[1][1] or 0:<3}"
            f"({table[1][0] / len(second):.3f})  Fisher p={p:.4f}  "
            f"controlled {json.dumps(result)}"
        )
    payload["turn_order"] = turn_order
    print("\nv27 going second, game by game:")
    for row in by_group["v27"]:
        if row["went_first"] != "second":
            continue
        print(
            f"  ep {row['episode_id']} won={row['won']} "
            f"opp {row['opponent_rating']:.0f} {row['opponent_family']:<28} "
            f"turns={row['turns']:<3} our_prize_left={row['our_prize_left']} "
            f"opp_prize_left={row['opp_prize_left']} "
            f"shadow={row['shadow_attacks']} ready_t={row['own_first_ready_turn']}"
        )

    # ----------------------------------------------------------------- 13
    section("13. Every v27 loss")
    print(f"{'episode':<10}{'opp':>7} {'family':<26}{'ord':<7}{'turns':>6}"
          f"{'ourP':>5}{'oppP':>5}{'body':>5}{'atk':>4}{'shd':>4}"
          f"{'rdy':>4}{'ovr':>7}")
    for row in by_group["v27"]:
        if row["won"]:
            continue
        print(
            f"{row['episode_id']:<10}{(row['opponent_rating'] or 0):>7.0f} "
            f"{row['opponent_family']:<26}{row['went_first']:<7}"
            f"{row['turns']:>6}{row['our_prize_left']:>5}"
            f"{row['opp_prize_left']:>5}{row['our_bodies_left']:>5}"
            f"{row['attacks']:>4}{row['shadow_attacks']:>4}"
            f"{(row['own_first_ready_turn'] if row['own_first_ready_turn'] is not None else -1):>4}"
            f"{(row['our_overage_used'] or 0):>7.1f}"
        )
    print("\nand every v27 win, for contrast:")
    for row in by_group["v27"]:
        if not row["won"]:
            continue
        print(
            f"{row['episode_id']:<10}{(row['opponent_rating'] or 0):>7.0f} "
            f"{row['opponent_family']:<26}{row['went_first']:<7}"
            f"{row['turns']:>6}{row['our_prize_left']:>5}"
            f"{row['opp_prize_left']:>5}{row['our_bodies_left']:>5}"
            f"{row['attacks']:>4}{row['shadow_attacks']:>4}"
            f"{(row['own_first_ready_turn'] if row['own_first_ready_turn'] is not None else -1):>4}"
            f"{(row['our_overage_used'] or 0):>7.1f}"
        )

    # ----------------------------------------------------------------- 14
    section("14. Within-v27 gradients (what separates its own wins from losses)")
    print(f"{'metric':<28}{'win':>10}{'loss':>10}{'mwu p':>10}")
    wins = [r for r in by_group["v27"] if r["won"]]
    losses = [r for r in by_group["v27"] if not r["won"]]
    gradients = {}
    for metric in (
        "opponent_rating", "turns", "our_turns", "attacks", "shadow_attacks",
        "grim_evolutions", "rare_candies", "adrena_brains", "stamps", "bosses",
        "lillies", "froslass_actions", "own_first_ready_turn",
        "own_first_shadow_turn", "our_decisions", "our_overage_used",
        "our_deck_left",
    ):
        a = [float(r[metric]) for r in wins if r[metric] is not None]
        b = [float(r[metric]) for r in losses if r[metric] is not None]
        if len(a) < 4 or len(b) < 4:
            continue
        p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        gradients[metric] = {"win": mean(a), "loss": mean(b), "p": round(p, 4)}
        print(f"{metric:<28}{mean(a):>10.2f}{mean(b):>10.2f}{p:>10.4f}")
    payload["v27_gradients"] = gradients

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
