"""Did our policy get worse, or did a given opponent rating get stronger?

Section 11 of ``analyze_grimmsnarl_v27_vs_champions.py`` shows the whole
post-v22 lineage sitting ~147 Elo below v22 once opponent rating and turn
order are controlled.  That control assumes an 850-rated opponent on
2026-08-15 is the same thing as an 850-rated opponent on 2026-08-13.  Two days
is a long time on a live simulation ladder: every strong team that resubmits
re-enters at 600 and climbs back through exactly that band.

Three tests, in increasing strength:

1. *Per-run band rates against calendar time.*  If the decline is a field
   effect it should be monotone in time across all versions, not aligned with
   the version boundary.
2. *Time covariate.*  Refit the outcome on opponent rating, turn order and
   hours since the first stored game, then add the lineage dummy and see which
   survives.
3. *External underrating.*  For opponents we can map to a current top-60 team,
   ``team leaderboard score - rating at pairing`` measures how much the
   pairing rating understated the opponent.  This is independent of anything
   our agent did.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_field_freshness import submission_to_team  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

ELO = 400.0 / math.log(10.0)
GAMES = ROOT / "experiments/grimmsnarl_ml_v27/version_games.csv"
GROUPS = {
    "v22": ("v22_a", "v22_b", "v22_c", "v22_d"),
    "v23": ("v23",),
    "v24": ("v24_a", "v24_b"),
    "v25": ("v25_a", "v25_b"),
    "v26": ("v26",),
    "v27": ("v27",),
}
LINEAGE = {"v25", "v26", "v27"}


def parse_time(text: str) -> datetime | None:
    if not text:
        return None
    cleaned = text.replace("Z", "")[:26]
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def fit(
    rows: Sequence[dict[str, Any]],
    terms: dict[str, Callable[[dict[str, Any]], float]],
) -> dict[str, Any]:
    names = list(terms)
    X, y = [], []
    for row in rows:
        try:
            X.append([terms[name](row) for name in names])
        except (TypeError, ValueError):
            continue
        y.append(int(row["won"]))
    matrix = np.asarray(X, float)
    target = np.asarray(y, int)
    if len(target) < 20:
        return {"n": int(len(target)), "error": "too few rows"}
    model = LogisticRegression(penalty=None, max_iter=20000).fit(matrix, target)
    probabilities = model.predict_proba(matrix)[:, 1]
    design = np.hstack([matrix, np.ones((len(matrix), 1))])
    try:
        covariance = np.linalg.inv(
            design.T @ np.diag(probabilities * (1 - probabilities)) @ design
        )
    except np.linalg.LinAlgError:
        return {"n": int(len(target)), "error": "singular covariance"}
    se = np.sqrt(np.diag(covariance))
    out: dict[str, Any] = {"n": int(len(target))}
    for index, name in enumerate(names):
        coefficient = float(model.coef_[0][index])
        z = coefficient / float(se[index])
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        out[name] = {
            "elo": round(coefficient * ELO, 1), "z": round(z, 2), "p": round(p, 4)
        }
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/field_inflation.json",
    )
    args = parser.parse_args()

    mapping = submission_to_team()
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not raw["opponent_rating"] or raw["went_first"] not in ("first", "second"):
            continue
        moment = parse_time(raw["create_time"])
        if moment is None:
            continue
        try:
            info = mapping.get(int(raw["opponent_submission"]), {})
        except ValueError:
            info = {}
        group = next(
            (g for g, labels in GROUPS.items() if raw["version"] in labels),
            raw["version"],
        )
        rows.append({
            "version": raw["version"],
            "group": group,
            "time": moment,
            "won": int(raw["won"]),
            "opponent_rating": float(raw["opponent_rating"]),
            "went_first": 1.0 if raw["went_first"] == "first" else 0.0,
            "family": raw["opponent_family"],
            "team_score": info.get("team_score"),
        })
    rows.sort(key=lambda r: r["time"])
    origin = rows[0]["time"]
    for row in rows:
        row["hours"] = (row["time"] - origin).total_seconds() / 3600.0
    print(f"{len(rows)} rated games from {origin.isoformat()} to "
          f"{rows[-1]['time'].isoformat()} ({rows[-1]['hours']:.1f} h span)")

    payload: dict[str, Any] = {}

    # ------------------------------------------------------------------ 1
    print("\n=== 1. per-run rate in the 700-900 band against calendar time ===")
    print(f"{'run':<7}{'start':<18}{'700-900 n':>10}{'wr':>8}{'wilson':>18}"
          f"{'all n':>7}{'all wr':>8}")
    band_rows = []
    for version in sorted({r["version"] for r in rows}, key=lambda v: min(
        r["hours"] for r in rows if r["version"] == v
    )):
        subset = [r for r in rows if r["version"] == version]
        band = [r for r in subset if 700 <= r["opponent_rating"] < 900]
        wins = sum(r["won"] for r in band)
        low, high = wilson(wins, len(band)) if band else (0, 0)
        print(
            f"{version:<7}{subset[0]['time'].strftime('%m-%d %H:%M'):<18}"
            f"{len(band):>10}{(wins / len(band) if band else 0):>8.3f}"
            f"{f'[{low:.2f},{high:.2f}]':>18}"
            f"{len(subset):>7}{sum(r['won'] for r in subset) / len(subset):>8.3f}"
        )
        band_rows.append({
            "version": version, "band_games": len(band), "band_wins": wins,
            "start_hours": round(min(r["hours"] for r in subset), 2),
        })
    payload["band_by_run"] = band_rows

    early = [r for r in rows if r["hours"] < 24 and 700 <= r["opponent_rating"] < 900]
    late = [r for r in rows if r["hours"] >= 24 and 700 <= r["opponent_rating"] < 900]
    table = [
        [sum(r["won"] for r in late), len(late) - sum(r["won"] for r in late)],
        [sum(r["won"] for r in early), len(early) - sum(r["won"] for r in early)],
    ]
    print(f"\nfirst 24h  {table[1][0]}-{table[1][1]} "
          f"({table[1][0] / max(len(early), 1):.3f})   "
          f"after 24h {table[0][0]}-{table[0][1]} "
          f"({table[0][0] / max(len(late), 1):.3f})   "
          f"Fisher p={float(fisher_exact(table).pvalue):.4f}")

    # ------------------------------------------------------------------ 2
    print("\n=== 2. time covariate against the lineage dummy ===")
    base = {
        "opp_rating": lambda r: r["opponent_rating"] / 400.0,
        "went_first": lambda r: r["went_first"],
    }
    models = {
        "rating + order only": base,
        "+ lineage dummy": {
            **base, "is_lineage": lambda r: float(r["group"] in LINEAGE)
        },
        "+ hours": {**base, "hours_per_day": lambda r: r["hours"] / 24.0},
        "+ hours + lineage": {
            **base,
            "hours_per_day": lambda r: r["hours"] / 24.0,
            "is_lineage": lambda r: float(r["group"] in LINEAGE),
        },
        "+ hours + v24 + lineage": {
            **base,
            "hours_per_day": lambda r: r["hours"] / 24.0,
            "is_v24": lambda r: float(r["group"] == "v24"),
            "is_lineage": lambda r: float(r["group"] in LINEAGE),
        },
    }
    for name, terms in models.items():
        result = fit(rows, terms)
        payload.setdefault("models", {})[name] = result
        print(f"{name:<26} {json.dumps(result)}")

    print("\nsame, restricted to opponents rated 700-900 (the only band all "
          "versions share):")
    shared = [r for r in rows if 700 <= r["opponent_rating"] < 900]
    for name in ("+ lineage dummy", "+ hours", "+ hours + lineage"):
        result = fit(shared, models[name])
        payload.setdefault("models_700_900", {})[name] = result
        print(f"{name:<26} {json.dumps(result)}")

    # ------------------------------------------------------------------ 3
    print("\n=== 3. external underrating: team leaderboard score minus "
          "rating at pairing ===")
    print("Positive = the opponent was rated below what its team is currently "
          "worth, i.e. a strong agent early in its own climb.\n")
    print(f"{'group':<7}{'identified':>12}{'mean gap':>11}{'gap>100':>9}"
          f"{'our wr vs gap>100':>20}{'our wr vs gap<=0':>19}")
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        known = [r for r in subset if r["team_score"] is not None]
        if not known:
            continue
        gaps = [r["team_score"] - r["opponent_rating"] for r in known]
        high = [r for r in known if r["team_score"] - r["opponent_rating"] > 100]
        low = [r for r in known if r["team_score"] - r["opponent_rating"] <= 0]
        print(
            f"{group:<7}{len(known):>6}/{len(subset):<5}"
            f"{sum(gaps) / len(gaps):>11.1f}{len(high):>9}"
            f"{(sum(r['won'] for r in high) / len(high) if high else float('nan')):>20.3f}"
            f"{(sum(r['won'] for r in low) / len(low) if low else float('nan')):>19.3f}"
        )
    known_all = [r for r in rows if r["team_score"] is not None]
    print(f"\npooled over {len(known_all)} identified games, "
          f"outcome on the underrating gap:")
    print(json.dumps(fit(known_all, {
        "opp_rating": lambda r: r["opponent_rating"] / 400.0,
        "went_first": lambda r: r["went_first"],
        "underrating_per_400": lambda r: (
            (r["team_score"] - r["opponent_rating"]) / 400.0
        ),
    })))

    # ------------------------------------------------------------------ 4
    print("\n=== 4. matchup composition drift in the 700-900 band ===")
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group
                  and 700 <= r["opponent_rating"] < 900]
        if not subset:
            continue
        counts: dict[str, int] = defaultdict(int)
        for row in subset:
            counts[row["family"]] += 1
        top = ", ".join(
            f"{name.replace('other: ', '')} {count}"
            for name, count in sorted(counts.items(), key=lambda i: -i[1])[:6]
        )
        print(f"{group:<7} n={len(subset):<4} {top}")

    # ------------------------------------------------------------------ 5
    print("\n=== 5. the turn-order split, which is the one randomised "
          "comparison we have ===")
    print("Going first/second is a coin flip inside the episode, so within a "
          "single run the contrast is causally identified.  Whether the split "
          "*differs by version* is not randomised, so it is tested against "
          "calendar time as well.\n")
    print(f"{'run':<7}{'first':>12}{'second':>12}{'diff':>8}{'fisher p':>10}")
    for version in sorted({r["version"] for r in rows}, key=lambda v: min(
        r["hours"] for r in rows if r["version"] == v
    )):
        subset = [r for r in rows if r["version"] == version]
        first = [r for r in subset if r["went_first"] == 1.0]
        second = [r for r in subset if r["went_first"] == 0.0]
        if not first or not second:
            continue
        table = [
            [sum(r["won"] for r in first), len(first) - sum(r["won"] for r in first)],
            [sum(r["won"] for r in second), len(second) - sum(r["won"] for r in second)],
        ]
        a = table[0][0] / len(first)
        b = table[1][0] / len(second)
        print(
            f"{version:<7}{f'{table[0][0]}-{table[0][1]} {a:.3f}':>12}"
            f"{f'{table[1][0]}-{table[1][1]} {b:.3f}':>12}{a - b:>+8.3f}"
            f"{float(fisher_exact(table).pvalue):>10.4f}"
        )
    for label, predicate in (
        ("08-13 runs", lambda r: r["hours"] < 20),
        ("08-14 runs", lambda r: 20 <= r["hours"] < 38),
        ("08-15 runs", lambda r: r["hours"] >= 38),
        ("v26+v27", lambda r: r["group"] in {"v26", "v27"}),
        ("v22 only", lambda r: r["group"] == "v22"),
    ):
        subset = [r for r in rows if predicate(r)]
        first = [r for r in subset if r["went_first"] == 1.0]
        second = [r for r in subset if r["went_first"] == 0.0]
        if not first or not second:
            continue
        table = [
            [sum(r["won"] for r in first), len(first) - sum(r["won"] for r in first)],
            [sum(r["won"] for r in second), len(second) - sum(r["won"] for r in second)],
        ]
        print(
            f"{label:<12} first {table[0][0]:>3}-{table[0][1]:<3}"
            f"({table[0][0] / len(first):.3f})  second {table[1][0]:>3}-"
            f"{table[1][1]:<3}({table[1][0] / len(second):.3f})  "
            f"Fisher p={float(fisher_exact(table).pvalue):.4f}"
        )
    print("\ninteraction fits:")
    for name, terms in (
        ("order x hours", {
            "opp_rating": lambda r: r["opponent_rating"] / 400.0,
            "went_first": lambda r: r["went_first"],
            "hours_per_day": lambda r: r["hours"] / 24.0,
            "first_x_hours": lambda r: r["went_first"] * r["hours"] / 24.0,
        }),
        ("order x late-lineage", {
            "opp_rating": lambda r: r["opponent_rating"] / 400.0,
            "went_first": lambda r: r["went_first"],
            "is_v26_v27": lambda r: float(r["group"] in {"v26", "v27"}),
            "first_x_v26_v27": lambda r: (
                r["went_first"] * float(r["group"] in {"v26", "v27"})
            ),
        }),
    ):
        result = fit(rows, terms)
        payload.setdefault("turn_order", {})[name] = result
        print(f"  {name:<22} {json.dumps(result)}")

    # ------------------------------------------------------------------ 6
    print("\n=== 6. archetype share of the field over time ===")
    print("Ogerpon is the one family with no answer on this deck; its share "
          "is the tax we cannot play around.\n")
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        if not subset:
            continue
        counts: dict[str, int] = defaultdict(int)
        for row in subset:
            counts[row["family"]] += 1
        oger = counts.get("Ogerpon", 0)
        wall = oger + counts.get("Kangaskhan / Crustle", 0)
        wins_oger = sum(
            r["won"] for r in subset if r["family"] == "Ogerpon"
        )
        print(
            f"{group:<7} n={len(subset):<4} Ogerpon {oger:>3} "
            f"({oger / len(subset):>5.1%}, {wins_oger}-{oger - wins_oger})   "
            f"wall families {wall:>3} ({wall / len(subset):>5.1%})   "
            f"mirror {counts.get('Grimmsnarl (mirror)', 0):>3} "
            f"({counts.get('Grimmsnarl (mirror)', 0) / len(subset):>5.1%})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
