"""Does taking more Punk Up energy win games? The field's own answer.

``analyze_grimmsnarl_punk_budget`` established that v8's Punk Up search budget
is matchup-blind: against Alakazam the field pulls 3.01 energies where v8 pulls
2.66, while in every other matchup the two agree (2.70 against 2.75). That is a
real, matchup-specific divergence - but a divergence is only a lever if the
field's own winners are on the other side of it.

This asks that, with no model in the loop, so the whole archive is affordable
and the outcome and rating splits get their full sample:

* within the Alakazam cell, do winners take more than losers?
* does the count run with pilot rating? (a rank correlation with a real test,
  because "the elite do X" has been wrong on this deck before)
* is the elite band above or below v8's 2.66?

If the count neither tracks the outcome nor the rating, opening the tap is not
a fix and the Punk Up hypothesis closes here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8"))

import ml_features as mf  # noqa: E402

from analyze_grimmsnarl_alakazam_stage1 import (  # noqa: E402
    OUR_DECK_HASH, cohort_of, replay_meta,
)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta, continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    )
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _cf(a, b, x) / a
    return 1 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log(1 - x) + a * math.log(x)
    ) * _cf(b, a, 1 - x) / b


def _cf(a: float, b: float, x: float) -> float:
    tiny = 1e-30
    c, d = 1.0, 1 - (a + b) * x / (a + 1)
    d = tiny if abs(d) < tiny else d
    d, h = 1 / d, 1 / d
    for m in range(1, 300):
        m2 = 2 * m
        for numerator in (
            m * (b - m) * x / ((a + m2 - 1) * (a + m2)),
            -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1)),
        ):
            d = 1 + numerator * d
            d = tiny if abs(d) < tiny else d
            c = 1 + numerator / c
            c = tiny if abs(c) < tiny else c
            d = 1 / d
            h *= d * c
    return h


def welch(a: list[float], b: list[float]) -> dict[str, Any]:
    """Two-sample Welch t-test; the counts are 1-5 so normality is fine at n>30."""
    if len(a) < 3 or len(b) < 3:
        return {"t": None, "p": None}
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return {"t": None, "p": None}
    t = (statistics.fmean(a) - statistics.fmean(b)) / se
    df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1)
    )
    p = _betainc(df / 2, 0.5, df / (df + t * t))
    return {"t": round(t, 3), "df": round(df, 1), "p": round(p, 5)}


def spearman(xs: list[float], ys: list[float]) -> dict[str, Any]:
    n = len(xs)
    if n < 4:
        return {"rho": None, "p": None, "n": n}

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    if den == 0:
        return {"rho": None, "p": None, "n": n}
    rho = num / den
    if abs(rho) >= 1:
        return {"rho": round(rho, 4), "p": 0.0, "n": n}
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    df = n - 2
    return {
        "rho": round(rho, 4),
        "p": round(_betainc(df / 2, 0.5, df / (df + t * t)), 5),
        "n": n,
    }


def punk_rows(replay: dict[str, Any], seat: int) -> list[dict[str, Any]]:
    steps = replay.get("steps") or []
    out = []
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        effect = select.get("effect")
        if int(select.get("context", -1)) != mf.CTX_ATTACH_TO:
            continue
        if not isinstance(effect, dict):
            continue
        if mf._int(effect.get("id")) != mf.GRIMMSNARL_EX_ID:
            continue
        options = list(select.get("option") or [])
        action = (steps[index + 1][seat] or {}).get("action")
        if not options or not isinstance(action, list):
            continue
        current = observation.get("current") or {}
        out.append({
            "turn": int(current.get("turn", -1)),
            "offered": len(options),
            "max_count": int(select.get("maxCount") or 0),
            "taken": len(action),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument("--elite", type=float, default=1100.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ratings: dict[int, float] = {}
    for row in csv.DictReader(
        (args.data_root / "indexes" / "submissions.csv").open(
            encoding="utf-8-sig"
        )
    ):
        try:
            ratings[int(row["team_id"])] = float(row["submission_score"])
        except (KeyError, TypeError, ValueError):
            continue

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for raw in csv.DictReader(
        (args.data_root / "indexes" / "episodes.csv").open(encoding="utf-8-sig")
    ):
        if raw.get("download_status") != "success":
            continue
        if raw.get("deck_hash") != OUR_DECK_HASH:
            continue
        if raw.get("episode_type") != "EPISODE_TYPE_PUBLIC":
            continue
        episode_id, seat = int(raw["episode_id"]), int(raw["seat_index"])
        if (episode_id, seat) in seen:
            continue
        seen.add((episode_id, seat))
        path = args.data_root / "replays" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        meta = replay_meta(replay, seat)
        if meta is None:
            continue
        cohort = cohort_of(meta)
        team = int(raw["team_id"])
        for row in punk_rows(replay, seat):
            row.update({
                "cohort": cohort or "other_first",
                "alakazam": meta["opponent_family"] == "Alakazam",
                "won": meta["won"],
                "team": team,
                "rating": ratings.get(team),
            })
            rows.append(row)

    def counts(predicate) -> list[float]:
        return [float(r["taken"]) for r in rows if predicate(r)]

    report: dict[str, Any] = {
        "activations": len(rows),
        "elite_threshold": args.elite,
    }

    alak = counts(lambda r: r["alakazam"])
    other = counts(lambda r: not r["alakazam"])
    report["matchup"] = {
        "alakazam_mean": round(statistics.fmean(alak), 3),
        "alakazam_n": len(alak),
        "other_mean": round(statistics.fmean(other), 3),
        "other_n": len(other),
        "welch": welch(alak, other),
    }

    for label, predicate in (
        ("alakazam", lambda r: r["alakazam"]),
        ("alakazam_second", lambda r: r["cohort"] == "alakazam_second"),
        ("non_alakazam", lambda r: not r["alakazam"]),
    ):
        won = counts(lambda r: predicate(r) and r["won"])
        lost = counts(lambda r: predicate(r) and not r["won"])
        report[f"outcome_{label}"] = {
            "won_mean": round(statistics.fmean(won), 3) if won else None,
            "won_n": len(won),
            "lost_mean": round(statistics.fmean(lost), 3) if lost else None,
            "lost_n": len(lost),
            "welch": welch(won, lost),
        }

    for label, predicate in (
        ("alakazam", lambda r: r["alakazam"]),
        ("alakazam_second", lambda r: r["cohort"] == "alakazam_second"),
        ("non_alakazam", lambda r: not r["alakazam"]),
    ):
        per_pilot: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            if predicate(row) and row["rating"] is not None:
                per_pilot[row["team"]].append(float(row["taken"]))
        pilots = {t: v for t, v in per_pilot.items() if len(v) >= 5}
        report[f"rating_gradient_{label}"] = {
            "pilots": len(pilots),
            "spearman": spearman(
                [ratings[t] for t in pilots],
                [statistics.fmean(v) for v in pilots.values()],
            ),
            "per_pilot": {
                str(t): {
                    "rating": ratings[t], "n": len(pilots[t]),
                    "mean_taken": round(statistics.fmean(pilots[t]), 3),
                }
                for t in sorted(pilots, key=lambda t: -ratings[t])
            },
        }
        elite = [
            x for t, v in pilots.items() if ratings[t] >= args.elite for x in v
        ]
        rest = [
            x for t, v in pilots.items() if ratings[t] < args.elite for x in v
        ]
        report[f"band_{label}"] = {
            "elite_mean": round(statistics.fmean(elite), 3) if elite else None,
            "elite_n": len(elite),
            "rest_mean": round(statistics.fmean(rest), 3) if rest else None,
            "rest_n": len(rest),
            "welch": welch(elite, rest),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"activations {len(rows)}")
    print(f"matchup: alakazam {report['matchup']['alakazam_mean']} "
          f"(n={report['matchup']['alakazam_n']}) vs other "
          f"{report['matchup']['other_mean']} "
          f"(n={report['matchup']['other_n']})  {report['matchup']['welch']}")
    for key in (
        "outcome_alakazam", "outcome_alakazam_second", "outcome_non_alakazam",
    ):
        block = report[key]
        print(f"{key}: won {block['won_mean']} (n={block['won_n']}) vs lost "
              f"{block['lost_mean']} (n={block['lost_n']})  {block['welch']}")
    for key in (
        "rating_gradient_alakazam", "rating_gradient_alakazam_second",
        "rating_gradient_non_alakazam",
    ):
        print(f"{key}: {report[key]['spearman']}")
    for key in ("band_alakazam", "band_alakazam_second", "band_non_alakazam"):
        block = report[key]
        print(f"{key}: elite {block['elite_mean']} (n={block['elite_n']}) vs "
              f"rest {block['rest_mean']} (n={block['rest_n']})  "
              f"{block['welch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
