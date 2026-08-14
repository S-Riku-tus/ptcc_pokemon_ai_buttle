"""What does 194 games of one policy actually say?

Every previous Grimmsnarl verdict in this repo was read off 33-58 games, a
sample where the measured same-code spread is 20 win-rate points.  v22 has now
played 194 games under a byte-identical build across four submissions, which is
the largest single-policy corpus this project has had.  That is enough to do
three things no single run can:

1. calibrate the noise floor from the inside - fit the four same-code runs
   against each other and read the implied Elo of a difference that is exactly
   zero;
2. test behavioural levers with opponent rating and turn order held fixed,
   instead of the raw split that made Froslass and Punk Up look like gradients;
3. rank matchups by how much of the deficit they own, with Wilson bounds wide
   enough to be honest about which ones are still unmeasured.

Nothing here reads a model file.  Every column is an observed fact from a
stored replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

ELO = 400.0 / math.log(10.0)  # logit -> Elo


def fnum(row: dict, key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    return float(value)


def block(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"games": 0}
    wins = sum(1 for r in rows if r["won"])
    low, high = wilson(wins, n)
    opp = [r["opponent_rating"] for r in rows if r["opponent_rating"] is not None]
    return {
        "games": n, "wins": wins, "losses": n - wins,
        "win_rate": round(wins / n, 4),
        "wilson95": [round(low, 4), round(high, 4)],
        "opp_mean": round(sum(opp) / len(opp), 1) if opp else None,
    }


def fit(rows: list[dict], extra: dict[str, Callable[[dict], float]]) -> dict[str, Any]:
    """Logistic fit of won on opponent rating, turn order and extra terms.

    Coefficients are reported in Elo so a behavioural term can be compared with
    the rating term on the scale the ladder actually pays out in.
    """
    usable = [r for r in rows if r["opponent_rating"] is not None and r["went_first"] is not None]
    names = ["opp_rating/400", "went_first"] + list(extra)
    X, y = [], []
    for r in usable:
        row = [r["opponent_rating"] / 400.0, 1.0 if r["went_first"] else 0.0]
        row += [fn(r) for fn in extra.values()]
        if any(v is None for v in row):
            continue
        X.append(row)
        y.append(1 if r["won"] else 0)
    X, y = np.asarray(X, float), np.asarray(y, int)
    if len(set(y.tolist())) < 2 or len(y) < 12:
        return {"n": len(y), "error": "insufficient variation"}
    model = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
    beta = model.coef_[0]
    # Observed-information standard errors; the fit is unregularised so this is
    # the usual MLE covariance.
    p = model.predict_proba(X)[:, 1]
    Xd = np.hstack([X, np.ones((len(X), 1))])
    W = np.diag(p * (1 - p))
    try:
        cov = np.linalg.inv(Xd.T @ W @ Xd)
        se = np.sqrt(np.diag(cov))[:-1]
    except np.linalg.LinAlgError:
        se = np.full(len(beta), float("nan"))
    out = {"n": int(len(y)), "wins": int(y.sum()), "terms": {}}
    for name, b, s in zip(names, beta, se):
        z = b / s if s and not math.isnan(s) else float("nan")
        out["terms"][name] = {
            "coef": round(float(b), 4),
            "elo": round(float(b) * ELO, 1),
            "se": round(float(s), 4) if not math.isnan(s) else None,
            "z": round(float(z), 2) if not math.isnan(z) else None,
            "p": round(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))), 4)
            if not math.isnan(z) else None,
        }
    return out


def gradient(rows: list[dict], name: str, fn: Callable[[dict], float | None],
             cuts: list[tuple[str, Callable[[float], bool]]]) -> dict[str, Any]:
    """Raw split plus the same split with rating and turn order held fixed."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        value = fn(r)
        if value is None:
            buckets["missing"].append(r)
            continue
        for label, test in cuts:
            if test(value):
                buckets[label].append(r)
                break
    controlled = fit(rows, {name: lambda r, fn=fn: fn(r)})
    return {
        "raw": {label: block(items) for label, items in buckets.items()},
        "controlled": controlled.get("terms", {}).get(name, controlled),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v23" / "ladder_v22_v23_games.csv")
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v23" / "pool_analysis.json")
    args = parser.parse_args()

    raw = list(csv.DictReader(args.games.open(encoding="utf-8-sig")))
    rows = []
    for r in raw:
        row = dict(r)
        row["won"] = r["won"] == "True"
        row["went_first"] = {"True": True, "False": False}.get(r["went_first"])
        for key in ("opponent_rating", "own_first_shadow_turn", "own_first_ready_turn",
                    "shadow_attacks", "attacks", "grim_evolutions", "rare_candies",
                    "adrena_brains", "stamps", "bosses", "froslass_evolves",
                    "our_turns", "turns", "our_prize_left", "opp_prize_left",
                    "our_bodies_left", "our_deck_left", "opp_deck_left",
                    "first_shadow_turn", "first_ready_turn", "opp_first_attack_turn"):
            row[key] = fnum(r, key)
        rows.append(row)

    v22 = [r for r in rows if r["version"].startswith("v22")]
    v23 = [r for r in rows if r["version"] == "v23"]

    report: dict[str, Any] = {"n_v22": len(v22), "n_v23": len(v23)}

    # --- 1. noise floor from four byte-identical runs -----------------------
    runs = sorted({r["version"] for r in v22})
    per_run = {v: block([r for r in v22 if r["version"] == v]) for v in runs}
    extra = {
        f"is_{v}": (lambda r, v=v: 1.0 if r["version"] == v else 0.0)
        for v in runs[1:]
    }
    report["same_code_runs"] = {
        "per_run": per_run,
        "pooled": block(v22),
        "controlled_fit": fit(v22, extra),
        "note": "all four runs are the same build; every is_* term is a zero "
                "effect by construction, so its Elo is the floor.",
    }

    # --- 2. opponent strength ----------------------------------------------
    def band(r: dict) -> str:
        v = r["opponent_rating"]
        if v is None:
            return "unknown"
        for edge in (700, 800, 900, 1000, 1100):
            if v < edge:
                return f"<{edge}"
        return ">=1100"
    bands: dict[str, list[dict]] = defaultdict(list)
    for r in v22:
        bands[band(r)].append(r)
    order = ["<700", "<800", "<900", "<1000", "<1100", ">=1100", "unknown"]
    report["by_opponent_band"] = {
        k: block(bands[k]) for k in order if k in bands
    }

    # --- 3. matchups --------------------------------------------------------
    fams: dict[str, list[dict]] = defaultdict(list)
    for r in v22:
        fams[r["opponent_family"]].append(r)
    report["by_family"] = {
        k: block(v) for k, v in sorted(fams.items(), key=lambda i: -len(i[1]))
    }
    # deficit share: how many games below the pooled rate each family costs
    pooled_rate = block(v22)["win_rate"]
    deficit = []
    for k, v in fams.items():
        b = block(v)
        deficit.append({
            "family": k, "games": b["games"], "win_rate": b["win_rate"],
            "games_lost_vs_pool": round((pooled_rate - b["win_rate"]) * b["games"], 2),
        })
    report["deficit_ranking"] = sorted(deficit, key=lambda d: -d["games_lost_vs_pool"])

    # --- 4. turn order ------------------------------------------------------
    report["turn_order"] = {
        "first": block([r for r in v22 if r["went_first"] is True]),
        "second": block([r for r in v22 if r["went_first"] is False]),
        "controlled": fit(v22, {}),
    }
    report["turn_order_by_family"] = {
        fam: {
            "first": block([r for r in v if r["went_first"] is True]),
            "second": block([r for r in v if r["went_first"] is False]),
        }
        for fam, v in sorted(fams.items(), key=lambda i: -len(i[1]))[:6]
    }

    # --- 5. behavioural levers, rating and turn order held fixed ------------
    levers: dict[str, Any] = {}
    levers["own_first_shadow_turn"] = gradient(
        v22, "own_first_shadow_turn", lambda r: r["own_first_shadow_turn"],
        [("<=2", lambda v: v <= 2), ("3", lambda v: v == 3), (">=4", lambda v: v >= 4)])
    levers["own_first_ready_turn"] = gradient(
        v22, "own_first_ready_turn", lambda r: r["own_first_ready_turn"],
        [("<=2", lambda v: v <= 2), ("3", lambda v: v == 3), (">=4", lambda v: v >= 4)])
    for name, lo, hi in (
        ("shadow_attacks", 3, 4), ("attacks", 4, 5), ("grim_evolutions", 1, 2),
        ("rare_candies", 1, 2), ("adrena_brains", 2, 3), ("stamps", 1, 2),
        ("bosses", 1, 2), ("froslass_evolves", 1, 2), ("our_turns", 6, 7),
    ):
        levers[name] = gradient(
            v22, name, lambda r, n=name: r[n],
            [(f"<{lo}", lambda v, lo=lo: v < lo), (f"{lo}-{hi-1}", lambda v, lo=lo, hi=hi: lo <= v < hi),
             (f">={hi}", lambda v, hi=hi: v >= hi)])
    report["levers"] = levers

    # --- 6. regression gates from prior versions ----------------------------
    def rate(sel: Callable[[dict], bool], rs: list[dict] | None = None) -> dict:
        return block([r for r in (rs if rs is not None else v22) if sel(r)])
    gate_viol = [r for r in v22 if r["first_ready_turn"] is not None
                 and r["first_shadow_turn"] is not None
                 and r["first_shadow_turn"] > r["first_ready_turn"] + 1]
    report["gates"] = {
        "v15_attack_access": {
            "violations": len(gate_viol),
            "violation_share": round(len(gate_viol) / len(v22), 4),
            "violation_block": block(gate_viol),
            "clean_block": rate(lambda r: r not in gate_viol),
        },
        "never_shadowed": block([r for r in v22 if r["first_shadow_turn"] is None]),
        "board_by_own_turn_2": {
            "yes": rate(lambda r: (r["own_first_ready_turn"] or 99) <= 2),
            "no": rate(lambda r: (r["own_first_ready_turn"] or 99) > 2),
        },
        "dead_stamp_proxy_zero_stamps": {
            "zero": rate(lambda r: r["stamps"] == 0),
            "some": rate(lambda r: (r["stamps"] or 0) > 0),
        },
    }

    # --- 7. loss anatomy ----------------------------------------------------
    losses = [r for r in v22 if not r["won"]]
    wins = [r for r in v22 if r["won"]]
    report["loss_anatomy"] = {
        "losses": len(losses),
        "prizes_taken_on_loss": dict(sorted(Counter(
            int(6 - (r["our_prize_left"] or 6)) for r in losses).items())),
        "blowouts_5plus_left": sum(1 for r in losses if (r["our_prize_left"] or 0) >= 5),
        "one_prize_from_winning": sum(1 for r in losses if r["our_prize_left"] == 1),
        "board_out": sum(1 for r in losses if (r["our_bodies_left"] or 0) == 0),
        "deck_out": sum(1 for r in losses if (r["our_deck_left"] or 99) == 0),
        "mean_turns_loss": round(sum(r["turns"] for r in losses) / len(losses), 2),
        "mean_turns_win": round(sum(r["turns"] for r in wins) / len(wins), 2),
        "mean_own_first_shadow_loss": round(sum(
            r["own_first_shadow_turn"] for r in losses
            if r["own_first_shadow_turn"] is not None) / max(1, sum(
                1 for r in losses if r["own_first_shadow_turn"] is not None)), 2),
        "mean_own_first_shadow_win": round(sum(
            r["own_first_shadow_turn"] for r in wins
            if r["own_first_shadow_turn"] is not None) / max(1, sum(
                1 for r in wins if r["own_first_shadow_turn"] is not None)), 2),
        "by_family": {
            fam: {
                "losses": sum(1 for r in v if not r["won"]),
                "blowouts": sum(1 for r in v if not r["won"] and (r["our_prize_left"] or 0) >= 5),
                "close": sum(1 for r in v if not r["won"] and (r["our_prize_left"] or 9) <= 2),
            }
            for fam, v in sorted(fams.items(), key=lambda i: -len(i[1]))[:8]
        },
    }

    # --- 8. v23's 12 games, stated for completeness only -------------------
    report["v23"] = {
        "overall": block(v23),
        "by_family": {k: block([r for r in v23 if r["opponent_family"] == k])
                      for k in sorted({r["opponent_family"] for r in v23})},
        "own_first_shadow_turn": [r["own_first_shadow_turn"] for r in v23],
        "note": "12 games; the run was truncated. Not comparable on outcomes.",
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nreport: {args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
