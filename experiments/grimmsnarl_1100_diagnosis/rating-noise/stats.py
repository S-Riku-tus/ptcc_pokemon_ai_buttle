"""Statistics over episodes_all.csv: run table, v19 noise yardstick, rating vs
win rate, opponent-mix adjustment, monotone trend, power.

Pure stdlib + scipy (scipy only for fisher_exact / distributions); everything
is recomputed from the per-episode rows produced by build_table.py.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

try:
    from scipy import stats as sps
except Exception:  # noqa: BLE001
    sps = None


def load() -> list[dict[str, Any]]:
    rows = []
    with (HERE / "episodes_all.csv").open(encoding="utf-8", newline="") as h:
        for r in csv.DictReader(h):
            r["won"] = int(r["won"])
            r["order"] = int(r["order"])
            r["lineage"] = float(r["lineage"])
            r["opp_initial"] = float(r["opp_initial"]) if r["opp_initial"] else None
            r["our_initial"] = float(r["our_initial"]) if r["our_initial"] else None
            r["reported_rating"] = (
                float(r["reported_rating"]) if r["reported_rating"] else None
            )
            r["went_first"] = (
                None if r["went_first"] == "" else r["went_first"] == "True"
            )
            rows.append(r)
    return rows


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a,b],[c,d]]."""
    if sps is not None:
        return float(sps.fisher_exact([[a, b], [c, d]])[1])
    # fallback: exact enumeration
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c

    def logc(nn: int, kk: int) -> float:
        return (math.lgamma(nn + 1) - math.lgamma(kk + 1)
                - math.lgamma(nn - kk + 1))

    def p(x: int) -> float:
        return math.exp(logc(r1, x) + logc(r2, c1 - x) - logc(n, c1))

    obs = p(a)
    total = 0.0
    for x in range(max(0, c1 - r2), min(r1, c1) + 1):
        px = p(x)
        if px <= obs * (1 + 1e-9):
            total += px
    return min(1.0, total)


def spearman(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if sps is not None:
        r = sps.spearmanr(xs, ys)
        return float(r.statistic), float(r.pvalue)
    return float("nan"), float("nan")


def pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if sps is not None:
        r = sps.pearsonr(xs, ys)
        return float(r.statistic), float(r.pvalue)
    return float("nan"), float("nan")


def logistic_fit(
    X: list[list[float]], y: list[int], iters: int = 400
) -> tuple[list[float], list[float]]:
    """Newton-Raphson logistic regression; returns (beta, se)."""
    k = len(X[0])
    beta = [0.0] * k
    for _ in range(iters):
        grad = [0.0] * k
        hess = [[0.0] * k for _ in range(k)]
        for xi, yi in zip(X, y):
            z = sum(b * v for b, v in zip(beta, xi))
            z = max(-30.0, min(30.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            w = p * (1 - p)
            for a in range(k):
                grad[a] += (yi - p) * xi[a]
                for b_ in range(k):
                    hess[a][b_] += w * xi[a] * xi[b_]
        step = solve(hess, grad)
        if step is None:
            break
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-10:
            break
    cov = invert(hess)
    se = [math.sqrt(cov[i][i]) if cov and cov[i][i] > 0 else float("nan")
          for i in range(k)]
    return beta, se


def solve(A: list[list[float]], b: list[float]) -> list[float] | None:
    inv = invert(A)
    if inv is None:
        return None
    return [sum(inv[i][j] * b[j] for j in range(len(b))) for i in range(len(b))]


def invert(A: list[list[float]]) -> list[list[float]] | None:
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
         for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if f:
                M[r] = [v - f * w for v, w in zip(M[r], M[col])]
    return [row[n:] for row in M]


BUCKETS = [
    ("<800", -1e9, 800.0),
    ("800-900", 800.0, 900.0),
    ("900-1000", 900.0, 1000.0),
    ("1000-1100", 1000.0, 1100.0),
    ("1100+", 1100.0, 1e9),
]


def bucket(score: float | None) -> str | None:
    if score is None:
        return None
    for name, lo, hi in BUCKETS:
        if lo <= score < hi:
            return name
    return None


def main() -> int:
    rows = load()
    runs = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_run[r["label"]].append(r)
    meta = {r["label"]: r for r in runs}
    order = sorted(by_run, key=lambda lbl: meta[lbl]["order"])

    out: dict[str, Any] = {}

    # ---------- 1. run table ----------
    print("=" * 100)
    print("1. RUN TABLE  (win/loss from replay rewards)")
    print(f"{'ver':6} {'sub':>9} {'rating':>7} {'src':>4} {'N':>4} {'W':>4} "
          f"{'L':>4} {'wr':>6} {'wilson95':>16} {'meanOpp':>8} {'oppN':>5}")
    table = []
    for lbl in order:
        rs = by_run[lbl]
        n = len(rs)
        w = sum(r["won"] for r in rs)
        rating = meta[lbl]["reported_rating"]
        src = "rep"
        if rating is None:
            rating = meta[lbl]["last_updated_score"]
            src = "csv"
        opps = [r["opp_initial"] for r in rs if r["opp_initial"] is not None]
        lo, hi = wilson(w, n)
        row = {
            "label": lbl, "sub": meta[lbl]["sub_id"], "rating": rating,
            "rating_src": src, "n": n, "wins": w, "losses": n - w,
            "win_rate": w / n, "wilson": [lo, hi],
            "mean_opp": (sum(opps) / len(opps)) if opps else None,
            "opp_n": len(opps), "lineage": meta[lbl]["lineage"],
            "order": meta[lbl]["order"],
        }
        table.append(row)
        print(f"{lbl:6} {row['sub']:>9} "
              f"{(rating if rating is not None else float('nan')):>7.1f} {src:>4} "
              f"{n:>4} {w:>4} {n - w:>4} {w / n:>6.3f} "
              f"[{lo:.3f},{hi:.3f}]".ljust(0)
              + f"  {(row['mean_opp'] or float('nan')):>8.1f} {len(opps):>5}")
    out["table"] = table

    total_n = sum(r["n"] for r in table)
    total_w = sum(r["wins"] for r in table)
    print(f"POOLED: N={total_n} W={total_w} wr={total_w / total_n:.4f} "
          f"wilson={wilson(total_w, total_n)}")
    out["pooled"] = {"n": total_n, "wins": total_w,
                     "win_rate": total_w / total_n,
                     "wilson": wilson(total_w, total_n)}

    # ---------- 1b. all-pairs distinguishability ----------
    print()
    print("=" * 100)
    print("1b. ALL-PAIRS FISHER (378 pairs, win/loss)")
    pairs = []
    for i in range(len(table)):
        for j in range(i + 1, len(table)):
            a, b = table[i], table[j]
            p = fisher(a["wins"], a["losses"], b["wins"], b["losses"])
            pairs.append({
                "a": a["label"], "b": b["label"], "p": p,
                "wr_a": a["win_rate"], "wr_b": b["win_rate"],
                "d_wr": a["win_rate"] - b["win_rate"],
                "d_rating": (a["rating"] - b["rating"])
                if a["rating"] is not None and b["rating"] is not None else None,
            })
    pairs.sort(key=lambda d: d["p"])
    sig = [p for p in pairs if p["p"] < 0.05]
    m = len(pairs)
    bonf = [p for p in pairs if p["p"] < 0.05 / m]
    # Benjamini-Hochberg
    bh = []
    for rank, p in enumerate(pairs, start=1):
        if p["p"] <= 0.05 * rank / m:
            bh = pairs[:rank]
    print(f"pairs={m}  raw p<0.05: {len(sig)}  (expected by chance "
          f"{0.05 * m:.1f})   Bonferroni: {len(bonf)}   BH-FDR 0.05: {len(bh)}")
    for p in pairs[:8]:
        print(f"  {p['a']:>5} vs {p['b']:<5} wr {p['wr_a']:.3f} vs "
              f"{p['wr_b']:.3f}  p={p['p']:.4f}  dRating="
              f"{(p['d_rating'] if p['d_rating'] is not None else float('nan')):+.1f}")
    out["pairs"] = {"n_pairs": m, "raw_sig": len(sig), "bonf_sig": len(bonf),
                    "bh_sig": len(bh), "expected_by_chance": 0.05 * m,
                    "top": pairs[:12]}

    # ---------- 2. v19 pair ----------
    print()
    print("=" * 100)
    print("2. v19 NOISE YARDSTICK (same binary, two submissions)")
    a = next(r for r in table if r["label"] == "v19a")
    b = next(r for r in table if r["label"] == "v19b")
    p19 = fisher(a["wins"], a["losses"], b["wins"], b["losses"])
    d_wr = a["win_rate"] - b["win_rate"]
    d_rating = a["rating"] - b["rating"]
    print(f"v19a {a['wins']}/{a['n']} = {a['win_rate']:.4f} {a['wilson']}  "
          f"rating {a['rating']}")
    print(f"v19b {b['wins']}/{b['n']} = {b['win_rate']:.4f} {b['wilson']}  "
          f"rating {b['rating']}")
    print(f"delta win rate = {d_wr:+.4f}   Fisher p = {p19:.4f}   "
          f"delta rating = {d_rating:+.1f}")
    out["v19"] = {"a": a, "b": b, "fisher_p": p19, "d_win_rate": d_wr,
                  "d_rating": d_rating}

    # rating noise SD implied by the pair (half-normal MLE of |d|)
    sd_diff_hn = abs(d_rating) * math.sqrt(math.pi / 2)
    sd_single = sd_diff_hn / math.sqrt(2)
    need_p05 = 1.959963985 * sd_diff_hn
    need_80pow = (1.959963985 + 0.8416212) * sd_diff_hn
    print(f"implied SD of the run-to-run rating difference (half-normal MLE from "
          f"one pair) = {sd_diff_hn:.1f}")
    print(f"implied SD of a single run's rating = {sd_single:.1f}")
    print(f"=> rating gap needed for p<0.05 between two DIFFERENT versions "
          f"(1 run each) = {need_p05:.0f}")
    print(f"=> rating gap needed for p<0.05 AND 80% power = {need_80pow:.0f}")
    out["v19_noise"] = {"sd_diff_halfnormal": sd_diff_hn,
                        "sd_single_run": sd_single,
                        "rating_gap_for_p05": need_p05,
                        "rating_gap_for_p05_80pow": need_80pow}

    # ---------- 2b. independent estimate of rating SD ----------
    # residual scatter of rating around the win-rate regression across runs
    have = [r for r in table if r["rating"] is not None]
    xs = [r["win_rate"] for r in have]
    ys = [r["rating"] for r in have]
    n = len(have)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sd_resid = math.sqrt(sum(r * r for r in resid) / (n - 2))
    sd_rating_raw = math.sqrt(sum((y - my) ** 2 for y in ys) / (n - 1))
    print(f"\nindependent check: SD of reported rating across {n} runs = "
          f"{sd_rating_raw:.1f}; residual SD after removing the win-rate "
          f"regression = {sd_resid:.1f} (slope {slope:.1f} rating-pts per unit "
          f"win rate, i.e. {slope / 100:.2f} per pp)")
    out["rating_sd"] = {"raw_sd": sd_rating_raw, "resid_sd": sd_resid,
                        "ols_slope_rating_per_winrate": slope,
                        "ols_intercept": intercept, "n": n}

    # ---------- 3. rating vs win rate ----------
    print()
    print("=" * 100)
    print("3. REPORTED RATING vs OBSERVED WIN RATE")
    rho, prho = spearman(xs, ys)
    pr, ppr = pearson(xs, ys)
    print(f"N={n} runs  Spearman rho={rho:+.4f} p={prho:.4f}   "
          f"Pearson r={pr:+.4f} p={ppr:.4f}  R^2={pr * pr:.4f}")
    # the same with the pooled-game view: does a run's rating predict a game win?
    print(f"win rate explains {pr * pr * 100:.1f}% of the rating variance; "
          f"{(1 - pr * pr) * 100:.1f}% is unexplained")
    out["rating_vs_winrate"] = {"n": n, "spearman_rho": rho, "spearman_p": prho,
                                "pearson_r": pr, "pearson_p": ppr,
                                "r2": pr * pr}

    # ---------- 4. opponent buckets ----------
    print()
    print("=" * 100)
    print("4. OPPONENT-RATING BUCKETS")
    names = [b[0] for b in BUCKETS]
    print(f"{'ver':6} {'rating':>7} {'N':>4} " + " ".join(f"{nm:>13}" for nm in names)
          + f" {'meanOpp':>8} {'share>=1000':>11}")
    bucket_rows = []
    pooled_bucket: dict[str, list[int]] = {nm: [0, 0] for nm in names}
    for r in table:
        rs = [x for x in by_run[r["label"]] if x["opp_initial"] is not None]
        cells = {}
        for nm in names:
            sub = [x for x in rs if bucket(x["opp_initial"]) == nm]
            w = sum(x["won"] for x in sub)
            cells[nm] = (w, len(sub))
            pooled_bucket[nm][0] += w
            pooled_bucket[nm][1] += len(sub)
        strong = sum(cells[nm][1] for nm in ("1000-1100", "1100+"))
        share = strong / len(rs) if rs else float("nan")
        bucket_rows.append({"label": r["label"], "rating": r["rating"],
                            "n": len(rs), "cells": cells,
                            "mean_opp": r["mean_opp"], "share_ge_1000": share})
        cellstr = " ".join(
            f"{cells[nm][0]:>3}/{cells[nm][1]:<3}"
            f"{(cells[nm][0] / cells[nm][1] if cells[nm][1] else float('nan')):>5.2f}"
            for nm in names)
        print(f"{r['label']:6} "
              f"{(r['rating'] if r['rating'] is not None else float('nan')):>7.1f} "
              f"{len(rs):>4} {cellstr} "
              f"{(r['mean_opp'] or float('nan')):>8.1f} {share:>11.3f}")
    print("POOLED " + " " * 12 + " ".join(
        f"{pooled_bucket[nm][0]:>3}/{pooled_bucket[nm][1]:<3}"
        f"{(pooled_bucket[nm][0] / pooled_bucket[nm][1] if pooled_bucket[nm][1] else float('nan')):>5.2f}"
        for nm in names))
    for nm in names:
        w, t = pooled_bucket[nm]
        if t:
            print(f"   bucket {nm:>10}: {w}/{t} = {w / t:.4f} wilson={wilson(w, t)}")
    out["buckets"] = {"per_run": bucket_rows,
                      "pooled": {nm: pooled_bucket[nm] for nm in names}}

    # ---------- 4b. direct standardisation ----------
    print()
    print("4b. DIRECT STANDARDISATION to the pooled opponent mix")
    mix = {nm: pooled_bucket[nm][1] for nm in names}
    mix_total = sum(mix.values())
    base = {nm: (pooled_bucket[nm][0] / pooled_bucket[nm][1])
            if pooled_bucket[nm][1] else None for nm in names}
    print(f"{'ver':6} {'rating':>7} {'raw wr':>7} {'expected wr':>11} "
          f"{'adj wr':>7} {'std wr':>7}")
    adj_rows = []
    for br in bucket_rows:
        n_r = br["n"]
        if n_r == 0:
            continue
        # expected win rate of the POOLED policy against THIS run's mix
        exp = sum(base[nm] * br["cells"][nm][1] for nm in names
                  if base[nm] is not None) / n_r
        raw = sum(br["cells"][nm][0] for nm in names) / n_r
        # direct standardisation: this run's bucket rates, pooled mix weights
        num = 0.0
        den = 0.0
        for nm in names:
            w, t = br["cells"][nm]
            if t:
                num += (w / t) * mix[nm]
                den += mix[nm]
        std = num / den if den else float("nan")
        adj = raw - exp  # excess over what the average of our runs would do
        adj_rows.append({"label": br["label"], "rating": br["rating"],
                         "raw": raw, "expected": exp, "adj": adj, "std": std,
                         "n": n_r, "coverage": den / mix_total})
        print(f"{br['label']:6} "
              f"{(br['rating'] if br['rating'] is not None else float('nan')):>7.1f} "
              f"{raw:>7.3f} {exp:>11.3f} {adj:>+7.3f} {std:>7.3f}")
    out["standardised"] = adj_rows

    # how much of the rating spread does the mix explain?
    focus = [r for r in adj_rows if r["label"] in ("v4", "v8", "v15", "v18")]
    print("\n  focus set v4 / v8 / v15 / v18:")
    for r in focus:
        print(f"   {r['label']:5} rating {r['rating']:.1f}  raw wr {r['raw']:.3f}  "
              f"opponent-difficulty expectation {r['expected']:.3f}  "
              f"standardised wr {r['std']:.3f}")
    rr = [r["rating"] for r in adj_rows if r["rating"] is not None]
    ra = [r["raw"] for r in adj_rows if r["rating"] is not None]
    re_ = [r["expected"] for r in adj_rows if r["rating"] is not None]
    rstd = [r["std"] for r in adj_rows if r["rating"] is not None]
    rmix = [r for r in bucket_rows
            if r["rating"] is not None and r["mean_opp"] is not None]
    print(f"  Spearman(rating, mean opponent rating) = "
          f"{spearman([r['mean_opp'] for r in rmix], [r['rating'] for r in rmix])}")
    print(f"  Spearman(rating, share of opponents >=1000) = "
          f"{spearman([r['share_ge_1000'] for r in rmix], [r['rating'] for r in rmix])}")
    print(f"  Spearman(rating, raw win rate)          = {spearman(ra, rr)}")
    print(f"  Spearman(rating, standardised win rate) = {spearman(rstd, rr)}")
    print(f"  Spearman(rating, opponent-difficulty expectation) = "
          f"{spearman(re_, rr)}")
    out["mix_correlations"] = {
        "rating_vs_mean_opp": spearman([r["mean_opp"] for r in rmix],
                                       [r["rating"] for r in rmix]),
        "rating_vs_share_ge1000": spearman([r["share_ge_1000"] for r in rmix],
                                           [r["rating"] for r in rmix]),
        "rating_vs_raw_wr": spearman(ra, rr),
        "rating_vs_std_wr": spearman(rstd, rr),
        "rating_vs_expected": spearman(re_, rr),
    }

    # ---------- 5. monotone trend ----------
    print()
    print("=" * 100)
    print("5. MONOTONE TREND v1 -> v21")
    lin = [meta[r["label"]]["lineage"] for r in table]
    raww = [r["win_rate"] for r in table]
    print(f"  Spearman(lineage, raw win rate)  N={len(lin)}: {spearman(lin, raww)}")
    lin2 = [meta[r["label"]]["lineage"] for r in adj_rows]
    print(f"  Spearman(lineage, standardised win rate) N={len(lin2)}: "
          f"{spearman(lin2, [r['std'] for r in adj_rows])}")
    lin3 = [meta[r["label"]]["lineage"] for r in table if r["rating"] is not None]
    print(f"  Spearman(lineage, reported rating) N={len(lin3)}: "
          f"{spearman(lin3, [r['rating'] for r in table if r['rating'] is not None])}")

    # per-game logistic: win ~ 1 + opp_rating(centred, /100) + lineage(centred)
    games = [r for r in rows if r["opp_initial"] is not None]
    mean_opp_all = sum(r["opp_initial"] for r in games) / len(games)
    mean_lin = sum(r["lineage"] for r in games) / len(games)
    X = [[1.0, (r["opp_initial"] - mean_opp_all) / 100.0,
          (r["lineage"] - mean_lin)] for r in games]
    y = [r["won"] for r in games]
    beta, se = logistic_fit(X, y)
    zs = [b / s if s == s and s > 0 else float("nan") for b, s in zip(beta, se)]
    pvals = [2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
             if z == z else float("nan") for z in zs]
    print(f"\n  per-game logistic (N={len(games)}): "
          f"win ~ 1 + opp_rating/100 + lineage")
    for nm, b_, s_, z_, p_ in zip(
            ["intercept", "opp_rating/100", "lineage"], beta, se, zs, pvals):
        print(f"    {nm:>16}: beta={b_:+.5f} se={s_:.5f} z={z_:+.2f} p={p_:.4f}")
    # cluster-robust SE by run (sandwich)
    k = len(beta)
    bread = [[0.0] * k for _ in range(k)]
    scores: dict[str, list[float]] = defaultdict(lambda: [0.0] * k)
    for xi, yi, r in zip(X, y, games):
        z = sum(b * v for b, v in zip(beta, xi))
        p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
        w = p * (1 - p)
        for aa in range(k):
            for bb in range(k):
                bread[aa][bb] += w * xi[aa] * xi[bb]
            scores[r["label"]][aa] += (yi - p) * xi[aa]
    binv = invert(bread)
    meat = [[0.0] * k for _ in range(k)]
    for s in scores.values():
        for aa in range(k):
            for bb in range(k):
                meat[aa][bb] += s[aa] * s[bb]
    cov = [[sum(binv[i][a] * meat[a][b] for a in range(k)) for b in range(k)]
           for i in range(k)]
    cov = [[sum(cov[i][a] * binv[a][j] for a in range(k)) for j in range(k)]
           for i in range(k)]
    cse = [math.sqrt(cov[i][i]) for i in range(k)]
    print(f"  cluster-robust (by run, G={len(scores)}) SEs: " +
          ", ".join(f"{nm}={s:.5f} (z={b / s:+.2f}, p="
                    f"{2 * (1 - 0.5 * (1 + math.erf(abs(b / s) / math.sqrt(2)))):.4f})"
                    for nm, b, s in zip(
                        ["intercept", "opp/100", "lineage"], beta, cse)))
    out["trend"] = {
        "spearman_lineage_raw_wr": spearman(lin, raww),
        "spearman_lineage_std_wr": spearman(lin2, [r["std"] for r in adj_rows]),
        "spearman_lineage_rating": spearman(
            lin3, [r["rating"] for r in table if r["rating"] is not None]),
        "logit_beta": beta, "logit_se": se, "logit_p": pvals,
        "logit_cluster_se": cse, "n_games": len(games),
        "mean_opp": mean_opp_all,
    }

    # early half vs late half
    early = [r for r in table if meta[r["label"]]["lineage"] <= 11]
    late = [r for r in table if meta[r["label"]]["lineage"] > 11]
    ew, en = sum(r["wins"] for r in early), sum(r["n"] for r in early)
    lw, ln_ = sum(r["wins"] for r in late), sum(r["n"] for r in late)
    print(f"\n  v1-v11 pooled: {ew}/{en} = {ew / en:.4f} {wilson(ew, en)}")
    print(f"  v12-v21 pooled: {lw}/{ln_} = {lw / ln_:.4f} {wilson(lw, ln_)}")
    print(f"  Fisher p = {fisher(ew, en - ew, lw, ln_ - lw):.4f}")
    out["halves"] = {"early": [ew, en], "late": [lw, ln_],
                     "fisher_p": fisher(ew, en - ew, lw, ln_ - lw)}

    # ---------- 6. rating <-> win rate scale, and power ----------
    print()
    print("=" * 100)
    print("6. RATING SCALE AND REQUIRED SAMPLE SIZE")
    # empirical: win prob vs (our rating - opp rating) per game
    g2 = [r for r in rows if r["opp_initial"] is not None
          and r["our_initial"] is not None]
    X2 = [[1.0, (r["our_initial"] - r["opp_initial"]) / 400.0] for r in g2]
    y2 = [r["won"] for r in g2]
    b2, s2 = logistic_fit(X2, y2)
    print(f"  per-game logistic win ~ 1 + (ourRating-oppRating)/400  N={len(g2)}: "
          f"intercept {b2[0]:+.4f} (se {s2[0]:.4f}), slope {b2[1]:+.4f} "
          f"(se {s2[1]:.4f})")
    # convert: d(win prob)/d(rating) at the observed base rate
    p0 = total_w / total_n
    dpr_400 = b2[1] * p0 * (1 - p0)          # per 400 rating points
    dpr_50 = dpr_400 * 50 / 400
    dpr_100 = dpr_400 * 100 / 400
    print(f"  observed base win rate p0={p0:.4f}; empirical marginal effect: "
          f"+50 rating = {dpr_50 * 100:+.2f} pp, +100 rating = "
          f"{dpr_100 * 100:+.2f} pp")
    # Elo reference
    elo50 = 1 / (1 + 10 ** (-50 / 400)) - 0.5
    elo100 = 1 / (1 + 10 ** (-100 / 400)) - 0.5
    print(f"  Elo-400 reference at p=0.5: +50 = {elo50 * 100:+.2f} pp, "
          f"+100 = {elo100 * 100:+.2f} pp")

    def n_per_arm(p1: float, p2: float, power: float = 0.80,
                  alpha: float = 0.05) -> float:
        za = 1.959963985 if alpha == 0.05 else 2.5758
        zb = 0.8416212 if power == 0.80 else 1.2815516
        return ((za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
                / (p1 - p2) ** 2)

    print("\n  two-proportion test, alpha=0.05 two-sided, 80% power, "
          "games PER ARM (each arm = one version):")
    for tag, dpr in (("empirical(this repo)", (dpr_50, dpr_100)),
                     ("Elo-400", (elo50, elo100))):
        for delta_rating, d in zip((50, 100), dpr):
            p1 = p0
            p2 = p0 + d
            n = n_per_arm(p1, p2)
            print(f"    {tag:>22}  +{delta_rating:<4} rating -> "
                  f"{p1:.3f} vs {p2:.3f}  n={n:,.0f} per arm "
                  f"({2 * n:,.0f} total)  [we run "
                  f"{total_n / len(table):.0f}/run]")
    out["power"] = {
        "logit_rating_slope_per400": b2[1], "logit_se": s2[1],
        "p0": p0,
        "empirical_pp_per_50": dpr_50 * 100,
        "empirical_pp_per_100": dpr_100 * 100,
        "elo_pp_per_50": elo50 * 100, "elo_pp_per_100": elo100 * 100,
        "n_per_arm": {
            "empirical_50": n_per_arm(p0, p0 + dpr_50),
            "empirical_100": n_per_arm(p0, p0 + dpr_100),
            "elo_50": n_per_arm(p0, p0 + elo50),
            "elo_100": n_per_arm(p0, p0 + elo100),
        },
        "mean_games_per_run": total_n / len(table),
    }

    # power actually achieved by a 50-game run
    print("\n  power of a typical 47-game-per-arm comparison:")
    for tag, d in (("+50 empirical", dpr_50), ("+100 empirical", dpr_100),
                   ("+50 Elo", elo50), ("+100 Elo", elo100)):
        p1, p2 = p0, p0 + d
        nn = 47
        se_d = math.sqrt(p1 * (1 - p1) / nn + p2 * (1 - p2) / nn)
        z = abs(p2 - p1) / se_d
        power = 0.5 * (1 + math.erf((z - 1.959963985) / math.sqrt(2)))
        print(f"    {tag:>16}: power = {power * 100:.1f}%")
        out.setdefault("power_at_47", {})[tag] = power

    (HERE / "stats.json").write_text(json.dumps(out, indent=2, default=str),
                                     encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
