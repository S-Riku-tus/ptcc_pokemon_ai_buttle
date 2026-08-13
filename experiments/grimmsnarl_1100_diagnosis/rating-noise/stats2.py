"""Part 2: heterogeneity, model-based opponent adjustment, within-run rating
volatility, and a simulation of the Kaggle rating process.

Answers the questions the bucket table cannot: is the between-run spread in win
rate larger than binomial noise (Cochran's Q), how much of the final-rating
spread mean-opponent-strength alone reproduces, and how far the rating itself
wanders inside one run at fixed true strength.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402
from scipy import stats as sps  # noqa: E402

from stats import invert, load, logistic_fit, spearman, pearson  # noqa: E402

RUNS_DIR = ROOT / "data" / "runs" / "grimmsnarl"


def main() -> int:
    rows = load()
    runs = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))
    meta = {r["label"]: r for r in runs}
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_run[r["label"]].append(r)
    order = sorted(by_run, key=lambda lbl: meta[lbl]["order"])
    out: dict[str, Any] = {}

    def rating(lbl: str) -> float:
        v = meta[lbl]["reported_rating"]
        return v if v is not None else meta[lbl]["last_updated_score"]

    # ---------- A. heterogeneity across the 28 runs ----------
    print("=" * 100)
    print("A. IS THERE ANY BETWEEN-RUN HETEROGENEITY AT ALL?")
    ks = [sum(r["won"] for r in by_run[l]) for l in order]
    ns = [len(by_run[l]) for l in order]
    N, K = sum(ns), sum(ks)
    p_bar = K / N
    # Cochran's Q / Pearson chi-square for homogeneity of proportions
    chi2 = sum((k - n * p_bar) ** 2 / (n * p_bar * (1 - p_bar))
               for k, n in zip(ks, ns))
    df = len(ns) - 1
    p_het = float(sps.chi2.sf(chi2, df))
    print(f"  28 runs, N={N}, pooled p={p_bar:.4f}")
    print(f"  chi-square homogeneity: chi2={chi2:.2f}, df={df}, p={p_het:.4f}")
    print(f"  I^2 = {max(0.0, (chi2 - df) / chi2) * 100:.1f}%")
    # observed vs binomial-expected SD of the per-run win rate
    obs_sd = math.sqrt(sum((k / n - p_bar) ** 2 for k, n in zip(ks, ns))
                       / (len(ns) - 1))
    exp_sd = math.sqrt(sum(p_bar * (1 - p_bar) / n for n in ns) / len(ns))
    print(f"  SD of per-run win rate: observed {obs_sd:.4f}, "
          f"binomial-expected {exp_sd:.4f}  (ratio {obs_sd / exp_sd:.3f})")
    # method-of-moments between-run variance (DerSimonian-Laird style)
    tau2 = max(0.0, (chi2 - df) / (N - sum(n * n for n in ns) / N)
               * p_bar * (1 - p_bar))
    print(f"  implied between-run SD of TRUE win rate (tau) = "
          f"{math.sqrt(tau2):.4f}  ({math.sqrt(tau2) * 100:.2f} pp)")
    out["heterogeneity"] = {"chi2": chi2, "df": df, "p": p_het,
                            "obs_sd": obs_sd, "exp_sd": exp_sd,
                            "ratio": obs_sd / exp_sd, "tau": math.sqrt(tau2),
                            "n": N, "pooled_p": p_bar}

    # ---------- B. model-based opponent adjustment ----------
    print()
    print("=" * 100)
    print("B. MODEL-BASED OPPONENT ADJUSTMENT (continuous logistic)")
    g = [r for r in rows if r["opp_initial"] is not None]
    m_opp = sum(r["opp_initial"] for r in g) / len(g)
    X = [[1.0, (r["opp_initial"] - m_opp) / 100.0] for r in g]
    y = [r["won"] for r in g]
    beta, se = logistic_fit(X, y)
    print(f"  N={len(g)} games with a known opponent rating "
          f"(v1/v2 episodes.csv carries no scores)")
    print(f"  win ~ 1 + (oppRating-{m_opp:.0f})/100: intercept "
          f"{beta[0]:+.4f} (se {se[0]:.4f}), slope {beta[1]:+.4f} "
          f"(se {se[1]:.4f}), z={beta[1] / se[1]:.2f}")
    print(f"  => each +100 opponent rating costs "
          f"{-(beta[1] * p_bar * (1 - p_bar)) * 100:.2f} pp of win rate "
          f"at p={p_bar:.3f}")

    def predict(opp: float) -> float:
        z = beta[0] + beta[1] * (opp - m_opp) / 100.0
        return 1.0 / (1.0 + math.exp(-z))

    print(f"\n{'ver':6} {'rating':>7} {'N':>4} {'meanOpp':>8} {'raw':>6} "
          f"{'expected':>8} {'excess':>7} {'adjWR':>6} {'wilsonAdj':>16}")
    adj = []
    for lbl in order:
        rs = [r for r in by_run[lbl] if r["opp_initial"] is not None]
        if not rs:
            continue
        n = len(rs)
        w = sum(r["won"] for r in rs)
        exp = sum(predict(r["opp_initial"]) for r in rs) / n
        raw = w / n
        excess = raw - exp
        adjwr = p_bar + excess
        lo, hi = wilson(w, n)
        adj.append({"label": lbl, "rating": rating(lbl), "n": n, "raw": raw,
                    "expected": exp, "excess": excess, "adj": adjwr,
                    "mean_opp": sum(r["opp_initial"] for r in rs) / n,
                    "wilson_adj": [lo - raw + adjwr, hi - raw + adjwr],
                    "lineage": meta[lbl]["lineage"]})
        print(f"{lbl:6} {rating(lbl):>7.1f} {n:>4} "
              f"{adj[-1]['mean_opp']:>8.1f} {raw:>6.3f} {exp:>8.3f} "
              f"{excess:>+7.3f} {adjwr:>6.3f} "
              f"[{adj[-1]['wilson_adj'][0]:.3f},{adj[-1]['wilson_adj'][1]:.3f}]")
    out["adjusted"] = adj
    out["opp_logit"] = {"beta": beta, "se": se, "mean_opp": m_opp}

    # heterogeneity of the EXCESS (opponent-adjusted) across runs
    chi2b = sum((a["raw"] - a["expected"]) ** 2 * a["n"]
                / (a["expected"] * (1 - a["expected"])) for a in adj)
    dfb = len(adj) - 1
    print(f"\n  opponent-adjusted homogeneity: chi2={chi2b:.2f}, df={dfb}, "
          f"p={float(sps.chi2.sf(chi2b, dfb)):.4f}")
    out["heterogeneity_adjusted"] = {
        "chi2": chi2b, "df": dfb, "p": float(sps.chi2.sf(chi2b, dfb))}

    # how much of the final-rating spread does the schedule alone reproduce?
    rr = [a["rating"] for a in adj]
    mo = [a["mean_opp"] for a in adj]
    ex = [a["excess"] for a in adj]
    raww = [a["raw"] for a in adj]
    r_mo, p_mo = pearson(mo, rr)
    r_ex, p_ex = pearson(ex, rr)
    r_raw, p_raw = pearson(raww, rr)
    print(f"\n  Pearson(mean opponent rating, final rating) r={r_mo:+.4f} "
          f"p={p_mo:.2e}  R^2={r_mo ** 2:.4f}   N={len(adj)}")
    print(f"  Pearson(raw win rate, final rating)          r={r_raw:+.4f} "
          f"p={p_raw:.4f}  R^2={r_raw ** 2:.4f}")
    print(f"  Pearson(opponent-adjusted excess, final rating) r={r_ex:+.4f} "
          f"p={p_ex:.4f}  R^2={r_ex ** 2:.4f}")
    print(f"  Spearman(mean opponent rating, final rating) = "
          f"{spearman(mo, rr)}")
    # two-predictor OLS: rating ~ meanOpp + rawWinRate
    Xo = [[1.0, a["mean_opp"] / 100.0, a["raw"]] for a in adj]
    yo = rr
    XtX = [[sum(Xo[i][a] * Xo[i][b] for i in range(len(Xo)))
            for b in range(3)] for a in range(3)]
    Xty = [sum(Xo[i][a] * yo[i] for i in range(len(Xo))) for a in range(3)]
    inv = invert(XtX)
    b_ols = [sum(inv[a][b] * Xty[b] for b in range(3)) for a in range(3)]
    fitted = [sum(b_ols[a] * Xo[i][a] for a in range(3)) for i in range(len(Xo))]
    ybar = sum(yo) / len(yo)
    ss_tot = sum((v - ybar) ** 2 for v in yo)
    ss_res = sum((v - f) ** 2 for v, f in zip(yo, fitted))
    print(f"  OLS rating ~ meanOpp/100 + rawWR : R^2={1 - ss_res / ss_tot:.4f}, "
          f"betas {[round(v, 2) for v in b_ols]}, resid SD "
          f"{math.sqrt(ss_res / (len(yo) - 3)):.1f}")
    out["rating_vs_schedule"] = {
        "pearson_meanopp": [r_mo, p_mo], "pearson_raw_wr": [r_raw, p_raw],
        "pearson_excess": [r_ex, p_ex],
        "ols_r2": 1 - ss_res / ss_tot, "ols_beta": b_ols,
        "ols_resid_sd": math.sqrt(ss_res / (len(yo) - 3)), "n": len(adj)}

    focus = {a["label"]: a for a in adj}
    print("\n  focus set:")
    for lbl in ("v4", "v8", "v15", "v18"):
        a = focus[lbl]
        print(f"   {lbl:5} rating {a['rating']:7.1f}  meanOpp {a['mean_opp']:7.1f}  "
              f"raw {a['raw']:.3f}  expected-if-average {a['expected']:.3f}  "
              f"excess {a['excess']:+.3f}  adj {a['adj']:.3f}")
    # pairwise adjusted comparisons within the focus set
    print("\n  focus-set pairwise on opponent-adjusted excess (z-test):")
    for i, la in enumerate(("v4", "v8", "v15", "v18")):
        for lb in ("v4", "v8", "v15", "v18")[i + 1:]:
            a, b = focus[la], focus[lb]
            sea = math.sqrt(a["expected"] * (1 - a["expected"]) / a["n"])
            seb = math.sqrt(b["expected"] * (1 - b["expected"]) / b["n"])
            d = a["excess"] - b["excess"]
            z = d / math.sqrt(sea ** 2 + seb ** 2)
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
            print(f"    {la:>4} vs {lb:<4}: d(excess)={d:+.3f} z={z:+.2f} "
                  f"p={p:.4f}   d(rating)={a['rating'] - b['rating']:+.1f}")

    # ---------- C. within-run rating volatility ----------
    print()
    print("=" * 100)
    print("C. WITHIN-RUN RATING VOLATILITY (same binary, same run, over time)")
    print(f"{'ver':6} {'games':>5} {'final':>7} {'min':>7} {'max':>7} "
          f"{'range':>6} {'sd(last half)':>13} {'peak-final':>10}")
    vols = []
    for lbl in order:
        run_dir = RUNS_DIR / meta[lbl]["run"]
        sub = meta[lbl]["sub_id"]
        traj: list[tuple[str, float]] = []
        with (run_dir / "episodes.csv").open(encoding="utf-8-sig",
                                             newline="") as h:
            for r in csv.DictReader(h):
                if "agent_0_updated_score" not in r:
                    continue
                seat = 0 if int(r["agent_0_submission_id"]) == sub else 1
                try:
                    traj.append((r["end_time"],
                                 float(r[f"agent_{seat}_updated_score"])))
                except (ValueError, KeyError, TypeError):
                    pass
        if len(traj) < 10:
            continue
        traj.sort()
        vals = [v for _, v in traj]
        half = vals[len(vals) // 2:]
        mu = sum(half) / len(half)
        sd = math.sqrt(sum((v - mu) ** 2 for v in half) / (len(half) - 1))
        vols.append({"label": lbl, "n": len(vals), "final": vals[-1],
                     "min": min(vals), "max": max(vals),
                     "range": max(vals) - min(vals), "sd_last_half": sd,
                     "peak_minus_final": max(vals) - vals[-1]})
        print(f"{lbl:6} {len(vals):>5} {vals[-1]:>7.1f} {min(vals):>7.1f} "
              f"{max(vals):>7.1f} {max(vals) - min(vals):>6.1f} {sd:>13.1f} "
              f"{max(vals) - vals[-1]:>10.1f}")
    if vols:
        mr = sum(v["range"] for v in vols) / len(vols)
        ms = sum(v["sd_last_half"] for v in vols) / len(vols)
        mp = sum(v["peak_minus_final"] for v in vols) / len(vols)
        print(f"  mean intra-run range {mr:.1f}, mean SD over the last half "
              f"{ms:.1f}, mean (peak - final) {mp:.1f}   N={len(vols)} runs")
        out["volatility"] = {"rows": vols, "mean_range": mr,
                             "mean_sd_last_half": ms,
                             "mean_peak_minus_final": mp, "n_runs": len(vols)}

    # ---------- D. simulate the Kaggle rating process ----------
    print()
    print("=" * 100)
    print("D. SIMULATED FINAL RATING AT FIXED TRUE STRENGTH")
    # fit the per-game rating step from the data: delta = updated - initial
    steps: list[tuple[int, float, float, int]] = []
    for lbl in order:
        run_dir = RUNS_DIR / meta[lbl]["run"]
        sub = meta[lbl]["sub_id"]
        with (run_dir / "episodes.csv").open(encoding="utf-8-sig",
                                             newline="") as h:
            recs = list(csv.DictReader(h))
        if not recs or "agent_0_updated_score" not in recs[0]:
            continue
        outcome = {r["episode_id"]: r["won"] for r in by_run[lbl]}
        recs.sort(key=lambda r: r["end_time"])
        for idx, r in enumerate(recs):
            seat = 0 if int(r["agent_0_submission_id"]) == sub else 1
            key = str(int(r["episode_id"]))
            try:
                mine0 = float(r[f"agent_{seat}_initial_score"])
                mine1 = float(r[f"agent_{seat}_updated_score"])
                opp0 = float(r[f"agent_{1 - seat}_initial_score"])
            except (ValueError, KeyError, TypeError):
                continue
            won = None
            for rr2 in by_run[lbl]:
                if rr2["episode_id"] == key:
                    won = rr2["won"]
                    break
            if won is None:
                continue
            steps.append((idx, mine1 - mine0, mine0 - opp0, won))
    wins = [s for s in steps if s[3] == 1]
    losses = [s for s in steps if s[3] == 0]
    print(f"  per-game rating step, N={len(steps)}: "
          f"win mean {sum(s[1] for s in wins) / len(wins):+.2f} "
          f"(sd {statsd([s[1] for s in wins]):.2f}), "
          f"loss mean {sum(s[1] for s in losses) / len(losses):+.2f} "
          f"(sd {statsd([s[1] for s in losses]):.2f})")
    # nonparametric bootstrap of the whole run: resample games with replacement
    random.seed(20260813)
    sims = []
    for _ in range(20000):
        r0 = 1000.0
        for _ in range(47):
            won = random.random() < p_bar
            pool = wins if won else losses
            r0 += random.choice(pool)[1]
        sims.append(r0)
    sims.sort()
    mu = sum(sims) / len(sims)
    sd = math.sqrt(sum((v - mu) ** 2 for v in sims) / (len(sims) - 1))
    print(f"  bootstrap of a 47-game run at the pooled true win rate "
          f"{p_bar:.3f}: final rating mean {mu:.1f}, SD {sd:.1f}, "
          f"5-95% [{sims[1000]:.0f},{sims[19000]:.0f}], "
          f"full range [{sims[0]:.0f},{sims[-1]:.0f}]")
    print(f"  => two identical binaries differ by more than "
          f"{1.96 * sd * math.sqrt(2):.0f} rating points 5% of the time")
    out["simulation"] = {
        "n_steps": len(steps),
        "win_step_mean": sum(s[1] for s in wins) / len(wins),
        "loss_step_mean": sum(s[1] for s in losses) / len(losses),
        "sim_sd": sd, "sim_mean": mu,
        "sim_p05_p95": [sims[1000], sims[19000]],
        "identical_pair_95pct_gap": 1.96 * sd * math.sqrt(2)}

    # ---------- E. how much of the observed spread is noise? ----------
    print()
    print("=" * 100)
    print("E. OBSERVED vs SIMULATED RATING SPREAD")
    all_r = [rating(l) for l in order]
    mu_r = sum(all_r) / len(all_r)
    sd_r = math.sqrt(sum((v - mu_r) ** 2 for v in all_r) / (len(all_r) - 1))
    print(f"  observed SD of the 28 final ratings = {sd_r:.1f}")
    print(f"  SD expected from pure noise at constant true strength = "
          f"{sd:.1f}")
    print(f"  => noise accounts for {min(1.0, sd ** 2 / sd_r ** 2) * 100:.0f}% "
          f"of the observed rating variance; residual true-skill SD = "
          f"{math.sqrt(max(0.0, sd_r ** 2 - sd ** 2)):.1f} rating points")
    out["spread"] = {"observed_sd": sd_r, "noise_sd": sd,
                     "noise_share": min(1.0, sd ** 2 / sd_r ** 2),
                     "true_sd": math.sqrt(max(0.0, sd_r ** 2 - sd ** 2))}

    (HERE / "stats2.json").write_text(json.dumps(out, indent=2, default=str),
                                      encoding="utf-8")
    return 0


def statsd(v: list[float]) -> float:
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


if __name__ == "__main__":
    raise SystemExit(main())
