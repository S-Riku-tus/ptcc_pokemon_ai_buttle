"""Part 3: the rating process itself.

The naive bootstrap in stats2 (resample i.i.d. per-game steps) is wrong because
the step is mean-reverting - it depends on the rating gap to the opponent, and
the matchmaker feeds you opponents near your own rating.  Here every piece is
fitted from the data: the update rule delta = K * (S - E(gap)), the matchmaker
(opp - own) distribution, and our own win curve versus opponent rating.  Then a
full run is simulated from the same 600 start every Kaggle submission gets.
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

from scipy import stats as sps  # noqa: E402

from stats import load, logistic_fit, spearman, pearson  # noqa: E402

RUNS_DIR = ROOT / "data" / "runs" / "grimmsnarl"


def sd(v: list[float]) -> float:
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main() -> int:
    rows = load()
    runs = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))
    meta = {r["label"]: r for r in runs}
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_run[r["label"]].append(r)
    order = sorted(by_run, key=lambda lbl: meta[lbl]["order"])

    def rating(lbl: str) -> float:
        v = meta[lbl]["reported_rating"]
        return v if v is not None else meta[lbl]["last_updated_score"]

    out: dict[str, Any] = {}

    # ---------- per-game step records, in play order ----------
    recs: list[dict[str, Any]] = []
    traj: dict[str, list[float]] = {}
    for lbl in order:
        run_dir = RUNS_DIR / meta[lbl]["run"]
        sub = meta[lbl]["sub_id"]
        won_by_ep = {r["episode_id"]: r["won"] for r in by_run[lbl]}
        with (run_dir / "episodes.csv").open(encoding="utf-8-sig",
                                             newline="") as h:
            csv_rows = list(csv.DictReader(h))
        if not csv_rows or "agent_0_updated_score" not in csv_rows[0]:
            continue
        csv_rows.sort(key=lambda r: r["end_time"])
        path: list[float] = []
        for idx, r in enumerate(csv_rows):
            seat = 0 if int(r["agent_0_submission_id"]) == sub else 1
            try:
                mine0 = float(r[f"agent_{seat}_initial_score"])
                mine1 = float(r[f"agent_{seat}_updated_score"])
                opp0 = float(r[f"agent_{1 - seat}_initial_score"])
            except (ValueError, KeyError, TypeError):
                continue
            path.append(mine1)
            won = won_by_ep.get(str(int(r["episode_id"])))
            if won is None:
                continue
            recs.append({"label": lbl, "idx": idx, "n_run": len(csv_rows),
                         "own": mine0, "opp": opp0, "gap": mine0 - opp0,
                         "delta": mine1 - mine0, "won": won})
        traj[lbl] = path

    print("=" * 100)
    print("F. THE KAGGLE UPDATE RULE, FITTED")
    print(f"  usable step records: {len(recs)}")
    # every submission starts at 600
    starts = [t[0] for t in traj.values() if t]
    print(f"  first recorded rating per run: min {min(starts):.1f} "
          f"max {max(starts):.1f} (Kaggle seeds every submission at 600)")

    # fit K and the Elo scale: delta = K * (S - 1/(1+10^(-gap/scale)))
    best = None
    for scale in range(100, 1201, 10):
        num = den = 0.0
        for r in recs:
            e = 1.0 / (1.0 + 10 ** (-r["gap"] / scale))
            resid = r["won"] - e
            num += resid * r["delta"]
            den += resid * resid
        k = num / den
        ss = sum((r["delta"] - k * (r["won"]
                                    - 1.0 / (1.0 + 10 ** (-r["gap"] / scale))))
                 ** 2 for r in recs)
        if best is None or ss < best[2]:
            best = (scale, k, ss)
    scale, K, ss = best
    tot = sum((r["delta"] - sum(x["delta"] for x in recs) / len(recs)) ** 2
              for r in recs)
    print(f"  best fit: delta = K*(S - E), K={K:.2f}, Elo scale={scale} "
          f"(logistic base-10), R^2={1 - ss / tot:.4f}, "
          f"residual SD {math.sqrt(ss / len(recs)):.2f}")
    out["update_rule"] = {"K": K, "scale": scale, "r2": 1 - ss / tot,
                          "resid_sd": math.sqrt(ss / len(recs)),
                          "n": len(recs)}

    # does K decay with games played?
    for lo, hi in ((0, 10), (10, 25), (25, 50), (50, 200)):
        sub = [r for r in recs if lo <= r["idx"] < hi]
        if len(sub) < 30:
            continue
        num = den = 0.0
        for r in sub:
            e = 1.0 / (1.0 + 10 ** (-r["gap"] / scale))
            resid = r["won"] - e
            num += resid * r["delta"]
            den += resid * resid
        print(f"    games {lo:>3}-{hi:<3}: N={len(sub):>4} K={num / den:6.2f} "
              f"mean|delta|={sum(abs(r['delta']) for r in sub) / len(sub):5.2f}")

    # ---------- matchmaking ----------
    print()
    print("=" * 100)
    print("G. MATCHMAKING")
    gaps = [r["opp"] - r["own"] for r in recs]
    print(f"  opponent minus own rating: mean {sum(gaps) / len(gaps):+.1f}, "
          f"SD {sd(gaps):.1f}, "
          f"deciles {[round(float(x), 1) for x in sps.mstats.mquantiles(gaps, [0.1, 0.25, 0.5, 0.75, 0.9])]}")
    # does the matchmaker chase your rating?
    r_mm, p_mm = pearson([r["own"] for r in recs], [r["opp"] for r in recs])
    print(f"  Pearson(own rating at pairing, opponent rating) r={r_mm:+.4f} "
          f"p={p_mm:.2e}  N={len(recs)}")
    # residual: opp = a + b*own + eps
    xs = [r["own"] for r in recs]
    ys = [r["opp"] for r in recs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
         / sum((x - mx) ** 2 for x in xs))
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    print(f"  opp = {a:.1f} + {b:.4f}*own + eps, eps SD = {sd(resid):.1f}")
    out["matchmaking"] = {"mean_gap": sum(gaps) / len(gaps), "sd_gap": sd(gaps),
                          "pearson_own_opp": [r_mm, p_mm], "slope": b,
                          "intercept": a, "resid_sd": sd(resid)}

    # ---------- our win curve vs opponent rating ----------
    g = [r for r in rows if r["opp_initial"] is not None]
    m_opp = sum(r["opp_initial"] for r in g) / len(g)
    X = [[1.0, (r["opp_initial"] - m_opp) / 100.0] for r in g]
    beta, se = logistic_fit(X, [r["won"] for r in g])

    def p_win(opp: float) -> float:
        z = beta[0] + beta[1] * (opp - m_opp) / 100.0
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    # ---------- H. generative simulation of a run ----------
    print()
    print("=" * 100)
    print("H. SIMULATED LADDER RUN (all components fitted above)")

    def sim_run(n_games: int, skill_shift: float = 0.0,
                seed: int | None = None) -> float:
        """Return the final rating of one simulated run.

        ``skill_shift`` is added to the log-odds of every game, so a positive
        value is a genuinely stronger agent.
        """
        rng = random.Random(seed)
        r0 = 600.0
        for _ in range(n_games):
            opp = a + b * r0 + rng.gauss(0.0, sd(resid))
            p = p_win(opp)
            z = math.log(p / (1 - p)) + skill_shift
            p = 1.0 / (1.0 + math.exp(-z))
            won = 1 if rng.random() < p else 0
            e = 1.0 / (1.0 + 10 ** (-(r0 - opp) / scale))
            r0 += K * (won - e)
        return r0

    random.seed(20260813)
    resid_sd = sd(resid)
    for n_games in (34, 47, 96, 200, 500, 1000):
        vals = sorted(sim_run(n_games, 0.0, seed=None) for _ in range(4000))
        m = sum(vals) / len(vals)
        print(f"  n={n_games:>4} games: mean {m:7.1f}  SD {sd(vals):6.1f}  "
              f"5-95% [{vals[200]:.0f},{vals[3800]:.0f}]")
        out.setdefault("sim_by_n", {})[str(n_games)] = {
            "mean": m, "sd": sd(vals), "p05": vals[200], "p95": vals[3800]}

    # calibration: does the simulation reproduce the observed spread?
    obs_r = [rating(l) for l in order]
    obs_n = [len(by_run[l]) for l in order]
    sims = []
    for nn in obs_n:
        sims.append(sim_run(nn, 0.0, seed=None))
    print(f"\n  observed 28 final ratings: mean {sum(obs_r) / len(obs_r):.1f}, "
          f"SD {sd(obs_r):.1f}, range [{min(obs_r):.1f},{max(obs_r):.1f}]")
    reps = []
    for _ in range(2000):
        s = [sim_run(nn, 0.0, seed=None) for nn in obs_n]
        reps.append((sum(s) / len(s), sd(s), max(s) - min(s)))
    msd = sorted(r[1] for r in reps)
    mmean = sorted(r[0] for r in reps)
    mrange = sorted(r[2] for r in reps)
    print(f"  simulated (identical agent, same game counts, 2000 reps): "
          f"mean {sum(mmean) / len(mmean):.1f} "
          f"[{mmean[100]:.0f},{mmean[1900]:.0f}], "
          f"SD {sum(msd) / len(msd):.1f} [{msd[100]:.1f},{msd[1900]:.1f}], "
          f"range {sum(mrange) / len(mrange):.0f} "
          f"[{mrange[100]:.0f},{mrange[1900]:.0f}]")
    obs_sd_r = sd(obs_r)
    pct = sum(1 for v in msd if v >= obs_sd_r) / len(msd)
    print(f"  P(simulated SD >= observed SD {obs_sd_r:.1f} | identical agent) "
          f"= {pct:.3f}")
    obs_range = max(obs_r) - min(obs_r)
    pctr = sum(1 for v in mrange if v >= obs_range) / len(mrange)
    print(f"  P(simulated range >= observed range {obs_range:.0f} | identical "
          f"agent) = {pctr:.3f}")
    out["sim_calibration"] = {
        "obs_mean": sum(obs_r) / len(obs_r), "obs_sd": obs_sd_r,
        "obs_range": obs_range,
        "sim_mean": sum(mmean) / len(mmean), "sim_sd": sum(msd) / len(msd),
        "sim_range": sum(mrange) / len(mrange),
        "p_sd_ge_obs": pct, "p_range_ge_obs": pctr}

    # the v19 pair under the null
    d19 = []
    for _ in range(20000):
        d19.append(abs(sim_run(43, 0.0, seed=None) - sim_run(43, 0.0, seed=None)))
    d19.sort()
    obs_d = abs(978.3 - 904.6)
    print(f"\n  |rating difference| between two identical 43-game runs: "
          f"median {d19[10000]:.0f}, 90th pct {d19[18000]:.0f}, "
          f"95th pct {d19[19000]:.0f}")
    print(f"  observed v19a-v19b gap {obs_d:.1f} is at percentile "
          f"{sum(1 for v in d19 if v <= obs_d) / len(d19) * 100:.1f} of that "
          f"null - i.e. p={sum(1 for v in d19 if v >= obs_d) / len(d19):.3f}")
    out["v19_null"] = {"median": d19[10000], "p90": d19[18000],
                       "p95": d19[19000], "observed": obs_d,
                       "p": sum(1 for v in d19 if v >= obs_d) / len(d19)}

    # ---------- I. detectable improvement, on the RATING readout ----------
    print()
    print("=" * 100)
    print("I. HOW BIG AN IMPROVEMENT DOES A SINGLE 47-GAME RUN DETECT?")
    # convert a rating target to the skill_shift that produces it in the limit
    print(f"  {'target':>8} {'shift':>7} {'meanR(47g)':>11} {'SD':>7} "
          f"{'power vs a control run':>22}")
    null47 = sorted(sim_run(47, 0.0, seed=None) for _ in range(6000))
    for target, shift in (("+0", 0.0), ("+50", 0.0), ("+100", 0.0)):
        pass
    # find the shift that produces +50 / +100 in the converged (1000-game) mean
    conv = {}
    for shift in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
        vals = [sim_run(600, shift, seed=None) for _ in range(600)]
        conv[shift] = sum(vals) / len(vals)
        print(f"    log-odds shift {shift:+.2f} -> converged rating "
              f"{conv[shift]:.1f} (delta {conv[shift] - conv[0.0]:+.1f})")
    base = conv[0.0]

    def shift_for(delta: float) -> float:
        keys = sorted(conv)
        for i in range(len(keys) - 1):
            lo, hi = keys[i], keys[i + 1]
            if conv[lo] - base <= delta <= conv[hi] - base:
                t = (delta - (conv[lo] - base)) / ((conv[hi] - base)
                                                   - (conv[lo] - base))
                return lo + t * (hi - lo)
        return keys[-1]

    for delta in (50.0, 100.0):
        s = shift_for(delta)
        for n_games in (47, 100, 200, 400, 800):
            treat = sorted(sim_run(n_games, s, seed=None) for _ in range(3000))
            ctrl = sorted(sim_run(n_games, 0.0, seed=None) for _ in range(3000))
            # power of "treatment rating > control rating" being significant
            # under a two-sample z-test with the simulated SD
            mt, mc = sum(treat) / len(treat), sum(ctrl) / len(ctrl)
            st, sc = sd(treat), sd(ctrl)
            sed = math.sqrt(st ** 2 + sc ** 2)
            z = (mt - mc) / sed
            power = 1 - float(sps.norm.cdf(1.959963985 - z)) \
                + float(sps.norm.cdf(-1.959963985 - z))
            # and the crude "did the number go up" hit rate
            hits = sum(1 for i in range(3000)
                       if treat[random.randrange(3000)]
                       > ctrl[random.randrange(3000)]) / 3000
            print(f"  true +{delta:.0f} rating (shift {s:.3f}), "
                  f"{n_games:>4} games/arm: meanR {mt:.0f} vs {mc:.0f}, "
                  f"SD {st:.0f}/{sc:.0f}, power(1 run each)={power * 100:5.1f}%,"
                  f" P(number goes up)={hits * 100:.1f}%")
            out.setdefault("power_sim", {})[f"{delta:.0f}_{n_games}"] = {
                "shift": s, "mean_treat": mt, "mean_ctrl": mc,
                "sd_treat": st, "sd_ctrl": sc, "power": power,
                "p_up": hits}

    # ---------- J. is the rating converged at the end of a run? ----------
    print()
    print("=" * 100)
    print("J. IS THE RATING CONVERGED WHEN WE READ IT?")
    print(f"  {'ver':6} {'n':>4} {'final':>7} {'slope/last20':>12} "
          f"{'mean last10':>11} {'final-mean10':>12}")
    slopes = []
    for lbl in order:
        path = traj.get(lbl) or []
        if len(path) < 25:
            continue
        tail = path[-20:]
        xs2 = list(range(len(tail)))
        mx2 = sum(xs2) / len(xs2)
        my2 = sum(tail) / len(tail)
        sl = (sum((x - mx2) * (y - my2) for x, y in zip(xs2, tail))
              / sum((x - mx2) ** 2 for x in xs2))
        m10 = sum(path[-10:]) / 10
        slopes.append(sl)
        print(f"  {lbl:6} {len(path):>4} {path[-1]:>7.1f} {sl:>+12.2f} "
              f"{m10:>11.1f} {path[-1] - m10:>+12.1f}")
    print(f"  mean slope over the last 20 games = "
          f"{sum(slopes) / len(slopes):+.2f} rating pts/game "
          f"(N={len(slopes)} runs); a converged rating would be 0")
    out["convergence"] = {"mean_slope_last20": sum(slopes) / len(slopes),
                          "n": len(slopes), "slopes": slopes}

    # games played vs final rating
    ns = [len(by_run[l]) for l in order]
    rs = [rating(l) for l in order]
    print(f"  Spearman(games fetched, final rating) = {spearman(ns, rs)}")
    out["games_vs_rating"] = spearman(ns, rs)

    (HERE / "stats3.json").write_text(json.dumps(out, indent=2, default=str),
                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
