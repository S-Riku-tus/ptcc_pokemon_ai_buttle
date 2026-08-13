"""Put the field corpora and our own ladder runs on one strength scale.

Our 418 recent games average an 861.8-rated opponent; the archived 1100+ pilots
average a 1090-rated opponent, so raw win rates are not comparable. A
Bradley-Terry fit over the union of both game sets (they share 104 opponent
teams) gives a common scale, and regressing the fitted strengths on the Kaggle
leaderboard rating for the 114 teams whose rating we know converts the scale
back into rating points.

P(A beats B) = sigmoid(s_A - s_B + h * first_A)
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

OUT = Path(__file__).resolve().parent
OUR = "9714ab5c3996f6cc"
ELO = 400.0 / math.log(10.0)


def rows(p):
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def main() -> int:
    field = rows(OUT / "games.jsonl")
    ours = rows(OUT / "our_games.jsonl")
    pilots = json.loads((OUT / "pilots_by_team.json").read_text(encoding="utf-8"))
    rating = {t: p["rating"] for t, p in pilots.items() if p["rating"] is not None}

    # ---- edges: (winner, loser, first_is_winner in {+1,-1,0}) ----
    edges = []
    for g in field:
        a, b = g["team"]
        if not a or not b or a == b:
            continue
        rw = g["rewards"]
        if rw[0] is None or rw[1] is None or rw[0] == rw[1]:
            continue
        w, l = (0, 1) if rw[0] > rw[1] else (1, 0)
        f = 0 if g["first"] < 0 else (1 if g["first"] == w else -1)
        edges.append((g["team"][w], g["team"][l], f, "field"))

    for g in ours:
        me = f"OURS:{g['version']}"
        opp = g["opp_team"] or f"sub:{g['opp_sub']}"
        if g["won"]:
            w, l = me, opp
            f = 0 if g["went_first"] is None else (1 if g["went_first"] else -1)
        else:
            w, l = opp, me
            f = 0 if g["went_first"] is None else (-1 if g["went_first"] else 1)
        edges.append((w, l, f, "ours"))

    names = sorted({e[0] for e in edges} | {e[1] for e in edges})
    idx = {n: i for i, n in enumerate(names)}
    print(f"nodes={len(names)} edges={len(edges)}", file=sys.stderr)

    # connectivity of the component containing OURS:v21
    adj = defaultdict(set)
    for w, l, _, _ in edges:
        adj[w].add(l)
        adj[l].add(w)
    seen = {"OURS:v21"}
    stack = ["OURS:v21"]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    print(f"component containing OURS:v21 = {len(seen)} / {len(names)} nodes",
          file=sys.stderr)

    n = len(names)
    s = [0.0] * n
    h = 0.0
    sigma2 = 1.0        # L2 prior variance on strengths (in log-odds)
    ei = [(idx[w], idx[l], f) for w, l, f, _ in edges]
    games = Counter()
    for w, l, _, _ in edges:
        games[w] += 1
        games[l] += 1
    # diagonal Newton preconditioner: Hessian_ii <= 0.25 * games_i + 1/sigma2
    prec = [1.0 / (0.25 * games[names[i]] + 1.0 / sigma2) for i in range(n)]
    prec_h = 1.0 / (0.25 * len(ei) + 1.0)
    for it in range(3000):
        gs = [0.0] * n
        gh = 0.0
        for wi, li, f in ei:
            z = s[wi] - s[li] + h * f
            p = 1.0 / (1.0 + math.exp(-z))
            r = 1.0 - p
            gs[wi] += r
            gs[li] -= r
            gh += r * f
        for i in range(n):
            gs[i] -= s[i] / sigma2
        for i in range(n):
            s[i] += prec[i] * gs[i]
        h += prec_h * gh
        if it % 500 == 0:
            ll = 0.0
            for wi, li, f in ei:
                z = s[wi] - s[li] + h * f
                ll -= math.log(1.0 + math.exp(-z))
            print(f"  it={it} loglik={ll:.1f} h={h:.4f}", file=sys.stderr)

    strength = {names[i]: s[i] for i in range(n)}

    # ---- calibrate strength -> Kaggle rating on the teams we have ratings for ----
    pts = [(strength[t], rating[t]) for t in names
           if t in rating and games[t] >= 30]
    if len(pts) >= 5:
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        sxy = sum((x - mx) * (y - my) for x, y in pts)
        sxx = sum((x - mx) ** 2 for x, y in pts)
        syy = sum((y - my) ** 2 for x, y in pts)
        beta = sxy / sxx
        alpha = my - beta * mx
        r = sxy / math.sqrt(sxx * syy)
    else:
        beta = alpha = r = float("nan")
    print(f"calibration on {len(pts)} rated teams (>=30 games): "
          f"rating = {alpha:.1f} + {beta:.1f} * strength   r={r:.3f}",
          file=sys.stderr)

    def pred(t):
        return alpha + beta * strength[t]

    report = {
        "nodes": n, "edges": len(edges), "home_first_advantage_logodds": h,
        "first_player_win_prob_equal_strength": 1 / (1 + math.exp(-h)),
        "component_with_v21": len(seen),
        "calibration": {"n_rated_teams": len(pts), "alpha": alpha,
                        "beta": beta, "pearson_r": r},
        "ours": {}, "field_top": [],
    }
    for v in sorted({f"OURS:{g['version']}" for g in ours}):
        report["ours"][v] = {
            "games": games[v], "strength": round(strength[v], 4),
            "strength_elo": round(strength[v] * ELO, 1),
            "predicted_kaggle_rating": round(pred(v), 1),
        }
    scored = sorted(
        [t for t in names if games[t] >= 60 and not t.startswith("OURS:")],
        key=lambda t: -strength[t])
    for t in scored[:40]:
        report["field_top"].append({
            "team": t, "games": games[t],
            "strength": round(strength[t], 4),
            "strength_elo": round(strength[t] * ELO, 1),
            "kaggle_rating": rating.get(t),
            "predicted_kaggle_rating": round(pred(t), 1),
            "deck_hash": pilots.get(t, {}).get("deck_hash"),
            "arch": pilots.get(t, {}).get("arch"),
        })

    # policy prediction comparison: v21 vs each top Grimmsnarl pilot
    grimm = [t for t in names if pilots.get(t, {}).get("deck_hash") == OUR
             and rating.get(t) is not None and games[t] >= 100]
    grimm.sort(key=lambda t: -rating[t])
    report["v21_vs_grimm_pilots"] = [
        {"team": t, "kaggle_rating": rating[t],
         "p_v21_wins": round(1 / (1 + math.exp(-(strength["OURS:v21"] - strength[t]))), 4),
         "elo_gap": round((strength[t] - strength["OURS:v21"]) * ELO, 1)}
        for t in grimm[:10]]

    json.dump(report, (OUT / "bt.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
