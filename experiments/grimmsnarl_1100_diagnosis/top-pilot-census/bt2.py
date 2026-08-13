"""numpy Bradley-Terry with a bootstrap CI on the v21-vs-top-pilot gap,
plus the common-opponent check and the pool-anchoring diagnostics.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

OUT = Path(__file__).resolve().parent
OUR = "9714ab5c3996f6cc"
ELO = 400.0 / math.log(10.0)


def rows(p):
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def build_edges(field, ours):
    edges = []
    for g in field:
        a, b = g["team"]
        if not a or not b or a == b:
            continue
        rw = g["rewards"]
        if rw[0] is None or rw[1] is None or rw[0] == rw[1]:
            continue
        w, lo = (0, 1) if rw[0] > rw[1] else (1, 0)
        f = 0 if g["first"] < 0 else (1 if g["first"] == w else -1)
        edges.append((g["team"][w], g["team"][lo], f, "field"))
    for g in ours:
        me = f"OURS:{g['version']}"
        opp = g["opp_team"] or f"sub:{g['opp_sub']}"
        if g["won"]:
            w, lo = me, opp
            f = 0 if g["went_first"] is None else (1 if g["went_first"] else -1)
        else:
            w, lo = opp, me
            f = 0 if g["went_first"] is None else (-1 if g["went_first"] else 1)
        edges.append((w, lo, f, "ours"))
    return edges


def fit(wi, li, fv, n, sigma2=1.0, iters=600):
    s = np.zeros(n)
    h = 0.0
    cnt = np.bincount(np.concatenate([wi, li]), minlength=n).astype(float)
    prec = 1.0 / (0.25 * cnt + 1.0 / sigma2)
    prec_h = 1.0 / (0.25 * len(wi) + 1.0)
    for _ in range(iters):
        z = s[wi] - s[li] + h * fv
        r = 1.0 - 1.0 / (1.0 + np.exp(-z))
        g = np.bincount(wi, weights=r, minlength=n) - np.bincount(li, weights=r, minlength=n)
        g -= s / sigma2
        s += prec * g
        h += prec_h * float(np.dot(r, fv))
    return s, h


def main() -> int:
    field = rows(OUT / "games.jsonl")
    ours = rows(OUT / "our_games.jsonl")
    pilots = json.loads((OUT / "pilots_by_team.json").read_text(encoding="utf-8"))
    rating = {t: p["rating"] for t, p in pilots.items() if p["rating"] is not None}

    edges = build_edges(field, ours)
    names = sorted({e[0] for e in edges} | {e[1] for e in edges})
    idx = {t: i for i, t in enumerate(names)}
    wi = np.array([idx[e[0]] for e in edges])
    li = np.array([idx[e[1]] for e in edges])
    fv = np.array([e[2] for e in edges], dtype=float)
    n = len(names)

    s, h = fit(wi, li, fv, n)
    st = {names[i]: s[i] for i in range(n)}

    rep = {"n_nodes": n, "n_edges": len(edges),
           "first_player_logodds": round(h, 4),
           "first_player_winprob": round(1 / (1 + math.exp(-h)), 4)}

    # ---- anchoring diagnostics ----
    field_games = Counter()
    for a, b, _, src in edges:
        if src == "field":
            field_games[a] += 1
            field_games[b] += 1
    anchor = {}
    for v in sorted({g["version"] for g in ours}):
        gs = [g for g in ours if g["version"] == v]
        opp = [g["opp_team"] or f"sub:{g['opp_sub']}" for g in gs]
        anchor[v] = {
            "games": len(gs),
            "opp_seen_in_field_corpus": sum(1 for o in opp if field_games[o] > 0),
            "opp_with_ge20_field_games": sum(1 for o in opp if field_games[o] >= 20),
            "distinct_opponents": len(set(opp)),
        }
    rep["anchoring"] = anchor

    # ---- targets ----
    targets = [t for t in names
               if pilots.get(t, {}).get("deck_hash") == OUR
               and rating.get(t) is not None and field_games[t] >= 100]
    targets.sort(key=lambda t: -rating[t])
    targets = targets[:8]

    # ---- bootstrap over episodes ----
    B = 200
    rng = np.random.default_rng(20260813)
    m = len(edges)
    boot = defaultdict(list)
    for b in range(B):
        pick = rng.integers(0, m, m)
        sb, _ = fit(wi[pick], li[pick], fv[pick], n, iters=300)
        for t in targets + [f"OURS:{v}" for v in ("v21", "v20", "v15")]:
            boot[t].append(sb[idx[t]])
        boot["_gap_best"].append(sb[idx[targets[0]]] - sb[idx["OURS:v21"]])
        if b % 50 == 0:
            print(f"  boot {b}", file=sys.stderr)

    def ci(vals):
        a = np.sort(np.array(vals))
        return [round(float(a[int(0.025 * len(a))]), 4),
                round(float(a[int(0.975 * len(a))]), 4)]

    rep["strengths"] = {}
    for t in targets + [f"OURS:{v}" for v in ("v21", "v20", "v15")]:
        rep["strengths"][t] = {
            "strength": round(float(st[t]), 4),
            "elo": round(float(st[t]) * ELO, 1),
            "boot_ci_elo": [round(x * ELO, 1) for x in ci(boot[t])],
            "kaggle_rating": rating.get(t),
            "field_games": field_games[t],
        }
    gap = [float(st[targets[0]] - st["OURS:v21"])]
    rep["gap_v21_to_best_grimm"] = {
        "team": targets[0], "elo": round(gap[0] * ELO, 1),
        "boot_ci_elo": [round(x * ELO, 1) for x in ci(boot["_gap_best"])],
        "p_v21_wins": round(1 / (1 + math.exp(gap[0])), 4),
    }

    # ---- common-opponent check ----
    our_by_opp = defaultdict(list)
    for g in ours:
        if g["version"] in ("v15", "v15b", "v16", "v17", "v18", "v19a",
                            "v19b", "v20", "v21"):
            our_by_opp[g["opp_team"]].append(g["won"])
    field_by_team_opp = defaultdict(lambda: defaultdict(list))
    for g in field:
        a, b = g["team"]
        if not a or not b or a == b:
            continue
        rw = g["rewards"]
        if rw[0] is None or rw[1] is None:
            continue
        for seat in (0, 1):
            field_by_team_opp[g["team"][seat]][g["team"][1 - seat]].append(
                bool(rw[seat] > rw[1 - seat]))

    co = []
    for t in targets:
        shared = [o for o in our_by_opp if o in field_by_team_opp[t]]
        ow = sum(sum(our_by_opp[o]) for o in shared)
        on = sum(len(our_by_opp[o]) for o in shared)
        tw = sum(sum(field_by_team_opp[t][o]) for o in shared)
        tn = sum(len(field_by_team_opp[t][o]) for o in shared)
        co.append({
            "team": t, "kaggle_rating": rating[t],
            "shared_opponents": len(shared),
            "our_games": on, "our_wr": round(ow / on, 4) if on else None,
            "our_ci": wilson(ow, on),
            "their_games": tn, "their_wr": round(tw / tn, 4) if tn else None,
            "their_ci": wilson(tw, tn),
        })
    rep["common_opponent"] = co

    json.dump(rep, (OUT / "bt2.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
