"""THE CENTRAL QUESTION, answered inside the archive where N is large.

Among pilots on our exact 60 (hash 9714ab5c3996f6cc):
  * which per-pilot rate correlates with the pilot's Kaggle rating
    (overall / going first / going second / mirror / non-mirror / vs 1050+)
  * elite band (>=1100) vs low band (<1075), standardised on shared opponent
    teams so opponent strength cannot drive the difference, split by turn
    order and opponent family.
Also: deck-hash neighbourhood of our list.
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


def rows(p):
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def blk(gs):
    n = len(gs)
    w = sum(1 for g in gs if g["won"])
    return {"n": n, "w": w, "wr": round(w / n, 4) if n else None,
            "ci": wilson(w, n)}


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    r = sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")
    if n > 3 and abs(r) < 1:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        # two-sided p from the normal approx
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    else:
        p = float("nan")
    return round(r, 4), round(p, 4), n


def main() -> int:
    field = rows(OUT / "games.jsonl")
    pilots = json.loads((OUT / "pilots_by_team.json").read_text(encoding="utf-8"))
    rating = {t: p["rating"] for t, p in pilots.items() if p["rating"] is not None}

    # ---- all same-hash rows ----
    G = []
    for g in field:
        for seat in (0, 1):
            t = g["team"][seat]
            if not t or g["team"][0] == g["team"][1]:
                continue
            if g["hash"][seat] != OUR:
                continue
            if g["rewards"][seat] is None or g["rewards"][1 - seat] is None:
                continue
            G.append({
                "team": t, "opp": g["team"][1 - seat],
                "won": bool(g["rewards"][seat] > g["rewards"][1 - seat]),
                "went_first": (g["first"] == seat) if g["first"] >= 0 else None,
                "opp_family": g["family"][1 - seat],
                "opp_hash": g["hash"][1 - seat],
                "opp_rating": rating.get(g["team"][1 - seat]),
            })
    by_team = defaultdict(list)
    for r in G:
        by_team[r["team"]].append(r)

    rep = {"same_hash_rows": len(G),
           "same_hash_pilots": len(by_team),
           "same_hash_pilots_rated": sum(1 for t in by_team if t in rating)}

    # ---- per-pilot rate table + spearman vs rating ----
    tbl = []
    for t, gs in by_team.items():
        if t not in rating or len(gs) < 100:
            continue
        first = [g for g in gs if g["went_first"] is True]
        second = [g for g in gs if g["went_first"] is False]
        mirror = [g for g in gs if g["opp_hash"] == OUR]
        nonm = [g for g in gs if g["opp_hash"] != OUR]
        strong = [g for g in gs if (g["opp_rating"] or 0) >= 1050]
        tbl.append({
            "team": t, "rating": rating[t], "n": len(gs),
            "overall": blk(gs)["wr"], "first": blk(first)["wr"],
            "second": blk(second)["wr"], "mirror": blk(mirror)["wr"],
            "nonmirror": blk(nonm)["wr"], "vs1050plus": blk(strong)["wr"],
            "n_first": len(first), "n_second": len(second),
            "n_mirror": len(mirror), "n_nonmirror": len(nonm),
            "n_vs1050plus": len(strong),
            "first_minus_second": (round(blk(first)["wr"] - blk(second)["wr"], 4)
                                   if first and second else None),
        })
    tbl.sort(key=lambda r: -r["rating"])
    rep["per_pilot"] = tbl
    rats = [r["rating"] for r in tbl]
    rep["spearman_vs_rating"] = {}
    for k in ("overall", "first", "second", "mirror", "nonmirror",
              "vs1050plus", "first_minus_second"):
        pairs = [(r["rating"], r[k]) for r in tbl if r[k] is not None]
        if len(pairs) >= 6:
            rep["spearman_vs_rating"][k] = spearman(
                [p[0] for p in pairs], [p[1] for p in pairs])

    # ---- elite vs low band, standardised on shared opponent teams ----
    elite = {t for t in by_team if rating.get(t, 0) >= 1100 and len(by_team[t]) >= 100}
    low = {t for t in by_team
           if t in rating and rating[t] < 1075 and len(by_team[t]) >= 100}
    E = [g for t in elite for g in by_team[t]]
    L = [g for t in low for g in by_team[t]]
    rep["bands"] = {"elite_teams": sorted(elite), "low_teams": sorted(low),
                    "elite": blk(E), "low": blk(L)}

    def std_compare(Ea, La, label):
        eo, lo = defaultdict(list), defaultdict(list)
        for g in Ea:
            eo[g["opp"]].append(g)
        for g in La:
            lo[g["opp"]].append(g)
        sh = [o for o in eo if o in lo]
        if not sh:
            return None
        w = {o: len(eo[o]) + len(lo[o]) for o in sh}
        er = {o: sum(1 for g in eo[o] if g["won"]) / len(eo[o]) for o in sh}
        lr = {o: sum(1 for g in lo[o] if g["won"]) / len(lo[o]) for o in sh}
        tot = sum(w.values())
        es = sum(w[o] * er[o] for o in sh) / tot
        ls = sum(w[o] * lr[o] for o in sh) / tot
        rng = np.random.default_rng(7)
        ds = []
        for _ in range(2000):
            pick = rng.integers(0, len(sh), len(sh))
            ss = [sh[i] for i in pick]
            tt = sum(w[o] for o in ss)
            ds.append(sum(w[o] * er[o] for o in ss) / tt
                      - sum(w[o] * lr[o] for o in ss) / tt)
        ds.sort()
        return {
            "label": label, "shared_opponents": len(sh),
            "elite": blk([g for o in sh for g in eo[o]]),
            "low": blk([g for o in sh for g in lo[o]]),
            "elite_std": round(es, 4), "low_std": round(ls, 4),
            "diff": round(es - ls, 4),
            "diff_ci": [round(ds[50], 4), round(ds[1949], 4)],
        }

    rep["standardised"] = {"all": std_compare(E, L, "all")}
    rep["standardised"]["first"] = std_compare(
        [g for g in E if g["went_first"] is True],
        [g for g in L if g["went_first"] is True], "going first")
    rep["standardised"]["second"] = std_compare(
        [g for g in E if g["went_first"] is False],
        [g for g in L if g["went_first"] is False], "going second")
    fams = [f for f, c in Counter(g["opp_family"] for g in E).most_common() if c >= 40]
    rep["standardised_by_family"] = [
        x for x in (std_compare([g for g in E if g["opp_family"] == f],
                                [g for g in L if g["opp_family"] == f], f)
                    for f in fams) if x]

    # ---- deck-hash neighbourhood ----
    hashes = Counter()
    for t, p in pilots.items():
        if "Grimmsnarl" in p["arch"] and p["games"] >= 10:
            hashes[p["deck_hash"]] += 1
    rep["grimmsnarl_hashes"] = [
        {"hash": h, "teams": c,
         "example_teams": [t for t, p in pilots.items()
                           if p["deck_hash"] == h and p["games"] >= 10][:4],
         "best_rating": max((rating[t] for t, p in pilots.items()
                             if p["deck_hash"] == h and t in rating), default=None)}
        for h, c in hashes.most_common(12)]

    json.dump(rep, (OUT / "report4.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_pilot"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
