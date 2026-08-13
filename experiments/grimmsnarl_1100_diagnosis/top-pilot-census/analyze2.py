"""Opponent-strength controls, the 1300/1177 question, and the gap decomposition."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

OUT = Path(__file__).resolve().parent
OUR = "9714ab5c3996f6cc"


def rows(p):
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def blk(gs, key="won"):
    n = len(gs); w = sum(1 for g in gs if g[key])
    return {"n": n, "w": w, "wr": round(w / n, 4) if n else None, "ci": wilson(w, n)}


def main() -> int:
    field = rows(OUT / "games.jsonl")
    ours = rows(OUT / "our_games.jsonl")
    pilots = json.loads((OUT / "pilots_by_team.json").read_text(encoding="utf-8"))
    rating = {t: p["rating"] for t, p in pilots.items() if p["rating"] is not None}

    rep = {}

    # ---------------- 1. our own runs: win rate by opponent rating ----------------
    recent = [g for g in ours if g["version"] in
              ("v15", "v15b", "v16", "v17", "v18", "v19a", "v19b", "v20", "v21")]
    have = [g for g in recent if g["opp_rating"] is not None]
    rep["our_opponent_rating"] = {
        "games": len(recent), "with_opp_rating": len(have),
        "mean_opp_rating": round(sum(g["opp_rating"] for g in have) / len(have), 1),
        "median_opp_rating": round(sorted(g["opp_rating"] for g in have)[len(have) // 2], 1),
        "pct_opp_ge_1050": round(sum(1 for g in have if g["opp_rating"] >= 1050) / len(have), 4),
        "pct_opp_ge_1100": round(sum(1 for g in have if g["opp_rating"] >= 1100) / len(have), 4),
        "buckets": {},
    }
    for lo, hi in ((0, 850), (850, 950), (950, 1050), (1050, 1100), (1100, 9999)):
        sub = [g for g in have if lo <= g["opp_rating"] < hi]
        rep["our_opponent_rating"]["buckets"][f"{lo}-{hi}"] = blk(sub)
    v21 = [g for g in ours if g["version"] == "v21" and g["opp_rating"] is not None]
    rep["v21_opponent_rating"] = {
        "games": len(v21),
        "mean_opp_rating": round(sum(g["opp_rating"] for g in v21) / len(v21), 1),
        "pct_opp_ge_1050": round(sum(1 for g in v21 if g["opp_rating"] >= 1050) / len(v21), 4),
        "mean_own_rating": round(
            sum(g["own_rating"] for g in v21 if g["own_rating"]) /
            sum(1 for g in v21 if g["own_rating"]), 1),
    }

    # ---------------- 2. field pilot rows with a rated opponent ----------------
    prows = []
    for g in field:
        for seat in (0, 1):
            t = g["team"][seat]
            if not t or g["rewards"][seat] is None:
                continue
            if g["team"][0] == g["team"][1]:
                continue
            o = g["team"][1 - seat]
            prows.append({
                "team": t, "opp": o, "opp_rating": rating.get(o),
                "own_rating": rating.get(t),
                "won": bool(g["rewards"][seat] >
                            (g["rewards"][1 - seat] if g["rewards"][1 - seat] is not None else 0)),
                "went_first": (g["first"] == seat) if g["first"] >= 0 else None,
                "own_hash": g["hash"][seat], "own_arch": g["arch"][seat],
                "opp_hash": g["hash"][1 - seat], "opp_family": g["family"][1 - seat],
                "episode_id": g["episode_id"], "corpus": g["corpus"],
            })
    by_team = defaultdict(list)
    for r in prows:
        by_team[r["team"]].append(r)

    rated = [r for r in prows if r["opp_rating"] is not None]
    rep["field_rows"] = {"total": len(prows), "with_rated_opponent": len(rated),
                         "distinct_teams": len(by_team)}

    # ---------------- 3. top Grimmsnarl pilots, opponent-strength controlled ----
    grimm = sorted(
        [(t, p) for t, p in pilots.items()
         if "Grimmsnarl" in p["arch"] and p["rating"] is not None and p["games"] >= 100],
        key=lambda kv: -kv[1]["rating"])
    table = []
    for t, p in grimm:
        gs = [r for r in by_team[t] if r["own_hash"] == OUR]
        rr = [r for r in gs if r["opp_rating"] is not None]
        row = {
            "team": t, "rating": p["rating"], "games": len(gs),
            "overall": blk(gs),
            "first": blk([g for g in gs if g["went_first"] is True]),
            "second": blk([g for g in gs if g["went_first"] is False]),
            "mirror": blk([g for g in gs if g["opp_family"] == "Grimmsnarl (mirror)"]),
            "nonmirror": blk([g for g in gs if g["opp_family"] != "Grimmsnarl (mirror)"]),
            "rated_opp_n": len(rr),
            "mean_rated_opp": round(sum(r["opp_rating"] for r in rr) / len(rr), 1) if rr else None,
            "vs_opp_ge_1050": blk([r for r in rr if r["opp_rating"] >= 1050]),
            "vs_opp_lt_1050": blk([r for r in rr if r["opp_rating"] < 1050]),
        }
        table.append(row)
    rep["grimm_pilots"] = table

    # ---------------- 4. the very top of the board ----------------
    # who are the top 10 teams on each leaderboard snapshot, what do they play,
    # and how do they do vs Grimmsnarl
    def lb(path, nk="team_name", sk="leaderboard_score"):
        out = []
        with (ROOT / path).open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                out.append((int(r["rank"]), r[nk], float(r[sk])))
        return out

    tops = {}
    for label, path in (
            ("20260805", "data/kaggle_top100/20260805_113507_JST/leaderboard_top50.csv"),
            ("20260807", "data/kaggle_top100/20260807_104146_JST/leaderboard_top100.csv")):
        entries = []
        for rank, team, score in lb(path)[:15]:
            gs = by_team.get(team, [])
            arch = Counter(g["own_arch"] for g in gs).most_common(1)[0][0] if gs else None
            h = Counter(g["own_hash"] for g in gs).most_common(1)[0][0] if gs else None
            vsg = [g for g in gs if g["opp_family"] == "Grimmsnarl (mirror)"]
            entries.append({
                "rank": rank, "team": team, "score": score,
                "games_in_corpus": len(gs), "arch": arch, "hash": h,
                "is_our_hash": h == OUR,
                "vs_grimmsnarl": blk(vsg),
            })
        tops[label] = entries
    rep["leaderboard_top15"] = tops

    # ---------------- 5. staleness ----------------
    lb0807 = lb("data/kaggle_top100/20260807_104146_JST/leaderboard_top100.csv")
    covered = []
    for rank, team, score in lb0807:
        gs = by_team.get(team, [])
        covered.append({"rank": rank, "team": team, "score": score, "games": len(gs)})
    rep["top100_coverage"] = {
        "teams": len(lb0807),
        "with_zero_games": sum(1 for c in covered if c["games"] == 0),
        "with_lt_20_games": sum(1 for c in covered if c["games"] < 20),
        "with_ge_100_games": sum(1 for c in covered if c["games"] >= 100),
        "detail": covered,
    }
    # churn between the two snapshots
    n0805 = {t for _, t, _ in lb("data/kaggle_top100/20260805_113507_JST/leaderboard_top50.csv")}
    n0807_50 = {t for r, t, _ in lb0807 if r <= 50}
    rep["leaderboard_churn_0805_to_0807"] = {
        "top50_overlap": len(n0805 & n0807_50), "top50_new": len(n0807_50 - n0805),
        "days": 2,
    }

    # ---------------- 6. gap decomposition vs best Grimmsnarl pilot ----------
    best = table[0]
    bteam = best["team"]
    bgs = [r for r in by_team[bteam] if r["own_hash"] == OUR]
    ourrec = recent
    fam_b = Counter(g["opp_family"] for g in bgs)
    fam_o = Counter(g["opp_family"] for g in ourrec)
    fams = sorted(set(fam_b) | set(fam_o), key=lambda f: -(fam_o[f] + fam_b[f]))
    decomp = []
    for f in fams:
        b = blk([g for g in bgs if g["opp_family"] == f])
        o = blk([g for g in ourrec if g["opp_family"] == f])
        decomp.append({
            "family": f,
            "best_share": round(fam_b[f] / len(bgs), 4),
            "our_share": round(fam_o[f] / len(ourrec), 4),
            "best": b, "ours": o,
            "wr_delta": (round(b["wr"] - o["wr"], 4)
                         if b["wr"] is not None and o["wr"] is not None else None),
            "pts_at_our_mix": (round((b["wr"] - o["wr"]) * fam_o[f] / len(ourrec), 4)
                               if b["wr"] is not None and o["wr"] is not None else None),
            "pts_mix_effect": (round((fam_b[f] / len(bgs) - fam_o[f] / len(ourrec))
                                     * (o["wr"] if o["wr"] is not None else 0), 4)),
        })
    rep["decomposition_vs_best"] = {
        "best_team": bteam, "best_rating": best["rating"],
        "best_overall": best["overall"], "our_overall": blk(ourrec),
        "rows": decomp,
        "sum_pts_at_our_mix": round(
            sum(d["pts_at_our_mix"] or 0 for d in decomp), 4),
        "sum_pts_mix_effect": round(sum(d["pts_mix_effect"] for d in decomp), 4),
    }

    # ---------------- 7. opponent-pool comparison -------------------------------
    our_opp_teams = Counter(g["opp_team"] for g in ourrec)
    field_opp_teams = Counter(r["opp"] for r in prows if r["own_hash"] == OUR)
    rep["opponent_pool_overlap"] = {
        "our_distinct_opponents": len(our_opp_teams),
        "field_grimm_distinct_opponents": len(field_opp_teams),
        "shared": len(set(our_opp_teams) & set(field_opp_teams)),
        "our_games_vs_shared": sum(v for k, v in our_opp_teams.items()
                                   if k in field_opp_teams),
        "our_opponents_with_known_rating": sum(
            v for k, v in our_opp_teams.items() if k in rating),
    }

    json.dump(rep, (OUT / "report2.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps(
        {k: v for k, v in rep.items() if k != "top100_coverage"},
        ensure_ascii=False, indent=1))
    print("\ntop100 coverage:", json.dumps(
        {k: v for k, v in rep["top100_coverage"].items() if k != "detail"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
