"""Date coverage of every replay we hold, and how much of the live field we
have no games for. Episode dates come from the index CSVs where present and by
nearest-neighbour interpolation on episode_id elsewhere (episode ids are
monotone in time).
"""
from __future__ import annotations

import bisect
import csv
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def rows(p):
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def parse(s):
    return dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")


def main() -> int:
    known: dict[int, dt.datetime] = {}
    for idx in (ROOT / "data/kaggle_top50_meta/indexes/episodes.csv",
                ROOT / "data/kaggle_grimmsnarl_top50/indexes/episodes.csv",
                ROOT / "data/kaggle_top40_alakazam/indexes/episodes.csv"):
        if not idx.exists():
            continue
        with idx.open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("created_at") and r.get("episode_id", "").isdigit():
                    known[int(r["episode_id"])] = parse(r["created_at"])
    for run in (ROOT / "data" / "runs" / "grimmsnarl").glob("*/episodes.csv"):
        with run.open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("create_time") and r.get("episode_id", "").isdigit():
                    known[int(r["episode_id"])] = parse(r["create_time"])
    ks = sorted(known)
    print(f"anchors: {len(ks)}  id {ks[0]}..{ks[-1]}  "
          f"{known[ks[0]]}..{known[ks[-1]]}", file=sys.stderr)

    def when(eid):
        i = bisect.bisect_left(ks, eid)
        cands = [ks[j] for j in (i - 1, i) if 0 <= j < len(ks)]
        return known[min(cands, key=lambda k: abs(k - eid))]

    field = rows(OUT / "games.jsonl")
    per = defaultdict(list)
    for g in field:
        per[g["corpus"]].append(when(g["episode_id"]))
    ours = rows(OUT / "our_games.jsonl")
    per_o = defaultdict(list)
    for g in ours:
        per_o[g["version"]].append(parse(g["create_time"]) if g["create_time"]
                                   else None)

    rep = {"corpus_date_range": {}, "our_run_dates": {}}
    for k, v in per.items():
        v = sorted(v)
        rep["corpus_date_range"][k] = {
            "episodes": len(v), "min": str(v[0]), "max": str(v[-1]),
            "median": str(v[len(v) // 2]),
            "days_stale_vs_20260813": round(
                (dt.datetime(2026, 8, 13) - v[-1]).total_seconds() / 86400, 2),
        }
    for k, v in per_o.items():
        vv = sorted(x for x in v if x)
        if vv:
            rep["our_run_dates"][k] = {"episodes": len(vv), "min": str(vv[0]),
                                       "max": str(vv[-1])}

    # how much of the 08-07 top-100 do we have games for, and how stale
    pilots = json.loads((OUT / "pilots_by_team.json").read_text(encoding="utf-8"))
    lb = []
    with (ROOT / "data/kaggle_top100/20260807_104146_JST/leaderboard_top100.csv").open(
            encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            lb.append((int(r["rank"]), r["team_name"], float(r["leaderboard_score"])))
    g_by_team = defaultdict(list)
    for g in field:
        for seat in (0, 1):
            if g["team"][seat]:
                g_by_team[g["team"][seat]].append(when(g["episode_id"]))
    cov = []
    for rank, t, sc in lb:
        d = sorted(g_by_team.get(t, []))
        cov.append({"rank": rank, "team": t, "score": sc, "games": len(d),
                    "last_game": str(d[-1]) if d else None})
    rep["top100_coverage_20260807"] = {
        "n": len(cov),
        "zero_games": sum(1 for c in cov if c["games"] == 0),
        "lt_20_games": sum(1 for c in cov if c["games"] < 20),
        "ge_100_games": sum(1 for c in cov if c["games"] >= 100),
        "median_games": sorted(c["games"] for c in cov)[len(cov) // 2],
        "top10_zero_games": sum(1 for c in cov[:10] if c["games"] == 0),
        "top25_zero_games": sum(1 for c in cov[:25] if c["games"] == 0),
    }
    rep["top100_detail"] = cov

    # what v21 actually met, vs what we hold
    v21 = [g for g in ours if g["version"] == "v21"]
    lbnames = {t for _, t, _ in lb}
    lbscore = {t: s for _, t, s in lb}
    rep["v21_opponents"] = {
        "games": len(v21),
        "distinct_opponents": len({g["opp_team"] for g in v21}),
        "opp_in_top100_20260807": sum(1 for g in v21 if g["opp_team"] in lbnames),
        "opp_with_any_archived_game": sum(1 for g in v21 if g_by_team.get(g["opp_team"])),
        "max_opp_rating": max(g["opp_rating"] for g in v21 if g["opp_rating"]),
        "opp_rating_deciles": [
            round(sorted(g["opp_rating"] for g in v21 if g["opp_rating"])[
                int(p * (len(v21) - 1))], 1)
            for p in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)],
    }

    # leaderboard churn 08-05 -> 08-07
    lb5 = []
    with (ROOT / "data/kaggle_top100/20260805_113507_JST/leaderboard_top50.csv").open(
            encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            lb5.append((int(r["rank"]), r["team_name"], float(r["leaderboard_score"])))
    m5 = {t: s for _, t, s in lb5}
    m7 = {t: s for _, t, s in lb if _ <= 100}
    both = [t for t in m5 if t in m7]
    rep["churn"] = {
        "top50_0805": len(m5), "top100_0807": len(m7),
        "0805_top50_still_in_0807_top50": sum(
            1 for r7, t, _ in lb if r7 <= 50 and t in m5),
        "0805_top50_still_in_0807_top100": len(both),
        "mean_abs_score_change": round(
            sum(abs(m7[t] - m5[t]) for t in both) / len(both), 1),
        "mean_signed_score_change": round(
            sum(m7[t] - m5[t] for t in both) / len(both), 1),
        "biggest_drops": sorted(
            [{"team": t, "d0805": m5[t], "d0807": m7[t],
              "delta": round(m7[t] - m5[t], 1)} for t in both],
            key=lambda x: x["delta"])[:8],
    }

    json.dump(rep, (OUT / "staleness.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k != "top100_detail"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
