"""Robustness: redo the elite-vs-low standardised comparison using ONLY the
games actually played by the submission id the rating belongs to.

Team-name attribution pools several submissions per team; the corpus-native
indexes (kaggle_grimmsnarl_top50 / kaggle_top50_meta) carry both the rated
submission_id and its score, so those rows can be matched exactly.
"""
from __future__ import annotations

import csv
import json
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


def main() -> int:
    # episode -> (sub0, sub1)
    sub = {}
    for p in ("data/kaggle_grimmsnarl_top50/indexes/episodes.csv",
              "data/kaggle_top50_meta/indexes/episodes.csv",
              "data/kaggle_top40_alakazam/indexes/episodes.csv"):
        f = ROOT / p
        if not f.exists():
            continue
        for r in csv.DictReader(f.open(encoding="utf-8-sig")):
            if r.get("episode_id", "").isdigit():
                sub[int(r["episode_id"])] = (r.get("agent_0_submission_id", ""),
                                             r.get("agent_1_submission_id", ""))
    for corpus in ("kaggle_grimmsnarl_top50", "kaggle_top50_meta",
                   "kaggle_top40_alakazam", "kaggle_top100_current"):
        d = ROOT / "data" / corpus / "submissions"
        if not d.exists():
            continue
        for sd in d.iterdir():
            f = sd / "episodes.json"
            if not f.exists():
                continue
            try:
                e = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for ep in (e.get("episodes", []) if isinstance(e, dict) else e):
                if str(ep.get("episode_id", "")).isdigit():
                    sub[int(ep["episode_id"])] = (
                        str(ep.get("agent_0_submission_id", "")),
                        str(ep.get("agent_1_submission_id", "")))

    # submission_id -> rating, from every index that carries both
    srate = {}
    for p in ("data/kaggle_grimmsnarl_top50/indexes/submissions.csv",
              "data/kaggle_top50_meta/indexes/submissions.csv"):
        f = ROOT / p
        if not f.exists():
            continue
        for r in csv.DictReader(f.open(encoding="utf-8-sig")):
            try:
                srate[str(r["submission_id"])] = (float(r["submission_score"]),
                                                  r["team_name"])
            except (KeyError, ValueError, TypeError):
                continue
    import subprocess
    blob = subprocess.run(
        ["git", "show", "HEAD:data/kaggle_top50_meta/indexes/submissions.csv"],
        cwd=ROOT, capture_output=True).stdout.decode("utf-8-sig")
    for r in csv.DictReader(blob.splitlines()):
        try:
            srate.setdefault(str(r["submission_id"]),
                             (float(r["submission_score"]), r["team_name"]))
        except (KeyError, ValueError, TypeError):
            continue
    for p, sk, ik in (("data/kaggle_top100/20260807_104146_JST/leaderboard_top100.csv",
                       "leaderboard_score", "leaderboard_submission_id"),
                      ("data/kaggle_top100/20260805_113507_JST/leaderboard_top50.csv",
                       "leaderboard_score", "leaderboard_submission_id")):
        for r in csv.DictReader((ROOT / p).open(encoding="utf-8-sig")):
            srate.setdefault(str(r[ik]), (float(r[sk]), r["team_name"]))

    field = rows(OUT / "games.jsonl")
    G = []
    for g in field:
        s = sub.get(g["episode_id"])
        if not s:
            continue
        for seat in (0, 1):
            sid = s[seat]
            if sid not in srate or g["hash"][seat] != OUR:
                continue
            if s[0] == s[1]:
                continue
            if g["rewards"][seat] is None or g["rewards"][1 - seat] is None:
                continue
            G.append({
                "sub": sid, "rating": srate[sid][0], "team": srate[sid][1],
                "opp": g["team"][1 - seat], "opp_sub": s[1 - seat],
                "won": bool(g["rewards"][seat] > g["rewards"][1 - seat]),
                "went_first": (g["first"] == seat) if g["first"] >= 0 else None,
                "opp_family": g["family"][1 - seat],
                "opp_hash": g["hash"][1 - seat],
            })
    by_sub = defaultdict(list)
    for r in G:
        by_sub[r["sub"]].append(r)

    rep = {"rows": len(G), "rated_submissions_on_our_hash": len(by_sub)}
    tbl = sorted(
        [{"sub": s, "team": v[0]["team"], "rating": v[0]["rating"],
          **blk(v),
          "first": blk([g for g in v if g["went_first"] is True])["wr"],
          "second": blk([g for g in v if g["went_first"] is False])["wr"],
          "mirror": blk([g for g in v if g["opp_hash"] == OUR])["wr"]}
         for s, v in by_sub.items() if len(v) >= 50],
        key=lambda r: -r["rating"])
    rep["per_submission"] = tbl

    E = [g for s, v in by_sub.items() if len(v) >= 50 and v[0]["rating"] >= 1100
         for g in v]
    L = [g for s, v in by_sub.items() if len(v) >= 50 and v[0]["rating"] < 1075
         for g in v]
    rep["bands"] = {
        "elite_subs": sorted({(g["sub"], g["team"], g["rating"]) for g in E}),
        "low_subs": sorted({(g["sub"], g["team"], g["rating"]) for g in L}),
        "elite": blk(E), "low": blk(L)}

    def std(Ea, La, label):
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
        rng = np.random.default_rng(11)
        ds = []
        for _ in range(2000):
            pick = rng.integers(0, len(sh), len(sh))
            ss = [sh[i] for i in pick]
            tt = sum(w[o] for o in ss)
            ds.append(sum(w[o] * er[o] for o in ss) / tt
                      - sum(w[o] * lr[o] for o in ss) / tt)
        ds.sort()
        return {"label": label, "shared_opponents": len(sh),
                "elite": blk([g for o in sh for g in eo[o]]),
                "low": blk([g for o in sh for g in lo[o]]),
                "elite_std": round(es, 4), "low_std": round(ls, 4),
                "diff": round(es - ls, 4),
                "diff_ci": [round(ds[50], 4), round(ds[1949], 4)]}

    rep["standardised"] = {
        "all": std(E, L, "all"),
        "first": std([g for g in E if g["went_first"] is True],
                     [g for g in L if g["went_first"] is True], "going first"),
        "second": std([g for g in E if g["went_first"] is False],
                      [g for g in L if g["went_first"] is False], "going second"),
        "mirror": std([g for g in E if g["opp_hash"] == OUR],
                      [g for g in L if g["opp_hash"] == OUR], "mirror"),
        "nonmirror": std([g for g in E if g["opp_hash"] != OUR],
                         [g for g in L if g["opp_hash"] != OUR], "non-mirror"),
    }
    fams = [f for f, c in Counter(g["opp_family"] for g in E).most_common() if c >= 30]
    rep["by_family"] = [x for x in
                        (std([g for g in E if g["opp_family"] == f],
                             [g for g in L if g["opp_family"] == f], f)
                         for f in fams) if x]

    json.dump(rep, (OUT / "robust.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"rows={rep['rows']} subs={rep['rated_submissions_on_our_hash']}")
    print(f"{'sub':>10} {'team':28s} {'rating':>7} {'n':>5} {'wr':>6} "
          f"{'1st':>6} {'2nd':>6} {'mirror':>6}")
    for r in tbl:
        print(f"{r['sub']:>10} {r['team'][:27]:28s} {r['rating']:7.1f} {r['n']:5d} "
              f"{r['wr']:6.3f} {str(r['first']):>6} {str(r['second']):>6} "
              f"{str(r['mirror']):>6}")
    print()
    for k, v in rep["standardised"].items():
        if v:
            print(f"{v['label']:12s} shared={v['shared_opponents']:3d} "
                  f"elite {v['elite_std']:.4f} (n={v['elite']['n']}) "
                  f"low {v['low_std']:.4f} (n={v['low']['n']}) "
                  f"diff {v['diff']:+.4f} {v['diff_ci']}")
    print()
    for v in rep["by_family"]:
        print(f"{v['label'][:28]:29s} shared={v['shared_opponents']:3d} "
              f"elite {v['elite_std']:.3f}({v['elite']['n']:4d}) "
              f"low {v['low_std']:.3f}({v['low']['n']:4d}) "
              f"diff {v['diff']:+.4f} {v['diff_ci']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
