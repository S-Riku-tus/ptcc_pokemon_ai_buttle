"""Census + top-pilot decomposition. Reads games.jsonl / our_games.jsonl."""
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


def rows(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_ratings():
    """team_name -> {source: (date, rating, rank)} ."""
    out = defaultdict(dict)
    srcs = [
        ("lb_20260805", ROOT / "data/kaggle_top100/20260805_113507_JST/leaderboard_top50.csv",
         "team_name", "leaderboard_score", "rank"),
        ("lb_20260807", ROOT / "data/kaggle_top100/20260807_104146_JST/leaderboard_top100.csv",
         "team_name", "leaderboard_score", "rank"),
        ("top50_meta", ROOT / "data/kaggle_top50_meta/indexes/submissions.csv",
         "team_name", "submission_score", "leaderboard_rank"),
        ("grimm50", ROOT / "data/kaggle_grimmsnarl_top50/indexes/submissions.csv",
         "team_name", "submission_score", "leaderboard_rank"),
        ("teams_csv", ROOT / "data/kaggle_top50_meta/analysis/teams.csv",
         "team_name", "score", "rank"),
    ]
    for label, path, nk, sk, rk in srcs:
        for r in read_csv(path):
            try:
                sc = float(r[sk])
            except (KeyError, TypeError, ValueError):
                continue
            out[r[nk]][label] = (sc, r.get(rk, ""))
    # HEAD version of the overwritten top50_meta index (100 teams)
    try:
        blob = subprocess.run(
            ["git", "show", "HEAD:data/kaggle_top50_meta/indexes/submissions.csv"],
            cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8-sig")
        for r in csv.DictReader(blob.splitlines()):
            try:
                sc = float(r["submission_score"])
            except (KeyError, TypeError, ValueError):
                continue
            out[r["team_name"]].setdefault(
                "top50_meta_head", (sc, r.get("leaderboard_rank", "")))
    except Exception as exc:  # noqa: BLE001
        print("git show failed:", exc, file=sys.stderr)
    return out


def best_rating(entry):
    """Prefer the 08-07 leaderboard, then 08-05, then the corpus indexes."""
    for k in ("lb_20260807", "lb_20260805", "grimm50", "top50_meta",
              "top50_meta_head", "teams_csv"):
        if k in entry:
            return entry[k][0], k
    return None, None


def block(games, key="won"):
    n = len(games)
    w = sum(1 for g in games if g[key])
    return {"n": n, "w": w, "wr": round(w / n, 4) if n else None,
            "ci": wilson(w, n)}


def main() -> int:
    field = rows(OUT / "games.jsonl")
    ours = rows(OUT / "our_games.jsonl")
    ratings = load_ratings()

    # ---------- per (team, seat) game rows ----------
    prows = []
    for g in field:
        for seat in (0, 1):
            t = g["team"][seat]
            if not t:
                continue
            rw = g["rewards"]
            if rw[seat] is None:
                continue
            prows.append({
                "team": t, "episode_id": g["episode_id"], "corpus": g["corpus"],
                "seat": seat,
                "won": bool(rw[seat] > (rw[1 - seat] if rw[1 - seat] is not None else 0)),
                "went_first": (g["first"] == seat) if g["first"] >= 0 else None,
                "own_hash": g["hash"][seat], "own_arch": g["arch"][seat],
                "opp_hash": g["hash"][1 - seat], "opp_family": g["family"][1 - seat],
                "opp_arch": g["arch"][1 - seat], "opp_team": g["team"][1 - seat],
                "mirror_self": g["team"][0] == g["team"][1],
            })

    by_team = defaultdict(list)
    for r in prows:
        by_team[r["team"]].append(r)

    # ---------- deck-hash census ----------
    pilots = {}
    for team, gs in by_team.items():
        hc = Counter(g["own_hash"] for g in gs)
        top, _ = hc.most_common(1)[0]
        rt, src = best_rating(ratings.get(team, {}))
        pilots[team] = {
            "team": team, "games": len(gs), "deck_hash": top,
            "hash_share": round(hc[top] / len(gs), 3),
            "distinct_hashes": len(hc),
            "arch": Counter(g["own_arch"] for g in gs).most_common(1)[0][0],
            "rating": rt, "rating_source": src,
            "rating_all": {k: v[0] for k, v in ratings.get(team, {}).items()},
        }

    json.dump(pilots, (OUT / "pilots_by_team.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # Grimmsnarl pilots: archetype string contains Grimmsnarl
    grimm = {t: p for t, p in pilots.items() if "Grimmsnarl" in p["arch"]}
    print(f"total teams seen in replays: {len(pilots)}   "
          f"Grimmsnarl teams: {len(grimm)}   "
          f"our exact hash: {sum(1 for p in pilots.values() if p['deck_hash'] == OUR)}")

    # ---------- report ----------
    rep = {}

    # exact hash vs near neighbour
    hashes = Counter(p["deck_hash"] for p in grimm.values())
    rep["grimmsnarl_hash_counts"] = hashes.most_common()

    ranked = sorted(
        [p for p in grimm.values() if p["rating"] is not None and p["games"] >= 20],
        key=lambda p: -p["rating"])
    rep["grimmsnarl_pilots_ranked"] = [
        {k: p[k] for k in ("team", "rating", "rating_source", "games", "deck_hash",
                           "hash_share", "arch")}
        for p in ranked]

    print("\n== Grimmsnarl pilots by rating (>=20 games in corpus) ==")
    print(f"{'team':34s} {'rating':>7} {'src':>13} {'games':>6} {'hash':>17} {'wr':>6} {'ci':>16}")
    for p in ranked:
        gs = by_team[p["team"]]
        b = block(gs)
        print(f"{p['team'][:33]:34s} {p['rating']:7.1f} {p['rating_source']:>13} "
              f"{p['games']:6d} {p['deck_hash']:>17} {b['wr'] or 0:6.3f} "
              f"{str(b['ci']):>16}")

    # ---------- top 5 detail ----------
    top5 = ranked[:5]
    detail = {}
    for p in top5:
        gs = [g for g in by_team[p["team"]] if not g["mirror_self"]]
        d = {
            "rating": p["rating"], "deck_hash": p["deck_hash"],
            "overall": block(gs),
            "went_first": block([g for g in gs if g["went_first"] is True]),
            "went_second": block([g for g in gs if g["went_first"] is False]),
            "turn_order_unknown": sum(1 for g in gs if g["went_first"] is None),
            "by_family": {},
        }
        fam = Counter(g["opp_family"] for g in gs)
        for f, _ in fam.most_common():
            d["by_family"][f] = {
                **block([g for g in gs if g["opp_family"] == f]),
                "share": round(fam[f] / len(gs), 4),
            }
        detail[p["team"]] = d
    rep["top5"] = detail

    # ---------- our v21 ----------
    v21 = [g for g in ours if g["version"] == "v21"]
    our_block = {
        "overall": block(v21),
        "went_first": block([g for g in v21 if g["went_first"] is True]),
        "went_second": block([g for g in v21 if g["went_first"] is False]),
        "by_family": {},
    }
    fam = Counter(g["opp_family"] for g in v21)
    for f, _ in fam.most_common():
        our_block["by_family"][f] = {
            **block([g for g in v21 if g["opp_family"] == f]),
            "share": round(fam[f] / len(v21), 4),
        }
    rep["v21"] = our_block

    # pooled recent us (v15..v21) for a thicker denominator
    recent = [g for g in ours if g["version"] in
              ("v15", "v15b", "v16", "v17", "v18", "v19a", "v19b", "v20", "v21")]
    pooled = {"overall": block(recent),
              "went_first": block([g for g in recent if g["went_first"] is True]),
              "went_second": block([g for g in recent if g["went_first"] is False]),
              "by_family": {}}
    fam = Counter(g["opp_family"] for g in recent)
    for f, _ in fam.most_common():
        pooled["by_family"][f] = {
            **block([g for g in recent if g["opp_family"] == f]),
            "share": round(fam[f] / len(recent), 4)}
    rep["ours_v15_v21_pooled"] = pooled

    json.dump(rep, (OUT / "report.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print("\n== our v21 (58 games) ==")
    print(json.dumps(our_block, ensure_ascii=False, indent=1))
    print("\n== pooled v15..v21 ==")
    print(json.dumps(pooled, ensure_ascii=False, indent=1))
    print("\n== top5 ==")
    print(json.dumps(detail, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
