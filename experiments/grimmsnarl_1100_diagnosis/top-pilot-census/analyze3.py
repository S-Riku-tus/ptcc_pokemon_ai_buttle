"""Opponent-matched (direct standardisation) comparison of us vs the elite
Grimmsnarl pilots, decomposed into mirror / matchup / turn order.
"""
from __future__ import annotations

import csv
import json
import math
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


def blk(gs):
    n = len(gs)
    w = sum(1 for g in gs if g["won"])
    return {"n": n, "w": w, "wr": round(w / n, 4) if n else None,
            "ci": wilson(w, n)}


def main() -> int:
    field = rows(OUT / "games.jsonl")
    ours = rows(OUT / "our_games.jsonl")
    pilots = json.loads((OUT / "pilots_by_team.json").read_text(encoding="utf-8"))
    rating = {t: p["rating"] for t, p in pilots.items() if p["rating"] is not None}

    elite = [t for t, p in pilots.items()
             if p["deck_hash"] == OUR and (p["rating"] or 0) >= 1100
             and p["games"] >= 100]
    rep = {"elite_pilots": sorted(
        [{"team": t, "rating": rating[t], "games": pilots[t]["games"]} for t in elite],
        key=lambda x: -x["rating"])}

    # elite game rows
    egs = []
    for g in field:
        for seat in (0, 1):
            t = g["team"][seat]
            if t not in elite or g["team"][0] == g["team"][1]:
                continue
            if g["hash"][seat] != OUR or g["rewards"][seat] is None:
                continue
            if g["rewards"][1 - seat] is None:
                continue
            egs.append({
                "team": t, "opp": g["team"][1 - seat],
                "won": bool(g["rewards"][seat] > g["rewards"][1 - seat]),
                "went_first": (g["first"] == seat) if g["first"] >= 0 else None,
                "opp_family": g["family"][1 - seat],
                "opp_hash": g["hash"][1 - seat],
            })

    ogs = [{"team": "ours", "opp": g["opp_team"], "won": g["won"],
            "went_first": g["went_first"], "opp_family": g["opp_family"],
            "opp_hash": g["opp_hash"], "version": g["version"],
            "opp_rating": g["opp_rating"]}
           for g in ours if g["version"] in
           ("v15", "v15b", "v16", "v17", "v18", "v19a", "v19b", "v20", "v21")]

    rep["raw"] = {"elite": blk(egs), "ours": blk(ogs),
                  "elite_first": blk([g for g in egs if g["went_first"] is True]),
                  "elite_second": blk([g for g in egs if g["went_first"] is False]),
                  "ours_first": blk([g for g in ogs if g["went_first"] is True]),
                  "ours_second": blk([g for g in ogs if g["went_first"] is False])}

    # ---- direct standardisation over shared opponent TEAMS ----
    e_by_opp = defaultdict(list)
    for g in egs:
        e_by_opp[g["opp"]].append(g)
    o_by_opp = defaultdict(list)
    for g in ogs:
        o_by_opp[g["opp"]].append(g)
    shared = [o for o in o_by_opp if o in e_by_opp]

    def standardise(weights_by_opp, rates_by_opp):
        tot = sum(weights_by_opp.values())
        return sum(weights_by_opp[o] * rates_by_opp[o] for o in weights_by_opp) / tot

    ow = {o: len(o_by_opp[o]) for o in shared}
    our_rate = {o: sum(1 for g in o_by_opp[o] if g["won"]) / len(o_by_opp[o])
                for o in shared}
    eli_rate = {o: sum(1 for g in e_by_opp[o] if g["won"]) / len(e_by_opp[o])
                for o in shared}
    ew = {o: len(e_by_opp[o]) for o in shared}

    our_games_shared = [g for o in shared for g in o_by_opp[o]]
    eli_games_shared = [g for o in shared for g in e_by_opp[o]]

    # bootstrap the standardised difference, resampling opponents
    import numpy as np
    rng = np.random.default_rng(20260813)
    diffs = []
    sh = list(shared)
    for _ in range(2000):
        pick = rng.integers(0, len(sh), len(sh))
        os_ = [sh[i] for i in pick]
        tot = sum(ow[o] for o in os_)
        if tot == 0:
            continue
        a = sum(ow[o] * our_rate[o] for o in os_) / tot
        b = sum(ow[o] * eli_rate[o] for o in os_) / tot
        diffs.append(b - a)
    diffs.sort()

    rep["standardised_on_shared_opponents"] = {
        "shared_opponent_teams": len(shared),
        "our_games_on_shared": len(our_games_shared),
        "elite_games_on_shared": len(eli_games_shared),
        "our_raw_wr_on_shared": round(
            sum(1 for g in our_games_shared if g["won"]) / len(our_games_shared), 4),
        "our_ci": wilson(sum(1 for g in our_games_shared if g["won"]),
                         len(our_games_shared)),
        "elite_raw_wr_on_shared": round(
            sum(1 for g in eli_games_shared if g["won"]) / len(eli_games_shared), 4),
        "elite_ci": wilson(sum(1 for g in eli_games_shared if g["won"]),
                           len(eli_games_shared)),
        "elite_wr_standardised_to_our_opponent_mix": round(
            standardise(ow, eli_rate), 4),
        "our_wr_standardised_to_elite_opponent_mix": round(
            standardise(ew, our_rate), 4),
        "diff_elite_minus_ours_at_our_mix": round(
            standardise(ow, eli_rate) - standardise(ow, our_rate), 4),
        "diff_boot_ci": [round(diffs[int(0.025 * len(diffs))], 4),
                         round(diffs[int(0.975 * len(diffs))], 4)],
    }

    # ---- same, split by turn order and by opponent family ----
    def split(pred, label):
        o2 = defaultdict(list)
        e2 = defaultdict(list)
        for g in our_games_shared:
            if pred(g):
                o2[g["opp"]].append(g)
        for g in eli_games_shared:
            if pred(g):
                e2[g["opp"]].append(g)
        sh2 = [o for o in o2 if o in e2]
        if not sh2:
            return None
        w = {o: len(o2[o]) for o in sh2}
        orate = {o: sum(1 for g in o2[o] if g["won"]) / len(o2[o]) for o in sh2}
        erate = {o: sum(1 for g in e2[o] if g["won"]) / len(e2[o]) for o in sh2}
        og = [g for o in sh2 for g in o2[o]]
        eg = [g for o in sh2 for g in e2[o]]
        return {
            "label": label, "shared_opponents": len(sh2),
            "our": blk(og), "elite": blk(eg),
            "elite_std_to_our_mix": round(standardise(w, erate), 4),
            "our_std": round(standardise(w, orate), 4),
            "diff": round(standardise(w, erate) - standardise(w, orate), 4),
        }

    rep["by_turn_order"] = [
        split(lambda g: g["went_first"] is True, "we go first"),
        split(lambda g: g["went_first"] is False, "we go second"),
    ]
    fams = [f for f, _ in Counter(g["opp_family"] for g in our_games_shared).most_common()]
    rep["by_family"] = [x for x in
                        (split(lambda g, f=f: g["opp_family"] == f, f) for f in fams)
                        if x]

    # ---- what does the current top of the board play ----
    def lb(path):
        out = []
        with (ROOT / path).open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                out.append((int(r["rank"]), r["team_name"],
                            float(r["leaderboard_score"])))
        return out

    lb07 = lb("data/kaggle_top100/20260807_104146_JST/leaderboard_top100.csv")
    arch = Counter()
    hashc = Counter()
    known = 0
    for rank, t, sc in lb07:
        p = pilots.get(t)
        if not p or p["games"] < 5:
            arch["<no games in archive>"] += 1
            continue
        known += 1
        arch[p["arch"]] += 1
        hashc[p["deck_hash"]] += 1
    rep["top100_20260807_archetypes"] = {
        "teams_with_ge5_games": known, "archetypes": arch.most_common(),
        "our_hash_share_of_known": round(hashc[OUR] / known, 4) if known else None,
        "our_hash_count": hashc[OUR],
    }
    # top 10 only
    arch10 = Counter()
    for rank, t, sc in lb07[:10]:
        p = pilots.get(t)
        arch10[(p["arch"] if p and p["games"] >= 5 else "<no games>")] += 1
    rep["top10_20260807_archetypes"] = arch10.most_common()

    # Majkel / anti-Grimmsnarl decks
    anti = []
    for t in ("Majkel1337", "Rmy", "keidroid", "flg", "Yushin Ito",
              "LiamK", "James Cox & Henry Chao", "Oshbocker"):
        gs = []
        for g in field:
            for seat in (0, 1):
                if g["team"][seat] != t or g["team"][0] == g["team"][1]:
                    continue
                if g["rewards"][seat] is None or g["rewards"][1 - seat] is None:
                    continue
                gs.append({"won": bool(g["rewards"][seat] > g["rewards"][1 - seat]),
                           "opp_family": g["family"][1 - seat],
                           "opp_hash": g["hash"][1 - seat]})
        vg = [g for g in gs if g["opp_hash"] == OUR]
        anti.append({"team": t, "rating": rating.get(t),
                     "arch": pilots.get(t, {}).get("arch"),
                     "all": blk(gs), "vs_our_exact_hash": blk(vg)})
    rep["anti_grimmsnarl"] = anti

    json.dump(rep, (OUT / "report3.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
