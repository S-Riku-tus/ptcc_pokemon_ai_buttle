"""Top-5 pilot detail, deck-hash neighbourhood with card distance, and the
rating-per-win-rate-point calibration.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ml.core.replay_io import deck_hash, extract_fast_header_from_file  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import wilson, archetype  # noqa: E402

OUT = Path(__file__).resolve().parent
OUR = "9714ab5c3996f6cc"
CARDS = {int(c["cardId"]): c for c in json.loads(
    (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8"))}


def rows(p):
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def blk(gs):
    n = len(gs)
    w = sum(1 for g in gs if g["won"])
    return {"n": n, "w": w, "wr": round(w / n, 4) if n else None,
            "ci": wilson(w, n)}


def main() -> int:
    field = rows(OUT / "games.jsonl")
    pilots = json.loads((OUT / "pilots_by_team.json").read_text(encoding="utf-8"))
    rating = {t: p["rating"] for t, p in pilots.items() if p["rating"] is not None}
    r4 = json.loads((OUT / "report4.json").read_text(encoding="utf-8"))

    # ---------- deck neighbourhood: get one full deck list per hash ----------
    # walk replays until we have a sample deck for each Grimmsnarl-family hash
    want = {p["deck_hash"] for t, p in pilots.items()
            if "Grimmsnarl" in p["arch"] and p["games"] >= 10}
    samples = {}
    for corpus in ("kaggle_grimmsnarl_top50", "kaggle_top50_meta",
                   "kaggle_top40_alakazam"):
        d = ROOT / "data" / corpus / "replays"
        if not d.exists():
            continue
        for path in d.glob("episode_*.json"):
            if set(want) <= set(samples):
                break
            h = extract_fast_header_from_file(path)
            for deck in h["decks"]:
                if not deck:
                    continue
                hh = deck_hash(deck)
                if hh in want and hh not in samples:
                    samples[hh] = deck
    ours_deck = samples.get(OUR)

    def diff(deck):
        a, b = Counter(deck), Counter(ours_deck)
        add = [(CARDS.get(c, {}).get("name", str(c)), n)
               for c, n in (a - b).items()]
        rem = [(CARDS.get(c, {}).get("name", str(c)), n)
               for c, n in (b - a).items()]
        return sum(n for _, n in add), sorted(add), sorted(rem)

    hashes = Counter()
    for t, p in pilots.items():
        if "Grimmsnarl" in p["arch"] and p["games"] >= 10:
            hashes[p["deck_hash"]] += 1
    nb = []
    for h, c in hashes.most_common():
        d = samples.get(h)
        n, add, rem = diff(d) if (d and ours_deck) else (None, [], [])
        teams = [t for t, p in pilots.items()
                 if p["deck_hash"] == h and p["games"] >= 10]
        nb.append({"hash": h, "teams": c, "cards_different": n,
                   "adds": add, "removes": rem,
                   "best_rating": max((rating[t] for t in teams if t in rating),
                                      default=None),
                   "rated_teams": sum(1 for t in teams if t in rating),
                   "example": sorted(teams, key=lambda t: -(rating.get(t) or 0))[:3]})

    # ---------- top-5 pilots, per family ----------
    top = sorted([t for t in pilots
                  if pilots[t]["deck_hash"] == OUR and t in rating
                  and pilots[t]["games"] >= 100],
                 key=lambda t: -rating[t])[:5]
    detail = {}
    for t in top:
        gs = []
        for g in field:
            for seat in (0, 1):
                if g["team"][seat] != t or g["team"][0] == g["team"][1]:
                    continue
                if g["hash"][seat] != OUR:
                    continue
                if g["rewards"][seat] is None or g["rewards"][1 - seat] is None:
                    continue
                gs.append({"won": bool(g["rewards"][seat] > g["rewards"][1 - seat]),
                           "went_first": (g["first"] == seat) if g["first"] >= 0 else None,
                           "fam": g["family"][1 - seat],
                           "opp": g["team"][1 - seat]})
        fam = Counter(g["fam"] for g in gs)
        detail[t] = {
            "rating": rating[t], "overall": blk(gs),
            "first": blk([g for g in gs if g["went_first"] is True]),
            "second": blk([g for g in gs if g["went_first"] is False]),
            "by_family": {f: {**blk([g for g in gs if g["fam"] == f]),
                              "share": round(fam[f] / len(gs), 4)}
                          for f, c in fam.most_common() if c >= 8},
        }

    # ---------- rating per win-rate point ----------
    er = [rating[t] for t in r4["bands"]["elite_teams"]]
    lr = [rating[t] for t in r4["bands"]["low_teams"]]
    d = r4["standardised"]["all"]
    mean_e, mean_l = sum(er) / len(er), sum(lr) / len(lr)
    calib = {
        "elite_mean_rating": round(mean_e, 1),
        "low_mean_rating": round(mean_l, 1),
        "rating_gap": round(mean_e - mean_l, 1),
        "standardised_wr_gap": d["diff"],
        "rating_points_per_wr_point": round((mean_e - mean_l) / d["diff"] / 100, 2),
        "wr_points_needed_for_1100_from_v21_9482": round(
            (1100 - 948.2) / ((mean_e - mean_l) / d["diff"]), 4),
    }

    rep = {"deck_neighbourhood": nb, "top5": detail, "calibration": calib}
    json.dump(rep, (OUT / "final.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print("=== Grimmsnarl deck-hash neighbourhood (teams with >=10 archived games) ===")
    for x in nb[:10]:
        print(f"{x['hash']}  teams={x['teams']:3d} rated={x['rated_teams']:2d} "
              f"best={x['best_rating']} cards_diff_from_ours={x['cards_different']}")
        if x["cards_different"] not in (None, 0):
            print(f"    +{x['adds']}  -{x['removes']}")
    print()
    print("=== calibration ===")
    print(json.dumps(calib, indent=1))
    print()
    for t, v in detail.items():
        print(f"--- {t} rating={v['rating']} overall={v['overall']['wr']} "
              f"(n={v['overall']['n']}) 1st={v['first']['wr']}({v['first']['n']}) "
              f"2nd={v['second']['wr']}({v['second']['n']})")
        for f, b in v["by_family"].items():
            print(f"      {f[:32]:33s} share={b['share']:.3f} "
                  f"wr={b['wr']} n={b['n']} {b['ci']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
