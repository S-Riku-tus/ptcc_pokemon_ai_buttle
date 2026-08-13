"""Weight the elite-minus-low per-matchup gaps by (a) our current opponent mix
and (b) the mix the elite pilots actually face at 1100+, so each matchup gets a
number in points of overall win rate.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OUR = "9714ab5c3996f6cc"


def rows(p):
    return [json.loads(x) for x in p.open(encoding="utf-8") if x.strip()]


def main() -> int:
    r4 = json.loads((OUT / "report4.json").read_text(encoding="utf-8"))
    ours = rows(OUT / "our_games.jsonl")
    field = rows(OUT / "games.jsonl")
    pilots = json.loads((OUT / "pilots_by_team.json").read_text(encoding="utf-8"))
    rating = {t: p["rating"] for t, p in pilots.items() if p["rating"] is not None}

    elite = set(r4["bands"]["elite_teams"])

    ourrec = [g for g in ours if g["version"] in
              ("v15", "v15b", "v16", "v17", "v18", "v19a", "v19b", "v20", "v21")]
    v21 = [g for g in ours if g["version"] == "v21"]
    our_mix = Counter(g["opp_family"] for g in ourrec)
    v21_mix = Counter(g["opp_family"] for g in v21)

    elite_mix = Counter()
    for g in field:
        for seat in (0, 1):
            if g["team"][seat] in elite and g["hash"][seat] == OUR \
                    and g["team"][0] != g["team"][1]:
                elite_mix[g["family"][1 - seat]] += 1

    diffs = {x["label"]: x for x in r4["standardised_by_family"]}
    fams = sorted(set(our_mix) | set(elite_mix),
                  key=lambda f: -(our_mix[f] + elite_mix[f]))
    tot_o, tot_v, tot_e = sum(our_mix.values()), sum(v21_mix.values()), sum(elite_mix.values())

    out = []
    for f in fams:
        d = diffs.get(f)
        out.append({
            "family": f,
            "our_share": round(our_mix[f] / tot_o, 4),
            "v21_share": round(v21_mix[f] / tot_v, 4),
            "elite_share": round(elite_mix[f] / tot_e, 4),
            "elite_std": d["elite_std"] if d else None,
            "low_std": d["low_std"] if d else None,
            "diff": d["diff"] if d else None,
            "diff_ci": d["diff_ci"] if d else None,
            "elite_n": d["elite"]["n"] if d else 0,
            "low_n": d["low"]["n"] if d else 0,
            "pts_at_our_mix": (round(d["diff"] * our_mix[f] / tot_o, 4)
                               if d else None),
            "pts_at_elite_mix": (round(d["diff"] * elite_mix[f] / tot_e, 4)
                                 if d else None),
        })
    covered_o = sum(o["our_share"] for o in out if o["diff"] is not None)
    covered_e = sum(o["elite_share"] for o in out if o["diff"] is not None)
    rep = {
        "rows": out,
        "coverage_of_our_mix": round(covered_o, 4),
        "coverage_of_elite_mix": round(covered_e, 4),
        "sum_pts_at_our_mix": round(sum(o["pts_at_our_mix"] or 0 for o in out), 4),
        "sum_pts_at_elite_mix": round(sum(o["pts_at_elite_mix"] or 0 for o in out), 4),
        "mix_effect_our_to_elite": round(
            sum((o["elite_share"] - o["our_share"]) * (o["low_std"] or 0)
                for o in out), 4),
        "overall_standardised_gap": r4["standardised"]["all"],
    }
    json.dump(rep, (OUT / "decompose.json").open("w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    print(f"{'family':30s} {'ourSh':>6} {'v21Sh':>6} {'eliSh':>6} "
          f"{'low':>6} {'elite':>6} {'diff':>7} {'ci':>18} "
          f"{'ptsOur':>7} {'ptsEli':>7} {'nE':>5} {'nL':>5}")
    for o in out:
        if o["our_share"] < 0.005 and o["elite_share"] < 0.005:
            continue
        print(f"{o['family'][:29]:30s} {o['our_share']:6.3f} {o['v21_share']:6.3f} "
              f"{o['elite_share']:6.3f} "
              f"{(o['low_std'] if o['low_std'] is not None else float('nan')):6.3f} "
              f"{(o['elite_std'] if o['elite_std'] is not None else float('nan')):6.3f} "
              f"{(o['diff'] if o['diff'] is not None else float('nan')):+7.3f} "
              f"{str(o['diff_ci']):>18} "
              f"{(o['pts_at_our_mix'] if o['pts_at_our_mix'] is not None else float('nan')):+7.4f} "
              f"{(o['pts_at_elite_mix'] if o['pts_at_elite_mix'] is not None else float('nan')):+7.4f} "
              f"{o['elite_n']:5d} {o['low_n']:5d}")
    print()
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
