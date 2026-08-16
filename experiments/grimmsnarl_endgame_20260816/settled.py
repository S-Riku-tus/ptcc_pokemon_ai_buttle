"""Join our stored games to the newest leaderboard by opponent submission id.

The rating an opponent carried at pairing time is provisional: a strong agent
that resubmitted starts at 600 and climbs. The leaderboard score its team
carries now is the settled estimate. Comparing the two answers the question
"did the field get stronger, or did the pool get less converged".
"""
import csv
import json
import math
import collections

lb = json.load(open("experiments/grimmsnarl_endgame_20260816/leaderboard_full_20260816.json",
                    encoding="utf-8"))["publicLeaderboard"]
by_sub = {}
for r in lb:
    sid = r.get("submissionId")
    if sid is not None:
        by_sub[str(sid)] = (float(r["displayScore"]), r["rank"])

rows = list(csv.DictReader(open("experiments/grimmsnarl_endgame_20260816/version_games.csv",
                                encoding="utf-8-sig")))

def block(rs, label):
    n = len(rs)
    if not n:
        return
    matched = [r for r in rs if r["opponent_submission"] in by_sub]
    wins = sum(int(r["won"]) for r in rs)
    paired = sum(float(r["opponent_rating"]) for r in rs) / n
    line = (f"{label:8s} n={n:4d} matched={len(matched):3d} "
            f"({len(matched)/n:5.1%})  paired_mean={paired:7.1f}")
    if matched:
        m = len(matched)
        mw = sum(int(r["won"]) for r in matched)
        mp = sum(float(r["opponent_rating"]) for r in matched) / m
        ms = sum(by_sub[r["opponent_submission"]][0] for r in matched) / m
        line += (f"  [matched: paired={mp:7.1f} settled={ms:7.1f} "
                 f"delta={ms-mp:+7.1f} wr={mw/m:.3f}]")
    print(line)

print("=== per version: paired vs settled opponent rating ===")
by = collections.defaultdict(list)
for r in rows:
    by[r["version"]].append(r)
for v, rs in by.items():
    block(rs, v)

print()
print("=== per day, all versions pooled ===")
day = collections.defaultdict(list)
for r in rows:
    day[r["create_time"][:10]].append(r)
for d in sorted(day):
    block(day[d], d)

print()
print("=== v22 only: old runs vs the 08-16 rerun, settled bands ===")
bands = [(0, 800), (800, 900), (900, 1000), (1000, 1100), (1100, 9999)]
groups = {
    "v22_early(13/14)": [r for r in rows if r["version"] in
                         {"v22_a", "v22_b", "v22_c", "v22_d"}],
    "v22_e(08-16)": [r for r in rows if r["version"] == "v22_e"],
    "v29(08-16)": [r for r in rows if r["version"] == "v29"],
}
for g, rs in groups.items():
    print(f"-- {g}")
    for lo, hi in bands:
        sub = [r for r in rs if r["opponent_submission"] in by_sub
               and lo <= by_sub[r["opponent_submission"]][0] < hi]
        if not sub:
            continue
        w = sum(int(r["won"]) for r in sub)
        print(f"   settled {lo:4d}-{hi:<5d} n={len(sub):3d} "
              f"{w:2d}-{len(sub)-w:<2d} wr={w/len(sub):.3f}")
    sub = [r for r in rs if r["opponent_submission"] in by_sub]
    if sub:
        w = sum(int(r["won"]) for r in sub)
        ms = sum(by_sub[r["opponent_submission"]][0] for r in sub) / len(sub)
        wr = min(max(w / len(sub), 1e-6), 1 - 1e-6)
        print(f"   ALL matched n={len(sub):3d} wr={wr:.3f} settled_mean={ms:.1f}"
              f"  implied={ms + 400*math.log10(wr/(1-wr)):.1f}")
