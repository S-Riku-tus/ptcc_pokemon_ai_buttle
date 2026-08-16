"""How fast does a fresh submission accumulate rated games, and how fast does
its rating converge? Both decide whether a late resubmission can pay for the
600-point reset it costs."""
import csv
import collections
import datetime as dt

rows = list(csv.DictReader(open("experiments/grimmsnarl_endgame_20260816/version_games.csv",
                                encoding="utf-8-sig")))
by = collections.defaultdict(list)
for r in rows:
    by[r["version"]].append(r)

print(f"{'ver':7s} {'n':>3s} {'first game (UTC)':20s} {'last game':20s} "
      f"{'hours':>6s} {'g/h':>5s} {'final':>7s}")
for v, rs in by.items():
    rs.sort(key=lambda r: r["create_time"])
    t0 = dt.datetime.strptime(rs[0]["create_time"][:19], "%Y-%m-%dT%H:%M:%S")
    t1 = dt.datetime.strptime(rs[-1]["create_time"][:19], "%Y-%m-%dT%H:%M:%S")
    hrs = (t1 - t0).total_seconds() / 3600
    print(f"{v:7s} {len(rs):3d} {str(t0)[:19]:20s} {str(t1)[:19]:20s} "
          f"{hrs:6.1f} {len(rs)/max(hrs,1e-6):5.2f} "
          f"{float(rs[-1]['our_rating_after']):7.1f}")

print()
print("=== rating trajectory: mean rating after game N, across runs ===")
traj = collections.defaultdict(list)
for v, rs in by.items():
    rs.sort(key=lambda r: r["create_time"])
    for i, r in enumerate(rs, 1):
        traj[i].append(float(r["our_rating_after"]))
for n in (1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45):
    if n in traj:
        vals = traj[n]
        print(f"  after {n:2d} games: n_runs={len(vals):2d} "
              f"mean={sum(vals)/len(vals):7.1f} "
              f"min={min(vals):7.1f} max={max(vals):7.1f}")

print()
print("=== how much did rating still move after game 20? ===")
for v, rs in by.items():
    rs.sort(key=lambda r: r["create_time"])
    if len(rs) > 20:
        print(f"  {v:7s} at20={float(rs[19]['our_rating_after']):7.1f} "
              f"final={float(rs[-1]['our_rating_after']):7.1f} "
              f"delta={float(rs[-1]['our_rating_after'])-float(rs[19]['our_rating_after']):+7.1f}"
              f"  ({len(rs)-20} more games)")
