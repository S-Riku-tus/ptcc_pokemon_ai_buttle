"""What is a submission slot reroll worth, and when is it safe?

Mechanics established elsewhere in this directory and in
`experiments/grimmsnarl_ml_v27/WHY_THE_RATING.md`:

* a team has exactly two live submission slots;
* the public board shows the MAX of the two;
* a new submission truncates the OLDEST live slot;
* a run's final rating is a draw with sd ~= 63 around the agent's fixed point,
  and it converges in 2-12 hours.

The consequence is a stopping rule rather than a resubmission cadence. With
state (old, new) the displayed score is max(old, new); submitting replaces the
pair with (new, fresh), so the post-submission floor is `new`. Submitting is
free of downside exactly when new >= old, and it destroys `old - new` points of
displayed score otherwise.

This script prices that rule against the live board.
"""
import bisect
import json
import random
import statistics

random.seed(20260816)

BOARD = "experiments/grimmsnarl_endgame_20260816/leaderboard_full_20260816.json"
SD = 63.0                      # single-run noise, WHY_THE_RATING section 6
OLD, NEW = 847.5, 886.2        # live slots at 2026-08-16 15:0x JST
TRIALS = 200000

scores = sorted(
    (float(r["displayScore"]) for r in
     json.load(open(BOARD, encoding="utf-8"))["publicLeaderboard"]),
    reverse=True,
)


def rank_of(score: float) -> int:
    lo, hi = 0, len(scores)
    while lo < hi:
        mid = (lo + hi) // 2
        if scores[mid] > score:
            lo = mid + 1
        else:
            hi = mid
    return lo + 1


def simulate(mu: float, cycles: int, old: float, new: float,
             both_slots: bool = False) -> list[float]:
    """Return the displayed score at the deadline over TRIALS runs."""
    out = []
    for _ in range(TRIALS):
        o, n = old, new
        if both_slots:
            # Burn both slots at once: (F1, F2), no floor kept.
            a = random.gauss(mu, SD)
            b = random.gauss(mu, SD)
            out.append(max(a, b))
            continue
        for _ in range(cycles):
            if n < o:
                break            # submitting here would destroy the max
            o, n = n, random.gauss(mu, SD)
        out.append(max(o, n))
    return out


def block(label: str, vals: list[float]) -> None:
    vals_sorted = sorted(vals)
    mean = statistics.fmean(vals)
    p10 = vals_sorted[int(0.10 * len(vals))]
    p90 = vals_sorted[int(0.90 * len(vals))]
    worse = sum(1 for v in vals if v < max(OLD, NEW)) / len(vals)
    print(f"{label:34s} mean={mean:7.1f} (rank ~{rank_of(mean):4d})  "
          f"p10={p10:7.1f}  p90={p90:7.1f}  "
          f"P(worse than now)={worse:5.1%}")


print(f"live slots: old={OLD} new={NEW}  displayed={max(OLD, NEW)} "
      f"(rank {rank_of(max(OLD, NEW))} of {len(scores)})")
print(f"single-run sd assumed {SD}\n")

for mu in (860.0, 890.0, 920.0, 950.0):
    print(f"=== assumed fixed point on today's scale: {mu:.0f} ===")
    block("do nothing", [max(OLD, NEW)] * 1000)
    block("burn both slots now (2 fresh)", simulate(mu, 0, OLD, NEW, True))
    for c in (1, 2, 3, 4, 6):
        block(f"reroll, up to {c} cycle(s)", simulate(mu, c, OLD, NEW))
    print()

print("Reading it: 'reroll' follows the stopping rule - submit only while the "
      "newer live slot is at least the older one. Under that rule the floor "
      "never falls below today's 886.2, which is why P(worse than now) is 0.")
