"""Live routing and per-context agreement against the held-out teacher block.

Every offline number for this agent is measured on teacher replays.  The ladder
is a different distribution, and the deployed runtime has gates (OOD, optional,
single-semantic) that can fire on one and not the other.  This puts the two
reports side by side so a difference in *routing* is not mistaken for a
difference in *skill*.

Usage:
  python scripts/compare_dragapult_live_vs_test.py \
      --live experiments/dragapult_ml_v2/runtime_eval_live_v2_26g.json \
      --test experiments/dragapult_ml_v2/runtime_eval_v2full_newtest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONTEXTS = {
    0: "MAIN", 1: "mulligan", 2: "starter", 3: "search-card", 4: "tool",
    5: "energy-card", 7: "TO_HAND", 8: "DISCARD", 13: "DAMAGE_COUNTER",
    14: "DAMAGE_COUNTER_ANY (Phantom Dive)", 16: "coin/other",
    21: "ATTACH_FROM", 22: "ATTACH_TO", 30: "DISCARD_ENERGY",
    40: "bench-out", 41: "prize", 43: "misc",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--min-decisions", type=int, default=15)
    args = parser.parse_args()

    live = json.loads(args.live.read_text(encoding="utf-8"))
    test = json.loads(args.test.read_text(encoding="utf-8"))

    print(f"live  {live['episodes']:>4} episodes  {live['decisions']:>6} "
          f"decisions  agreement {live['semantic_agreement']:.4f} "
          f"{live['semantic_agreement_wilson95']}")
    print(f"test  {test['episodes']:>4} episodes  {test['decisions']:>6} "
          f"decisions  agreement {test['semantic_agreement']:.4f} "
          f"{test['semantic_agreement_wilson95']}")

    print("\nrouting (share of decisions seen by the ranker):")
    print(f"{'':30} {'live':>10} {'test':>10}")
    live_stats = live["diagnostics"]["ml"]
    test_stats = test["diagnostics"]["ml"]
    live_total = max(1, live_stats["decisions_seen"])
    test_total = max(1, test_stats["decisions_seen"])
    for key in ("decisions_seen", "ranker_used", "unrouted", "optional_fallback",
                "ood_fallback", "single_semantic_fallback", "feature_errors",
                "score_errors"):
        left = live_stats.get(key, 0)
        right = test_stats.get(key, 0)
        if key == "decisions_seen":
            print(f"{key:30} {left:>10} {right:>10}")
            continue
        print(f"{key:30} {left / live_total:>10.4f} {right / test_total:>10.4f}"
              f"   ({left} / {right})")

    print("\nper-context agreement:")
    print(f"{'ctx':>4} {'name':34} {'live_n':>7} {'live':>7} "
          f"{'test_n':>7} {'test':>7} {'delta':>8}")
    keys = sorted(set(live["by_context"]) | set(test["by_context"]), key=int)
    for key in keys:
        left = live["by_context"].get(key)
        right = test["by_context"].get(key)
        if not left or left["decisions"] < args.min_decisions:
            continue
        name = CONTEXTS.get(int(key), "")
        if not right:
            print(f"{key:>4} {name:34} {left['decisions']:>7} "
                  f"{left['semantic_top1']:>7.4f} {'-':>7} {'-':>7} {'-':>8}")
            continue
        delta = left["semantic_top1"] - right["semantic_top1"]
        print(f"{key:>4} {name:34} {left['decisions']:>7} "
              f"{left['semantic_top1']:>7.4f} {right['decisions']:>7} "
              f"{right['semantic_top1']:>7.4f} {delta:>+8.4f}")

    print("\nper-slice agreement:")
    print(f"{'slice':20} {'live_n':>7} {'live':>7} {'test_n':>7} {'test':>7}")
    for key in sorted(set(live["by_slice"]) | set(test["by_slice"])):
        left = live["by_slice"].get(key, {})
        right = test["by_slice"].get(key, {})
        print(f"{key:20} {left.get('decisions', 0):>7} "
              f"{left.get('semantic_top1', 0):>7.4f} "
              f"{right.get('decisions', 0):>7} "
              f"{right.get('semantic_top1', 0):>7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
