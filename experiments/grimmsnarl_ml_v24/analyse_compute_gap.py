"""Two joins the project has never made: compute spend, and top-40 deck share.

**Compute.**  The episode configuration is `actTimeout: 0` with a 600s
per-episode overage bank.  v22/v24 have no deadline, no lookahead and no timing
code, and spend 12s of 600.  Opponents spend up to 439s.  If the seats that
burn the bank are the seats that beat us, the unused 98% is not an
optimisation detail - it is the lever.

**Deck share.**  `top40_decks_20260814.csv` gives the deck hash behind every
top-40 submission.  Joining it with our 281 pooled games says whether the
deficit sits in lists we are guaranteed to meet while climbing, or in the tail.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

RUNS = ROOT / "data/runs/grimmsnarl"
GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_v24_games.csv"
TOP40 = ROOT / "experiments/grimmsnarl_ml_v24/top40_decks_20260814.csv"
OUT = ROOT / "experiments/grimmsnarl_ml_v24/compute_gap.json"
ELO = 400.0 / math.log(10.0)
OUR_HASH = "9714ab5c3996f6cc"


def replay_index() -> dict[str, tuple[Path, int]]:
    index: dict[str, tuple[Path, int]] = {}
    for run_dir in sorted(RUNS.iterdir()):
        manifest = run_dir / "manifest.csv"
        if not run_dir.is_dir() or not manifest.exists():
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            if row.get("detected_submission_agent_index") in {"0", "1"}:
                index[row["episode_id"]] = (
                    run_dir, int(row["detected_submission_agent_index"]))
    return index


def spend(replay: dict[str, Any], seat: int) -> float | None:
    values = []
    for step in replay.get("steps") or []:
        if seat < len(step) and isinstance(step[seat], dict):
            obs = step[seat].get("observation") or {}
            value = obs.get("remainingOverageTime")
            if isinstance(value, (int, float)):
                values.append(float(value))
    return values[0] - values[-1] if values else None


def block(rows: list[dict]) -> str:
    n = len(rows)
    if not n:
        return "n=0"
    wins = sum(1 for r in rows if r["won"])
    low, high = wilson(wins, n)
    opp = [r["opponent_rating"] for r in rows if r["opponent_rating"]]
    return (f"n={n:>3} {wins:>3}-{n - wins:<3} {wins / n:.3f} "
            f"[{low:.3f},{high:.3f}]  opp {sum(opp) / len(opp):.0f}"
            if opp else f"n={n:>3} {wins:>3}-{n - wins:<3} {wins / n:.3f}")


def fit(rows: list[dict], name: str, value) -> dict[str, Any]:
    X, y = [], []
    for r in rows:
        v = value(r)
        if r["opponent_rating"] is None or r["went_first"] is None or v is None:
            continue
        X.append([r["opponent_rating"] / 400.0, float(r["went_first"]), float(v)])
        y.append(int(r["won"]))
    X, y = np.asarray(X, float), np.asarray(y, int)
    if len(y) < 12 or len(set(y.tolist())) < 2 or len(set(X[:, 2].tolist())) < 2:
        return {"n": int(len(y)), "error": "insufficient variation"}
    model = LogisticRegression(penalty=None, max_iter=8000).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    design = np.hstack([X, np.ones((len(X), 1))])
    try:
        cov = np.linalg.inv(design.T @ np.diag(p * (1 - p)) @ design)
    except np.linalg.LinAlgError:
        return {"n": int(len(y)), "error": "singular"}
    se = float(np.sqrt(np.diag(cov))[2])
    beta = float(model.coef_[0][2])
    z = beta / se
    return {
        "n": int(len(y)), "term": name, "elo": round(beta * ELO, 1),
        "z": round(z, 2),
        "p": round(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))), 4),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    index = replay_index()
    top40 = list(csv.DictReader(TOP40.open(encoding="utf-8-sig")))
    slots = Counter(r["deck_hash"] for r in top40)
    best = {}
    for r in top40:
        best[r["deck_hash"]] = max(
            best.get(r["deck_hash"], 0.0), float(r["leaderboard_score"]))

    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not raw["version"].startswith(("v22", "v24")):
            continue
        entry = index.get(raw["episode_id"])
        if entry is None:
            continue
        run_dir, seat = entry
        path = (run_dir / "episodes" / raw["episode_id"] / "replay"
                / f"episode_{raw['episode_id']}.json")
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "episode_id": raw["episode_id"],
            "version": raw["version"],
            "won": raw["won"] == "True",
            "went_first": raw["went_first"] == "True",
            "opponent_rating": float(raw["opponent_rating"]) if raw["opponent_rating"] else None,
            "family": raw["opponent_family"] or "unknown",
            "deck_hash": raw["opponent_deck_hash"] or "unknown",
            "our_spend": spend(replay, seat),
            "opp_spend": spend(replay, 1 - seat),
        })

    timed = [r for r in rows if r["opp_spend"] is not None]
    print(f"games with timing: {len(timed)} / {len(rows)}\n")

    print("=== our record by opponent compute spend ===")
    def bucket(r):
        s = r["opp_spend"]
        if s < 20:
            return "opp <20s (no search)"
        if s < 60:
            return "opp 20-60s"
        if s < 150:
            return "opp 60-150s"
        return "opp >=150s (deep search)"
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in timed:
        buckets[bucket(r)].append(r)
    for label in ("opp <20s (no search)", "opp 20-60s", "opp 60-150s",
                  "opp >=150s (deep search)"):
        if label in buckets:
            print(f"  {label:<26}{block(buckets[label])}")
    print()
    print("  controlled for opponent rating and turn order:")
    print("   ", json.dumps(fit(timed, "opp_spend_seconds",
                                lambda r: r["opp_spend"]), ensure_ascii=False))
    print("   ", json.dumps(fit(timed, "opp_spend>=60s",
                                lambda r: r["opp_spend"] >= 60), ensure_ascii=False))
    print()

    print("=== opponent compute spend by their rating band ===")
    for lo, hi in ((0, 800), (800, 900), (900, 1000), (1000, 1100), (1100, 9999)):
        items = [r for r in timed
                 if r["opponent_rating"] and lo <= r["opponent_rating"] < hi]
        if not items:
            continue
        spends = [r["opp_spend"] for r in items]
        deep = sum(1 for s in spends if s >= 60)
        print(f"  {lo:>4}-{hi:<5} n={len(items):>3}  mean {np.mean(spends):7.1f}s  "
              f"median {np.median(spends):7.1f}s  max {max(spends):7.1f}s  "
              f">=60s: {deep} ({deep / len(items):.0%})")
    print(f"\n  our own spend: mean {np.mean([r['our_spend'] for r in rows if r['our_spend']]):.1f}s")
    print()

    print("=== our record vs the decks that hold the top 40 ===")
    record: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    family: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        record[r["deck_hash"]][0] += 1
        record[r["deck_hash"]][1] += int(r["won"])
        family[r["deck_hash"]][r["family"]] += 1
    total = sum(slots.values())
    covered = 0.0
    expected_terms = []
    print(f"  {'deck_hash':<20}{'slots':>6}{'share':>7}{'best':>7}"
          f"{'n':>5}{'record':>9}{'wr':>7}  family")
    for h, n_slots in slots.most_common():
        n, wins = record.get(h, [0, 0])
        share = n_slots / total
        wr = wins / n if n else None
        if n >= 3:
            covered += share
            expected_terms.append(share * wr)
        print(f"  {h:<20}{n_slots:>6}{share:>7.3f}{best[h]:>7.0f}{n:>5}"
              f"{f'{wins}-{n - wins}':>9}"
              f"{(f'{wr:.3f}' if wr is not None else '-'):>7}"
              f"  {family[h].most_common(1)[0][0] if n else ''}")
    expected = sum(expected_terms) / covered if covered else None
    print(f"\n  top-40 slot-weighted expected win rate: "
          f"{expected:.4f} (covering {covered:.0%} of slots)")

    payload = {
        "compute": {
            "our_mean_spend_s": round(float(np.mean(
                [r["our_spend"] for r in rows if r["our_spend"]])), 2),
            "opp_mean_spend_s": round(float(np.mean(
                [r["opp_spend"] for r in timed])), 2),
            "bank_s": 600.0,
            "buckets": {k: {"games": len(v),
                            "wins": sum(1 for r in v if r["won"])}
                        for k, v in buckets.items()},
            "controlled_seconds": fit(timed, "opp_spend_seconds",
                                      lambda r: r["opp_spend"]),
            "controlled_deep": fit(timed, "opp_spend>=60s",
                                   lambda r: r["opp_spend"] >= 60),
        },
        "top40_expected_win_rate": expected,
        "by_deck": {
            h: {"slots": slots[h], "best": best[h],
                "games": record.get(h, [0, 0])[0],
                "wins": record.get(h, [0, 0])[1]}
            for h in slots
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
