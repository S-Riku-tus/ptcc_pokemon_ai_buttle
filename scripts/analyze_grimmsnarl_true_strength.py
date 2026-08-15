"""Our record against opponents graded by their settled leaderboard standing.

``opponent_rating`` in ``version_games.csv`` is the rating the opponent held at
the moment we met them, which on this ladder is a noisy and often unconverged
number - a strong agent twenty games into its run is still sitting at 700.
The newest full leaderboard snapshot gives a much better strength estimate for
the same submissions: the team's *current* score, after hundreds of games.

About 40% of our 445 opponents can be matched to a team that way.  On that
subset this script asks the only question that matters for climbing:

* what is our win rate against opponents who are genuinely strong, as opposed
  to opponents who merely happened to be rated highly that afternoon;
* how much of the field is genuinely strong, i.e. how much exposure to those
  opponents a 34-game run even gets;
* which deck families make up the strong end, and where our record against
  them actually is.

It also runs the within-day bootstrap that the pooled version cannot: resample
34 games from the 08-13 ladder and from the 08-15 ladder separately, with the
same fitted K schedule, so "the same policy on a different day" is priced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402
from analyze_grimmsnarl_rating_mechanics import (  # noqa: E402
    fit_k, load, replay,
)

TOP = ROOT / "data/kaggle_top100/latest/raw/api"
POOL = ("v22_a", "v22_b", "v22_c", "v22_d", "v26", "v27")


def elo(w: float) -> float:
    w = min(max(w, 1e-4), 1 - 1e-4)
    return 400 * math.log10(w / (1 - w))


def submission_index() -> tuple[dict[int, int], dict[int, float]]:
    board = json.loads(
        (TOP / "leaderboard_full.json").read_text(encoding="utf-8")
    )["publicLeaderboard"]
    sub_to_team: dict[int, int] = {}
    team_score: dict[int, float] = {}
    for row in board:
        try:
            score = float(row["displayScore"])
        except (TypeError, ValueError):
            continue
        team = int(row["teamId"])
        team_score[team] = score
        if row.get("submissionId"):
            sub_to_team[int(row["submissionId"])] = team
    for path in (TOP / "team_public_submissions").glob("team_*.json"):
        team = int(path.stem.split("_")[1])
        for sub in json.loads(path.read_text(encoding="utf-8"))["submissions"]:
            sub_to_team.setdefault(int(sub["id"]), team)
    return sub_to_team, team_score


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/true_strength.json",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    payload: dict = {}

    by_run = load()
    schedule, _ = fit_k(by_run)
    sub_to_team, team_score = submission_index()

    rows = [row for games in by_run.values() for row in games]
    for row in rows:
        team = sub_to_team.get(int(row["opponent_submission"] or 0))
        row["true"] = team_score.get(team) if team else None
        row["day"] = row["create_time"][:10]
    matched = [row for row in rows if row["true"] is not None]

    print("=== 1. how well does the rating we saw predict real strength? ===")
    print(f"matched {len(matched)} of {len(rows)} games "
          f"({len(matched) / len(rows):.1%}) to a current leaderboard team")
    seen = np.array([row["opp"] for row in matched])
    true = np.array([row["true"] for row in matched])
    print(f"  rating at the time : mean {seen.mean():.1f}  sd {seen.std():.1f}")
    print(f"  current team score : mean {true.mean():.1f}  sd {true.std():.1f}")
    print(f"  correlation        : {float(np.corrcoef(seen, true)[0, 1]):.3f}")
    print(f"  mean understatement: {float((true - seen).mean()):+.1f} points")
    payload["match_rate"] = len(matched) / len(rows)
    payload["understatement"] = float((true - seen).mean())
    print("\nthe ladder rating we faced understates the opponent, because most")
    print("opponents are met mid-climb. Grading by settled score is stricter.")

    print("\n=== 2. our record by the opponent's settled strength ===")
    bands = ((0, 700), (700, 850), (850, 1000), (1000, 1100), (1100, 9999))
    print(f"{'settled score':<16}{'n':>5}{'record':>10}{'wr':>8}"
          f"{'wilson':>18}{'seen rating':>13}")
    for low, high in bands:
        group = [r for r in matched if low <= r["true"] < high]
        if not group:
            continue
        wins = sum(r["w"] for r in group)
        n = len(group)
        lo, hi = wilson(wins, n)
        print(f"{f'{low}-{high}':<16}{n:>5}{f'{wins}-{n - wins}':>10}"
              f"{wins / n:>8.3f}{f'[{lo:.2f},{hi:.2f}]':>18}"
              f"{np.mean([r['opp'] for r in group]):>13.0f}")
        payload.setdefault("by_true", {})[f"{low}-{high}"] = {
            "n": n, "wins": wins, "wr": wins / n,
        }

    print("\nsame, split by the seat we drew:")
    for low, high in bands:
        group = [r for r in matched if low <= r["true"] < high]
        if len(group) < 8:
            continue
        line = f"{f'{low}-{high}':<16}"
        for order in ("first", "second"):
            sub = [r for r in group if r["went_first"] == order]
            if not sub:
                line += f"  {order} n/a"
                continue
            wins = sum(r["w"] for r in sub)
            line += (f"  {order} {wins:>2}-{len(sub) - wins:<2} "
                     f"{wins / len(sub):.3f}")
        print(line)

    print("\n=== 3. exposure: how much of a run is against real strength? ===")
    print(f"{'run':<8}{'matched':>9}{'>=1000':>9}{'>=1100':>9}{'mean true':>11}")
    for run in sorted(by_run):
        group = [r for r in by_run[run] if r["true"] is not None]
        if not group:
            continue
        strong = sum(1 for r in group if r["true"] >= 1000)
        elite = sum(1 for r in group if r["true"] >= 1100)
        print(f"{run:<8}{len(group):>9}{strong:>9}{elite:>9}"
              f"{np.mean([r['true'] for r in group]):>11.0f}")

    print("\n=== 4. which decks make up the strong end ===")
    strong = [r for r in matched if r["true"] >= 1000]
    print(f"opponents with a settled score >= 1000: {len(strong)} games")
    by_family: dict[str, list] = defaultdict(list)
    for row in strong:
        by_family[row["opponent_family"]].append(row)
    print(f"{'family':<34}{'n':>4}{'record':>9}{'wr':>8}")
    for family, group in sorted(by_family.items(), key=lambda i: -len(i[1])):
        wins = sum(r["w"] for r in group)
        print(f"{family:<34}{len(group):>4}"
              f"{f'{wins}-{len(group) - wins}':>9}{wins / len(group):>8.3f}")

    print("\nsame families, but every game we played against them "
          "(any strength):")
    print(f"{'family':<34}{'n':>4}{'record':>9}{'wr':>8}{'vs>=1000':>10}")
    for family in sorted(by_family, key=lambda f: -len(by_family[f])):
        group = [r for r in rows if r["opponent_family"] == family]
        wins = sum(r["w"] for r in group)
        elite = by_family[family]
        ewins = sum(r["w"] for r in elite)
        print(f"{family:<34}{len(group):>4}"
              f"{f'{wins}-{len(group) - wins}':>9}{wins / len(group):>8.3f}"
              f"{f'{ewins}-{len(elite) - ewins}':>10}")

    print("\n=== 5. same policy, different day: within-day bootstrap ===")
    pool = [r for run in POOL for r in by_run[run]]
    for label, subset in (
        ("08-13/14 ladder", [r for r in pool if r["day"] < "2026-08-15"]),
        ("08-15 ladder", [r for r in pool if r["day"] >= "2026-08-15"]),
    ):
        wins = sum(r["w"] for r in subset)
        pairs = np.array([(r["opp"], r["w"]) for r in subset], dtype=float)
        finals = np.empty(args.draws)
        for draw in range(args.draws):
            finals[draw] = replay(
                pairs[rng.integers(0, len(pairs), 34)], schedule
            )
        lo, hi = wilson(wins, len(subset))
        print(f"{label:<18} {wins}-{len(subset) - wins} "
              f"({wins / len(subset):.3f}) over {len(subset)} games, "
              f"Wilson [{lo:.3f}, {hi:.3f}]")
        print(f"{'':18} opponent mean {np.mean([r['opp'] for r in subset]):.0f}"
              f"  ->  34-game rating {finals.mean():.0f} "
              f"+/- {finals.std():.0f}  "
              f"[p05 {np.percentile(finals, 5):.0f}, "
              f"p95 {np.percentile(finals, 95):.0f}]")
        payload.setdefault("within_day", {})[label] = {
            "n": len(subset), "wins": wins,
            "mean": float(finals.mean()), "sd": float(finals.std()),
        }
    early = [r for r in pool if r["day"] < "2026-08-15"]
    late = [r for r in pool if r["day"] >= "2026-08-15"]
    table = [
        [sum(r["w"] for r in early), len(early) - sum(r["w"] for r in early)],
        [sum(r["w"] for r in late), len(late) - sum(r["w"] for r in late)],
    ]
    print(f"\nsame policy, the two days differ at Fisher "
          f"p = {float(fisher_exact(table).pvalue):.4f}")

    print("\n=== 6. what win rate buys what rank ===")
    print("a settled rating is opponent mean + 400 log10(w/(1-w)). Against an")
    print("880-rated slice of the field, which is what a converged run meets:")
    print(f"  {'win rate':>10}{'rating':>10}{'rank on 08-14':>16}")
    board = json.loads(
        (TOP / "leaderboard_full.json").read_text(encoding="utf-8")
    )["publicLeaderboard"]
    scores = np.array(sorted(
        (float(r["displayScore"]) for r in board
         if r["displayScore"] not in (None, "")), reverse=True
    ))
    for wr in (0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80):
        rating = 880 + elo(wr)
        rank = int((scores > rating).sum()) + 1
        print(f"  {wr:>10.2f}{rating:>10.1f}{rank:>16}")
        payload.setdefault("wr_to_rank", {})[f"{wr:.2f}"] = rank

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
