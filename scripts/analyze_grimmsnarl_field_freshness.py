"""Was v27's low-rated pairing draw genuinely weak, or just freshly submitted?

A Kaggle simulation rating starts at 600 and climbs, so at any moment the
600-900 pool contains two very different populations: agents that belong there
and strong agents that resubmitted an hour ago.  ``initialScore`` cannot tell
them apart, which matters because every controlled comparison in this repo
uses ``initialScore`` as the strength control.

Two independent proxies are computed here, both from data already on disk:

1. *Rating volatility.*  Kaggle's update size shrinks as a submission
   accumulates games, so ``|updatedScore - initialScore|`` for the opponent is
   a monotone proxy for how early in its own run the opponent was.
2. *Top-60 membership.*  ``data/kaggle_top100/latest`` stores each top-60
   team's public submissions, which maps an opponent submission id onto a team
   and that team's current leaderboard score - regardless of what the opponent
   was rated when we played it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

GROUPS = {
    "v22": ("v22_a", "v22_b", "v22_c", "v22_d"),
    "v24": ("v24_a", "v24_b"),
    "v25": ("v25_a", "v25_b"),
    "v26": ("v26",),
    "v27": ("v27",),
}
RUNS = {
    "v22_a": "data/runs/grimmsnarl/20260813_grimmsnarl_ml_v22_sub55479857",
    "v22_b": "data/runs/grimmsnarl/20260814_grimmsnarl_ml_v22_b_sub55483874",
    "v22_c": "data/runs/grimmsnarl/20260814_grimmsnarl_ml_v22_c_sub55486680",
    "v22_d": "data/runs/grimmsnarl/20260814_grimmsnarl_ml_v22_d_sub55486691",
    "v24_a": "data/runs/grimmsnarl/20260814_grimmsnarl_ml_v24_a_sub55496021",
    "v24_b": "data/runs/grimmsnarl/20260814_grimmsnarl_ml_v24_b_sub55496665",
    "v25_a": "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v25_sub55507909",
    "v25_b": "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v25_b_sub55517142",
    "v26": "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v26_sub55520389",
    "v27": "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v27_sub55521760",
}
SUBMISSIONS = {
    "v22_a": 55479857, "v22_b": 55483874, "v22_c": 55486680, "v22_d": 55486691,
    "v24_a": 55496021, "v24_b": 55496665, "v25_a": 55507909, "v25_b": 55517142,
    "v26": 55520389, "v27": 55521760,
}


def submission_to_team() -> dict[int, dict[str, Any]]:
    """Map every known public submission id onto its team and team score."""
    root = ROOT / "data/kaggle_top100/latest"
    scores: dict[int, float] = {}
    board = root / "raw/api/leaderboard_full.json"
    if board.exists():
        data = json.loads(board.read_text(encoding="utf-8"))
        for row in data.get("publicLeaderboard") or []:
            try:
                scores[int(row["teamId"])] = float(row["displayScore"])
            except (KeyError, TypeError, ValueError):
                continue
    mapping: dict[int, dict[str, Any]] = {}
    folder = root / "raw/api/team_public_submissions"
    for path in sorted(folder.glob("team_*.json")) if folder.exists() else []:
        try:
            team_id = int(path.stem.split("_")[1])
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError):
            continue
        rows = payload if isinstance(payload, list) else (
            payload.get("submissions")
            or payload.get("list")
            or payload.get("publicSubmissions")
            or []
        )
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            sid = row.get("id") or row.get("submissionId")
            try:
                sid = int(sid)
            except (TypeError, ValueError):
                continue
            mapping[sid] = {
                "team_id": team_id,
                "team_score": scores.get(team_id),
            }
    # A team's leaderboard submission is also a known submission of that team.
    if board.exists():
        data = json.loads(board.read_text(encoding="utf-8"))
        for row in data.get("publicLeaderboard") or []:
            try:
                sid = int(row["submissionId"])
                tid = int(row["teamId"])
            except (KeyError, TypeError, ValueError):
                continue
            mapping.setdefault(sid, {"team_id": tid, "team_score": scores.get(tid)})
    return mapping


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/field_freshness.json",
    )
    args = parser.parse_args()

    mapping = submission_to_team()
    print(f"known submission -> team entries: {len(mapping)}")

    rows: list[dict[str, Any]] = []
    for label, path in RUNS.items():
        run = ROOT / path
        if not (run / "episodes.csv").exists():
            continue
        submission = SUBMISSIONS[label]
        for meta in csv.DictReader((run / "episodes.csv").open(encoding="utf-8-sig")):
            if meta.get("state") != "COMPLETED":
                continue
            if meta.get("agent_0_submission_id") == str(submission):
                seat = 0
            elif meta.get("agent_1_submission_id") == str(submission):
                seat = 1
            else:
                continue

            def number(key: str) -> float | None:
                try:
                    return float(meta[key])
                except (KeyError, TypeError, ValueError):
                    return None

            opp_initial = number(f"agent_{1 - seat}_initial_score")
            opp_updated = number(f"agent_{1 - seat}_updated_score")
            our_initial = number(f"agent_{seat}_initial_score")
            our_updated = number(f"agent_{seat}_updated_score")
            opp_submission = meta.get(f"agent_{1 - seat}_submission_id", "")
            try:
                info = mapping.get(int(opp_submission), {})
            except ValueError:
                info = {}
            rows.append({
                "version": label,
                "group": next(g for g, labels in GROUPS.items() if label in labels),
                "episode_id": meta["episode_id"],
                "create_time": meta.get("create_time", ""),
                "opp_submission": opp_submission,
                "opp_initial": opp_initial,
                "opp_swing": (
                    abs(opp_updated - opp_initial)
                    if opp_initial is not None and opp_updated is not None else None
                ),
                "our_initial": our_initial,
                "our_swing": (
                    abs(our_updated - our_initial)
                    if our_initial is not None and our_updated is not None else None
                ),
                "our_delta": (
                    our_updated - our_initial
                    if our_initial is not None and our_updated is not None else None
                ),
                "won": int((our_updated or 0) > (our_initial or 0)),
                "opp_team": info.get("team_id"),
                "opp_team_score": info.get("team_score"),
            })

    def mean(values):
        numbers = [v for v in values if v is not None]
        return sum(numbers) / len(numbers) if numbers else None

    print("\n--- opponent rating volatility (proxy for how fresh the field was) ---")
    print(f"{'group':<8}{'n':>5}{'opp mean':>10}{'opp |swing|':>13}"
          f"{'opp<700 share':>15}{'our |swing|':>13}")
    payload: dict[str, Any] = {"groups": {}}
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        if not subset:
            continue
        fresh = sum(1 for r in subset if (r["opp_initial"] or 0) < 700)
        print(
            f"{group:<8}{len(subset):>5}"
            f"{mean(r['opp_initial'] for r in subset):>10.1f}"
            f"{mean(r['opp_swing'] for r in subset):>13.2f}"
            f"{fresh / len(subset):>15.1%}"
            f"{mean(r['our_swing'] for r in subset):>13.2f}"
        )
        payload["groups"][group] = {
            "games": len(subset),
            "opp_mean": round(mean(r["opp_initial"] for r in subset), 1),
            "opp_swing": round(mean(r["opp_swing"] for r in subset), 2),
            "our_swing": round(mean(r["our_swing"] for r in subset), 2),
        }

    print("\n--- how much rating each game was worth to us ---")
    print("Elo income per game, split by whether we won, per group.")
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group and r["our_delta"] is not None]
        if not subset:
            continue
        wins = [r["our_delta"] for r in subset if r["our_delta"] > 0]
        losses = [r["our_delta"] for r in subset if r["our_delta"] <= 0]
        print(
            f"{group:<8} win {mean(wins):+6.2f} (n={len(wins):>3})   "
            f"loss {mean(losses):+6.2f} (n={len(losses):>3})   "
            f"net {sum(r['our_delta'] for r in subset):+8.1f}"
        )

    print("\n--- opponents that belong to a current top-60 team ---")
    for group in GROUPS:
        subset = [r for r in rows if r["group"] == group]
        if not subset:
            continue
        known = [r for r in subset if r["opp_team_score"] is not None]
        strong = [r for r in known if r["opp_team_score"] >= 1000]
        beat = sum(r["won"] for r in strong)
        print(
            f"{group:<8} identified {len(known):>3}/{len(subset):<3} "
            f"({len(known) / len(subset):.0%})   of those team>=1000: "
            f"{len(strong):>3}   our record there {beat}-{len(strong) - beat}"
        )
        if strong:
            gap = mean(
                r["opp_team_score"] - (r["opp_initial"] or 0) for r in strong
            )
            print(f"           mean (team score - rating at pairing) = {gap:+.1f}")

    print("\n--- v27 game by game ---")
    print(f"{'episode':<10}{'time':<21}{'opp sub':<11}{'opp rating':>11}"
          f"{'opp swing':>11}{'our rating':>11}{'delta':>8}{'team':>10}{'teamLB':>9}")
    for r in sorted(
        (r for r in rows if r["group"] == "v27"), key=lambda r: r["create_time"]
    ):
        print(
            f"{r['episode_id']:<10}{r['create_time'][:19]:<21}"
            f"{r['opp_submission']:<11}"
            f"{(r['opp_initial'] or 0):>11.1f}{(r['opp_swing'] or 0):>11.2f}"
            f"{(r['our_initial'] or 0):>11.1f}{(r['our_delta'] or 0):>+8.1f}"
            f"{str(r['opp_team'] or ''):>10}{str(r['opp_team_score'] or ''):>9}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({**payload, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
