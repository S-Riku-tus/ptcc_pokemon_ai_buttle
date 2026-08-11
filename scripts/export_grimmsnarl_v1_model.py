"""Distil a trained Grimmsnarl ranker into the standard-library runtime model.

The runtime cannot import LightGBM, so the booster is dumped to the compact
tree JSON that ``ml_runtime.tree_score`` walks. For a teacher-conditioned
model the chosen pilot's dense category code is baked in, because the runtime
has no team id to read off the observation - it *is* the pilot now.

The exported feature order is the booster's own, and the runtime builds its
vectors from that list by name, so a feature reordering cannot silently
misalign the two.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.distill import compact_booster  # noqa: E402


def team_codes(corpus_path: Path) -> dict[int, int]:
    """Same dense mapping the trainer builds: sorted team ids to 0..N-1."""
    data = np.load(corpus_path, allow_pickle=False)
    teams = sorted({int(x) for x in data["team_ids"]})
    return {team: index for index, team in enumerate(teams)}


def routed_contexts(
    report: dict,
    min_support: int,
    min_top1: float,
) -> list[int]:
    """Contexts the ranker is allowed to decide, from measured support.

    The first cut of v2 routed every context in ``SCORABLE_CONTEXTS`` through
    the model unconditionally. Context 8 (DISCARD) had 9 held-out decisions and
    scored 22%, which is a coin flip on a sample too small to have an opinion
    about - shipping it means replacing a rule with noise. A context earns the
    ranker by having enough training decisions to fit, or by scoring well
    enough on held-out data that the thin sample is not in doubt.

    Returned rather than hard-coded so the list re-derives itself every retrain
    instead of drifting away from the corpus it describes.
    """
    support = {
        int(k): int(v) for k, v in
        (report.get("train_context_support") or {}).items()
    }
    scored = {
        int(k): float(v["top1"]) for k, v in
        ((report.get("validation") or {}).get("top1_by_context") or {}).items()
    }
    routed = []
    dropped = []
    for context in sorted(set(support) | set(scored)):
        count = support.get(context, 0)
        top1 = scored.get(context)
        if count >= min_support or (top1 is not None and top1 >= min_top1
                                    and count >= 50):
            routed.append(context)
        else:
            dropped.append({
                "context": context, "train_decisions": count,
                "validation_top1": top1,
            })
    if dropped:
        print(f"contexts left to the rule policy: {dropped}")
    return routed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-team", type=int)
    parser.add_argument(
        "--route-teachers",
        default="",
        help=(
            "Optional public-route:team-id pairs.  The conditioned runtime "
            "may then select one learned pilot for each matchup without "
            "mixing scores inside an argmax."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--num-iteration", type=int,
        help="Export only the first N trees of the trained booster.",
    )
    parser.add_argument(
        "--report", type=Path,
        help="Training report; used to bake the routed-context allow list.",
    )
    parser.add_argument(
        "--min-context-support", type=int, default=400,
        help="Training decisions a context needs before the ranker owns it.",
    )
    parser.add_argument(
        "--min-context-top1", type=float, default=0.60,
        help="Held-out Top-1 a thin context needs before the ranker owns it.",
    )
    args = parser.parse_args()

    booster = lgb.Booster(model_file=str(args.model))
    if args.num_iteration is not None and args.num_iteration <= 0:
        parser.error("--num-iteration must be positive")
    model = compact_booster(
        booster,
        kind="grimmsnarl_ranker",
        num_iteration=args.num_iteration,
    )

    if args.report and args.report.exists():
        model["routed_contexts"] = routed_contexts(
            json.loads(args.report.read_text(encoding="utf-8")),
            args.min_context_support,
            args.min_context_top1,
        )
        # Keep the gate reproducible in the deployed artifact.  v2.1 uses a
        # deliberately lower threshold for context 8 after an independent
        # wall-heavy replay set showed that its inherited rule policy was far
        # worse than the ranker despite the tiny validation sample.
        model["routing_min_context_support"] = args.min_context_support
        model["routing_min_context_top1"] = args.min_context_top1

    if "teacher_team_id" in model["feature_names"]:
        if args.teacher_team is None:
            raise SystemExit(
                "model is teacher-conditioned; pass --teacher-team"
            )
        codes = team_codes(args.corpus)
        if args.teacher_team not in codes:
            raise SystemExit(
                f"team {args.teacher_team} is not in the corpus"
            )
        model["teacher_team_id"] = args.teacher_team
        model["teacher_team_code"] = codes[args.teacher_team]
        route_teams: dict[str, int] = {}
        for pair in args.route_teachers.split(","):
            if not pair.strip():
                continue
            if ":" not in pair:
                raise SystemExit(f"invalid --route-teachers pair: {pair}")
            route, raw_team = pair.split(":", 1)
            team = int(raw_team)
            if team not in codes:
                raise SystemExit(
                    f"route teacher {team} is not in the corpus"
                )
            route_teams[route.strip()] = team
        if route_teams:
            model["route_teacher_teams"] = route_teams
            model["route_teacher_codes"] = {
                route: codes[team] for route, team in route_teams.items()
            }
    elif args.teacher_team is not None:
        raise SystemExit(
            "model has no teacher_team_id feature; --teacher-team is unused"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(model, separators=(",", ":")), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "trees": len(model["trees"]),
        "features": len(model["feature_names"]),
        "teacher_team_id": model.get("teacher_team_id"),
        "teacher_team_code": model.get("teacher_team_code"),
        "route_teacher_teams": model.get("route_teacher_teams"),
        "route_teacher_codes": model.get("route_teacher_codes"),
        "routed_contexts": model.get("routed_contexts"),
        "bytes": args.output.stat().st_size,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
