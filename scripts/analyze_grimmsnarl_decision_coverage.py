"""Which of our decisions is any model actually choosing?

The behaviour probe shows v22, v26 and v27 make the same move on 99.7% of
stored decisions, and the offer/take split shows we already take essentially
every Shadow Bullet we are offered.  If the policy is saturated on the
decisions the ranker owns, the remaining room has to be in the decisions it
does not own.

For every stored decision of our seat this counts, per select context:

* how many options were on the menu and whether ``maxCount`` was 1 - a
  multi-pick select never reaches the ranker, it falls to the rule policy;
* whether the ranker considered the select scorable at all;
* how often the answer was forced (one legal option), which is not a decision.

Contexts are reported with the card most often involved so the numbers can be
read without the engine source.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor", ROOT / "scripts",
             ROOT / "agents/grimmsnarl/grimmsnarl_ml_v22"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402
from ml_runtime import Ranker  # noqa: E402

RUN_ROOT = ROOT / "data/runs/grimmsnarl"
POOL = (
    ("v22_a", 55479857, "20260813_grimmsnarl_ml_v22_sub55479857"),
    ("v26", 55520389, "20260815_grimmsnarl_ml_v26_sub55520389"),
    ("v27", 55521760, "20260815_grimmsnarl_ml_v27_sub55521760"),
)
NAMES = {
    7: "DarkEnergy", 646: "Impidimp", 647: "Morgrem", 648: "GrimmsnarlEX",
    860: "Snorunt", 104: "Froslass", 112: "Munkidori", 1079: "RareCandy",
    1080: "UnfairStamp", 1086: "Poffin", 1097: "NightStretcher",
    1122: "Pokegear", 1137: "ToolScrapper", 1152: "PokePad", 1182: "Boss",
    1219: "Petrel", 1227: "Lillie", 1231: "Dawn", 1259: "SpikemuthGym",
}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/decision_coverage.json",
    )
    args = parser.parse_args()

    ranker = Ranker()
    stats: dict[int, Counter] = defaultdict(Counter)
    cards_seen: dict[int, Counter] = defaultdict(Counter)
    games = 0

    for label, submission, folder in POOL:
        run = RUN_ROOT / folder
        for meta in csv.DictReader(
            (run / "episodes.csv").open(encoding="utf-8-sig")
        ):
            if meta.get("state") != "COMPLETED":
                continue
            if meta.get("episode_type") == "EPISODE_TYPE_VALIDATION":
                continue
            if meta.get("agent_0_submission_id") == str(submission):
                seat = 0
            elif meta.get("agent_1_submission_id") == str(submission):
                seat = 1
            else:
                continue
            episode_id = int(meta["episode_id"])
            path = (
                run / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            if not path.exists():
                continue
            steps = json.loads(path.read_text(encoding="utf-8")).get("steps") or []
            games += 1
            for index, step in enumerate(steps[:-1]):
                if seat >= len(step):
                    continue
                record = step[seat] or {}
                if record.get("status") != "ACTIVE":
                    continue
                observation = record.get("observation") or {}
                select = observation.get("select") or {}
                options = list(select.get("option") or [])
                if not options:
                    continue
                context = int(select.get("context", -1))
                max_count = int(select.get("maxCount", 1) or 1)
                bucket = stats[context]
                bucket["decisions"] += 1
                bucket["options"] += len(options)
                if len(options) == 1:
                    bucket["forced"] += 1
                if max_count > 1:
                    bucket["multi_pick"] += 1
                if ranker.is_scorable(select):
                    bucket["scorable"] += 1
                else:
                    bucket["not_scorable"] += 1
                current = observation.get("current") or {}
                for option in options[:12]:
                    card = mf.candidate_card(current, option, select) or {}
                    ident = int(card.get("id", -1))
                    if ident > 0:
                        cards_seen[context][ident] += 1

    total = sum(bucket["decisions"] for bucket in stats.values())
    real = sum(
        bucket["decisions"] - bucket["forced"] for bucket in stats.values()
    )
    print(f"games {games}   decisions {total}   "
          f"real choices (>1 option) {real}   per game {real / games:.1f}\n")
    print(f"{'ctx':>4}{'decisions':>11}{'share':>8}{'forced':>8}"
          f"{'multi':>7}{'scorable':>10}{'ranker owns':>13}{'mean opts':>10}"
          f"  top cards")
    payload = {}
    for context, bucket in sorted(
        stats.items(), key=lambda item: -item[1]["decisions"]
    ):
        decisions = bucket["decisions"]
        real_here = decisions - bucket["forced"]
        # A multi-pick select is never scored even when ``is_scorable`` says
        # yes, so the ranker's share is the overlap, floored at zero.
        owned = max(bucket["scorable"] - bucket["multi_pick"], 0)
        top = ", ".join(
            NAMES.get(ident, str(ident))
            for ident, _ in cards_seen[context].most_common(3)
        )
        print(
            f"{context:>4}{decisions:>11}{decisions / total:>8.1%}"
            f"{bucket['forced']:>8}{bucket['multi_pick']:>7}"
            f"{bucket['scorable']:>10}"
            f"{(owned / real_here if real_here else 0):>13.1%}"
            f"{bucket['options'] / decisions:>10.1f}  {top}"
        )
        payload[str(context)] = dict(bucket)

    owned_total = sum(
        max(bucket["scorable"] - bucket["multi_pick"], 0)
        for bucket in stats.values()
    )
    print(f"\nranker owns {owned_total} of {real} real choices "
          f"({owned_total / real:.1%}); the remaining "
          f"{real - owned_total} ({1 - owned_total / real:.1%}) are decided by "
          f"the hand-written rule policy.")
    print(f"per game that is {(real - owned_total) / games:.1f} rule-decided "
          f"choices against {owned_total / games:.1f} ranker-decided ones.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"games": games, "contexts": payload},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
