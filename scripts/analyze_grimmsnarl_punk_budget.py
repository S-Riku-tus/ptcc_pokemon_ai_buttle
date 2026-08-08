"""How many energies Punk Up takes: v8's hand-written budget against the field's.

Punk Up's deck search is a multi-pick select, so ``Ranker.is_scorable`` rejects
it and the choice is made by ``fallback_policy.punk_search_budget`` - a rule v5
fitted to the *whole* archive:

    wanted = max(PUNK_MIN_SEARCH, deficit_of_trigger + hungry_line_bodies)

The pin panel is what makes this the interesting number. Swapping v8's teacher
for any of the five pilots with the best Alakazam records moves the Alakazam
agreement by at most +0.003 and usually down, so the matchup-specific attach
divergence is *not* something the ranker can express - which is exactly what
you would expect if the decision upstream of it never reaches the ranker at
all. Punk Up is the deck's only energy acceleration, and how much it pulls is
decided by a rule that was never conditioned on the opponent.

So: does the field's Punk Up count depend on the matchup, and does v8's rule
track it? Two counts per activation, on identical boards:

* ``offered`` / ``taken`` from the replay - what the field pilot did
* ``v8_taken`` - the agent run on that same observation

The agent is only invoked on the Punk Up selects themselves; every other
decision is still walked and fed to ``observe_external`` so the intra-turn
history is identical, which is what makes three cohorts cheap here when the
full-fidelity probe costs an hour.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))

from agent_loader import load_dir_agent  # noqa: E402
from analyze_grimmsnarl_alakazam_stage1 import (  # noqa: E402
    OUR_DECK_HASH, cohort_of, replay_meta,
)


class Probe:
    def __init__(self, agent_dir: Path):
        self.agent, _, self.module = load_dir_agent(agent_dir)
        self.ranker = getattr(self.module, "_RANKER", None)
        if self.ranker is None:
            raise SystemExit(f"{agent_dir}: no ranker loaded")
        self.ranker.teacher_forced = True
        self.observe = getattr(self.module, "observe_external", None)

    def reset(self) -> None:
        self.module.diag_reset()

    def ask_multi(self, observation: dict[str, Any]) -> list[int] | None:
        try:
            answer = self.agent(observation)
        except Exception:  # noqa: BLE001
            return None
        return [int(x) for x in answer] if isinstance(answer, list) else None

    def advance(self, observation: dict[str, Any], played: Any) -> None:
        if self.observe is not None:
            self.observe(observation, played)
        else:
            self.ranker.observe_external(observation, played)


def walk(
    probe: Probe, replay: dict[str, Any], seat: int, mf: Any, max_turn: int
) -> list[dict[str, Any]]:
    steps = replay.get("steps") or []
    probe.reset()
    out: list[dict[str, Any]] = []
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        action = (steps[index + 1][seat] or {}).get("action")
        if not options or not isinstance(action, list):
            continue
        current = observation.get("current") or {}
        if len(current.get("players") or []) < 2:
            continue
        turn = int(current.get("turn", -1))
        if max_turn and turn > max_turn:
            break
        effect = select.get("effect")
        is_punk = (
            int(select.get("context", -1)) == mf.CTX_ATTACH_TO
            and isinstance(effect, dict)
            and mf._int(effect.get("id")) == mf.GRIMMSNARL_EX_ID
        )
        if is_punk:
            answer = probe.ask_multi(observation)
            out.append({
                "turn": turn,
                "offered": len(options),
                "max_count": int(select.get("maxCount") or 0),
                "taken": len(action),
                "v8_taken": len(answer) if answer is not None else None,
            })
        played = action[0] if len(action) == 1 else action
        probe.advance(observation, played)
    return out


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"activations": 0}
    taken = [r["taken"] for r in rows]
    v8 = [r["v8_taken"] for r in rows if r["v8_taken"] is not None]
    both = [r for r in rows if r["v8_taken"] is not None]
    return {
        "activations": len(rows),
        "mean_offered": round(statistics.fmean(r["offered"] for r in rows), 3),
        "mean_taken": round(statistics.fmean(taken), 3),
        "mean_v8_taken": round(statistics.fmean(v8), 3) if v8 else None,
        "takes_everything_offered": round(sum(
            int(r["taken"] >= min(r["offered"], r["max_count"] or r["offered"]))
            for r in rows
        ) / len(rows), 4),
        "v8_takes_everything": (round(sum(
            int(r["v8_taken"] >= min(r["offered"],
                                     r["max_count"] or r["offered"]))
            for r in both
        ) / len(both), 4) if both else None),
        "v8_minus_field": (
            round(statistics.fmean(r["v8_taken"] - r["taken"] for r in both), 3)
            if both else None
        ),
        "v8_exact_match": (round(sum(
            int(r["v8_taken"] == r["taken"]) for r in both
        ) / len(both), 4) if both else None),
        "count_histogram": dict(sorted(Counter(taken).items())),
        "v8_count_histogram": dict(sorted(Counter(v8).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument(
        "--agent", type=Path,
        default=ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8",
    )
    parser.add_argument(
        "--max-turn", type=int, default=0,
        help="0 walks the whole game; Punk Up matters after turn 8 too.",
    )
    parser.add_argument("--control-cap", type=int, default=400)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ratings: dict[int, float] = {}
    for row in csv.DictReader(
        (args.data_root / "indexes" / "submissions.csv").open(
            encoding="utf-8-sig"
        )
    ):
        try:
            ratings[int(row["team_id"])] = float(row["submission_score"])
        except (KeyError, TypeError, ValueError):
            continue

    catalogue: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for raw in csv.DictReader(
        (args.data_root / "indexes" / "episodes.csv").open(encoding="utf-8-sig")
    ):
        if raw.get("download_status") != "success":
            continue
        if raw.get("deck_hash") != OUR_DECK_HASH:
            continue
        if raw.get("episode_type") != "EPISODE_TYPE_PUBLIC":
            continue
        episode_id, seat = int(raw["episode_id"]), int(raw["seat_index"])
        if (episode_id, seat) in seen:
            continue
        seen.add((episode_id, seat))
        path = args.data_root / "replays" / f"episode_{episode_id}.json"
        if path.exists():
            catalogue.append({
                "episode_id": episode_id, "seat": seat, "path": path,
                "team": int(raw["team_id"]),
            })
    catalogue.sort(key=lambda r: (r["episode_id"], r["seat"]))

    probe = Probe(args.agent)
    mf = sys.modules["ml_features"]

    kept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    walked: Counter = Counter()
    for entry in catalogue:
        try:
            replay = json.loads(entry["path"].read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        meta = replay_meta(replay, entry["seat"])
        if meta is None:
            continue
        cohort = cohort_of(meta)
        if cohort is None:
            continue
        if cohort != "alakazam_second" and walked[cohort] >= args.control_cap:
            continue
        rows = walk(probe, replay, entry["seat"], mf, args.max_turn)
        walked[cohort] += 1
        rating = ratings.get(entry["team"])
        for row in rows:
            row["team"] = entry["team"]
            row["rating"] = rating
            row["won"] = meta["won"]
        kept[cohort].extend(rows)
        if cohort == "alakazam_second":
            kept["alakazam_second_won" if meta["won"]
                 else "alakazam_second_lost"].extend(rows)
        if rating is not None and rating >= 1100:
            kept[f"elite_{cohort}"].extend(rows)
        print(f"{entry['episode_id']} {cohort:16s} punk={len(rows)} "
              f"walked={dict(walked)}", flush=True)

    report = {
        "games": dict(walked),
        "cohorts": {
            name: describe(rows) for name, rows in sorted(kept.items())
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"{'cohort':26s} {'acts':>5} {'offer':>6} {'field':>6} {'v8':>6} "
          f"{'v8-field':>9} {'fieldAll':>9} {'v8All':>7} {'exact':>7}")
    for name, block in report["cohorts"].items():
        if not block.get("activations"):
            continue
        print(f"{name:26s} {block['activations']:5d} "
              f"{block['mean_offered']:6.2f} {block['mean_taken']:6.2f} "
              f"{(block['mean_v8_taken'] or 0):6.2f} "
              f"{(block['v8_minus_field'] or 0):+9.2f} "
              f"{block['takes_everything_offered']:9.3f} "
              f"{(block['v8_takes_everything'] or 0):7.3f} "
              f"{(block['v8_exact_match'] or 0):7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
