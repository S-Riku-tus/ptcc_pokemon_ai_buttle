"""Teacher-forced footprint of v18's mirror Prize guard.

Every exact-60 mirror action is replayed as the action that actually happened.
The report counts where the guard would redirect that stored action and keeps
the deployed and high-rated reference groups separate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agents/grimmsnarl/grimmsnarl_ml_v18"
for path in (ROOT, ROOT / "scripts", AGENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mirror_prize import MirrorPrizeGuard  # noqa: E402
from analyze_grimmsnarl_v18_mirror_endgame import (  # noqa: E402
    DEFAULT_RUNS,
    OUR_DECK,
    RunSpec,
    deck_at,
    selected_indices,
)


def scan(spec: RunSpec) -> Counter:
    counts: Counter = Counter()
    path = spec.directory / "episodes.csv"
    if not path.exists():
        return counts
    for raw in csv.DictReader(path.open(encoding="utf-8-sig")):
        if raw.get("state") != "COMPLETED":
            continue
        a0 = raw.get("agent_0_submission_id", "")
        a1 = raw.get("agent_1_submission_id", "")
        if spec.submission not in (a0, a1) or a0 == a1:
            continue
        seat = 0 if a0 == spec.submission else 1
        episode_id = int(raw["episode_id"])
        replay_path = (
            spec.directory / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not replay_path.exists():
            continue
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        if deck_at(steps, seat) != OUR_DECK or deck_at(steps, 1 - seat) != OUR_DECK:
            continue
        counts["games"] += 1
        guard = MirrorPrizeGuard()
        guard.set_mirror(True)
        for index, step in enumerate(steps[:-1]):
            if seat >= len(step):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            select = observation.get("select") or {}
            options = list(select.get("option") or [])
            chosen = selected_indices(steps, index, seat)
            if len(chosen) != 1 or not 0 <= chosen[0] < len(options):
                continue
            guard.note(observation)
            moved = guard.adjust(observation, select, chosen[0])
            counts["single_pick_decisions"] += 1
            if moved != chosen[0]:
                counts["differences"] += 1
                counts[f"episode_{episode_id}"] += 1
        snapshot = guard.snapshot()
        for key in (
            "post_shadow_target_prompts",
            "immediate_ko_prompts",
            "already_max_prizes",
            "overrides",
            "shadow_bench_overrides",
            "adrena_overrides",
            "errors",
        ):
            counts[key] += snapshot[key]
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    per_run: dict[str, dict[str, int]] = {}
    groups = {"deployed": Counter(), "teacher": Counter()}
    for name, group, submission, directory in DEFAULT_RUNS:
        spec = RunSpec(name, group, submission, ROOT / directory)
        result = scan(spec)
        per_run[name] = dict(result)
        groups[group].update(result)
    output = {
        "definition": "actual-action replay over exact-60 mirror games",
        "per_run": per_run,
        "groups": {key: dict(value) for key, value in groups.items()},
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
