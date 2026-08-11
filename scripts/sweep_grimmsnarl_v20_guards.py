"""Sweep every legal input through v20's retained safety guards.

Replaying only the stored selected index cannot distinguish a binding guard
from a saturated invariant.  This audit feeds every legal single-pick index to
each guard on the same public board, while advancing persistent guard state
with the action that was actually played.  It therefore reports both actual
overrides and counterfactual inputs the guard would correct.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agents/grimmsnarl/grimmsnarl_ml_v20"
for path in (AGENT_DIR, ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from attack_access import AttackAccessGuard  # noqa: E402
from mirror_prize import MirrorPrizeGuard  # noqa: E402
from policy_router import MIRROR, PolicyRouter  # noqa: E402
from probe_grimmsnarl_v20_footprint import games, single  # noqa: E402
from wall_break import WallBreakGuard  # noqa: E402


def sweep_guard(
    guard: Any,
    observation: dict[str, Any],
    select: dict[str, Any],
    played: int,
    kind: str,
) -> list[int]:
    options = list(select.get("option") or [])
    outputs = []
    for incoming in range(len(options)):
        probe = copy.deepcopy(guard)
        if kind == "mirror_prize":
            output = probe.adjust(
                observation,
                select,
                incoming,
                {slot: -float(slot) for slot in range(len(options))},
            )
        else:
            output = probe.adjust(
                observation, select, incoming, [played]
            )
        outputs.append(int(output))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    selected = games(set())
    aggregate: dict[str, Counter] = {
        name: Counter() for name in
        ("attack_access", "wall_break", "mirror_prize", "pipeline")
    }
    by_context: dict[str, dict[int, Counter]] = {
        name: defaultdict(Counter) for name in aggregate
    }
    by_matchup: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {name: Counter() for name in aggregate}
    )
    actual_examples: list[dict[str, Any]] = []

    for episode_id, replay, seat, matchup in selected:
        router = PolicyRouter()
        guards = {
            "attack_access": AttackAccessGuard(),
            "wall_break": WallBreakGuard(),
            "mirror_prize": MirrorPrizeGuard(),
        }
        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[step_index + 1]):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            select = observation.get("select") or {}
            current = observation.get("current")
            if not isinstance(current, dict) or not current.get("players"):
                continue
            played = single((steps[step_index + 1][seat] or {}).get("action"))
            options = list(select.get("option") or [])
            if played is None or len(options) < 1:
                continue
            route = router.choose(observation)
            guards["mirror_prize"].set_mirror(route == MIRROR)
            for guard in guards.values():
                guard.note(observation)

            if int(select.get("maxCount") or 0) == 1:
                outputs: dict[str, list[int]] = {}
                for name, guard in guards.items():
                    outputs[name] = sweep_guard(
                        guard, observation, select, played, name
                    )
                # The real call order, evaluated on independent cloned state.
                piped = []
                for incoming in range(len(options)):
                    clones = {
                        name: copy.deepcopy(guard)
                        for name, guard in guards.items()
                    }
                    value = clones["attack_access"].adjust(
                        observation, select, incoming, [played]
                    )
                    value = clones["wall_break"].adjust(
                        observation, select, value, [played]
                    )
                    value = clones["mirror_prize"].adjust(
                        observation, select, value,
                        {slot: -float(slot) for slot in range(len(options))},
                    )
                    piped.append(int(value))
                outputs["pipeline"] = piped

                context = int(select.get("context", -1))
                for name, values in outputs.items():
                    moved = sum(
                        int(output != incoming)
                        for incoming, output in enumerate(values)
                    )
                    actual_moved = int(
                        0 <= played < len(values) and values[played] != played
                    )
                    block = aggregate[name]
                    block["decisions"] += 1
                    block["legal_inputs"] += len(values)
                    block["moved_inputs"] += moved
                    block["binding_prompts"] += int(moved > 0)
                    block["actual_overrides"] += actual_moved
                    by_context[name][context]["legal_inputs"] += len(values)
                    by_context[name][context]["moved_inputs"] += moved
                    by_context[name][context]["binding_prompts"] += int(moved > 0)
                    by_context[name][context]["actual_overrides"] += actual_moved
                    match = by_matchup[matchup][name]
                    match["legal_inputs"] += len(values)
                    match["moved_inputs"] += moved
                    match["binding_prompts"] += int(moved > 0)
                    match["actual_overrides"] += actual_moved
                    if actual_moved and len(actual_examples) < 100:
                        actual_examples.append({
                            "episode_id": episode_id,
                            "matchup": matchup,
                            "step": step_index,
                            "turn": int(current.get("turn", -1)),
                            "context": context,
                            "guard": name,
                            "incoming": played,
                            "output": values[played],
                        })

            # Advance persistent state with the submitted action, never with
            # one of the counterfactual sweep inputs.
            guards["attack_access"].adjust(
                observation, select, played, [played]
            )
            guards["wall_break"].adjust(
                observation, select, played, [played]
            )
            guards["mirror_prize"].record(
                observation, select, played
            )

    report = {
        "scope": {
            "games": len(selected),
            "runs": [run for run, _ in __import__(
                "probe_grimmsnarl_v20_footprint"
            ).RUNS],
            "principle": "all legal single-pick inputs; stored action advances state",
        },
        "guards": {name: dict(values) for name, values in aggregate.items()},
        "by_context": {
            name: {str(ctx): dict(values) for ctx, values in sorted(block.items())}
            for name, block in by_context.items()
        },
        "by_matchup": {
            matchup: {
                name: dict(values) for name, values in guards_block.items()
            }
            for matchup, guards_block in sorted(by_matchup.items())
        },
        "actual_override_examples": actual_examples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "scope": report["scope"],
        "guards": report["guards"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
