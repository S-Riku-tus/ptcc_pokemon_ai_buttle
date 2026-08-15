"""Run one real-engine v26 H2 comparison on a stored public mirror board."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agents/grimmsnarl/grimmsnarl_ml_v26"
RUNS = (
    ROOT / "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v25_sub55507909",
    ROOT / "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v25_b_sub55517142",
)
for path in (ROOT / "vendor", AGENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def single(action: Any) -> int | None:
    return (
        int(action[0])
        if isinstance(action, list) and len(action) == 1
        and isinstance(action[0], int) else None
    )


def ids(player: dict[str, Any]) -> set[int]:
    output = set()
    for area in ("active", "bench", "discard"):
        for card in player.get(area) or []:
            if isinstance(card, dict):
                output.add(int(card.get("id", -1)))
                output.update(
                    int(previous.get("id", -1))
                    for previous in card.get("preEvolution") or []
                    if isinstance(previous, dict)
                )
    return output


def mirror_episode() -> tuple[int, int, list] | None:
    for run in RUNS:
        for entry in csv.DictReader((run / "manifest.csv").open(encoding="utf-8-sig")):
            seat_text = entry.get("detected_submission_agent_index", "")
            if seat_text not in {"0", "1"}:
                continue
            seat = int(seat_text)
            episode_id = int(entry["episode_id"])
            path = run / "episodes" / str(episode_id) / "replay" / f"episode_{episode_id}.json"
            if not path.exists():
                continue
            steps = json.loads(path.read_text(encoding="utf-8")).get("steps") or []
            for step in steps:
                observation = (step[seat] or {}).get("observation") or {}
                current = observation.get("current") or {}
                players = current.get("players") or []
                if len(players) >= 2 and ids(players[1 - seat]) & {646, 647, 648}:
                    return episode_id, seat, steps
    return None


def main() -> int:
    found = mirror_episode()
    if found is None:
        raise RuntimeError("no public mirror replay found")
    episode_id, seat, steps = found

    # Load without constructing Search while the teacher history is replayed.
    os.environ["GRIMMSNARL_H2_DISABLE"] = "1"
    import main as agent_main

    agent_main.diag_reset()
    agent_main._RANKER.teacher_forced = True
    agent_main._PEER.teacher_forced = True
    target = None
    for position, step in enumerate(steps[:-1]):
        observation = (step[seat] or {}).get("observation") or {}
        select = observation.get("select") or {}
        current = observation.get("current") or {}
        played = single((steps[position + 1][seat] or {}).get("action"))
        if played is None or observation.get("select") is None:
            continue
        answer = single(agent_main.agent(observation))
        trace = dict(agent_main._LAST_TRACE)
        if (
            int(current.get("turn", -1)) >= 5
            and int(select.get("context", -1)) == 0
            and observation.get("search_begin_input")
            and trace.get("ranker") is not None
            and trace.get("peer") is not None
            and (
                trace.get("ranker") != trace.get("peer")
                or len(agent_main._RANKER.last_scores) >= 2
            )
        ):
            target = (observation, trace, answer)
            break
        agent_main.observe_external(observation, played)
    if target is None:
        raise RuntimeError("no searchable mirror root found")

    observation, trace, answer = target
    # main.agent's teacher-forced commit cleared pending; score once more to
    # recreate the root snapshot H2 branches must advance.
    base = agent_main._RANKER.choose(observation)
    peer = agent_main._PEER.choose(observation)
    if base is None:
        raise RuntimeError("root ranker unexpectedly declined")

    os.environ.pop("GRIMMSNARL_H2_DISABLE", None)
    from h2_search import H2SearchPlanner

    planner = H2SearchPlanner()
    chosen = planner.adjust(
        observation,
        base,
        agent_main._RANKER.last_scores,
        agent_main._RANKER,
        peer,
        is_mirror=True,
    )
    report = {
        "episode_id": episode_id,
        "turn": int(observation["current"].get("turn", -1)),
        "stored_answer": answer,
        "initial_trace": trace,
        "v22": base,
        "v25_peer": peer,
        "v26_h2": chosen,
        "search": planner.snapshot(),
    }
    target_path = ROOT / "experiments/grimmsnarl_ml_v26/h2_engine_probe.json"
    target_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
