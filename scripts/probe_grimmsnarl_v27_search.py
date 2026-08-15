"""Probe v27's legal belief and real-engine H3 on one stored mirror board."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agents/grimmsnarl/grimmsnarl_ml_v27"
RUNS = (
    ROOT / "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v26_sub55520389",
    ROOT / "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v25_b_sub55517142",
    ROOT / "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v25_sub55507909",
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


def public_ids(player: dict[str, Any]) -> set[int]:
    output: set[int] = set()
    for area in ("active", "bench", "discard"):
        for card in player.get(area) or []:
            if not isinstance(card, dict):
                continue
            output.add(int(card.get("id", -1)))
            output.update(
                int(previous.get("id", -1))
                for previous in card.get("preEvolution") or []
                if isinstance(previous, dict)
            )
    return output


def is_mirror_step(step: list[Any], seat: int) -> bool:
    observation = (step[seat] or {}).get("observation") or {}
    current = observation.get("current") or {}
    players = current.get("players") or []
    return bool(
        len(players) >= 2
        and isinstance(players[1 - seat], dict)
        and public_ids(players[1 - seat]) & {646, 647, 648}
    )


def mirror_episodes():
    for run in RUNS:
        manifest = run / "manifest.csv"
        if not manifest.exists():
            continue
        for entry in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            seat_text = entry.get("detected_submission_agent_index", "")
            if seat_text not in {"0", "1"}:
                continue
            seat = int(seat_text)
            episode_id = int(entry["episode_id"])
            replay = (
                run / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json"
            )
            if not replay.exists():
                continue
            steps = json.loads(replay.read_text(encoding="utf-8")).get("steps") or []
            if any(is_mirror_step(step, seat) for step in steps):
                yield episode_id, seat, steps


def find_root(agent_main):
    for episode_id, seat, steps in mirror_episodes():
        agent_main.diag_reset()
        agent_main._RANKER.teacher_forced = True
        agent_main._PEER.teacher_forced = True
        for position, step in enumerate(steps[:-1]):
            observation = (step[seat] or {}).get("observation") or {}
            played = single((steps[position + 1][seat] or {}).get("action"))
            if played is None or observation.get("select") is None:
                continue
            agent_main.agent(observation)
            trace = dict(agent_main._LAST_TRACE)
            current = observation.get("current") or {}
            select = observation.get("select") or {}
            if (
                int(current.get("turn", -1)) >= 5
                and int(select.get("context", -1)) == 0
                and observation.get("search_begin_input")
                and trace.get("ranker") is not None
                and trace.get("peer") is not None
                and agent_main._search_mirror(observation, trace.get("route", ""))
                and (
                    trace.get("ranker") != trace.get("peer")
                    or trace.get("mirror_froslass") != trace.get("ranker")
                )
            ):
                return episode_id, observation, trace
            agent_main.observe_external(observation, played)
    raise RuntimeError("no v27-searchable public mirror root found")


def main() -> int:
    os.environ["GRIMMSNARL_H2_DISABLE"] = "1"
    import main as agent_main

    episode_id, observation, trace = find_root(agent_main)
    base = agent_main._RANKER.choose(observation)
    peer = agent_main._PEER.choose(observation)
    if base is None or peer is None:
        raise RuntimeError("root ranker unexpectedly declined")
    proposed = int(trace.get("mirror_froslass", base))

    os.environ.pop("GRIMMSNARL_H2_DISABLE", None)
    from h2_search import H2SearchPlanner, _hidden_state

    # First prove that the actual engine can finish the longer H3 horizon.
    h3 = H2SearchPlanner()
    ranker_state = agent_main._RANKER.save_dynamic_state()
    peer_state = agent_main._PEER.save_dynamic_state()
    hidden = _hidden_state(observation, int(observation["current"]["yourIndex"]), 0)
    started = time.perf_counter()
    root = h3.search.begin(observation, hidden)
    h3_values: dict[int, float] = {}
    try:
        for candidate in dict.fromkeys((proposed, peer)):
            result = h3._rollout(
                int(root["searchId"]), candidate,
                int(observation["current"]["yourIndex"]),
                agent_main._RANKER, ranker_state,
                future_own_turns=2,
                opponent_ranker=agent_main._PEER,
                opponent_state=peer_state,
                opponent_policy="peer",
            )
            if result is None:
                raise RuntimeError(f"H3 candidate {candidate} did not complete")
            h3_values[candidate] = round(float(result[1]), 6)
    finally:
        h3.search.end()
        agent_main._RANKER.restore_dynamic_state(ranker_state)
        agent_main._PEER.restore_dynamic_state(peer_state)
    h3_seconds = time.perf_counter() - started

    # Then run the complete adaptive gate from the same root.
    planner = H2SearchPlanner()
    chosen = planner.adjust(
        observation, proposed, agent_main._RANKER.last_scores,
        agent_main._RANKER, peer,
        is_mirror=True,
        opponent_ranker=agent_main._PEER,
        allow_non_top=proposed != base,
    )
    report = {
        "episode_id": episode_id,
        "turn": int(observation["current"].get("turn", -1)),
        "root_trace": trace,
        "v22": base,
        "v24_guarded": proposed,
        "v25_peer": peer,
        "direct_h3": {
            "completed_branches": len(h3_values),
            "values": h3_values,
            "seconds": round(h3_seconds, 3),
        },
        "v27": chosen,
        "adaptive_search": planner.snapshot(),
    }
    target = ROOT / "experiments/grimmsnarl_ml_v27/search_engine_probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
