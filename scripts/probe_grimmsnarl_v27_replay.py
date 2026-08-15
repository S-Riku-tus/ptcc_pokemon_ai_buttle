"""Teacher-forced v22/v27 comparison on both v25 ladder submissions.

H2 is disabled here so the comparison isolates deterministic recovery.  Every
stored teacher action, rather than either candidate's proposal, advances the
ranker history.  The report therefore measures exactly where v27's inherited
wall/deck guards and restored v24 mirror-Froslass veto would intervene on real
reachable boards, including the ordinary non-wall preservation boundary.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loader import load_dir_agent_module  # noqa: E402


RUNS = (
    ROOT / "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v25_sub55507909",
    ROOT / "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v25_b_sub55517142",
)
WALL_IDS = {117, 330, 344, 345}
MIRROR_IDS = {646, 647, 648}


def single(action: Any) -> int | None:
    return (
        int(action[0])
        if isinstance(action, list) and len(action) == 1
        and isinstance(action[0], int) else None
    )


def cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    return [card for card in (player.get(area) or []) if isinstance(card, dict)]


def public_ids(player: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for area in ("active", "bench", "discard"):
        for card in cards(player, area):
            result.add(int(card.get("id", -1)))
            for previous in card.get("preEvolution") or []:
                if isinstance(previous, dict):
                    result.add(int(previous.get("id", -1)))
    return result


def decisions() -> list[dict[str, Any]]:
    output = []
    for run in RUNS:
        for entry in csv.DictReader((run / "manifest.csv").open(encoding="utf-8-sig")):
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
            episode = []
            sticky_wall = False
            sticky_mirror = False
            for position, step in enumerate(steps[:-1]):
                if seat >= len(step) or seat >= len(steps[position + 1]):
                    continue
                observation = (step[seat] or {}).get("observation") or {}
                if not isinstance(observation, dict) or observation.get("select") is None:
                    continue
                played = single((steps[position + 1][seat] or {}).get("action"))
                current = observation.get("current") or {}
                players = current.get("players") or []
                if played is None or len(players) < 2:
                    continue
                opponent = players[1 - seat]
                ids = public_ids(opponent)
                sticky_wall = sticky_wall or bool(ids & WALL_IDS)
                sticky_mirror = sticky_mirror or bool(ids & MIRROR_IDS)
                active = (cards(opponent, "active") or [{}])[0]
                episode.append({
                    "observation": observation,
                    "played": played,
                    "wall_public": sticky_wall,
                    "wall_active": int(active.get("id", -1)) in WALL_IDS,
                    "mirror_public": sticky_mirror,
                    "episode_id": episode_id,
                    "turn": int(current.get("turn", -1)),
                })
            output.append({"episode_id": episode_id, "decisions": episode})
    return output


def run_agent(agent_dir: Path, episodes) -> tuple[
    dict[int, list[int | None]], dict[int, list[int | None]], dict
]:
    for name in (
        "deck_clock", "fallback_policy", "h2_search", "main", "ml_features",
        "ml_planner", "ml_runtime", "policy_base", "policy_router",
        "mirror_froslass", "value_features", "wall_break", "wall_trajectory",
    ):
        sys.modules.pop(name, None)
    previous = os.environ.get("GRIMMSNARL_H2_DISABLE")
    os.environ["GRIMMSNARL_H2_DISABLE"] = "1"
    try:
        module = load_dir_agent_module(agent_dir)
    finally:
        if previous is None:
            os.environ.pop("GRIMMSNARL_H2_DISABLE", None)
        else:
            os.environ["GRIMMSNARL_H2_DISABLE"] = previous
    # With H2 disabled, the peer is observational only.  Removing it makes the
    # replay probe measure the same action in roughly half the tree scoring
    # time without changing any v27 choice.
    if agent_dir.name == "grimmsnarl_ml_v27":
        module._PEER = None
    answers: dict[int, list[int | None]] = {}
    baselines: dict[int, list[int | None]] = {}
    aggregate = Counter()
    for episode in episodes:
        if hasattr(module, "diag_reset"):
            module.diag_reset()
        for name in ("_RANKER", "_PEER"):
            ranker = getattr(module, name, None)
            if ranker is not None:
                ranker.teacher_forced = True
        picks = []
        bases = []
        for decision in episode["decisions"]:
            if decision.get("evaluate"):
                picks.append(single(module.agent(decision["observation"])))
                trace = getattr(module, "_LAST_TRACE", {})
                base = trace.get("planner")
                bases.append(int(base) if isinstance(base, int) else None)
            else:
                picks.append(None)
                bases.append(None)
            if hasattr(module, "observe_external"):
                module.observe_external(
                    decision["observation"], decision["played"]
                )
        answers[episode["episode_id"]] = picks
        baselines[episode["episode_id"]] = bases
        if hasattr(module, "diag_snapshot"):
            snapshot = module.diag_snapshot()
            for section in (
                "wall_trajectory", "wall_break", "deck_clock",
                "mirror_froslass",
            ):
                for key, value in (snapshot.get(section) or {}).items():
                    if isinstance(value, int):
                        aggregate[f"{section}.{key}"] += value
    return answers, baselines, dict(aggregate)


def main() -> int:
    all_episodes = decisions()
    wall_episodes = [
        episode for episode in all_episodes
        if any(d["wall_public"] for d in episode["decisions"])
    ]
    controls = [
        episode for episode in all_episodes
        if not any(d["wall_public"] for d in episode["decisions"])
    ][:12]
    # Full coverage of the defect plus a pre-registered ordinary control is
    # enough for this counterfactual. Scoring all 87 games twice takes over
    # four minutes without changing the cell this guard can enter.
    episodes = wall_episodes + controls
    control_ids = {episode["episode_id"] for episode in controls}
    for episode in episodes:
        remaining_control = 8
        for decision in episode["decisions"]:
            select = decision["observation"].get("select") or {}
            single_pick = (
                int(select.get("maxCount", 0) or 0) == 1
                and len(select.get("option") or []) >= 2
            )
            evaluate = bool(decision["wall_public"] and single_pick)
            if (
                episode["episode_id"] in control_ids
                and single_pick and remaining_control > 0
            ):
                evaluate = True
                remaining_control -= 1
            decision["evaluate"] = evaluate
    v27, v22, guard_stats = run_agent(
        ROOT / "agents/grimmsnarl/grimmsnarl_ml_v27", episodes
    )
    counts = Counter()
    differences = []
    for episode in episodes:
        episode_id = episode["episode_id"]
        for offset, decision in enumerate(episode["decisions"]):
            if not decision.get("evaluate"):
                continue
            counts["decisions"] += 1
            changed = v22[episode_id][offset] != v27[episode_id][offset]
            cell = (
                "wall_active" if decision["wall_active"]
                else "wall_public" if decision["wall_public"]
                else "mirror" if decision["mirror_public"]
                else "ordinary"
            )
            counts[f"{cell}_decisions"] += 1
            counts[f"{cell}_changes"] += int(changed)
            if changed and len(differences) < 100:
                differences.append({
                    "episode_id": episode_id,
                    "turn": decision["turn"],
                    "cell": cell,
                    "played": decision["played"],
                    "v22": v22[episode_id][offset],
                    "v27": v27[episode_id][offset],
                })
    report = {
        "runs": [str(path.relative_to(ROOT)) for path in RUNS],
        "available_episodes": len(all_episodes),
        "wall_episodes": len(wall_episodes),
        "control_episodes": len(controls),
        "episodes": len(episodes),
        "counts": dict(counts),
        "guard_stats": guard_stats,
        "differences": differences,
    }
    target = ROOT / "experiments/grimmsnarl_ml_v27/replay_probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
