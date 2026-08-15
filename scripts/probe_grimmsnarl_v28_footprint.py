"""Measure whether v28 is a policy-sized change before spending a ladder slot.

Replay stored games with teacher forcing and compare the two synchronized
rankers inside v28 on exactly the same boards and history.  Unlike v26/v27's
eight-decision footprint, v28 is promotable only if its final answer differs
from the v22 ranker at least 50 times per 35 games.

The stored action always advances both histories.  Agreement with that action
is reported only as a reproduction/control statistic; it is not an outcome
value estimate for the counterfactual action.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loader import load_dir_agent_module  # noqa: E402

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
            result.update(
                int(old.get("id", -1))
                for old in card.get("preEvolution") or []
                if isinstance(old, dict)
            )
    return result


def load_episodes(run: Path, submission_id: int) -> list[dict[str, Any]]:
    rows = {
        row["episode_id"]: row
        for row in csv.DictReader((run / "episodes.csv").open(encoding="utf-8-sig"))
    }
    episodes: list[dict[str, Any]] = []
    for entry in csv.DictReader((run / "manifest.csv").open(encoding="utf-8-sig")):
        seat_text = entry.get("detected_submission_agent_index", "")
        if seat_text not in {"0", "1"}:
            continue
        seat = int(seat_text)
        episode_id = int(entry["episode_id"])
        replay_path = (
            run / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not replay_path.exists():
            continue
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        rewards = replay.get("rewards") or [None, None]
        won = int(bool((rewards[seat] or 0) > (rewards[1 - seat] or 0)))
        decisions: list[dict[str, Any]] = []
        sticky_wall = False
        sticky_mirror = False
        for position, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[position + 1]):
                continue
            observation = (step[seat] or {}).get("observation") or {}
            if not isinstance(observation, dict) or observation.get("select") is None:
                continue
            action = (steps[position + 1][seat] or {}).get("action")
            played = single(action)
            current = observation.get("current") or {}
            players = current.get("players") or []
            if len(players) < 2:
                continue
            opponent = players[1 - seat]
            ids = public_ids(opponent)
            sticky_wall = sticky_wall or bool(ids & WALL_IDS)
            sticky_mirror = sticky_mirror or bool(ids & MIRROR_IDS)
            active = (cards(opponent, "active") or [{}])[0]
            select = observation.get("select") or {}
            decisions.append({
                "observation": observation,
                "action": action,
                "played": played,
                "turn": int(current.get("turn", -1)),
                "context": int(select.get("context", -1)),
                "options": len(select.get("option") or []),
                "wall_public": sticky_wall,
                "wall_active": int(active.get("id", -1)) in WALL_IDS,
                "mirror_public": sticky_mirror,
                "evaluate": (
                    played is not None
                    and int(select.get("maxCount", 0) or 0) == 1
                    and len(select.get("option") or []) >= 2
                ),
            })
        episodes.append({
            "episode_id": episode_id,
            "won": won,
            "decisions": decisions,
            "meta": rows.get(str(episode_id), {}),
        })
    episodes.sort(key=lambda item: item["meta"].get("create_time", ""))
    return episodes


def cell(decision: dict[str, Any]) -> str:
    if decision["wall_active"]:
        return "wall_active"
    if decision["wall_public"]:
        return "wall_public"
    if decision["mirror_public"]:
        return "mirror"
    return "ordinary"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path,
        default=ROOT / "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v27_sub55521760",
    )
    parser.add_argument("--submission", type=int, default=55521760)
    parser.add_argument(
        "--agent", type=Path,
        default=ROOT / "agents/grimmsnarl/grimmsnarl_ml_v28",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v28/footprint_v27_run.json",
    )
    args = parser.parse_args()

    episodes = load_episodes(args.run, args.submission)
    if args.limit:
        episodes = episodes[:args.limit]
    module = load_dir_agent_module(args.agent)
    load_errors = module.diag_snapshot().get("load_errors", {})
    print(f"episodes={len(episodes)} load_errors={json.dumps(load_errors)}")

    counts: Counter[str] = Counter()
    per_episode: dict[int, Counter[str]] = defaultdict(Counter)
    differences: list[dict[str, Any]] = []
    elapsed = 0.0

    for number, episode in enumerate(episodes, 1):
        module.diag_reset()
        for name in ("_RACE", "_WALL"):
            ranker = getattr(module, name, None)
            if ranker is not None:
                ranker.teacher_forced = True
        start = time.perf_counter()
        for decision in episode["decisions"]:
            proposed = module.agent(decision["observation"])
            if decision["evaluate"]:
                final = single(proposed)
                trace = dict(getattr(module, "_LAST_TRACE", {}))
                v25 = trace.get("v25_race")
                v22 = trace.get("v22_wall")
                group = cell(decision)
                policy = str(trace.get("policy", "unknown"))
                counts["evaluated"] += 1
                counts[f"cell_{group}"] += 1
                counts[f"policy_{policy}"] += 1
                counts["final_matches_played"] += int(final == decision["played"])
                counts["v22_matches_played"] += int(v22 == decision["played"])
                counts["v25_matches_played"] += int(v25 == decision["played"])
                if isinstance(v22, int) and isinstance(v25, int):
                    counts["comparable"] += 1
                    counts["ranker_disagreements"] += int(v22 != v25)
                if isinstance(v22, int) and final != v22:
                    counts["final_vs_v22"] += 1
                    counts[f"final_vs_v22_{group}"] += 1
                    per_episode[episode["episode_id"]]["final_vs_v22"] += 1
                    if len(differences) < 300:
                        differences.append({
                            "episode_id": episode["episode_id"],
                            "won": episode["won"],
                            "turn": decision["turn"],
                            "context": decision["context"],
                            "cell": group,
                            "played": decision["played"],
                            "v22": v22,
                            "v25": v25,
                            "final": final,
                            "trace": trace,
                        })
            # The proposal is counterfactual; all state advances with replay.
            if decision["played"] is not None:
                module.observe_external(decision["observation"], decision["played"])
        elapsed += time.perf_counter() - start
        print(
            f"[{number:02d}/{len(episodes):02d}] episode={episode['episode_id']} "
            f"changes={per_episode[episode['episode_id']]['final_vs_v22']}"
        )

    evaluated = max(1, counts["evaluated"])
    comparable = max(1, counts["comparable"])
    normalized_35 = counts["final_vs_v22"] * 35 / max(1, len(episodes))
    passed = normalized_35 >= 50
    summary = {
        "episodes": len(episodes),
        "evaluated": counts["evaluated"],
        "comparable": counts["comparable"],
        "final_vs_v22": counts["final_vs_v22"],
        "final_vs_v22_pct": counts["final_vs_v22"] / evaluated,
        "normalized_changes_per_35_games": normalized_35,
        "promotion_footprint_gate": "PASS" if passed else "FAIL",
        "ranker_disagreements": counts["ranker_disagreements"],
        "ranker_disagreement_pct": counts["ranker_disagreements"] / comparable,
        "final_played_agreement": counts["final_matches_played"] / evaluated,
        "v22_played_agreement": counts["v22_matches_played"] / evaluated,
        "v25_played_agreement": counts["v25_matches_played"] / evaluated,
        "seconds": elapsed,
        "seconds_per_game": elapsed / max(1, len(episodes)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    payload = {
        "run": str(args.run),
        "submission": args.submission,
        "summary": summary,
        "counts": dict(counts),
        "per_episode": {str(key): dict(value) for key, value in per_episode.items()},
        "differences": differences,
        "load_errors": load_errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"JSON: {args.output}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
