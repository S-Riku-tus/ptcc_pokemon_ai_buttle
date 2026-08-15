"""How often did v27 actually depart from v22 in the 35 games it played?

The v27 pre-submission probe measured the guard footprint on *v25's* replays.
This one measures it on v27's own ladder corpus, which is the only board
distribution that can explain v27's own result.

Method: replay every stored decision of our seat, teacher-forced on the action
that was really played, and read ``main._LAST_TRACE``.  The trace records the
index at each layer, so the v22 answer (``planner``) and the final v27 answer
come from the same pass over the same history - no second agent, no drift.
H2 is disabled by default because the Kaggle search API is not available
offline; ``--h2`` re-enables it for the mirror games if an engine is present.

Output: per-layer override counts, the cell each override landed in, and the
win/loss record of the games that contain at least one override.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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
LAYERS = (
    "planner", "mirror_froslass", "h2", "deck_clock",
    "wall_trajectory", "wall_break",
)


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


def load_episodes(run: Path, submission_id: int) -> list[dict[str, Any]]:
    outcomes = {}
    for row in csv.DictReader((run / "episodes.csv").open(encoding="utf-8-sig")):
        outcomes[row["episode_id"]] = row
    episodes: list[dict[str, Any]] = []
    for entry in csv.DictReader((run / "manifest.csv").open(encoding="utf-8-sig")):
        seat_text = entry.get("detected_submission_agent_index", "")
        if seat_text not in {"0", "1"}:
            continue
        seat = int(seat_text)
        episode_id = int(entry["episode_id"])
        path = (
            run / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        rewards = replay.get("rewards") or [None, None]
        won = int(bool((rewards[seat] or 0) > (rewards[1 - seat] or 0)))
        decisions: list[dict[str, Any]] = []
        sticky_wall = False
        sticky_mirror = False
        for position, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[position + 1]):
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
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
            select = observation.get("select") or {}
            decisions.append({
                "observation": observation,
                "played": played,
                "turn": int(current.get("turn", -1)),
                "context": int(select.get("context", -1)),
                "options": len(select.get("option") or []),
                "wall_public": sticky_wall,
                "wall_active": int(active.get("id", -1)) in WALL_IDS,
                "mirror_public": sticky_mirror,
                "evaluate": (
                    int(select.get("maxCount", 0) or 0) == 1
                    and len(select.get("option") or []) >= 2
                ),
            })
        episodes.append({
            "episode_id": episode_id,
            "seat": seat,
            "won": won,
            "decisions": decisions,
            "meta": outcomes.get(str(episode_id), {}),
        })
    episodes.sort(key=lambda item: item["meta"].get("create_time", ""))
    return episodes


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT / "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v27_sub55521760",
    )
    parser.add_argument("--submission", type=int, default=55521760)
    parser.add_argument(
        "--agent", type=Path,
        default=ROOT / "agents/grimmsnarl/grimmsnarl_ml_v27",
    )
    parser.add_argument("--h2", action="store_true", help="Leave H2 enabled.")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v27/ladder_footprint.json",
    )
    args = parser.parse_args()

    episodes = load_episodes(args.run, args.submission)
    print(f"episodes: {len(episodes)}  "
          f"decisions: {sum(len(e['decisions']) for e in episodes)}  "
          f"evaluated: {sum(sum(d['evaluate'] for d in e['decisions']) for e in episodes)}")

    if not args.h2:
        os.environ["GRIMMSNARL_H2_DISABLE"] = "1"
    module = load_dir_agent_module(args.agent)
    print("load errors:", json.dumps(module.diag_snapshot()["load_errors"]))

    counts: Counter[str] = Counter()
    per_layer: Counter[str] = Counter()
    per_layer_cell: Counter[str] = Counter()
    per_episode: dict[int, Counter[str]] = defaultdict(Counter)
    guard_stats: Counter[str] = Counter()
    differences: list[dict[str, Any]] = []

    for episode in episodes:
        module.diag_reset()
        for name in ("_RANKER", "_PEER"):
            ranker = getattr(module, name, None)
            if ranker is not None:
                ranker.teacher_forced = True
        for decision in episode["decisions"]:
            if decision["evaluate"]:
                final = single(module.agent(decision["observation"]))
                trace = dict(getattr(module, "_LAST_TRACE", {}))
                baseline = trace.get("planner")
                counts["evaluated"] += 1
                cell = (
                    "wall_active" if decision["wall_active"]
                    else "wall_public" if decision["wall_public"]
                    else "mirror" if decision["mirror_public"]
                    else "ordinary"
                )
                counts[f"cell_{cell}"] += 1
                counts["matches_played"] += int(final == decision["played"])
                counts["baseline_matches_played"] += int(
                    baseline == decision["played"]
                )
                if isinstance(baseline, int) and final != baseline:
                    counts["overrides"] += 1
                    counts[f"override_{cell}"] += 1
                    per_episode[episode["episode_id"]]["overrides"] += 1
                    previous = baseline
                    for layer in LAYERS[1:]:
                        value = trace.get(layer)
                        if isinstance(value, int) and value != previous:
                            per_layer[layer] += 1
                            per_layer_cell[f"{layer}|{cell}"] += 1
                            per_episode[episode["episode_id"]][layer] += 1
                            previous = value
                    if len(differences) < 200:
                        differences.append({
                            "episode_id": episode["episode_id"],
                            "won": episode["won"],
                            "turn": decision["turn"],
                            "context": decision["context"],
                            "cell": cell,
                            "options": decision["options"],
                            "played": decision["played"],
                            "v22": baseline,
                            "v27": final,
                            "trace": {
                                k: v for k, v in trace.items()
                                if k in ("rule", "ranker", "peer", *LAYERS, "route")
                            },
                        })
            module.observe_external(
                decision["observation"], decision["played"]
            )
        snapshot = module.diag_snapshot()
        for sec in ("wall_trajectory", "wall_break", "deck_clock",
                    "mirror_froslass", "h2_search", "router"):
            for key, value in (snapshot.get(sec) or {}).items():
                if isinstance(value, int) and not isinstance(value, bool):
                    guard_stats[f"{sec}.{key}"] += value

    print("\n--- reproduction ---")
    print(f"evaluated single-pick decisions: {counts['evaluated']}")
    print(f"v27 reproduces the played action: {counts['matches_played']} "
          f"({counts['matches_played'] / max(counts['evaluated'], 1):.4f})")
    print(f"v22 baseline reproduces it:      {counts['baseline_matches_played']} "
          f"({counts['baseline_matches_played'] / max(counts['evaluated'], 1):.4f})")

    print("\n--- override footprint (v27 final != v22 planner answer) ---")
    print(f"total overrides: {counts['overrides']} of {counts['evaluated']} "
          f"({counts['overrides'] / max(counts['evaluated'], 1):.4%})")
    for cell in ("ordinary", "mirror", "wall_public", "wall_active"):
        total = counts.get(f"cell_{cell}", 0)
        over = counts.get(f"override_{cell}", 0)
        print(f"  {cell:<12} {over:>4} / {total:<5} "
              f"({over / total:.4%})" if total else f"  {cell:<12}    0 / 0")
    print("\nby layer:")
    for layer, count in per_layer.most_common():
        print(f"  {layer:<18} {count}")
    print("\nby layer x cell:")
    for key, count in per_layer_cell.most_common():
        print(f"  {key:<32} {count}")

    print("\n--- games containing at least one override ---")
    touched = [e for e in episodes if per_episode[e["episode_id"]]["overrides"]]
    clean = [e for e in episodes if not per_episode[e["episode_id"]]["overrides"]]
    for label, group in (("with override", touched), ("without", clean)):
        if group:
            wins = sum(e["won"] for e in group)
            print(f"  {label:<14} n={len(group):>3}  {wins}-{len(group) - wins}  "
                  f"{wins / len(group):.3f}")
    print("\nper-game override counts (episode, won, overrides):")
    for episode in episodes:
        n = per_episode[episode["episode_id"]]["overrides"]
        if n:
            print(f"  {episode['episode_id']}  won={episode['won']}  overrides={n}  "
                  f"{dict(per_episode[episode['episode_id']])}")

    print("\n--- guard internal counters (summed over games) ---")
    for key, value in sorted(guard_stats.items()):
        if value:
            print(f"  {key:<44} {value}")

    payload = {
        "run": str(args.run.relative_to(ROOT)),
        "h2_enabled": bool(args.h2),
        "counts": dict(counts),
        "per_layer": dict(per_layer),
        "per_layer_cell": dict(per_layer_cell),
        "guard_stats": dict(guard_stats),
        "per_episode": {str(k): dict(v) for k, v in per_episode.items()},
        "differences": differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nJSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
