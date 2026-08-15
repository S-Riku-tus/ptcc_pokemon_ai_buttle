"""Teacher-forced v22-vs-v25 probe on the v25 ladder boards that met a wall.

Every decision from the v25 run is replayed through both shipped runtimes with
the *stored* action advancing their history, so a changed answer never invents
a future board.  The question is narrow: on the boards where the opponent's
Active was damage-immune to Marnie's Grimmsnarl ex (Crustle / Cornerstone Mask
Ogerpon ex), does the v22 ranker leave the wall alone where the v25 ranker
attacks it for zero?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loader import load_dir_agent_module  # noqa: E402

IMMUNE = {345, 117}          # Crustle, Cornerstone Mask Ogerpon ex
GRIMMSNARL_EX = 648
MORGREM = 647
RUN = ROOT / "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v25_sub55507909"


def single(action: Any) -> int | None:
    if isinstance(action, list) and len(action) == 1 and isinstance(action[0], int):
        return action[0]
    return None


def _cards(player: dict[str, Any], area: str) -> list[dict[str, Any]]:
    value = player.get(area)
    if isinstance(value, list):
        return [c for c in value if isinstance(c, dict)]
    return []


def describe(observation: dict[str, Any], index: int | None) -> str:
    select = observation.get("select") or {}
    options = list(select.get("option") or [])
    if index is None or not 0 <= index < len(options):
        return "multi/none"
    option = options[index]
    kind = int(option.get("type", -1))
    names = {13: "attack", 12: "retreat", 14: "end", 9: "evolve", 10: "ability",
             8: "energy", 7: "play", 3: "select", 0: "number", 15: "skill"}
    return f"{names.get(kind, 'type' + str(kind))}:{option.get('attackId', '')}"


def wall_boards() -> list[tuple[int, list[dict[str, Any]]]]:
    """(episode_id, [decision]) for every v25 episode that ever met a wall."""
    out = []
    for entry in csv.DictReader((RUN / "manifest.csv").open(encoding="utf-8-sig")):
        seat_text = entry.get("detected_submission_agent_index", "")
        if seat_text not in {"0", "1"}:
            continue
        seat = int(seat_text)
        episode_id = int(entry["episode_id"])
        path = (RUN / "episodes" / str(episode_id) / "replay"
                / f"episode_{episode_id}.json")
        if not path.exists():
            continue
        steps = json.loads(path.read_text(encoding="utf-8")).get("steps") or []
        decisions: list[dict[str, Any]] = []
        met_wall = False
        for i, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[i + 1]):
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
            if not isinstance(observation, dict) or observation.get("select") is None:
                continue
            current = observation.get("current")
            if not isinstance(current, dict) or not current.get("players"):
                continue
            played = single((steps[i + 1][seat] or {}).get("action"))
            if played is None:
                continue
            them = current["players"][1 - seat]
            active = (_cards(them, "active") or [{}])[0]
            walled = int(active.get("id", -1)) in IMMUNE
            met_wall = met_wall or walled
            us = current["players"][seat]
            decisions.append({
                "observation": observation,
                "played": played,
                "walled": walled,
                "turn": int(current.get("turn", -1)),
                "our_active": int((_cards(us, "active") or [{}])[0].get("id", -1)),
                "morgrem_benched": any(
                    int(c.get("id", -1)) == MORGREM for c in _cards(us, "bench")
                ),
            })
        if met_wall:
            out.append((episode_id, decisions))
    return out


def run(agent_dir: Path, boards) -> dict[int, list[int | None]]:
    for name in ("ml_runtime", "ml_features", "fallback_policy", "ml_planner",
                 "policy_base", "mirror_froslass", "main"):
        sys.modules.pop(name, None)
    module = load_dir_agent_module(agent_dir)
    answers: dict[int, list[int | None]] = {}
    for episode_id, decisions in boards:
        for attr in ("reset", "_reset"):
            if hasattr(module, attr):
                getattr(module, attr)()
                break
        picks = []
        for decision in decisions:
            picks.append(single(module.agent(decision["observation"])))
            module.observe_external(decision["observation"], decision["played"])
        answers[episode_id] = picks
    return answers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v25/wall_probe.json",
    )
    args = parser.parse_args()

    boards = wall_boards()
    print(f"wall episodes: {len(boards)}  "
          f"decisions: {sum(len(d) for _, d in boards)}")

    v25 = run(ROOT / "agents/grimmsnarl/grimmsnarl_ml_v25", boards)
    v22 = run(ROOT / "agents/grimmsnarl/grimmsnarl_ml_v22", boards)

    stats: dict[str, Counter] = defaultdict(Counter)
    rows = []
    for episode_id, decisions in boards:
        for k, decision in enumerate(decisions):
            if not decision["walled"]:
                continue
            observation = decision["observation"]
            options = list((observation.get("select") or {}).get("option") or [])
            for label, index in (("played", decision["played"]),
                                 ("v25", v25[episode_id][k]),
                                 ("v22", v22[episode_id][k])):
                stats[label]["decisions"] += 1
                if index is None or not 0 <= index < len(options):
                    stats[label]["multi_or_none"] += 1
                    continue
                kind = int(options[index].get("type", -1))
                if kind == 13:
                    stats[label]["attack"] += 1
                    if decision["our_active"] == GRIMMSNARL_EX:
                        stats[label]["attack_with_ex_into_wall"] += 1
                    if decision["our_active"] == MORGREM:
                        stats[label]["attack_with_morgrem"] += 1
                elif kind == 12:
                    stats[label]["retreat"] += 1
                elif kind == 14:
                    stats[label]["end"] += 1
                else:
                    stats[label]["other"] += 1
            if v22[episode_id][k] != v25[episode_id][k]:
                rows.append({
                    "episode_id": episode_id,
                    "turn": decision["turn"],
                    "our_active": decision["our_active"],
                    "morgrem_benched": decision["morgrem_benched"],
                    "played": describe(observation, decision["played"]),
                    "v25": describe(observation, v25[episode_id][k]),
                    "v22": describe(observation, v22[episode_id][k]),
                })

    print("\n=== decisions where the opponent Active is damage-immune ===")
    for label in ("played", "v25", "v22"):
        print(f"{label:7} {dict(stats[label])}")
    print(f"\nv22 != v25 on {len(rows)} walled decisions")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        {"stats": {k: dict(v) for k, v in stats.items()}, "differences": rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
