"""Replay the live v1 run and ask v2 what it would have done, decision by decision.

Aggregate teacher agreement says v2 imitates better on held-out teacher games.
It does not say v2 fixes the decisions that actually lost the ladder run.  This
walks the downloaded run in order, keeps every agent on the *real* trajectory by
forcing the action that was actually played, and reports what each agent would
have chosen at the same observation.

The decisions of interest are the ones the live probe isolated: a route
attachment onto a body that already holds that colour, while the same decision
offers an attachment that completes the Fire+Psychic pair.

Usage:
  python scripts/counterfactual_dragapult_v2.py \
      data/submissions/submission_55545828_dragapult_v1 \
      --baseline data/submissions/submission_55545828_dragapult_v1/submitted_v1_0 \
      --candidate agents/dragapult/dragapult_ml_v2 \
      --report experiments/dragapult_ml_v2/counterfactual_live_v2.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loader import load_dir_agent_module  # noqa: E402

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}

DREEPY, DRAKLOAK, DRAGAPULT = 119, 120, 121
LINE = (DREEPY, DRAKLOAK, DRAGAPULT)
FIRE, PSYCHIC = 2, 5
OPT_ATTACH = 8


def name(card_id: int) -> str:
    return str(CARDS.get(int(card_id), {}).get("name") or card_id)


def attach_facts(observation: dict[str, Any], option: dict[str, Any]):
    """(source, target_id, energies) for an attachment, or None."""
    if int(option.get("type", -1)) != OPT_ATTACH:
        return None
    current = observation.get("current") or {}
    players = current.get("players") or [{}, {}]
    your = int(current.get("yourIndex", 0))
    mine = players[your] if your in (0, 1) else {}
    hand = mine.get("hand") or []
    index = int(option.get("index", -1))
    if not 0 <= index < len(hand) or not isinstance(hand[index], dict):
        return None
    source = int(hand[index].get("id", -1))
    area = int(option.get("inPlayArea", -1))
    target_index = int(option.get("inPlayIndex", -1))
    zone = (mine.get("active") if area == 4 else mine.get("bench") if area == 5 else []) or []
    if not isinstance(zone, list) or not 0 <= target_index < len(zone):
        return None
    target = zone[target_index]
    if not isinstance(target, dict):
        return None
    return source, int(target.get("id", -1)), [int(v) for v in target.get("energies") or []]


def classify(observation: dict[str, Any], option: dict[str, Any]) -> str:
    facts = attach_facts(observation, option)
    if facts is None:
        return "other"
    source, target_id, energies = facts
    if source not in (FIRE, PSYCHIC) or target_id not in LINE:
        return "other"
    other = PSYCHIC if source == FIRE else FIRE
    if source in energies:
        return "duplicate"
    if other in energies:
        return "completes"
    return "first_color"


def choice_of(module: Any, observation: dict[str, Any]) -> int | None:
    """What this agent would pick, without advancing its own history."""
    ranker = getattr(module, "_RANKER", None)
    if ranker is None:
        picked = list(module._fallback_agent(observation))
        return picked[0] if len(picked) == 1 else None
    index = ranker.choose(observation)
    if index is None:
        picked = list(module._fallback_agent(observation))
        return picked[0] if len(picked) == 1 else None
    guarded = getattr(module, "_guarded_index", None)
    if guarded is not None:
        # The guard returns None when it declines, not the unchanged index.
        replacement = guarded(observation, index)
        if replacement is not None:
            index = replacement
    return index


def force(module: Any, observation: dict[str, Any], actual: int) -> None:
    """Keep the agent's intra-turn history on the trajectory that was played."""
    ranker = getattr(module, "_RANKER", None)
    if ranker is not None:
        ranker.observe_external(observation, actual)


def walk(modules: dict[str, Any], replay: dict[str, Any], seat: int, episode: str):
    steps = replay.get("steps") or []
    for module in modules.values():
        module.diag_reset()
    rows: list[dict[str, Any]] = []
    for step_index, pair in enumerate(steps):
        payload = pair[seat]
        if payload.get("status") != "ACTIVE":
            continue
        observation = payload.get("observation") or {}
        if not isinstance(observation, dict):
            continue
        select = observation.get("select")
        actual = (
            steps[step_index + 1][seat].get("action")
            if step_index + 1 < len(steps) else None
        )
        if select is None:
            for module in modules.values():
                ranker = getattr(module, "_RANKER", None)
                if ranker is not None:
                    ranker.reset()
                module._fallback_agent(observation)
            continue
        if not isinstance(actual, list) or len(actual) != 1:
            continue

        options = select.get("option") or []
        played = int(actual[0])
        if not 0 <= played < len(options):
            continue
        picks = {tag: choice_of(module, observation) for tag, module in modules.items()}
        for module in modules.values():
            force(module, observation, played)

        classes = {
            tag: (classify(observation, options[index])
                  if index is not None and 0 <= index < len(options) else "invalid")
            for tag, index in picks.items()
        }
        available = {classify(observation, option) for option in options}
        rows.append({
            "episode": episode,
            "step": step_index,
            "turn": int((observation.get("current") or {}).get("turn") or 0),
            "context": int(select.get("context", -1)),
            "played": played,
            "played_class": classify(observation, options[played]),
            "picks": picks,
            "classes": classes,
            "completes_available": "completes" in available,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    modules = {
        "baseline": load_dir_agent_module(args.baseline),
        "candidate": load_dir_agent_module(args.candidate),
    }

    rows: list[dict[str, Any]] = []
    manifest = list(csv.DictReader(
        (args.run / "manifest.csv").read_text(encoding="utf-8-sig").splitlines()
    ))
    for entry in manifest:
        episode_id = str(entry["episode_id"])
        seat = int(entry["detected_submission_agent_index"])
        path = args.run / "episodes" / episode_id / "replay" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(walk(modules, replay, seat, episode_id))

    dup_rows = [r for r in rows if r["played_class"] == "duplicate" and r["completes_available"]]
    fixed = [r for r in dup_rows if r["classes"]["candidate"] == "completes"]
    still = [r for r in dup_rows if r["classes"]["candidate"] == "duplicate"]
    disagreements = [
        row for row in rows
        if row["picks"]["baseline"] != row["picks"]["candidate"]
    ]

    def rate(tag: str) -> dict[str, Any]:
        counts = Counter(row["classes"][tag] for row in rows)
        route = {
            key: counts.get(key, 0)
            for key in ("duplicate", "completes", "first_color")
        }
        agree = sum(1 for row in rows if row["picks"][tag] == row["played"])
        return {
            "route_classes": route,
            "agreement_with_live": round(agree / len(rows), 4),
        }

    summary = {
        "decisions": len(rows),
        "single_pick_only": True,
        "baseline": rate("baseline"),
        "candidate": rate("candidate"),
        "live_duplicate_with_completing_alternative": len(dup_rows),
        "candidate_takes_the_completing_attach": len(fixed),
        "candidate_repeats_the_duplicate": len(still),
        "candidate_other": len(dup_rows) - len(fixed) - len(still),
        "baseline_candidate_disagreements": len(disagreements),
    }
    print(json.dumps(summary, indent=2))
    print("\nlive duplicate attachments with a completing alternative:")
    for row in dup_rows:
        print(f"  {row['episode']} t{row['turn']:>3} v2 -> {row['classes']['candidate']}")
    print("\nbaseline/candidate disagreements:")
    for row in disagreements:
        print(
            f"  {row['episode']} t{row['turn']:>3} step {row['step']:>4} "
            f"context {row['context']:>2}: {row['picks']['baseline']} -> "
            f"{row['picks']['candidate']} (live {row['played']})"
        )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({
                "summary": summary,
                "duplicates": dup_rows,
                "disagreements": disagreements,
            }, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
