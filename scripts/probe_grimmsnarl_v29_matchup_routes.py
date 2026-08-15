"""Audit v29's targeted ranker switches on the v28 ladder histories.

The replay action teacher-forces both ranker histories.  This does not estimate
the counterfactual outcome, but it does prove the deployed footprint: which
public matchup route owned each decision and exactly how often v29 differs
from v28's recorded action, the v25 race ranker, and the v22 elite ranker.
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
for path in (ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loader import load_dir_agent_module  # noqa: E402
from probe_grimmsnarl_v28_footprint import load_episodes, single  # noqa: E402


DEFAULT_RUN = (
    ROOT / "data/runs/grimmsnarl/20260815_grimmsnarl_ml_v28_sub55526859"
)
DEFAULT_AGENT = ROOT / "agents/grimmsnarl/grimmsnarl_ml_v29"
DEFAULT_GAMES = ROOT / "experiments/grimmsnarl_ml_v28/version_games.csv"
DEFAULT_OUTPUT = ROOT / "experiments/grimmsnarl_ml_v29/matchup_routes.json"
AUDIT_FAMILIES = {
    "Mega Lopunny / Froslass",
    "other: Hydrapple ex",
    "Ogerpon",  # negative control: this must remain v25
}

CARDS = {
    int(card["cardId"]): str(card.get("name") or card["cardId"])
    for card in json.loads(
        (ROOT / "vendor/cg/cards.json").read_text(encoding="utf-8")
    )
}


def action_label(
    features: Any, observation: dict[str, Any], index: Any
) -> str:
    select = observation.get("select") or {}
    options = list(select.get("option") or [])
    if not isinstance(index, int) or not 0 <= index < len(options):
        return "unresolved"
    current = observation.get("current") or {}
    option = options[index]
    try:
        kind = features.action_type(current, option, select)
        card = features.candidate_card(current, option, select) or {}
        target = features.candidate_target(current, option) or {}
    except Exception:  # noqa: BLE001
        return f"type:{option.get('type', -1)}"
    card_id = int(card.get("id", -1))
    target_id = int(target.get("id", -1))
    parts = [str(kind)]
    if card_id >= 0:
        parts.append(CARDS.get(card_id, str(card_id)))
    attack_id = option.get("attackId")
    if isinstance(attack_id, int):
        parts.append(f"attack#{attack_id}")
    if target_id >= 0 and target_id != card_id:
        parts.append(f"to:{CARDS.get(target_id, str(target_id))}")
    return ":".join(parts)


def metadata(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        if row.get("version") != "v28":
            continue
        episode_id = int(row["episode_id"])
        result[episode_id] = {
            "family": row["opponent_family"],
            "won": int(row["won"]),
            "opponent_rating": float(row["opponent_rating"]),
        }
    return result


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--submission", type=int, default=55526859)
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--games", type=Path, default=DEFAULT_GAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    meta = metadata(args.games)
    wanted = {
        episode_id for episode_id, item in meta.items()
        if item["family"] in AUDIT_FAMILIES
    }
    episodes = [
        item for item in load_episodes(args.run, args.submission)
        if item["episode_id"] in wanted
    ]
    module = load_dir_agent_module(args.agent)
    features = sys.modules.get("ml_features")
    if features is None:
        raise RuntimeError("agent did not load ml_features")
    rows: list[dict[str, Any]] = []
    overall: Counter[str] = Counter()

    for episode in episodes:
        module.diag_reset()
        for name in ("_RACE", "_WALL"):
            ranker = getattr(module, name, None)
            if ranker is not None:
                ranker.teacher_forced = True
        counts: Counter[str] = Counter()
        examples: list[dict[str, Any]] = []
        for decision in episode["decisions"]:
            proposed = module.agent(decision["observation"])
            if decision["evaluate"]:
                final = single(proposed)
                trace = dict(getattr(module, "_LAST_TRACE", {}))
                route = str(trace.get("route", "unknown"))
                policy = str(trace.get("policy", "unknown"))
                played = decision["played"]
                v25 = trace.get("v25_race")
                v22 = trace.get("v22_wall")
                counts["evaluated"] += 1
                counts[f"route:{route}"] += 1
                counts[f"policy:{policy}"] += 1
                counts["final_diff_played"] += int(final != played)
                counts["final_diff_v25"] += int(final != v25)
                counts["final_diff_v22"] += int(final != v22)
                if final != played:
                    played_label = action_label(
                        features, decision["observation"], played
                    )
                    final_label = action_label(
                        features, decision["observation"], final
                    )
                    counts[f"change:{played_label} -> {final_label}"] += 1
                if final != played and len(examples) < 20:
                    examples.append({
                        "turn": decision["turn"],
                        "context": decision["context"],
                        "route": route,
                        "played_v28": played,
                        "v25": v25,
                        "v22": v22,
                        "final_v29": final,
                        "played_label": played_label,
                        "final_label": final_label,
                    })
            if decision["played"] is not None:
                module.observe_external(
                    decision["observation"], decision["played"]
                )
        overall.update(counts)
        info = meta[episode["episode_id"]]
        row = {
            "episode_id": episode["episode_id"],
            **info,
            "counts": dict(counts),
            "changed_examples": examples,
        }
        rows.append(row)
        print(
            f"ep={episode['episode_id']} family={info['family']:<27} "
            f"eval={counts['evaluated']:>3} changed={counts['final_diff_played']:>3} "
            f"v22={counts['policy:v22']:>3} v25={counts['policy:v25']:>3}"
        )

    payload = {
        "agent": str(args.agent),
        "run": str(args.run),
        "episodes": rows,
        "overall": dict(overall),
        "contract": {
            "lopunny_and_hydrapple_policy": "v22",
            "pure_ogerpon_policy": "v25",
            "teacher_forced_history": True,
            "counterfactual_outcome_claimed": False,
        },
        "load_errors": module.diag_snapshot().get("load_errors", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["overall"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
