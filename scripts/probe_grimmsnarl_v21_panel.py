"""Behaviour panel: what v21 would actually play on the stored v19/v20 boards.

The v20 promotion was decided on held-out Top-1, and Top-1 hid what
regressed: v20's Boss rate fell from 0.303 of offers to 0.180 against an elite
band at 0.385 while overall Top-1 rose 0.41 points. One accuracy number cannot
see a rare high-value action, so this panel measures the decision classes the
v19/v20 ladder autopsy named, per agent, on identical boards:

* ``boss`` / ``petrel_boss`` - the Prize-conversion behaviours with a rating
  gradient (``petrel_boss`` rho +0.581, BH significant; raw Boss play rate is
  not significant but v20 is far below every measured pilot);
* ``dead_shadow`` - a Shadow Bullet thrown into a damage-immune Active, 168 of
  1,903 stored swings and 8.8% of them;
* ``funds_active_retreat`` / ``leaves_retreat_locked`` - the retreat lock that
  strands a ready Grimmsnarl ex on the bench, 128 own turns in 82 of 529 games;
* ``attack`` - the commit axis where v20 moved back toward v8 (0.967 against an
  elite 0.941).

Every rate is offers-based: the denominator is decisions where the class was a
legal option, so an unavailable action never reads as a refusal. Answers come
from the deployed ``agent`` callable, safety shell included, because the shell
is what plays - not the ranker's raw argmax.
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

import importlib.util  # noqa: E402

from agent_loader import load_dir_agent_module  # noqa: E402


def load_reference_features(agent_dir: Path):
    """One fixed feature module for classification, whichever agent answers.

    The panel state ``retreat_lock_risk`` only exists in v21's extractor, so
    reading it from the agent under test would score v20 as never facing the
    decision.  Loading v21's module under its own name keeps the denominator
    identical for every agent.
    """
    spec = importlib.util.spec_from_file_location(
        "ml_features_reference", agent_dir / "ml_features.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ml_features_reference"] = module
    spec.loader.exec_module(module)
    return module


MODULES = (
    "main", "ml_runtime", "ml_features", "fallback_policy", "ml_planner",
    "ml_residual", "policy_router", "matchup_guard", "attack_access",
    "policy_base", "wall_break", "mirror_prize", "horizon_prize",
)


def load(agent_dir: Path) -> Any:
    for name in MODULES:
        sys.modules.pop(name, None)
    for entry in list(sys.path):
        if "grimmsnarl_ml_v" in entry:
            sys.path.remove(entry)
    return load_dir_agent_module(agent_dir)


REFERENCE: Any = None


def single(action: Any) -> int | None:
    if (
        isinstance(action, list)
        and len(action) == 1
        and isinstance(action[0], int)
    ):
        return action[0]
    return None


def reset(module: Any) -> None:
    for name in ("diag_reset", "reset_state"):
        hook = getattr(module, name, None)
        if callable(hook):
            hook()
            return


def classify(mf: Any, current: dict, select: dict, option: dict) -> set[str]:
    """Which panel classes this legal option belongs to."""
    tags: set[str] = set()
    try:
        action = mf.action_type(current, option, select)
    except Exception:  # noqa: BLE001
        return tags
    card = mf.candidate_card(current, option, select) or {}
    card_id = int(card.get("id", -1))
    context = int(select.get("context", -1))

    if context == mf.MAIN_CONTEXT:
        if action == "attack":
            tags.add("attack")
            if mf._int(option.get("attackId")) == mf.SHADOW_BULLET_ID:
                players = current.get("players") or []
                your = int(current.get("yourIndex", 0))
                if len(players) > 1:
                    opponent = players[1 - your]
                    active = (mf._cards(opponent, "active") or [None])[0]
                    if active is not None and mf.shadow_damage_to(
                        active, mf._stadium_id(current)
                    ) <= 0.0:
                        tags.add("dead_shadow")
        if card_id == mf.BOSS_ID:
            tags.add("boss")
        if action == "energy":
            # Basic Darkness is the deck's only Energy, so the attach target
            # area is the whole decision.
            area = mf._int(option.get("inPlayArea", option.get("area")))
            if area == mf.AREA_ACTIVE:
                tags.add("energy_to_active")
            elif area == mf.AREA_BENCH:
                tags.add("energy_to_bench")
    else:
        # Petrel resolves as a deck search; a Boss taken there is the
        # rating-correlated behaviour.
        if card_id == mf.BOSS_ID:
            tags.add("search_boss")
    return tags


def walk(
    module: Any,
    mf: Any,
    replay: dict[str, Any],
    seat: int,
    tally: Counter,
    by_matchup: Counter,
) -> None:
    """One row per own *turn*, not per decision.

    A card that stays legal through five MAIN prompts is one offer; counting it
    five times is what turned an 88.6% take rate into 18.9% the last time this
    line measured per decision.
    """
    steps = replay.get("steps") or []
    reset(module)
    offered_by_turn: dict[int, set[str]] = defaultdict(set)
    chosen_by_turn: dict[int, set[str]] = defaultdict(set)
    risk_turns: set[int] = set()
    for step_index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[step_index + 1]):
            continue
        record = step[seat] or {}
        observation = record.get("observation") or {}
        select = observation.get("select")
        if not isinstance(select, dict):
            continue
        current = observation.get("current")
        if not isinstance(current, dict) or not current.get("players"):
            continue
        options = list(select.get("option") or [])
        played = single((steps[step_index + 1][seat] or {}).get("action"))
        if played is None or not options:
            continue

        answer = single(module.agent(observation))
        module.observe_external(observation, played)
        tally["decisions"] += 1
        by_matchup["decisions"] += 1
        if answer == played:
            tally["agrees_with_stored"] += 1
            by_matchup["agrees_with_stored"] += 1

        turn = int(current.get("turn", -1))
        tags = [classify(mf, current, select, option) for option in options]
        for group in tags:
            offered_by_turn[turn] |= group
        if answer is not None and 0 <= answer < len(tags):
            chosen_by_turn[turn] |= tags[answer]
        try:
            if mf.state_features(current).get("retreat_lock_risk", 0):
                risk_turns.add(turn)
        except Exception:  # noqa: BLE001
            pass

    for turn, offered in offered_by_turn.items():
        chosen = chosen_by_turn.get(turn, set())
        for name in ("attack", "boss", "search_boss", "dead_shadow"):
            if name in offered:
                tally[f"{name}_offers"] += 1
                by_matchup[f"{name}_offers"] += 1
                if name in chosen:
                    tally[f"{name}_taken"] += 1
                    by_matchup[f"{name}_taken"] += 1
        if turn in risk_turns and "energy_to_active" in offered:
            tally["funds_active_retreat_offers"] += 1
            by_matchup["funds_active_retreat_offers"] += 1
            if "energy_to_active" in chosen:
                tally["funds_active_retreat_taken"] += 1
                by_matchup["funds_active_retreat_taken"] += 1
            elif "energy_to_bench" in chosen:
                tally["leaves_retreat_locked"] += 1
                by_matchup["leaves_retreat_locked"] += 1


def rates(tally: Counter) -> dict[str, Any]:
    out: dict[str, Any] = {"decisions": tally["decisions"]}
    if tally["decisions"]:
        out["top1_vs_stored"] = round(
            tally["agrees_with_stored"] / tally["decisions"], 4
        )
    for name in (
        "attack", "boss", "search_boss", "dead_shadow",
        "funds_active_retreat",
    ):
        offers = tally[f"{name}_offers"]
        taken = tally[f"{name}_taken"]
        out[name] = {
            "offers": offers,
            "taken": taken,
            "rate": round(taken / offers, 4) if offers else None,
        }
    out["leaves_retreat_locked"] = tally["leaves_retreat_locked"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", action="append", default=[], metavar="LABEL=DIR",
        help="Defaults to v20 and v21.",
    )
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v20"
        / "ladder_history_games.csv",
    )
    parser.add_argument(
        "--runs", type=Path, default=ROOT / "data" / "runs" / "grimmsnarl"
    )
    parser.add_argument("--versions", default="v19,v19_old,v20")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v21"
        / "behaviour_panel.json",
    )
    args = parser.parse_args()

    specs = args.agent or [
        "v20=agents/grimmsnarl/grimmsnarl_ml_v20",
        "v21=agents/grimmsnarl/grimmsnarl_ml_v21",
    ]
    wanted = {v for v in args.versions.split(",") if v}

    index: dict[str, tuple[Path, int]] = {}
    for run_dir in sorted(args.runs.iterdir()):
        manifest = run_dir / "manifest.csv"
        if not run_dir.is_dir() or not manifest.exists():
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            if row.get("detected_submission_agent_index") in {"0", "1"}:
                index[row["episode_id"]] = (
                    run_dir, int(row["detected_submission_agent_index"])
                )

    selected: list[tuple[str, dict, int, str]] = []
    for meta in csv.DictReader(args.games.open(encoding="utf-8-sig")):
        if wanted and meta["version"] not in wanted:
            continue
        entry = index.get(meta["episode_id"])
        if entry is None:
            continue
        run_dir, seat = entry
        path = (
            run_dir / "episodes" / meta["episode_id"] / "replay"
            / f"episode_{meta['episode_id']}.json"
        )
        if not path.exists():
            continue
        selected.append((meta["episode_id"], path, seat, meta["opponent_family"]))
        if args.max_games and len(selected) >= args.max_games:
            break

    print(f"boards: {len(selected)} games")
    global REFERENCE
    REFERENCE = load_reference_features(
        ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v21"
    )
    results: dict[str, Any] = {}
    for spec in specs:
        label, _, directory = spec.partition("=")
        module = load(ROOT / directory)
        mf = REFERENCE
        tally: Counter = Counter()
        matchups: dict[str, Counter] = defaultdict(Counter)
        started = time.perf_counter()
        for episode_id, path, seat, family in selected:
            replay = json.loads(path.read_text(encoding="utf-8"))
            walk(module, mf, replay, seat, tally, matchups[family])
        results[label] = {
            "agent_dir": directory,
            "elapsed_seconds": round(time.perf_counter() - started, 1),
            "overall": rates(tally),
            "by_matchup": {
                family: rates(counts)
                for family, counts in sorted(
                    matchups.items(), key=lambda item: -item[1]["decisions"]
                )
            },
        }
        print(f"  {label}: " + json.dumps(
            results[label]["overall"], ensure_ascii=False
        ))

    payload = {"games": len(selected), "agents": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
