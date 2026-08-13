"""Who actually decides a v21 move: the ranker's Top-1, or a rule shell?

Replicates ``main._choose`` decision-for-decision on stored replays and records
the index after every layer in the chain, so each override can be attributed to
the exact module that produced it.  Stored actions advance all persistent state
(teacher forcing), so a changed answer never invents a divergent future.

    python experiments/grimmsnarl_1100_diagnosis/code-evolution/layer_attribution.py \
        --agent agents/grimmsnarl/grimmsnarl_ml_v21 --runs 20260813_...v21... \
        --report out.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_loader import load_dir_agent_module  # noqa: E402

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


def single(action: Any) -> int | None:
    if (
        isinstance(action, list)
        and len(action) == 1
        and isinstance(action[0], int)
    ):
        return action[0]
    return None


# Order matches main._choose / main._route exactly.
ML_CHAIN = ("planner", "residual", "wall_guard")
ROUTE_CHAIN = ("attack_access", "wall_break", "horizon_prize", "mirror_prize")


def decide(m: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """One instrumented replication of ``main._choose``."""
    route = m._ROUTER.choose(observation)
    m._note_matchup(route)
    for layer in (m._RESIDUAL, m._ROUTE, m._WALL_BREAK, m._MIRROR_PRIZE,
                  m._HORIZON_PRIZE):
        if layer is not None:
            layer.note(observation)

    rule_choice = m._fallback_agent(observation)
    select = observation.get("select") or {}
    out: dict[str, Any] = {
        "route": route,
        "context": int(select.get("context", -1)),
        "n_options": len(select.get("option") or []),
        "max_count": int(select.get("maxCount") or 0),
        "changed_by": [],
    }
    rule_index = m._rule_index(rule_choice)
    out["rule_index"] = rule_index

    scorable = m._RANKER.is_scorable(select)
    out["scorable"] = bool(scorable)
    if not scorable:
        out["path"] = "rule_not_scorable"
        if rule_index is None:
            out["path"] = "rule_multipick"
            out["final"] = None
            return out
        index = rule_index
        for name, layer, disabled in (
            ("attack_access", m._ROUTE, m._ROUTE_DISABLED),
            ("wall_break", m._WALL_BREAK, m._WALL_BREAK_DISABLED),
            ("horizon_prize", m._HORIZON_PRIZE, m._HORIZON_PRIZE_DISABLED),
            ("mirror_prize", m._MIRROR_PRIZE, m._MIRROR_PRIZE_DISABLED),
        ):
            if disabled:
                continue
            before = index
            if name in ("horizon_prize", "mirror_prize"):
                index = layer.adjust(observation, select, index, None)
            else:
                index = layer.adjust(observation, select, index, rule_choice)
            if index != before:
                out["changed_by"].append(name)
        out["final"] = index
        return out

    top1 = m._RANKER.choose(observation)
    out["ranker_top1"] = top1
    if top1 is None:
        out["path"] = "rule_ranker_deferred"
        if rule_index is None:
            out["final"] = None
            return out
        index = rule_index
        for name, layer, disabled in (
            ("attack_access", m._ROUTE, m._ROUTE_DISABLED),
            ("wall_break", m._WALL_BREAK, m._WALL_BREAK_DISABLED),
            ("horizon_prize", m._HORIZON_PRIZE, m._HORIZON_PRIZE_DISABLED),
            ("mirror_prize", m._MIRROR_PRIZE, m._MIRROR_PRIZE_DISABLED),
        ):
            if disabled:
                continue
            before = index
            if name in ("horizon_prize", "mirror_prize"):
                index = layer.adjust(observation, select, index, None)
            else:
                index = layer.adjust(observation, select, index, rule_choice)
            if index != before:
                out["changed_by"].append(name)
        out["final"] = index
        return out

    out["path"] = "ml"
    index = top1
    if not m._PLANNER_DISABLED:
        before = index
        index = m._PLANNER.adjust(
            observation, select, index, m._RANKER.last_scores
        )
        if index != before:
            out["changed_by"].append("planner")
    if not m._RESIDUAL_DISABLED:
        before = index
        index = m._RESIDUAL.adjust(
            observation, select, index, m._RANKER, m._RANKER.last_scores
        )
        if index != before:
            out["changed_by"].append("residual")
    if not m._WALL_GUARD_DISABLED:
        before = index
        index = m._WALL_GUARD.adjust(observation, select, index, rule_choice)
        if index != before:
            out["changed_by"].append("wall_guard")
    if not m._ROUTE_DISABLED:
        before = index
        index = m._ROUTE.adjust(observation, select, index, rule_choice)
        if index != before:
            out["changed_by"].append("attack_access")
    if not m._WALL_BREAK_DISABLED:
        before = index
        index = m._WALL_BREAK.adjust(observation, select, index, rule_choice)
        if index != before:
            out["changed_by"].append("wall_break")
    if not m._HORIZON_PRIZE_DISABLED:
        before = index
        index = m._HORIZON_PRIZE.adjust(
            observation, select, index, m._RANKER.last_scores
        )
        if index != before:
            out["changed_by"].append("horizon_prize")
    if not m._MIRROR_PRIZE_DISABLED:
        before = index
        index = m._MIRROR_PRIZE.adjust(
            observation, select, index, m._RANKER.last_scores
        )
        if index != before:
            out["changed_by"].append("mirror_prize")
    out["final"] = index
    return out


def walk(m: Any, replay: dict, seat: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    steps = replay.get("steps") or []
    m.diag_reset()
    m._RANKER.teacher_forced = True
    for i, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[i + 1]):
            continue
        observation = (step[seat] or {}).get("observation") or {}
        select = observation.get("select")
        current = observation.get("current")
        if not isinstance(select, dict) or not isinstance(current, dict):
            continue
        if not current.get("players") or not (select.get("option") or []):
            continue
        action = (steps[i + 1][seat] or {}).get("action")
        played = single(action)
        row = decide(m, observation)
        row["turn"] = int(current.get("turn", -1))
        row["played"] = played
        row["played_is_single"] = played is not None
        rows.append(row)
        if played is not None:
            m.observe_external(observation, played)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default="agents/grimmsnarl/grimmsnarl_ml_v21")
    ap.add_argument("--runs", default="20260813_grimmsnarl_ml_v21_sub55456713")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    run_root = ROOT / "data" / "runs" / "grimmsnarl"
    selected: list[tuple[str, Path, int]] = []
    for run_name in args.runs.split(","):
        run_dir = run_root / run_name.strip()
        manifest = run_dir / "manifest.csv"
        if not manifest.exists():
            print(f"missing manifest: {manifest}")
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            if row.get("detected_submission_agent_index") not in {"0", "1"}:
                continue
            eid = row["episode_id"]
            path = (
                run_dir / "episodes" / eid / "replay" / f"episode_{eid}.json"
            )
            if path.exists():
                selected.append(
                    (eid, path, int(row["detected_submission_agent_index"]))
                )
    if args.limit:
        selected = selected[: args.limit]
    print(f"{len(selected)} episodes")

    m = load(ROOT / args.agent)
    started = time.perf_counter()
    totals: Counter = Counter()
    by_layer: Counter = Counter()
    by_layer_ml: Counter = Counter()
    by_context_ml: Counter = Counter()
    layer_ctx: Counter = Counter()
    examples: list[dict[str, Any]] = []
    games_touched: Counter = Counter()

    cumulative: dict[str, Counter] = {}
    for n, (eid, path, seat) in enumerate(selected, 1):
        replay = json.loads(path.read_text(encoding="utf-8"))
        rows = walk(m, replay, seat)
        for label, obj in (
            ("planner", m._PLANNER), ("residual", m._RESIDUAL),
            ("wall_guard", m._WALL_GUARD), ("attack_access", m._ROUTE),
            ("wall_break", m._WALL_BREAK), ("mirror_prize", m._MIRROR_PRIZE),
            ("horizon_prize", m._HORIZON_PRIZE), ("ranker", m._RANKER),
            ("router", m._ROUTER),
        ):
            snap = obj.snapshot()
            bucket = cumulative.setdefault(label, Counter())
            for key, value in snap.items():
                if isinstance(value, (int, float)) and not isinstance(
                    value, bool
                ):
                    bucket[key] += value
        seen_layers: set[str] = set()
        for row in rows:
            totals["decisions"] += 1
            totals["played_missing"] += int(not row["played_is_single"])
            totals[f"path_{row['path']}"] += 1
            if row["path"] == "ml":
                totals["ml_scored"] += 1
                if row["changed_by"]:
                    totals["ml_overridden"] += 1
                    by_layer_ml[row["changed_by"][0]] += 1
                    by_context_ml[row["context"]] += 1
                else:
                    totals["ml_top1_survived"] += 1
                if row["played_is_single"]:
                    totals["ml_agree_played"] += int(
                        row["final"] == row["played"]
                    )
                    totals["ml_top1_agree_played"] += int(
                        row["ranker_top1"] == row["played"]
                    )
                    totals["ml_with_played"] += 1
            for layer in row["changed_by"]:
                by_layer[layer] += 1
                seen_layers.add(layer)
                layer_ctx[f"{layer}|ctx{row['context']}"] += 1
                if len(examples) < 60:
                    examples.append({
                        "episode": eid, "turn": row["turn"],
                        "context": row["context"], "layer": layer,
                        "path": row["path"], "final": row["final"],
                        "played": row["played"],
                    })
        for layer in seen_layers:
            games_touched[layer] += 1
        if n % 10 == 0:
            print(f"  {n}/{len(selected)} "
                  f"({time.perf_counter() - started:.0f}s)")

    guard_stats = {
        "fallback": dict(m.fallback_policy.DIAG),
        "router": m._ROUTER.snapshot(),
        "ranker": m._RANKER.snapshot(),
        "planner": m._PLANNER.snapshot(),
        "residual": m._RESIDUAL.snapshot(),
        "wall_guard": m._WALL_GUARD.snapshot(),
        "attack_access": m._ROUTE.snapshot(),
        "wall_break": m._WALL_BREAK.snapshot(),
        "mirror_prize": m._MIRROR_PRIZE.snapshot(),
        "horizon_prize": m._HORIZON_PRIZE.snapshot(),
    }
    # snapshots are cumulative only within a game because walk() resets; take
    # the guard counters from a second pass that never resets instead.
    payload = {
        "agent": args.agent,
        "runs": args.runs,
        "episodes": len(selected),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "totals": dict(totals),
        "changed_by_layer": dict(by_layer),
        "ml_overrides_first_layer": dict(by_layer_ml),
        "ml_override_context": dict(sorted(by_context_ml.items())),
        "layer_by_context": dict(sorted(layer_ctx.items())),
        "games_touched_by_layer": dict(games_touched),
        "last_game_guard_snapshot": guard_stats,
        "cumulative_guard_counters": {
            label: dict(counts) for label, counts in cumulative.items()
        },
        "examples": examples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {k: v for k, v in payload.items()
         if k not in {"examples", "last_game_guard_snapshot",
                      "layer_by_context"}},
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
