"""Prove (or disprove) deterministic replay for the vendored cg engine.

This probe intentionally reports two hashes:

* ``observable_hash`` covers canonical JSON observations and selected actions;
* ``search_input_hash`` also covers the opaque native search payload.

The first-policy mode contains no Python randomness.  ``--agent-a`` and
``--agent-b`` additionally test whether real agents that consume the search
payload reproduce their complete visible trajectory.  A build is approved
for common-random-number evaluation only when every same-seed repeat has one
observable hash, one terminal result, and one action sequence hash, while
distinct seeds produce distinct observable hashes.  Opaque search-payload
byte stability is reported separately: the payload contains process-local
bytes in this build, but the real-agent probe demonstrates that they do not
change the policy action or visible game trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
# The order is important: do not accidentally load ROOT/cg, which is a
# different engine build.
for path in (ROOT / "vendor", ROOT / "scripts"):
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from agent_loader import load_dir_agent  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from cg_seed import EngineSeedController  # noqa: E402


def _deck(path: Path) -> list[int]:
    values = [int(value) for value in path.read_text(encoding="utf-8-sig").split()]
    if len(values) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(values)}")
    return values


def _resolve(spec: str) -> Path:
    direct = Path(spec)
    if direct.is_dir():
        return direct.resolve()
    for base in (ROOT / "agents", ROOT / "archive" / "agents"):
        if not base.is_dir():
            continue
        candidate = base / spec
        if candidate.is_dir():
            return candidate.resolve()
        for group in base.iterdir():
            nested = group / spec
            if group.is_dir() and nested.is_dir():
                return nested.resolve()
    raise FileNotFoundError(spec)


def _first_policy(observation: dict[str, Any]) -> list[int]:
    select = observation["select"]
    count = min(int(select["minCount"]), len(select.get("option") or []))
    return list(range(count))


def _canonical_observation(observation: dict[str, Any]) -> bytes:
    visible = {key: value for key, value in observation.items() if key != "search_begin_input"}
    return json.dumps(
        visible, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _play(
    policies: list[Callable[[dict[str, Any]], list[int]]],
    decks: list[list[int]],
    *,
    max_steps: int,
) -> dict[str, Any]:
    observation, start_data = battle_start(decks[0], decks[1])
    if observation is None:
        return {
            "result": "start_error",
            "error_player": start_data.errorPlayer,
            "error_type": start_data.errorType,
        }

    observable = hashlib.sha256()
    search_input = hashlib.sha256()
    actions = hashlib.sha256()
    steps = 0
    error = None
    try:
        for _ in range(max_steps):
            current = observation["current"]
            observable.update(_canonical_observation(observation))
            search_input.update(observation.get("search_begin_input", "").encode("ascii"))
            if current["result"] >= 0:
                break
            seat = int(current["yourIndex"])
            try:
                action = list(policies[seat](observation))
                actions.update(json.dumps(action, separators=(",", ":")).encode("ascii"))
                observation = battle_select(action)
            except Exception as exc:  # noqa: BLE001 - the probe must report agent/engine failures
                error = f"{type(exc).__name__}: {exc}"
                break
            steps += 1
        return {
            "result": observation["current"]["result"],
            "steps": steps,
            "observable_hash": observable.hexdigest(),
            "search_input_hash": search_input.hexdigest(),
            "action_hash": actions.hexdigest(),
            "error": error,
        }
    finally:
        battle_finish()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("observable_hash", "search_input_hash", "action_hash", "result", "steps", "error")
    return {
        "games": len(rows),
        **{
            f"distinct_{field}": len({str(row.get(field)) for row in rows})
            for field in fields
        },
        "results": dict(Counter(str(row.get("result")) for row in rows)),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-a")
    parser.add_argument("--agent-b")
    parser.add_argument("--deck-a", type=Path)
    parser.add_argument("--deck-b", type=Path)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--distinct-seeds", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if bool(args.agent_a) != bool(args.agent_b):
        parser.error("--agent-a and --agent-b must be supplied together")

    if args.agent_a:
        runtime_a, _diag_a, _module_a = load_dir_agent(_resolve(args.agent_a))
        runtime_b, _diag_b, _module_b = load_dir_agent(_resolve(args.agent_b))
        policies = [runtime_a, runtime_b]
        decks = [
            _deck(args.deck_a) if args.deck_a else list(runtime_a({"select": None})),
            _deck(args.deck_b) if args.deck_b else list(runtime_b({"select": None})),
        ]
        policy_label = [args.agent_a, args.agent_b]
    else:
        default_a = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v21" / "deck.csv"
        default_b = ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8" / "deck.csv"
        decks = [_deck(args.deck_a or default_a), _deck(args.deck_b or default_b)]
        policies = [_first_policy, _first_policy]
        policy_label = ["first", "first"]

    controller = EngineSeedController()
    started = time.perf_counter()
    try:
        repeated = []
        for _ in range(args.repeats):
            controller.set_seed(args.seed)
            repeated.append(_play(policies, decks, max_steps=args.max_steps))

        distinct = []
        for index in range(args.distinct_seeds):
            controller.set_seed(args.seed + (index + 1) * 7919)
            distinct.append(_play(policies, decks, max_steps=args.max_steps))

        same = _summarize(repeated)
        varied = _summarize(distinct)
        observable_replay_reproducible = (
            same["distinct_observable_hash"] == 1
            and same["distinct_action_hash"] == 1
            and same["distinct_result"] == 1
            and same["distinct_steps"] == 1
            and same["distinct_error"] == 1
            and repeated
            and repeated[0].get("error") is None
            and varied["distinct_observable_hash"] == len(distinct)
        )
        opaque_payload_byte_stable = same["distinct_search_input_hash"] == 1
        report = {
            "approved_for_crn": bool(observable_replay_reproducible),
            "observable_replay_reproducible": bool(observable_replay_reproducible),
            "opaque_search_payload_byte_stable": opaque_payload_byte_stable,
            "opaque_payload_note": (
                "The native search payload is not byte-stable, but visible observations, "
                "real-agent actions, and results are stable. Every evaluated agent pair "
                "must still pass duplicate-game calibration in paired_gauntlet.py."
                if not opaque_payload_byte_stable else None
            ),
            "policies": policy_label,
            "seed": args.seed,
            "wall_seconds": time.perf_counter() - started,
            "engine": controller.status(),
            "same_seed": same,
            "distinct_seeds": varied,
        }
    finally:
        controller.restore()

    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["approved_for_crn"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
