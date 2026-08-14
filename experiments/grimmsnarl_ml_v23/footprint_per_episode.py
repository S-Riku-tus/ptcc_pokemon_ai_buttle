"""Per-episode v22 -> v23 footprint over the full pooled v22 ladder corpus.

The v23 report measured exposure on 44 games.  v22 has now played 194, of which
73 are losses, so the question that decides whether v23 is worth a slot can be
asked directly: on the boards where v22 actually lost, how often does v23 answer
differently, and where?

Two things this adds over ``parallel_policy_footprint.py``:

1. per-episode divergence counts, so footprint can be joined to the outcome;
2. a validity check that costs nothing - v22 is the version that played these
   boards, so any decision where v22's own answer differs from the stored
   action means the walker, not the agent, is wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "vendor", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from probe_grimmsnarl_v21_footprint import load, single  # noqa: E402

_M: dict[str, Any] = {}
_ERR: str | None = None


def _init(base: str, candidate: str) -> None:
    global _ERR
    try:
        _M["base"] = load(Path(base))
        _M["cand"] = load(Path(candidate))
    except Exception as exc:  # noqa: BLE001
        _ERR = f"{type(exc).__name__}: {exc}"


def _walk(module: Any, replay: dict, seat: int) -> list[tuple[int, int, int, int | None, int]]:
    """(step, turn, context, proposed, played) for own single-pick decisions."""
    out = []
    steps = replay.get("steps") or []
    for hook in ("diag_reset", "reset_state"):
        fn = getattr(module, hook, None)
        if callable(fn):
            fn()
            break
    ranker = getattr(module, "_RANKER", None)
    if ranker is not None and hasattr(ranker, "teacher_forced"):
        ranker.teacher_forced = True
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        observation = (step[seat] or {}).get("observation") or {}
        select = observation.get("select")
        current = observation.get("current")
        if not isinstance(select, dict) or not isinstance(current, dict):
            continue
        if not current.get("players") or not (select.get("option") or []):
            continue
        played = single((steps[index + 1][seat] or {}).get("action"))
        if played is None:
            continue
        out.append((
            index,
            int(current.get("turn", -1)),
            int(select.get("context", -1)),
            single(module.agent(observation)),
            played,
        ))
        module.observe_external(observation, played)
    return out


def _job(payload: tuple[str, str, int, dict]) -> dict[str, Any]:
    episode_id, path, seat, meta = payload
    if _ERR:
        return {"episode_id": episode_id, "error": _ERR}
    try:
        replay = json.loads(Path(path).read_text(encoding="utf-8"))
        base = _walk(_M["base"], replay, seat)
        cand = _walk(_M["cand"], replay, seat)
        if len(base) != len(cand):
            return {"episode_id": episode_id, "error": "decision_count_mismatch"}
        base_infidelity = sum(1 for (_, _, _, p, a) in base if p != a)
        diffs = []
        for (step, turn, ctx, b, played), (_, _, _, c, _) in zip(base, cand):
            if b != c:
                diffs.append({
                    "step": step, "turn": turn, "context": ctx,
                    "v22": b, "v23": c, "played": played,
                })
        return {
            "episode_id": episode_id,
            "meta": meta,
            "decisions": len(base),
            "base_infidelity": base_infidelity,
            "changed": len(diffs),
            "diffs": diffs,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"episode_id": episode_id, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="agents/grimmsnarl/grimmsnarl_ml_v22")
    parser.add_argument("--candidate", default="agents/grimmsnarl/grimmsnarl_ml_v23")
    parser.add_argument(
        "--games", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v23"
        / "ladder_v22_v23_games.csv",
    )
    parser.add_argument("--runs", type=Path, default=ROOT / "data" / "runs" / "grimmsnarl")
    parser.add_argument("--versions", default="v22_a,v22_b,v22_c,v22_d")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v23"
        / "footprint_per_episode.json",
    )
    args = parser.parse_args()

    wanted = {v for v in args.versions.split(",") if v}
    index: dict[str, tuple[Path, int]] = {}
    for run_dir in sorted(args.runs.iterdir()):
        manifest = run_dir / "manifest.csv"
        if not run_dir.is_dir() or not manifest.exists():
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            if row.get("detected_submission_agent_index") in {"0", "1"}:
                index[row["episode_id"]] = (run_dir, int(row["detected_submission_agent_index"]))

    selected = []
    for row in csv.DictReader(args.games.open(encoding="utf-8-sig")):
        if row["version"] not in wanted:
            continue
        entry = index.get(row["episode_id"])
        if entry is None:
            continue
        run_dir, seat = entry
        path = run_dir / "episodes" / row["episode_id"] / "replay" / f"episode_{row['episode_id']}.json"
        if not path.exists():
            continue
        selected.append((row["episode_id"], str(path), seat, {
            "version": row["version"],
            "won": row["won"] == "True",
            "went_first": row["went_first"],
            "family": row["opponent_family"],
            "opponent_rating": float(row["opponent_rating"]) if row["opponent_rating"] else None,
            "own_first_shadow_turn": int(row["own_first_shadow_turn"]) if row["own_first_shadow_turn"] else None,
        }))

    print(f"games: {len(selected)}", file=sys.stderr, flush=True)
    started = time.perf_counter()
    context = mp.get_context("spawn")
    rows: list[dict[str, Any]] = []
    with context.Pool(
        max(1, args.workers), initializer=_init,
        initargs=(str((ROOT / args.base).resolve()), str((ROOT / args.candidate).resolve())),
    ) as pool:
        for done, row in enumerate(pool.imap_unordered(_job, selected, chunksize=1), start=1):
            rows.append(row)
            if done % 10 == 0 or done == len(selected):
                print(f"progress {done}/{len(selected)}", file=sys.stderr, flush=True)

    errors = [r for r in rows if r.get("error")]
    good = [r for r in rows if not r.get("error")]
    infidelity = sum(r["base_infidelity"] for r in good)

    by_outcome: dict[str, Counter] = defaultdict(Counter)
    by_family: dict[str, Counter] = defaultdict(Counter)
    by_context: Counter = Counter()
    by_own_turn: Counter = Counter()
    per_episode = []
    for r in good:
        meta = r["meta"]
        key = "won" if meta["won"] else "lost"
        for bucket, name in ((by_outcome, key), (by_family, meta["family"])):
            bucket[name]["games"] += 1
            bucket[name]["decisions"] += r["decisions"]
            bucket[name]["changed"] += r["changed"]
            bucket[name]["games_touched"] += int(r["changed"] > 0)
        for d in r["diffs"]:
            by_context[d["context"]] += 1
            by_own_turn[d["turn"]] += 1
        per_episode.append({
            "episode_id": r["episode_id"], **meta,
            "decisions": r["decisions"], "changed": r["changed"],
        })
    per_episode.sort(key=lambda x: -x["changed"])

    payload = {
        "valid": not errors and infidelity == 0,
        "base": args.base, "candidate": args.candidate,
        "games": len(good), "errors": errors[:20],
        "base_infidelity": infidelity,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "totals": {
            "decisions": sum(r["decisions"] for r in good),
            "changed": sum(r["changed"] for r in good),
            "games_touched": sum(1 for r in good if r["changed"] > 0),
        },
        "by_outcome": {k: dict(v) for k, v in by_outcome.items()},
        "by_family": {k: dict(v) for k, v in sorted(by_family.items(), key=lambda i: -i[1]["games"])},
        "by_context": dict(sorted(by_context.items())),
        "by_shared_turn": dict(sorted(by_own_turn.items())),
        "per_episode": per_episode,
        "diffs": {r["episode_id"]: r["diffs"] for r in good if r["diffs"]},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in {"per_episode", "diffs", "by_family"}},
                     ensure_ascii=False, indent=2))
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
