"""Is the Adrena-Brain gap a decision we control, or a scoreboard?

Adrena-Brain separates wins from losses harder than anything else measured on
the 194-game pool (8.37 vs 4.12 uses in the mirror, t=7.4).  That is exactly the
shape of a metric this repo has already been burned by twice: uptake when
offered is 98.6%, so the count cannot be raised by "using it more".  Either the
count is downstream of winning - a longer, healthier board offers more
activations - or it is upstream, set by how early Munkidori is established with
a Darkness Energy.

This separates the two by walking the replay for *availability* rather than
usage:

* the own turn Munkidori first reaches the bench;
* the own turn Munkidori first carries a Darkness Energy, i.e. the first turn
  Adrena-Brain can legally fire at all;
* activations per own turn *after* that point, which is the part a policy could
  still be leaving on the table;
* Froslass evolutions in the mirror, which the win/loss contrast puts on the
  wrong side.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts", ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v22"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ml_features as mf  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import wilson  # noqa: E402

GAMES = ROOT / "experiments" / "grimmsnarl_ml_v23" / "ladder_v22_v23_games.csv"
RUNS = ROOT / "data" / "runs" / "grimmsnarl"
OUT = ROOT / "experiments" / "grimmsnarl_ml_v23" / "adrena_engine.json"


def own_turn(turn: int, went_first: bool | None) -> int:
    if went_first is None:
        return (turn + 1) // 2
    return (turn + 1) // 2 if went_first else turn // 2


def walk(replay: dict, seat: int, went_first: bool | None) -> dict[str, Any]:
    steps = replay.get("steps") or []
    munki_bench: int | None = None
    munki_energy: int | None = None
    adrena_turns: list[int] = []
    froslass_turns: list[int] = []
    own_turns: set[int] = set()
    max_own = 0

    for index, step in enumerate(steps[:-1]):
        for actor in (0, 1):
            if actor >= len(step) or actor >= len(steps[index + 1]):
                continue
            record = step[actor] or {}
            if record.get("status") != "ACTIVE":
                continue
            obs = record.get("observation") or {}
            select = obs.get("select") or {}
            options = list(select.get("option") or [])
            current = obs.get("current") or {}
            players = current.get("players") or []
            if len(players) < 2 or not options:
                continue
            turn = int(current.get("turn", -1))
            ot = own_turn(turn, went_first)
            if actor == seat:
                max_own = max(max_own, ot)
                if int(select.get("context", -1)) == mf.MAIN_CONTEXT:
                    own_turns.add(ot)
                # availability, read off the board rather than off a choice
                for card in mf._cards(players[actor], "bench") + mf._cards(players[actor], "active"):
                    if int(card.get("id", -1)) != mf.MUNKIDORI_ID:
                        continue
                    if munki_bench is None:
                        munki_bench = ot
                    if munki_energy is None and mf._dark_energy_count(card) >= 1:
                        munki_energy = ot
            action = (steps[index + 1][actor] or {}).get("action")
            picked = [int(v) for v in action
                      if isinstance(v, int) and 0 <= int(v) < len(options)] \
                if isinstance(action, list) else []
            if actor != seat:
                continue
            for choice in picked:
                option = options[choice]
                try:
                    kind = mf.action_type(current, option, select)
                except Exception:  # noqa: BLE001
                    continue
                card = mf.candidate_card(current, option, select) or {}
                cid = int(card.get("id", -1))
                if kind == "ability" and cid == mf.MUNKIDORI_ID:
                    adrena_turns.append(ot)
                elif kind not in {"attack", "ability", "end", "retreat"} and cid == mf.FROSLASS_ID:
                    froslass_turns.append(ot)
    return {
        "munki_bench_own_turn": munki_bench,
        "munki_energy_own_turn": munki_energy,
        "adrena_turns": adrena_turns,
        "first_adrena_own_turn": min(adrena_turns) if adrena_turns else None,
        "froslass_turns": froslass_turns,
        "own_main_turns": len(own_turns),
        "max_own_turn": max_own,
    }


def blk(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"games": 0}
    w = sum(1 for r in rows if r["won"])
    lo, hi = wilson(w, n)
    return {"games": n, "wins": w, "win_rate": round(w / n, 3),
            "wilson95": [round(lo, 3), round(hi, 3)]}


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def welch(a: list[float], b: list[float]) -> dict:
    a = [v for v in a if v is not None]
    b = [v for v in b if v is not None]
    if len(a) < 2 or len(b) < 2:
        return {}
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = (va / len(a) + vb / len(b)) ** 0.5
    t = (ma - mb) / se if se else float("nan")
    return {"win_mean": round(ma, 3), "loss_mean": round(mb, 3),
            "diff": round(ma - mb, 3),
            "t": round(t, 2) if not math.isnan(t) else None,
            "n_win": len(a), "n_loss": len(b)}


def main() -> int:
    index: dict[str, tuple[Path, int]] = {}
    for run_dir in sorted(RUNS.iterdir()):
        manifest = run_dir / "manifest.csv"
        if not run_dir.is_dir() or not manifest.exists():
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            if row.get("detected_submission_agent_index") in {"0", "1"}:
                index[row["episode_id"]] = (run_dir, int(row["detected_submission_agent_index"]))

    rows = []
    for meta in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not meta["version"].startswith("v22"):
            continue
        entry = index.get(meta["episode_id"])
        if entry is None:
            continue
        run_dir, seat = entry
        path = run_dir / "episodes" / meta["episode_id"] / "replay" / f"episode_{meta['episode_id']}.json"
        if not path.exists():
            continue
        went_first = {"True": True, "False": False}.get(meta["went_first"])
        data = walk(json.loads(path.read_text(encoding="utf-8")), seat, went_first)
        data.update({
            "episode_id": meta["episode_id"],
            "won": meta["won"] == "True",
            "went_first": went_first,
            "family": meta["opponent_family"],
            "opponent_rating": float(meta["opponent_rating"]) if meta["opponent_rating"] else None,
            "adrena_count": len(data["adrena_turns"]),
            "froslass_count": len(data["froslass_turns"]),
        })
        turns_after = max(0, data["max_own_turn"] - (data["munki_energy_own_turn"] or data["max_own_turn"]))
        data["adrena_per_turn_after_ready"] = (
            round(data["adrena_count"] / turns_after, 3) if turns_after > 0 else None)
        rows.append(data)

    wins = [r for r in rows if r["won"]]
    losses = [r for r in rows if not r["won"]]
    mirror = [r for r in rows if r["family"] == "Grimmsnarl (mirror)"]

    report: dict[str, Any] = {
        "games": len(rows),
        "availability": {
            "munki_bench_own_turn": welch([r["munki_bench_own_turn"] for r in wins],
                                          [r["munki_bench_own_turn"] for r in losses]),
            "munki_energy_own_turn": welch([r["munki_energy_own_turn"] for r in wins],
                                           [r["munki_energy_own_turn"] for r in losses]),
            "first_adrena_own_turn": welch([r["first_adrena_own_turn"] for r in wins],
                                           [r["first_adrena_own_turn"] for r in losses]),
            "adrena_count": welch([r["adrena_count"] for r in wins],
                                  [r["adrena_count"] for r in losses]),
            "adrena_per_turn_after_ready": welch(
                [r["adrena_per_turn_after_ready"] for r in wins],
                [r["adrena_per_turn_after_ready"] for r in losses]),
            "max_own_turn": welch([r["max_own_turn"] for r in wins],
                                  [r["max_own_turn"] for r in losses]),
        },
        "never_had_munkidori": {
            "bench": sum(1 for r in rows if r["munki_bench_own_turn"] is None),
            "energised": sum(1 for r in rows if r["munki_energy_own_turn"] is None),
            "block_no_energy": blk([r for r in rows if r["munki_energy_own_turn"] is None]),
            "block_energised": blk([r for r in rows if r["munki_energy_own_turn"] is not None]),
        },
        "by_energy_turn": {},
        "adrena_rate_terciles": {},
        "mirror": {
            "availability": {
                "munki_energy_own_turn": welch(
                    [r["munki_energy_own_turn"] for r in mirror if r["won"]],
                    [r["munki_energy_own_turn"] for r in mirror if not r["won"]]),
                "adrena_per_turn_after_ready": welch(
                    [r["adrena_per_turn_after_ready"] for r in mirror if r["won"]],
                    [r["adrena_per_turn_after_ready"] for r in mirror if not r["won"]]),
                "adrena_count": welch([r["adrena_count"] for r in mirror if r["won"]],
                                      [r["adrena_count"] for r in mirror if not r["won"]]),
            },
            "froslass": {
                "zero": blk([r for r in mirror if r["froslass_count"] == 0]),
                "one_plus": blk([r for r in mirror if r["froslass_count"] >= 1]),
                "counts_win": Counter(r["froslass_count"] for r in mirror if r["won"]),
                "counts_loss": Counter(r["froslass_count"] for r in mirror if not r["won"]),
            },
        },
        "froslass_all": {
            "zero": blk([r for r in rows if r["froslass_count"] == 0]),
            "one_plus": blk([r for r in rows if r["froslass_count"] >= 1]),
        },
    }
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        v = r["munki_energy_own_turn"]
        key = "never" if v is None else (f"own_turn_{int(v)}" if v <= 4 else "own_turn_5+")
        buckets[key].append(r)
    report["by_energy_turn"] = {k: blk(v) for k, v in sorted(buckets.items())}

    vals = sorted(r["adrena_per_turn_after_ready"] for r in rows
                  if r["adrena_per_turn_after_ready"] is not None)
    if len(vals) >= 9:
        lo, hi = vals[len(vals) // 3], vals[2 * len(vals) // 3]
        tb: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            v = r["adrena_per_turn_after_ready"]
            if v is None:
                continue
            tb["low" if v < lo else ("high" if v >= hi else "mid")].append(r)
        report["adrena_rate_terciles"] = {"cuts": [lo, hi],
                                          **{k: blk(v) for k, v in tb.items()}}

    report["per_game"] = [
        {k: r[k] for k in ("episode_id", "won", "family", "munki_bench_own_turn",
                           "munki_energy_own_turn", "first_adrena_own_turn",
                           "adrena_count", "adrena_per_turn_after_ready",
                           "froslass_count", "max_own_turn")}
        for r in rows]
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_game"},
                     ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
