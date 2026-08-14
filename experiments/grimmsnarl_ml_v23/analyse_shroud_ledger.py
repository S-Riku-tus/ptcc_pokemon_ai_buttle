"""Was the Froslass ledger already negative when we evolved it?

Freezing Shroud puts a counter on every Ability Pokemon at each checkup, on
both sides.  In the mirror both players run Grimmsnarl ex and Munkidori, so the
ledger is close to symmetric and the extra bench slot - Snorunt and Froslass
can neither attack nor retreat for free - is paid by us alone.  v22 already
computes ``shroud_net`` (opponent targets minus our own) and
``shroud_net_favourable``, so if the mirror evolutions happen at net <= 0 the
model is overriding a feature it already has, and a gate is enough.  If they
happen at net > 0, the ledger is not the mechanism and the finding needs a
different explanation.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
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
OUT = ROOT / "experiments" / "grimmsnarl_ml_v23" / "shroud_ledger.json"


def own_turn(turn: int, went_first: bool | None) -> int:
    if went_first is None:
        return (turn + 1) // 2
    return (turn + 1) // 2 if went_first else turn // 2


def ledger(current: dict, seat: int) -> tuple[int, int, int] | None:
    players = current.get("players") or []
    if len(players) < 2:
        return None
    stadium_id = mf._int((current.get("stadium") or {}).get("id", -1)) \
        if isinstance(current.get("stadium"), dict) else -1
    try:
        mine = mf.shroud_side(players[seat], stadium_id, is_own_side=True)
        theirs = mf.shroud_side(players[1 - seat], stadium_id, is_own_side=False)
    except Exception:  # noqa: BLE001
        return None
    return len(theirs), len(mine), len(theirs) - len(mine)


def walk(replay: dict, seat: int, went_first: bool | None) -> list[dict[str, Any]]:
    steps = replay.get("steps") or []
    events = []
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        obs = record.get("observation") or {}
        select = obs.get("select") or {}
        options = list(select.get("option") or [])
        current = obs.get("current") or {}
        if not (current.get("players") and options):
            continue
        action = (steps[index + 1][seat] or {}).get("action")
        picked = {int(v) for v in action
                  if isinstance(v, int) and 0 <= int(v) < len(options)} \
            if isinstance(action, list) else set()
        for position in picked:
            option = options[position]
            try:
                kind = mf.action_type(current, option, select)
            except Exception:  # noqa: BLE001
                continue
            card = mf.candidate_card(current, option, select) or {}
            if int(card.get("id", -1)) != mf.FROSLASS_ID:
                continue
            if kind in {"attack", "ability", "end", "retreat"}:
                continue
            led = ledger(current, seat)
            if led is None:
                continue
            events.append({
                "own_turn": own_turn(int(current.get("turn", -1)), went_first),
                "opp_targets": led[0], "our_targets": led[1], "net": led[2],
            })
    return events


def blk(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"games": 0}
    w = sum(1 for r in rows if r["won"])
    lo, hi = wilson(w, n)
    return {"games": n, "wins": w, "win_rate": round(w / n, 3),
            "wilson95": [round(lo, 3), round(hi, 3)]}


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
        events = walk(json.loads(path.read_text(encoding="utf-8")), seat, went_first)
        rows.append({
            "episode_id": meta["episode_id"], "won": meta["won"] == "True",
            "family": meta["opponent_family"], "events": events,
            "n_events": len(events),
            "worst_net": min((e["net"] for e in events), default=None),
            "any_nonpositive": any(e["net"] <= 0 for e in events),
        })

    mirror = [r for r in rows if r["family"] == "Grimmsnarl (mirror)"]
    mirror_events = [e for r in mirror for e in r["events"]]
    all_events = [e for r in rows for e in r["events"]]

    report = {
        "games": len(rows),
        "mirror": {
            "games": len(mirror),
            "evolutions": len(mirror_events),
            "net_histogram": dict(sorted(Counter(e["net"] for e in mirror_events).items())),
            "net_nonpositive": sum(1 for e in mirror_events if e["net"] <= 0),
            "net_positive": sum(1 for e in mirror_events if e["net"] > 0),
            "mean_our_targets": round(sum(e["our_targets"] for e in mirror_events)
                                      / max(1, len(mirror_events)), 2),
            "mean_opp_targets": round(sum(e["opp_targets"] for e in mirror_events)
                                      / max(1, len(mirror_events)), 2),
            "games_with_nonpositive": blk([r for r in mirror if r["any_nonpositive"]]),
            "games_all_positive": blk([r for r in mirror
                                       if r["n_events"] > 0 and not r["any_nonpositive"]]),
            "games_none": blk([r for r in mirror if r["n_events"] == 0]),
        },
        "all_matchups": {
            "evolutions": len(all_events),
            "net_histogram": dict(sorted(Counter(e["net"] for e in all_events).items())),
            "net_nonpositive": sum(1 for e in all_events if e["net"] <= 0),
        },
        "gate_preview_mirror": {
            "note": "how many mirror evolutions a 'require shroud_net > 0' gate "
                    "would have refused, and in how many games",
            "refused_evolutions": sum(1 for e in mirror_events if e["net"] <= 0),
            "games_touched": sum(1 for r in mirror if r["any_nonpositive"]),
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
