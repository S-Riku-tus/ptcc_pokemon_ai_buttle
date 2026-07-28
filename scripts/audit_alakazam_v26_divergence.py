"""Per-decision divergence audit between two Alakazam agents on a ladder run.

Teacher-forced: both agents see the recorded observation. For every decision
where they disagree it records the episode, turn, board state and the rule
context (board-count floor / bench threat) that explains the change.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_loader import load_dir_agent  # noqa: E402

CARDS = {
    c["cardId"]: c["name"]
    for c in json.loads((ROOT / "vendor/cg/cards.json").read_text(encoding="utf-8"))
}
TYPES = {0: "NUM", 3: "CARD", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY",
         12: "RETREAT", 13: "ATTACK", 14: "END"}
AREA_KEYS = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench"}


def _label(obs, option):
    current = obs.get("current") or {}
    players = current.get("players") or []
    seat = int(option.get("playerIndex", current.get("yourIndex", 0)))
    area = int(option.get("area", -1))
    index = option.get("index")
    key = AREA_KEYS.get(area)
    if key is None and int(option.get("type", -1)) in (7, 8, 9):
        key = "hand"
    cid = None
    if key and 0 <= seat < len(players) and isinstance(index, int):
        cards = players[seat].get(key) or []
        if 0 <= index < len(cards) and isinstance(cards[index], dict):
            cid = cards[index].get("id")
    tag = TYPES.get(int(option.get("type", -1)), str(option.get("type")))
    return f"{tag}:{CARDS.get(cid, cid)}" if cid else tag


def _seats(run_dir: Path, submission_id: int) -> dict[int, int]:
    out: dict[int, int] = {}
    with (run_dir / "episodes.csv").open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            for seat in (0, 1):
                if int(row[f"agent_{seat}_submission_id"]) == submission_id:
                    out[int(row["episode_id"])] = seat
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--submission-id", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    base_agent, _, _ = load_dir_agent(args.baseline.resolve())
    cand_agent, _, cand_module = load_dir_agent(args.candidate.resolve())

    seats = _seats(args.run_dir, args.submission_id)
    changes = []
    decisions = 0
    for replay_path in sorted((args.run_dir / "episodes").glob("*/replay/*.json")):
        eid = int(replay_path.parents[1].name)
        seat = seats.get(eid)
        if seat is None:
            continue
        base_agent({"select": None})
        cand_agent({"select": None})
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        steps = replay.get("steps") or []
        won = bool((steps[-1][seat].get("reward") or 0) > 0)
        for i, step in enumerate(steps[:-1]):
            record = step[seat] or {}
            obs = record.get("observation") or {}
            if record.get("status") != "ACTIVE" or not obs.get("select"):
                continue
            recorded = (steps[i + 1][seat] or {}).get("action")
            if not isinstance(recorded, list) or len(recorded) == 60:
                continue
            try:
                base = list(base_agent(obs))
                cand = list(cand_agent(obs))
            except Exception:
                continue
            decisions += 1
            if base == cand:
                continue
            options = (obs["select"].get("option") or [])
            state = obs.get("current") or {}
            us = state["players"][seat]
            bodies = sum(
                1 for zone in ("active", "bench")
                for c in (us.get(zone) or []) if isinstance(c, dict)
            )
            changes.append({
                "episode_id": eid,
                "won": won,
                "turn": state.get("turn"),
                "bodies": bodies,
                "context": obs["select"].get("context"),
                "baseline": _label(obs, options[base[0]]) if base and 0 <= base[0] < len(options) else str(base),
                "candidate": _label(obs, options[cand[0]]) if cand and 0 <= cand[0] < len(options) else str(cand),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"decisions": decisions, "changes": changes,
                    "diag": (getattr(cand_module, "diag_snapshot", lambda: None)() or {})},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")

    print(f"decisions {decisions}  changed {len(changes)} "
          f"({len(changes)/decisions:.4%})")
    print("changed games:",
          len({c['episode_id'] for c in changes}),
          "| in losses:",
          len({c['episode_id'] for c in changes if not c['won']}))
    print("by board size:", Counter(
        ("<=1 body" if c["bodies"] <= 1 else "2 bodies" if c["bodies"] == 2 else "3+ bodies")
        for c in changes).most_common())
    print("by transition:", Counter(
        f"{c['baseline']} -> {c['candidate']}" for c in changes).most_common(20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
