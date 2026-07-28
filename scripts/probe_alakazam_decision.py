#!/usr/bin/env python3
"""Replay one recorded observation through an agent and dump its option scores.

Usage:
  probe_alakazam_decision.py <replay.json> <agent_dir> --seat 0 --turn 9 [--all]

Prints, for every MAIN decision on the chosen turn, the option list with the
policy's own play/ability/attack scores so a "why did it not pick X" question is
answered from the live scorer rather than by reading the ranking code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_loader import load_dir_agent  # noqa: E402

CARDS = {
    c["cardId"]: c
    for c in json.loads((ROOT / "vendor/cg/cards.json").read_text(encoding="utf-8"))
}
TYPES = {7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY", 12: "RETREAT",
         13: "ATTACK", 14: "END"}


def _name(cid):
    c = CARDS.get(cid)
    return c["name"] if c else str(cid)


def _label(obs, option):
    current = obs.get("current") or {}
    players = current.get("players") or []
    seat = int(option.get("playerIndex", current.get("yourIndex", 0)))
    area = int(option.get("area", -1))
    index = option.get("index")
    key = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench"}.get(area)
    if key is None and int(option.get("type", -1)) in (7, 8, 9):
        key = "hand"
    cid = None
    if key and 0 <= seat < len(players) and isinstance(index, int):
        cards = players[seat].get(key) or []
        if 0 <= index < len(cards) and isinstance(cards[index], dict):
            cid = cards[index].get("id")
    tag = TYPES.get(int(option.get("type", -1)), str(option.get("type")))
    return f"{tag}:{_name(cid)}" if cid else tag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("replay", type=Path)
    ap.add_argument("agent_dir", type=Path)
    ap.add_argument("--seat", type=int, required=True)
    ap.add_argument("--turn", type=int)
    ap.add_argument("--context", type=int, default=0)
    args = ap.parse_args()

    agent, _, module = load_dir_agent(args.agent_dir.resolve())
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    steps = replay["steps"]
    agent({"select": None})

    for i, step in enumerate(steps[:-1]):
        record = step[args.seat] if args.seat < len(step) else {}
        obs = (record or {}).get("observation") or {}
        if record.get("status") != "ACTIVE" or not obs.get("select"):
            continue
        state = obs.get("current") or {}
        recorded = (steps[i + 1][args.seat] or {}).get("action")
        if not isinstance(recorded, list) or len(recorded) == 60:
            continue
        try:
            predicted = list(agent(obs))
        except Exception as exc:  # keep policy state advancing
            print(f"step {i}: policy raised {exc!r}")
            continue
        select = obs["select"]
        if select.get("context") != args.context:
            continue
        if args.turn is not None and state.get("turn") != args.turn:
            continue
        options = select.get("option") or []
        print(f"--- step {i} turn {state.get('turn')} ctx {select.get('context')}")
        rec = _label(obs, options[recorded[0]]) if recorded and 0 <= recorded[0] < len(options) else recorded
        pre = _label(obs, options[predicted[0]]) if predicted and 0 <= predicted[0] < len(options) else predicted
        print(f"    recorded={rec}   policy_now={pre}")
        scored = getattr(module, "last_option_scores", None)
        if callable(scored):
            for lab, sc in zip((_label(obs, o) for o in options), scored()):
                print(f"      {lab:<34} {sc}")
        else:
            print("      options:", [_label(obs, o) for o in options])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
