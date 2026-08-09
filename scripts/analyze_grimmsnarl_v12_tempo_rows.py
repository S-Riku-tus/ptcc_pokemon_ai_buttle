"""Pass 2: attack tempo per own turn, with a per-game cache.

Fixes two defects in pass 1: MAIN attack options are ``type == 13`` (there is no
``attack`` key), and opponent archetypes are named from the card DB so the
Alakazam cut is not silently empty. Rows are cached to JSONL so later questions
do not re-parse 3.6k replays.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = (Path(__file__).resolve().parents[1] / "experiments"
       / "grimmsnarl_ml_v12" / "tempo_rows.jsonl")
CARDS = {
    int(c["cardId"]): c
    for c in json.loads((ROOT / "vendor" / "cg" / "cards.json").read_text("utf-8"))
}
DECK_HASH = "9714ab5c3996f6cc"
GRIMM_EX = next(c for c, v in CARDS.items() if v.get("name") == "Marnie's Grimmsnarl ex")


def deck_hash(ids):
    counts = Counter(int(x) for x in ids)
    return hashlib.sha256(
        ";".join(f"{c}:{counts[c]}" for c in sorted(counts)).encode()
    ).hexdigest()[:16]


def archetype(deck):
    if not deck:
        return "unknown"
    pokemon = Counter(c for c in deck if CARDS.get(c, {}).get("cardType") == 0)
    if not pokemon:
        return "unknown"

    def key(item):
        cid, count = item
        card = CARDS[cid]
        return (bool(card.get("stage2")), bool(card.get("megaEx") or card.get("ex")),
                bool(card.get("stage1")), count, int(card.get("hp", 0)))

    return CARDS.get(max(pokemon.items(), key=key)[0], {}).get("name", "?")


def in_play(p):
    return [c for z in ("active", "bench") for c in (p.get(z) or [])
            if isinstance(c, dict) and isinstance(c.get("id"), int)]


def parse(path: Path, seat: int, tag: str):
    replay = json.loads(path.read_text(encoding="utf-8"))
    rewards = replay.get("rewards") or [None, None]
    if rewards[seat] is None:
        return None
    other = rewards[1 - seat]
    won = bool(rewards[seat] > (other if other is not None else 0))
    steps = replay.get("steps") or []
    decks = [None, None]
    if len(steps) > 1:
        for s in (0, 1):
            a = (steps[1][s] or {}).get("action")
            if isinstance(a, list) and len(a) == 60:
                decks[s] = [int(v) for v in a]
    if not decks[seat] or deck_hash(decks[seat]) != DECK_HASH:
        return None
    went_first = None
    for step in reversed(steps):
        if seat >= len(step):
            continue
        cur = ((step[seat] or {}).get("observation") or {}).get("current")
        if isinstance(cur, dict) and cur.get("players"):
            fp = int(cur.get("firstPlayer", -1))
            went_first = (fp == seat) if fp >= 0 else None
            break

    own_bodies: dict[int, int] = {}
    own_grimm: dict[int, int] = {}
    attack_turns: set[int] = set()
    my_turns: set[int] = set()
    for step in steps:
        if seat >= len(step) or not isinstance(step[seat], dict):
            continue
        entry = step[seat]
        if entry.get("status") != "ACTIVE":
            continue
        obs = entry.get("observation") or {}
        cur = obs.get("current") or {}
        players = cur.get("players") or []
        if len(players) < 2:
            continue
        turn = int(cur.get("turn", -1))
        sel = obs.get("select") or {}
        if int(sel.get("context", -1)) != 0:
            continue
        my_turns.add(turn)
        bodies = in_play(players[seat])
        own_bodies[turn] = len(bodies)
        own_grimm[turn] = int(any(int(c.get("id", -1)) == GRIMM_EX for c in bodies))
        action = entry.get("action")
        options = sel.get("option") or []
        if isinstance(action, list) and len(action) == 1:
            i = action[0]
            if isinstance(i, int) and 0 <= i < len(options):
                o = options[i]
                if isinstance(o, dict) and int(o.get("type", -1)) == 13:
                    attack_turns.add(turn)
    ordinal = {t: k + 1 for k, t in enumerate(sorted(my_turns))}
    return {
        "tag": tag,
        "won": won,
        "went_first": went_first,
        "opponent": archetype(decks[1 - seat]),
        "own_turns": len(my_turns),
        # keyed by *our own* turn ordinal, not the engine's shared turn counter
        "bodies": {str(ordinal[t]): own_bodies[t] for t in my_turns},
        "grimm": {str(ordinal[t]): own_grimm[t] for t in my_turns},
        "attacked": {str(ordinal[t]): int(t in attack_turns) for t in my_turns},
        "first_attack": min((ordinal[t] for t in attack_turns), default=None),
        "first_grimm": min((ordinal[t] for t in my_turns if own_grimm[t]), default=None),
    }


def field_rows():
    base = ROOT / "data" / "kaggle_grimmsnarl_top50"
    for raw in csv.DictReader((base / "indexes" / "replay_index.csv").open(encoding="utf-8-sig")):
        if raw["deck_hash"] != DECK_HASH:
            continue
        if raw["episode_type"] != "EPISODE_TYPE_PUBLIC" or raw["episode_state"] != "COMPLETED":
            continue
        if raw["download_status"] != "success":
            continue
        p = base / raw["replay_path"].replace("\\", "/")
        if p.exists():
            yield p, int(raw["seat_index"]), "field"


def run_rows():
    runs = ROOT / "data" / "runs" / "grimmsnarl"
    for name, sub in (
        ("20260807_grimmsnarl_ml_v8_sub_55317804", "55317804"),
        ("20260807_grimmsnarl_ml_v9_sub_55325029", "55325029"),
        ("20260808_grimmsnarl_ml_v11_a_sub55346539", "55346539"),
        ("20260808_grimmsnarl_ml_v11_b_sub55346548", "55346548"),
        ("20260809_grimmsnarl_ml_v11_sub55353978", "55353978"),
    ):
        d = runs / name
        if not d.exists():
            continue
        tag = name.split("_grimmsnarl_ml_")[1].split("_sub")[0]
        for raw in csv.DictReader((d / "episodes.csv").open(encoding="utf-8-sig")):
            if raw["episode_type"] != "EPISODE_TYPE_PUBLIC" or raw["state"] != "COMPLETED":
                continue
            a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
            if a0 == a1:
                continue
            seat = 0 if a0 == sub else 1
            p = d / "episodes" / raw["episode_id"] / "replay" / f"episode_{raw['episode_id']}.json"
            if p.exists():
                yield p, seat, tag


def main():
    rows = []
    for path, seat, tag in list(field_rows()) + list(run_rows()):
        try:
            row = parse(path, seat, tag)
        except Exception:
            row = None
        if row:
            rows.append(row)
    OUT.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"rows={len(rows)}")
    print("tags:", Counter(r["tag"] for r in rows))
    print("opponent archetypes (top 12):",
          Counter(r["opponent"] for r in rows).most_common(12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
