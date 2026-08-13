"""Play the same matchup N times with a fully deterministic policy and see how
many distinct games come out. If the engine were seeded, all N would be equal.

Policy: always take option indices [0..minCount) -- no Python RNG at all, so the
only source of variation is the engine.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "vendor"))

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from cg.sim import Battle  # noqa: E402


def deck(path: Path) -> list[int]:
    return [int(x) for x in path.read_text(encoding="utf-8-sig").split()]


def first_policy(obs: dict) -> list[int]:
    sel = obs["select"]
    return list(range(min(sel["minCount"], len(sel["option"]))))


def play(d0: list[int], d1: list[int], max_steps: int = 4000) -> dict:
    obs, sd = battle_start(d0, d1)
    ptr = int(Battle.battle_ptr)
    seed = ctypes.cast(ptr + 0x228, ctypes.POINTER(ctypes.c_uint32))[0]
    h = hashlib.sha1()
    steps = 0
    try:
        for _ in range(max_steps):
            cur = obs["current"]
            if cur["result"] >= 0:
                break
            h.update(obs["search_begin_input"].encode())
            action = first_policy(obs)
            steps += 1
            try:
                obs = battle_select(action)
            except Exception:
                return {"result": "illegal", "steps": steps,
                        "hash": h.hexdigest()[:16], "seed": seed}
        return {"result": obs["current"]["result"], "steps": steps,
                "hash": h.hexdigest()[:16], "seed": seed,
                "first": obs["current"]["firstPlayer"]}
    finally:
        battle_finish()


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    d0 = deck(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v21" / "deck.csv")
    d1 = deck(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8" / "deck.csv")
    rows = [play(d0, d1) for _ in range(n)]
    hashes = Counter(r["hash"] for r in rows)
    results = Counter(str(r["result"]) for r in rows)
    print(json.dumps({
        "games": n,
        "distinct_game_hashes": len(hashes),
        "most_common_hash_count": hashes.most_common(1)[0][1],
        "results": dict(results),
        "distinct_step_counts": len({r["steps"] for r in rows}),
        "step_count_range": [min(r["steps"] for r in rows),
                             max(r["steps"] for r in rows)],
        "distinct_seeds": len({r["seed"] for r in rows}),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
