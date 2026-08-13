"""Does seed_patch actually give common random numbers?

Test 1  same seed, N repeats  -> expect 1 distinct game hash
Test 2  N distinct seeds      -> expect N distinct game hashes
Test 3  flag patched but seed left alone -> expect N distinct (sanity)
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "vendor"))

import seed_patch  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402
from cg.sim import Battle  # noqa: E402


def deck(path: Path) -> list[int]:
    return [int(x) for x in path.read_text(encoding="utf-8-sig").split()]


def play(d0, d1, max_steps=4000):
    obs, _sd = battle_start(d0, d1)
    ptr = int(Battle.battle_ptr)
    seed = ctypes.cast(ptr + 0x228, ctypes.POINTER(ctypes.c_uint32))[0]
    flag = ctypes.cast(ptr + 0x233, ctypes.POINTER(ctypes.c_ubyte))[0]
    h = hashlib.sha1()
    steps = 0
    try:
        for _ in range(max_steps):
            cur = obs["current"]
            if cur["result"] >= 0:
                break
            h.update(obs["search_begin_input"].encode())
            sel = obs["select"]
            steps += 1
            try:
                obs = battle_select(list(range(min(sel["minCount"],
                                                   len(sel["option"])))))
            except Exception:
                break
        return {"hash": h.hexdigest()[:16], "steps": steps,
                "engine_seed": seed, "flag": flag,
                "result": obs["current"]["result"]}
    finally:
        battle_finish()


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    d0 = deck(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v21" / "deck.csv")
    d1 = deck(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8" / "deck.csv")

    out = {}

    # baseline, unpatched
    rows = [play(d0, d1) for _ in range(n)]
    out["unpatched"] = {
        "n": n, "distinct_hashes": len({r["hash"] for r in rows}),
        "flag_values": sorted({r["flag"] for r in rows}),
    }

    seed_patch.install()
    seed_patch.set_seed(12345)
    rows = [play(d0, d1) for _ in range(n)]
    out["patched_same_seed"] = {
        "n": n, "distinct_hashes": len({r["hash"] for r in rows}),
        "flag_values": sorted({r["flag"] for r in rows}),
        "engine_seeds": sorted({r["engine_seed"] for r in rows}),
        "steps": sorted({r["steps"] for r in rows}),
        "results": dict(Counter(r["result"] for r in rows)),
    }

    hashes = []
    for s in range(1, n + 1):
        seed_patch.set_seed(1000 + s * 7919)
        hashes.append(play(d0, d1)["hash"])
    out["patched_distinct_seeds"] = {
        "n": n, "distinct_hashes": len(set(hashes)),
    }

    # replay the same seed list a second time -> must reproduce exactly
    hashes2 = []
    for s in range(1, n + 1):
        seed_patch.set_seed(1000 + s * 7919)
        hashes2.append(play(d0, d1)["hash"])
    out["seed_list_reproducible"] = (hashes == hashes2)

    seed_patch.uninstall()
    rows = [play(d0, d1) for _ in range(5)]
    out["after_uninstall"] = {
        "n": 5, "distinct_hashes": len({r["hash"] for r in rows}),
        "flag_values": sorted({r["flag"] for r in rows}),
    }

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
