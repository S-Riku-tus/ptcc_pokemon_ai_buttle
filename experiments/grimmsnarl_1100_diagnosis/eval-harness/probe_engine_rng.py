"""Probe: is the vendored cabt engine's shuffle deterministic, and where is its RNG?

Static analysis of vendor/cg/libcg.so says:
  ApiBattleStart(int*) mallocs a 0x7028-byte ApiData, memsets it, seeds an
  inline std::mt19937 living at ApiData+0x238 (index at +0x15b8) and stores a
  std::random_device draw at ApiData+0x228, a flag byte at +0x233.
  ShuffleDeck(State&,int,bool) and SelectCoinSingle(State&,int) both branch on
  that flag byte: 0 -> use the inline mt19937, non-0 -> construct a fresh
  std::random_device("default") and shuffle with that.

This script checks the claim empirically from Python: it starts N battles with
identical decks and reports how many distinct opening states appear, and it
reads ApiData+0x228 / +0x233 / +0x238 straight out of the returned battle_ptr.
"""
from __future__ import annotations

import ctypes
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "vendor"))

from cg.game import battle_start, battle_finish  # noqa: E402
from cg.sim import Battle  # noqa: E402

OFF_SEED = 0x228
OFF_FLAG = 0x233
OFF_MT = 0x238
OFF_MT_IDX = 0x15B8


def deck(path: Path) -> list[int]:
    return [int(x) for x in path.read_text(encoding="utf-8-sig").split()]


def read_u32(addr: int) -> int:
    return ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint32))[0]


def read_u8(addr: int) -> int:
    return ctypes.cast(addr, ctypes.POINTER(ctypes.c_ubyte))[0]


def read_u64(addr: int) -> int:
    return ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint64))[0]


def opening_signature(obs: dict) -> str:
    """A fingerprint of everything the shuffle decided at battle start."""
    cur = obs["current"]
    parts = [str(cur.get("firstPlayer")), str(cur.get("turn"))]
    for key in ("your", "opponent", "player", "players"):
        if key in cur:
            parts.append(json.dumps(cur[key], sort_keys=True))
    parts.append(json.dumps(obs.get("select"), sort_keys=True))
    return "|".join(parts)


def main() -> int:
    d0 = deck(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v21" / "deck.csv")
    d1 = deck(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8" / "deck.csv")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    sigs: Counter[str] = Counter()
    seeds: list[int] = []
    flags: list[int] = []
    mt0: list[int] = []
    first_obs_keys = None
    for _ in range(n):
        obs, sd = battle_start(d0, d1)
        ptr = int(Battle.battle_ptr)
        seeds.append(read_u32(ptr + OFF_SEED))
        flags.append(read_u8(ptr + OFF_FLAG))
        mt0.append(read_u64(ptr + OFF_MT))
        if first_obs_keys is None:
            first_obs_keys = sorted(obs["current"].keys())
        sigs[opening_signature(obs)] += 1
        battle_finish()

    print(json.dumps({
        "battles": n,
        "distinct_opening_signatures": len(sigs),
        "most_common_count": sigs.most_common(1)[0][1],
        "obs_current_keys": first_obs_keys,
        "ApiData+0x228_seed_distinct": len(set(seeds)),
        "ApiData+0x228_seed_sample": seeds[:5],
        "ApiData+0x233_flag_values": sorted(set(flags)),
        "ApiData+0x238_mt_state0_distinct": len(set(mt0)),
        "ApiData+0x238_mt_state0_sample": mt0[:5],
        "mt_state0_equals_seed": [s == m for s, m in zip(seeds[:5], mt0[:5])],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
