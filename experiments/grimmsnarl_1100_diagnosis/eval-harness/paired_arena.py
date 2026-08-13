"""Seeded, paired, parallel comparator prototype.

For every seed s and every candidate C, play C vs a fixed control on the
*same* engine seed and the same seat. Because seed_patch pins the engine's
mt19937, every candidate sees the identical deal, identical coin flip and
identical prize placement, so the per-seed outcomes are correlated and the
paired difference has far less variance than two independent arenas.

Usage:
  python paired_arena.py --candidates grimmsnarl_ml_v21,grimmsnarl_ml_v20 \
      --control grimmsnarl_ml_v8 --seeds 64 --workers 12 --out report.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _init_paths():
    for p in (str(HERE), str(ROOT / "vendor"), str(ROOT / "scripts"),
              str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve(spec: str) -> Path:
    p = Path(spec)
    if p.is_dir():
        return p.resolve()
    for base in (ROOT / "agents", ROOT / "archive" / "agents"):
        if (base / spec).is_dir():
            return (base / spec).resolve()
        if base.is_dir():
            for group in sorted(base.iterdir()):
                if (group / spec).is_dir():
                    return (group / spec).resolve()
    raise FileNotFoundError(spec)


_STATE: dict = {}


def _worker_init(candidates: list[str], control: str):
    _init_paths()
    import seed_patch
    from agent_loader import load_dir_agent
    from cg.game import battle_start, battle_select, battle_finish

    seed_patch.install()
    agents = {}
    for spec in set(candidates) | {control}:
        agent, _diag, module = load_dir_agent(_resolve(spec))
        deck = list(agent({"select": None}))
        agents[spec] = (agent, deck, module)
    _STATE.update(agents=agents, seed_patch=seed_patch,
                  battle_start=battle_start, battle_select=battle_select,
                  battle_finish=battle_finish)


def _play(job):
    seed, cand, control, cand_seat = job
    st = _STATE
    st["seed_patch"].set_seed(seed)
    a_agent, a_deck, a_mod = st["agents"][cand]
    b_agent, b_deck, b_mod = st["agents"][control]
    for mod in (a_mod, b_mod):
        reset = getattr(mod, "diag_reset", None)
        if reset is not None:
            reset()
    if cand_seat == 0:
        agents, decks = [a_agent, b_agent], [a_deck, b_deck]
    else:
        agents, decks = [b_agent, a_agent], [b_deck, a_deck]
    t0 = time.perf_counter()
    obs, sd = st["battle_start"](decks[0], decks[1])
    if obs is None:
        return {"seed": seed, "candidate": cand, "seat": cand_seat,
                "win": None, "error": "battle_start"}
    moves = 0
    try:
        for _ in range(8000):
            cur = obs["current"]
            if cur["result"] >= 0:
                res = cur["result"]
                win = None if res not in (0, 1) else int(res == cand_seat)
                return {"seed": seed, "candidate": cand, "seat": cand_seat,
                        "win": win, "moves": moves,
                        "sec": time.perf_counter() - t0}
            s = cur["yourIndex"]
            moves += 1
            try:
                action = agents[s](obs)
                obs = st["battle_select"](list(action))
            except Exception as exc:  # noqa: BLE001
                return {"seed": seed, "candidate": cand, "seat": cand_seat,
                        "win": int(s != cand_seat), "moves": moves,
                        "error": f"{type(exc).__name__}",
                        "sec": time.perf_counter() - t0}
        return {"seed": seed, "candidate": cand, "seat": cand_seat,
                "win": None, "moves": moves, "sec": time.perf_counter() - t0}
    finally:
        st["battle_finish"]()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--both-seats", action="store_true",
                    help="play every seed from both seats (2x games)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cands = [c for c in args.candidates.split(",") if c]
    seeds = [1 + i * 7919 for i in range(args.seeds)]
    seats = (0, 1) if args.both_seats else (0,)
    jobs = [(s, c, args.control, seat)
            for s in seeds for c in cands for seat in seats]

    t0 = time.perf_counter()
    with mp.Pool(args.workers, initializer=_worker_init,
                 initargs=(cands, args.control)) as pool:
        rows = pool.map(_play, jobs, chunksize=1)
    wall = time.perf_counter() - t0

    report = {
        "control": args.control, "candidates": cands,
        "seeds": args.seeds, "seats": list(seats),
        "workers": args.workers, "games": len(jobs),
        "wall_seconds": round(wall, 2),
        "sec_per_game_wall": round(wall / len(jobs), 3),
        "cpu_seconds": round(sum(r.get("sec", 0.0) for r in rows), 1),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     indent=2))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
