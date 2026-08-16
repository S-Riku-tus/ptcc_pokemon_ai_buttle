"""Does the within-turn search actually run on a stored Kaggle observation?"""

from __future__ import annotations

import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "experiments" / "grimmsnarl_vfinal"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from turnsearch import SearchUnavailable, TurnSearch  # noqa: E402

DECK = [
    int(line) for line in
    (ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v22" / "deck.csv")
    .read_text(encoding="utf-8").split()
    if line.strip()
]

episode = (
    ROOT / "data" / "runs" / "grimmsnarl" / "20260816_grimmsnarl_sub55542305"
    / "episodes" / "93507293" / "replay" / "episode_93507293.json"
)
replay = json.loads(episode.read_text(encoding="utf-8"))
steps = replay["steps"]

# Which seat is ours?
seat = None
for candidate in (0, 1):
    action = (steps[1][candidate] or {}).get("action")
    if isinstance(action, list) and len(action) == 60 and sorted(action) == sorted(DECK):
        seat = candidate
print(f"our seat: {seat}")

searcher = TurnSearch(DECK, max_nodes=400, max_seconds=25.0, branch_cap=5)
tried = 0
seen_turns: set[int] = set()
for index, step in enumerate(steps):
    if seat >= len(step):
        continue
    entry = step[seat] or {}
    observation = entry.get("observation") or {}
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    if not observation.get("search_begin_input"):
        continue
    if int(select.get("context", -1)) != 0:
        continue
    turn = int(current.get("turn", -1))
    if turn < 3 or turn in seen_turns:
        continue
    seen_turns.add(turn)
    tried += 1
    started = time.monotonic()
    try:
        lines = searcher.search(observation, rng_shuffle=random.Random(7).shuffle)
    except SearchUnavailable as error:
        print(f"  step {index:4d} turn {turn:3d}  UNAVAILABLE  {error}")
        continue
    except Exception as error:  # noqa: BLE001
        print(f"  step {index:4d} turn {turn:3d}  ERROR  {type(error).__name__}: {error}")
        continue
    elapsed = time.monotonic() - started
    best = max(lines, key=lambda line: line["value"]) if lines else None
    print(
        f"  step {index:4d} turn {turn:3d}  {elapsed:5.2f}s  "
        f"nodes={searcher.stats['nodes']:4d} lines={len(lines):4d} "
        f"trunc={int(searcher.stats['truncated'])}  "
        f"best prizes={best['prizes'] if best else '-'} "
        f"damage={best['damage'] if best else '-'} "
        f"depth={best['depth'] if best else '-'}"
    )
    if tried >= 6:
        break
print(f"\nturn starts attempted: {tried}")
