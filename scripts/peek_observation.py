"""Dump one observation from a replay so field names can be verified.

Analyses in this repo read nested fields by name, and a wrong name fails as a
zero rather than an error.  Print the real shape before trusting a count.

Usage:
  python scripts/peek_observation.py \
      data/submissions/submission_55550682_dragapult_v2/episodes/93604864/replay/episode_93604864.json \
      --seat 0 --step 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--seat", type=int, default=0)
    parser.add_argument("--step", type=int, default=40)
    parser.add_argument("--path", default="current.players")
    args = parser.parse_args()

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    steps = replay.get("steps") or []
    payload = steps[args.step][args.seat]
    node = payload.get("observation") or {}
    for part in args.path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node.get(part)
        if node is None:
            print(f"path stops at {part}")
            return 1
    print(json.dumps(node, indent=1, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
