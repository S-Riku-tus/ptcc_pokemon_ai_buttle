"""Full scan of every replay on disk in the field corpora + our own ladder runs.

Attribution is by Kaggle TeamName from the replay header (present in 100% of
replays), because the index CSVs in data/kaggle_top50_meta were overwritten by
a later 1-episode-per-submission fetch and no longer cover the 622 replays that
sit in that corpus' replays/ directory.

Output: games.jsonl  (one row per (episode, seat) that we can attribute)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ml.core.replay_io import deck_hash, extract_fast_header_from_bytes  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import family, archetype  # noqa: E402

OUT = Path(__file__).resolve().parent
FIRST_PLAYER_RE = re.compile(rb'"firstPlayer"\s*:\s*([01])\b')
PREFIX = 3_000_000

CORPORA = {
    "top50_meta": ROOT / "data" / "kaggle_top50_meta" / "replays",
    "grimm_top50": ROOT / "data" / "kaggle_grimmsnarl_top50" / "replays",
    "top100_current": ROOT / "data" / "kaggle_top100_current" / "replays",
    "top40_alakazam": ROOT / "data" / "kaggle_top40_alakazam" / "replays",
}


def main() -> int:
    seen: set[int] = set()
    rows = []
    for corpus, root in CORPORA.items():
        if not root.exists():
            continue
        files = sorted(root.glob("episode_*.json"))
        print(f"{corpus}: {len(files)} replay files", file=sys.stderr)
        for n, path in enumerate(files):
            eid = int(path.stem.split("_")[1])
            if eid in seen:
                continue
            seen.add(eid)
            raw = path.read_bytes()[:PREFIX]
            h = extract_fast_header_from_bytes(raw)
            decks = h["decks"]
            if len(decks) < 2 or not decks[0] or not decks[1]:
                continue
            m = FIRST_PLAYER_RE.search(raw)
            first = int(m.group(1)) if m else -1
            hashes = [deck_hash(decks[0]), deck_hash(decks[1])]
            rows.append({
                "episode_id": eid, "corpus": corpus,
                "team": h["team_names"], "rewards": h["rewards"],
                "hash": hashes,
                "family": [family(decks[0]), family(decks[1])],
                "arch": [archetype(decks[0]), archetype(decks[1])],
                "first": first,
                "size": [len(decks[0]), len(decks[1])],
            })
            if n % 500 == 0:
                print(f"  {corpus} {n}", file=sys.stderr)

    with (OUT / "games.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
