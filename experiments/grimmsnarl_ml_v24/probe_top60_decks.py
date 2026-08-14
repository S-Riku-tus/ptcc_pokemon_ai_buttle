"""Which of today's top-60 submissions play our exact 60, and how do they rate?

Our v24 sits at 911-928.  Before asking "why is the policy bad" it has to be
established what this deck is *worth* right now: if the best pilot of the same
60 cards is at 1130 the gap is play, and if they are at 950 the gap is the
deck.  For each top-60 leaderboard submission this pulls one recent episode,
reads the 60-card deck action out of step 1, and hashes it with the same
``deck_hash`` the rest of the repo uses.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fetch_submission_logs import list_submission_episodes, read_json_url  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"
SNAPSHOT = ROOT / "data/kaggle_top100/20260814_215710_JST"
OUT = ROOT / "experiments/grimmsnarl_ml_v24/top60_decks.json"
REPLAY_CACHE = ROOT / "data/kaggle_top100_current/replays/probe_20260814"


def list_with_backoff(submission_id: int, attempts: int = 6):
    """EpisodeService rate-limits hard at ~40 calls; back off instead of dying."""
    delay = 5.0
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return list_submission_episodes(submission_id)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if "429" not in str(exc):
                raise
            time.sleep(delay)
            delay *= 2
    raise last  # type: ignore[misc]


def probe(submission_id: int) -> dict[str, Any]:
    result: dict[str, Any] = {"submission_id": submission_id}
    try:
        episodes = list_with_backoff(submission_id)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"list: {type(exc).__name__}: {exc}"
        return result
    result["episodes_available"] = len(episodes)
    completed = [e for e in episodes if e.state.upper() == "COMPLETED"]
    if not completed:
        result["error"] = "no completed episodes"
        return result

    episode = completed[0]
    result["probe_episode"] = episode.episode_id
    cache = REPLAY_CACHE / f"episode_{episode.episode_id}.json"
    try:
        if cache.exists():
            replay = json.loads(cache.read_text(encoding="utf-8"))
        else:
            replay = read_json_url(
                f"https://www.kaggleusercontent.com/episodes/{episode.episode_id}.json")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(replay, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"replay: {type(exc).__name__}: {exc}"
        return result

    seat = 0 if episode.agent_0_submission_id == str(submission_id) else 1
    steps = replay.get("steps") or []
    if len(steps) < 2:
        result["error"] = "replay too short"
        return result
    action = (steps[1][seat] or {}).get("action")
    if not (isinstance(action, list) and len(action) == 60):
        result["error"] = "no 60-card deck action at step 1"
        return result
    cards = [int(v) for v in action]
    result["deck_hash"] = deck_hash(cards)
    result["same_deck"] = result["deck_hash"] == OUR_DECK_HASH
    return result


def main() -> int:
    # Team names are arbitrary Unicode and this console is cp932.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = list(csv.DictReader(
        (SNAPSHOT / "leaderboard_top60.csv").open(encoding="utf-8-sig")))
    out: list[dict[str, Any]] = []
    for row in rows:
        entry = {
            "rank": int(row["rank"]),
            "team_id": row["team_id"],
            "team_name": row["team_name"],
            "score": float(row["leaderboard_score"]),
        }
        entry.update(probe(int(row["leaderboard_submission_id"])))
        out.append(entry)
        print(
            f"{entry['rank']:>3} {entry['score']:>7.1f} "
            f"{entry.get('deck_hash', entry.get('error', '?')):<20} "
            f"{'SAME DECK' if entry.get('same_deck') else ''}  "
            f"{entry['team_name'][:34]}"
        )
        time.sleep(0.3)

    same = [e for e in out if e.get("same_deck")]
    payload = {
        "our_deck_hash": OUR_DECK_HASH,
        "probed": len(out),
        "same_deck_count": len(same),
        "same_deck": same,
        "deck_hash_counts": {
            h: sum(1 for e in out if e.get("deck_hash") == h)
            for h in sorted({e.get("deck_hash") for e in out if e.get("deck_hash")})
        },
        "rows": out,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSame deck in top 60: {len(same)}")
    for entry in same:
        print(f"  rank {entry['rank']:>3}  {entry['score']:>7.1f}  "
              f"team {entry['team_id']}  sub {entry['submission_id']}  "
              f"{entry['team_name']}")
    print(f"\nReport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
