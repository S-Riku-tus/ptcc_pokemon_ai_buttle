"""Freeze the AlphaTCG replay run as a Grimmsnarl training manifest.

The generic corpus builder consumes the collector index schema.  Ladder-run
downloads use a smaller manifest, so this adapter verifies every selected
relation and writes the immutable schema used by the v25 training pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DECK_HASH = "9714ab5c3996f6cc"
TEAM_ID = 16381823
SUBMISSION_ID = 55350342


def _initial_decks(replay: dict[str, Any]) -> list[list[int]]:
    """Return the two 60-card setup actions from a downloaded replay."""
    decks: list[list[int]] = []
    for seat in (0, 1):
        found: list[int] = []
        for step in replay.get("steps") or []:
            if seat >= len(step):
                continue
            action = (step[seat] or {}).get("action")
            if (
                isinstance(action, list)
                and len(action) == 60
                and all(isinstance(value, int) for value in action)
            ):
                found = [int(value) for value in action]
                break
        decks.append(found)
    return decks


def _deck_hash(card_ids: list[int]) -> str:
    counts = Counter(card_ids)
    payload = ";".join(
        f"{card_id}:{counts[card_id]}" for card_id in sorted(counts)
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=(
            ROOT
            / "data/runs/grimmsnarl/20260814_peer_alphatcg_sub55350342"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "experiments/grimmsnarl_ml_v25/alphatcg_selection.csv"
        ),
    )
    args = parser.parse_args()

    source = args.run_dir / "manifest.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    output: list[dict[str, Any]] = []
    for row in rows:
        if int(row["submission_id"]) != SUBMISSION_ID:
            raise ValueError(f"unexpected submission: {row['submission_id']}")
        if row.get("episode_state") != "COMPLETED":
            continue
        if row.get("replay_status") != "downloaded":
            continue
        episode = int(row["episode_id"])
        seat = int(row["detected_submission_agent_index"])
        replay_path = Path("episodes") / str(episode) / "replay" / (
            f"episode_{episode}.json"
        )
        absolute = args.run_dir / replay_path
        replay = json.loads(absolute.read_text(encoding="utf-8"))
        decks = _initial_decks(replay)
        if seat not in (0, 1) or len(decks[seat]) != 60:
            raise ValueError(f"missing 60-card deck: episode={episode} seat={seat}")
        actual_hash = _deck_hash(decks[seat])
        if actual_hash != DECK_HASH:
            raise ValueError(
                f"deck drift: episode={episode} expected={DECK_HASH} "
                f"actual={actual_hash}"
            )
        output.append(
            {
                "download_status": "success",
                "deck_hash": actual_hash,
                "team_id": TEAM_ID,
                "submission_id": SUBMISSION_ID,
                "episode_id": episode,
                "seat_index": seat,
                "replay_path": str(replay_path),
            }
        )

    output.sort(key=lambda item: (item["episode_id"], item["seat_index"]))
    if len(output) != 120:
        raise ValueError(f"expected 120 complete relations, found {len(output)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output[0])
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "relations": len(output),
                "team_id": TEAM_ID,
                "submission_id": SUBMISSION_ID,
                "deck_hash": DECK_HASH,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
