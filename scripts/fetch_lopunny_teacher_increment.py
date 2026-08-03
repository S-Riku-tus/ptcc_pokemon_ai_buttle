"""Fetch a bounded Majkel replay increment concurrently and reproducibly.

This is intentionally narrower than ``fetch_submission_logs.py``: it stores
replays only, filters by a strict lower episode-ID bound, and writes the same
``episodes.csv``/``manifest.csv`` schema expected by the corpus builder.
Every worker owns a distinct episode directory, so resumable concurrent writes
cannot collide.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_submission_logs import (
    DownloadRecord,
    EpisodeRecord,
    HTTP_HEADERS,
    REPLAY_URL,
    list_submission_episodes,
    write_episode_list,
    write_manifest,
)


def _download_raw(episode_id: int, replay_dir: Path) -> str:
    """Save the public CDN response without a redundant JSON round trip."""
    replay_dir.mkdir(parents=True, exist_ok=True)
    output = replay_dir / f"episode_{episode_id}.json"
    if output.exists() and output.stat().st_size > 0:
        return "skipped_existing"
    temporary = output.with_suffix(".json.partial")
    request = urllib.request.Request(
        REPLAY_URL.format(episode_id=episode_id),
        headers={
            "Accept": "application/json",
            "User-Agent": HTTP_HEADERS["User-Agent"],
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    if not payload.strip().startswith(b"{"):
        raise ValueError("CDN response is not a JSON object")
    temporary.write_bytes(payload)
    temporary.replace(output)
    return "downloaded"


def _fetch(
    episode: EpisodeRecord,
    submission_id: int,
    output: Path,
) -> DownloadRecord:
    replay_dir = output / "episodes" / str(episode.episode_id) / "replay"
    status = "not_attempted"
    error = ""
    try:
        status = _download_raw(episode.episode_id, replay_dir)
    except Exception as exc:  # keep a complete manifest for resumability
        status = "failed"
        error = f"replay: {type(exc).__name__}: {exc}"
    seat = ""
    if episode.agent_0_submission_id == str(submission_id):
        seat = "0"
    elif episode.agent_1_submission_id == str(submission_id):
        seat = "1"
    return DownloadRecord(
        submission_id=submission_id,
        episode_id=episode.episode_id,
        episode_state=episode.state,
        detected_submission_agent_index=seat,
        replay_status=status,
        agent_0_log_status="skipped_replay_only",
        agent_1_log_status="skipped_replay_only",
        error=error,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=int, default=55137818)
    parser.add_argument("--after-episode-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-episodes", type=int, default=0,
        help="Keep only the newest N matching episodes; 0 keeps all.",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes = [
        episode for episode in list_submission_episodes(args.submission)
        if episode.episode_id > args.after_episode_id
    ]
    episodes.sort(key=lambda item: item.episode_id, reverse=True)
    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]
    write_episode_list(output / "episodes.csv", episodes)
    print(f"episodes={len(episodes)} workers={args.workers}", flush=True)

    records: list[DownloadRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(_fetch, episode, args.submission, output): episode
            for episode in episodes
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if completed % 20 == 0 or completed == len(futures):
                available = sum(
                    row.replay_status in {"downloaded", "skipped_existing"}
                    for row in records
                )
                print(
                    f"processed={completed}/{len(futures)} available={available}",
                    flush=True,
                )

    by_id = {record.episode_id: record for record in records}
    ordered = [by_id[episode.episode_id] for episode in episodes]
    write_manifest(output / "manifest.csv", ordered)
    failures = [record for record in ordered if record.error]
    print(
        f"complete={len(ordered) - len(failures)}/{len(ordered)} "
        f"manifest={output / 'manifest.csv'}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
