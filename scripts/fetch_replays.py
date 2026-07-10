"""Fetch Kaggle simulation Episode replay JSON files.

Episode-ID download uses Kaggle's public episode CDN.
Submission-ID discovery uses the authenticated Kaggle CLI.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

CDN = "https://www.kaggleusercontent.com/episodes/{}.json"
LIST_EPISODES_URL = (
    "https://www.kaggle.com/api/i/"
    "competitions.EpisodeService/ListEpisodes"
)
HTTP_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 PTCG-research",
}


def discover_submission_episodes(submission_id: int) -> list[int]:
    payload = json.dumps({"submissionId": submission_id}).encode("utf-8")
    request = urllib.request.Request(
        LIST_EPISODES_URL,
        data=payload,
        headers=HTTP_HEADERS,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Kaggle API failed with HTTP {exc.code}:\n{message}"
        ) from exc

    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(
            "Kaggle API response did not contain an episodes list:\n"
            + json.dumps(data, ensure_ascii=False)[:1000]
        )

    ids: list[int] = []
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        value = str(episode.get("id", "")).strip()
        if value.isdigit():
            ids.append(int(value))
    return ids


def download_episode(
    episode_id: int,
    output_dir: Path,
    overwrite: bool,
) -> Path:
    output = output_dir / f"episode_{episode_id}.json"
    if output.exists() and not overwrite:
        return output

    request = urllib.request.Request(
        CDN.format(episode_id),
        headers={"User-Agent": "Mozilla/5.0 PTCG-research"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()

    json.loads(data)
    output.write_bytes(data)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=int)
    parser.add_argument("--episode", nargs="*", type=int, default=[])
    parser.add_argument("--episode-file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/replays/manual"),
    )
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ids = list(args.episode)

    if args.submission is not None:
        discovered = discover_submission_episodes(args.submission)
        print(
            f"Submission {args.submission}: "
            f"{len(discovered)} episodes discovered"
        )
        ids.extend(discovered)

    if args.episode_file:
        ids.extend(
            int(value)
            for value in args.episode_file.read_text().split()
            if value.strip()
        )

    ids = list(dict.fromkeys(ids))
    if not ids:
        parser.error("Specify --submission, --episode, or --episode-file")

    args.output.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0
    for index, episode_id in enumerate(ids, start=1):
        try:
            path = download_episode(
                episode_id,
                args.output,
                args.overwrite,
            )
            obj = json.loads(path.read_text(encoding="utf-8"))
            print(
                f"[{index}/{len(ids)}] {episode_id}: "
                f"{len(obj.get('steps', []))} steps -> {path}"
            )
            succeeded += 1
        except Exception as exc:
            print(
                f"[{index}/{len(ids)}] {episode_id}: "
                f"FAILED {type(exc).__name__}: {exc}"
            )
            failed += 1

        if index < len(ids) and args.sleep > 0:
            time.sleep(args.sleep)

    print(f"done: {succeeded} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
