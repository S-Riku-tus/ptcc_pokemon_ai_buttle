"""Download all Kaggle simulation logs for one submission.

This script:

1. Lists every Episode associated with a Kaggle submission ID.
2. Downloads the replay JSON for each Episode.
3. Extracts replay observation logs for agent indexes 0 and 1.
4. Writes a manifest.csv summarizing successes and failures.
5. Can be rerun safely; already-downloaded files are skipped by default.
6. Optionally creates a ZIP archive for sharing.

Kaggle's public CLI no longer exposes the old simulation Episode commands.
This script uses Kaggle's current web EpisodeService for Episode discovery
and the public episode CDN for replay JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

LIST_EPISODES_URL = (
    "https://www.kaggle.com/api/i/"
    "competitions.EpisodeService/ListEpisodes"
)
REPLAY_URL = "https://www.kaggleusercontent.com/episodes/{episode_id}.json"
HTTP_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 PTCG-research",
}
IGNORED_SNAPSHOT_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
IGNORED_SNAPSHOT_SUFFIXES = {".pyc", ".pyo"}


@dataclass
class EpisodeRecord:
    episode_id: int
    create_time: str = ""
    end_time: str = ""
    state: str = ""
    episode_type: str = ""
    agent_0_submission_id: str = ""
    agent_1_submission_id: str = ""
    # Ratings as they stood when the match was paired. Without these, the only
    # way to bucket opponents by strength is to join the *current* public
    # leaderboard, which for the 59-game Grimmsnarl v2 run matched 12 of 59
    # opponents and could not say what any of them was rated at the time. The
    # EpisodeService returns them per agent; blank when it does not.
    agent_0_initial_score: str = ""
    agent_1_initial_score: str = ""
    agent_0_updated_score: str = ""
    agent_1_updated_score: str = ""


@dataclass
class DownloadRecord:
    submission_id: int
    episode_id: int
    episode_state: str
    detected_submission_agent_index: str
    replay_status: str
    agent_0_log_status: str
    agent_1_log_status: str
    error: str


def post_json(url: str, body: dict[str, object]) -> dict[str, object]:
    """POST JSON to Kaggle's current web API and return decoded JSON."""
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers=HTTP_HEADERS,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Kaggle API failed with HTTP {exc.code} for {url}:\n"
            f"{message}"
        ) from exc


def read_json_url(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": HTTP_HEADERS["User-Agent"],
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Kaggle download failed with HTTP {exc.code} for {url}:\n"
            f"{message}"
        ) from exc


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "run"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def snapshot_ignore(
    directory: str,
    names: list[str],
) -> set[str]:
    del directory
    ignored: set[str] = set()
    for name in names:
        path = Path(name)
        if name in IGNORED_SNAPSHOT_NAMES:
            ignored.add(name)
        elif path.suffix.lower() in IGNORED_SNAPSHOT_SUFFIXES:
            ignored.add(name)
    return ignored


def hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""

    files = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and not any(part in IGNORED_SNAPSHOT_NAMES for part in item.parts)
        and item.suffix.lower() not in IGNORED_SNAPSHOT_SUFFIXES
    ]
    for file_path in sorted(files, key=lambda item: item.relative_to(path).as_posix()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def write_run_metadata(
    output_root: Path,
    *,
    submission_id: int,
    run_name: str,
    deck_name: str,
    deck_dir: Path | None,
    notes: str,
) -> None:
    deck_hash = hash_directory(deck_dir) if deck_dir is not None else ""
    deck_snapshot_dir = output_root / "deck_snapshot"

    if deck_dir is not None:
        if not deck_dir.exists():
            raise FileNotFoundError(f"Deck directory does not exist: {deck_dir}")
        if deck_snapshot_dir.exists():
            shutil.rmtree(deck_snapshot_dir)
        shutil.copytree(
            deck_dir,
            deck_snapshot_dir,
            ignore=snapshot_ignore,
        )

    metadata = {
        "submission_id": submission_id,
        "run_name": run_name,
        "deck_name": deck_name,
        "deck_dir": str(deck_dir.resolve()) if deck_dir is not None else "",
        "deck_hash_sha256": deck_hash,
        "deck_snapshot_dir": str(deck_snapshot_dir.resolve())
        if deck_dir is not None
        else "",
        "fetched_at_local": datetime.now().astimezone().isoformat(),
        "git_commit": get_git_commit(),
        "output_dir": str(output_root),
        "notes": notes,
        "log_source": "Kaggle EpisodeService + replay.observation.logs",
    }
    (output_root / "run_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_agent_submission_ids(episode: dict[str, object]) -> dict[int, str]:
    return {
        index: str(agent.get("submissionId"))
        for index, agent in _agents_by_index(episode).items()
        if agent.get("submissionId") is not None
    }


def _agents_by_index(episode: dict[str, object]) -> dict[int, dict]:
    agents = episode.get("agents")
    if not isinstance(agents, list):
        return {}
    out: dict[int, dict] = {}
    for position, agent in enumerate(agents[:2]):
        if not isinstance(agent, dict):
            continue
        index = agent.get("index")
        out[index if isinstance(index, int) else position] = agent
    return out


def get_agent_scores(episode: dict[str, object], field: str) -> dict[int, str]:
    """Per-seat rating field, as a string so a missing value stays blank.

    Kaggle has used both ``initialScore``/``updatedScore`` and nested score
    objects here, so anything non-scalar is read through its ``value``.
    """
    out: dict[int, str] = {}
    for index, agent in _agents_by_index(episode).items():
        value = agent.get(field)
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, (int, float)):
            out[index] = f"{float(value):.4f}"
        elif isinstance(value, str) and value.strip():
            out[index] = value.strip()
    return out


def list_submission_episodes(submission_id: int) -> list[EpisodeRecord]:
    """Return all Episodes currently associated with a submission."""
    data = post_json(LIST_EPISODES_URL, {"submissionId": submission_id})
    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(
            "Kaggle API response did not contain an episodes list:\n"
            + json.dumps(data, ensure_ascii=False)[:1000]
        )

    records: list[EpisodeRecord] = []
    for episode in episodes:
        if not isinstance(episode, dict):
            continue

        raw_id = str(episode.get("id", "")).strip()
        if not raw_id.isdigit():
            continue

        submission_ids = get_agent_submission_ids(episode)
        initial = get_agent_scores(episode, "initialScore")
        updated = get_agent_scores(episode, "updatedScore")
        records.append(
            EpisodeRecord(
                episode_id=int(raw_id),
                create_time=str(episode.get("createTime") or ""),
                end_time=str(episode.get("endTime") or ""),
                state=str(episode.get("state") or ""),
                episode_type=str(episode.get("type") or ""),
                agent_0_submission_id=submission_ids.get(0, ""),
                agent_1_submission_id=submission_ids.get(1, ""),
                agent_0_initial_score=initial.get(0, ""),
                agent_1_initial_score=initial.get(1, ""),
                agent_0_updated_score=updated.get(0, ""),
                agent_1_updated_score=updated.get(1, ""),
            )
        )

    # Preserve the CLI ordering while removing accidental duplicates.
    unique: dict[int, EpisodeRecord] = {}
    for record in records:
        unique.setdefault(record.episode_id, record)

    return list(unique.values())


def directory_has_json(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.json"))


def download_replay(
    episode_id: int,
    output_dir: Path,
    *,
    overwrite: bool,
) -> tuple[str, Path | None]:
    """Download one Episode replay from Kaggle's public episode CDN."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / f"episode_{episode_id}.json"
    if output.exists() and not overwrite:
        return "skipped_existing", output

    if overwrite:
        output.unlink(missing_ok=True)

    data = read_json_url(REPLAY_URL.format(episode_id=episode_id))
    output.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )

    return "downloaded", output


def extract_agent_observation_logs(
    replay_path: Path,
    agent_index: int,
) -> list[dict[str, object]]:
    data = json.loads(replay_path.read_text(encoding="utf-8"))
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        return []

    entries: list[dict[str, object]] = []
    for step_index, step in enumerate(steps):
        if not isinstance(step, list) or agent_index >= len(step):
            continue
        agent_state = step[agent_index]
        if not isinstance(agent_state, dict):
            continue
        observation = agent_state.get("observation")
        if not isinstance(observation, dict):
            continue
        logs = observation.get("logs")
        if isinstance(logs, list) and logs:
            entries.append({"step": step_index, "logs": logs})

    return entries


def download_agent_log(
    episode_id: int,
    agent_index: int,
    output_dir: Path,
    *,
    replay_path: Path | None,
    overwrite: bool,
) -> str:
    """Extract agent observation logs from the replay JSON.

    Kaggle's current public CLI no longer exposes the old
    `competitions logs` command. The public replay contains per-agent
    observation logs, so this stores those in the same per-agent folders.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / f"agent_{agent_index}_observation_logs.json"
    if output.exists() and not overwrite:
        return "skipped_existing"

    if replay_path is None or not replay_path.exists():
        raise FileNotFoundError("Replay JSON is required to extract logs.")

    entries = extract_agent_observation_logs(replay_path, agent_index)
    payload = {
        "episode_id": episode_id,
        "agent_index": agent_index,
        "source": "replay.observation.logs",
        "entries": entries,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return "extracted_from_replay" if entries else "extracted_empty"


def scalar_matches_submission(value: object, submission_id: int) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    return str(value).strip() == str(submission_id)


def object_contains_submission_id(
    value: object,
    submission_id: int,
    *,
    parent_key: str = "",
) -> bool:
    """Recursively find a submission ID only under submission-like keys."""
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            if "submission" in key_text:
                if scalar_matches_submission(child, submission_id):
                    return True
                if isinstance(child, (dict, list)) and object_contains_submission_id(
                    child,
                    submission_id,
                    parent_key=key_text,
                ):
                    return True
            elif isinstance(child, (dict, list)):
                if object_contains_submission_id(
                    child,
                    submission_id,
                    parent_key=key_text,
                ):
                    return True
        return False

    if isinstance(value, list):
        return any(
            object_contains_submission_id(
                child,
                submission_id,
                parent_key=parent_key,
            )
            for child in value
        )

    return (
        "submission" in parent_key
        and scalar_matches_submission(value, submission_id)
    )


def detect_submission_agent_index(
    replay_path: Path | None,
    submission_id: int,
) -> int | None:
    """Best-effort detection of whether the submission was agent 0 or 1.

    Kaggle replay schemas can evolve. This intentionally checks several
    common top-level containers and then falls back to a bounded recursive
    search. Failure is harmless because both agent logs are downloaded.
    """
    if replay_path is None or not replay_path.exists():
        return None

    try:
        data = json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None

    candidate_lists: list[list[object]] = []

    if isinstance(data, dict):
        for key in ("agents", "participants", "players"):
            value = data.get(key)
            if isinstance(value, list):
                candidate_lists.append(value)

        info = data.get("info")
        if isinstance(info, dict):
            for key in ("agents", "participants", "players"):
                value = info.get(key)
                if isinstance(value, list):
                    candidate_lists.append(value)

        configuration = data.get("configuration")
        if isinstance(configuration, dict):
            for key in ("agents", "participants", "players"):
                value = configuration.get(key)
                if isinstance(value, list):
                    candidate_lists.append(value)

    for items in candidate_lists:
        for index, item in enumerate(items[:2]):
            if object_contains_submission_id(item, submission_id):
                return index

    return None


def write_episode_list(
    path: Path,
    episodes: Iterable[EpisodeRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_id",
        "create_time",
        "end_time",
        "state",
        "episode_type",
        "agent_0_submission_id",
        "agent_1_submission_id",
        "agent_0_initial_score",
        "agent_1_initial_score",
        "agent_0_updated_score",
        "agent_1_updated_score",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            writer.writerow(asdict(episode))


def write_manifest(path: Path, records: Iterable[DownloadRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "submission_id",
        "episode_id",
        "episode_state",
        "detected_submission_agent_index",
        "replay_status",
        "agent_0_log_status",
        "agent_1_log_status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def create_zip(source_dir: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.suffix.lower() == ".zip":
        base_name = zip_path.with_suffix("")
    else:
        base_name = zip_path

    produced = Path(
        shutil.make_archive(
            str(base_name),
            "zip",
            root_dir=source_dir.parent,
            base_dir=source_dir.name,
        )
    )
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download every Episode replay and both agents' logs for a "
            "Kaggle simulation submission."
        )
    )
    parser.add_argument(
        "--submission",
        type=int,
        required=True,
        help="Kaggle submission ID.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output directory. Default: "
            "data/submissions/submission_<ID>"
        ),
    )
    parser.add_argument(
        "--run-name",
        help=(
            "Experiment/run name. If --output is omitted, logs are saved "
            "under data/runs/<timestamp>_<run-name>_sub<ID>."
        ),
    )
    parser.add_argument(
        "--deck-name",
        default="",
        help="Human-readable deck name stored in run_meta.json.",
    )
    parser.add_argument(
        "--deck-dir",
        type=Path,
        help=(
            "Directory containing the deck/agent files to snapshot into "
            "the run output."
        ),
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional run notes stored in run_meta.json.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to wait between Episodes. Default: 1.0",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Process only the first N Episodes; 0 means all.",
    )
    parser.add_argument(
        "--after-episode-id",
        type=int,
        default=0,
        help=(
            "Process only Episodes whose ID is greater than this value. "
            "Useful for incremental corpus refreshes; 0 disables the filter."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files that already exist.",
    )
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help=(
            "Download replay JSON only and skip both large per-agent "
            "observation logs."
        ),
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a ZIP archive after downloading.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        help="Optional path for the ZIP archive.",
    )
    args = parser.parse_args()

    if args.output is not None:
        output_root = args.output
    elif args.run_name:
        output_root = (
            Path("data")
            / "runs"
            / f"{now_stamp()}_{slugify(args.run_name)}_sub{args.submission}"
        )
    else:
        output_root = (
            Path("data")
            / "submissions"
            / f"submission_{args.submission}"
        )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.run_name or args.deck_name or args.deck_dir or args.notes:
        write_run_metadata(
            output_root,
            submission_id=args.submission,
            run_name=args.run_name or output_root.name,
            deck_name=args.deck_name,
            deck_dir=args.deck_dir.resolve() if args.deck_dir is not None else None,
            notes=args.notes,
        )

    print(f"Submission ID: {args.submission}")
    print(f"Output directory: {output_root}")
    print("Discovering Episodes...")

    episodes = list_submission_episodes(args.submission)
    discovered = len(episodes)
    if args.after_episode_id > 0:
        episodes = [
            episode for episode in episodes
            if episode.episode_id > args.after_episode_id
        ]
        print(
            f"Incremental filter: {len(episodes)}/{discovered} Episodes have "
            f"ID > {args.after_episode_id}"
        )
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    if not episodes:
        print(
            "No Episodes were returned. The submission may not have played "
            "yet, or the Kaggle credentials may not have access."
        )
        return

    print(f"Episodes found: {len(episodes)}")
    write_episode_list(output_root / "episodes.csv", episodes)

    manifest: list[DownloadRecord] = []

    for number, episode in enumerate(episodes, start=1):
        episode_dir = output_root / "episodes" / str(episode.episode_id)
        replay_dir = episode_dir / "replay"
        log_0_dir = episode_dir / "agent_0"
        log_1_dir = episode_dir / "agent_1"

        print(
            f"\n[{number}/{len(episodes)}] "
            f"Episode {episode.episode_id} "
            f"(state={episode.state or 'unknown'})"
        )

        replay_status = "not_attempted"
        log_0_status = "not_attempted"
        log_1_status = "not_attempted"
        errors: list[str] = []
        replay_path: Path | None = None

        try:
            replay_status, replay_path = download_replay(
                episode.episode_id,
                replay_dir,
                overwrite=args.overwrite,
            )
            print(f"  Replay: {replay_status}")
        except Exception as exc:
            replay_status = "failed"
            errors.append(
                f"replay: {type(exc).__name__}: {exc}"
            )
            print(f"  Replay: FAILED: {exc}")

        detected_index = None
        if episode.agent_0_submission_id == str(args.submission):
            detected_index = 0
        elif episode.agent_1_submission_id == str(args.submission):
            detected_index = 1
        else:
            detected_index = detect_submission_agent_index(
                replay_path,
                args.submission,
            )
        detected_text = (
            str(detected_index)
            if detected_index is not None
            else ""
        )
        if detected_index is not None:
            print(f"  Submitted agent appears to be agent {detected_index}")
        else:
            print(
                "  Submitted agent index could not be detected; "
                "both logs will still be saved."
            )

        if args.replay_only:
            log_0_status = "skipped_replay_only"
            log_1_status = "skipped_replay_only"
            print("  Agent logs: skipped (--replay-only)")
        else:
            for agent_index, target_dir in (
                (0, log_0_dir),
                (1, log_1_dir),
            ):
                try:
                    status = download_agent_log(
                        episode.episode_id,
                        agent_index,
                        target_dir,
                        replay_path=replay_path,
                        overwrite=args.overwrite,
                    )
                    print(f"  Agent {agent_index} log: {status}")
                    if agent_index == 0:
                        log_0_status = status
                    else:
                        log_1_status = status
                except Exception as exc:
                    status = "failed"
                    errors.append(
                        f"agent_{agent_index}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(f"  Agent {agent_index} log: FAILED: {exc}")
                    if agent_index == 0:
                        log_0_status = status
                    else:
                        log_1_status = status

        manifest.append(
            DownloadRecord(
                submission_id=args.submission,
                episode_id=episode.episode_id,
                episode_state=episode.state,
                detected_submission_agent_index=detected_text,
                replay_status=replay_status,
                agent_0_log_status=log_0_status,
                agent_1_log_status=log_1_status,
                error=" | ".join(errors),
            )
        )
        write_manifest(output_root / "manifest.csv", manifest)

        if number < len(episodes) and args.sleep > 0:
            time.sleep(args.sleep)

    success_replays = sum(
        record.replay_status in {"downloaded", "skipped_existing"}
        for record in manifest
    )
    success_logs = sum(
        status in {
            "downloaded",
            "extracted_from_replay",
            "extracted_empty",
            "skipped_existing",
        }
        for record in manifest
        for status in (
            record.agent_0_log_status,
            record.agent_1_log_status,
        )
    )
    failures = sum(bool(record.error) for record in manifest)

    print("\n=== Summary ===")
    print(f"Episodes processed: {len(manifest)}")
    print(f"Replays available: {success_replays}/{len(manifest)}")
    print(f"Agent logs available: {success_logs}/{len(manifest) * 2}")
    print(f"Episodes with at least one failure: {failures}")
    print(f"Manifest: {output_root / 'manifest.csv'}")

    if args.zip or args.zip_path:
        zip_path = (
            args.zip_path
            if args.zip_path is not None
            else output_root.parent / f"{output_root.name}.zip"
        )
        produced = create_zip(output_root, zip_path.resolve())
        print(f"ZIP: {produced}")


if __name__ == "__main__":
    main()
