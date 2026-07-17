"""Collect replay data for the top-100 Kaggle public submission list.

This script reuses scripts.fetch_submission_logs for the actual Kaggle
EpisodeService and replay/log extraction steps, then adds:

* automatic input discovery from data/kaggle_top100
* submission-level bookkeeping for every public submission ID
* episode/replay de-duplication across submissions
* JSON validation and resumable skip logic
* consolidated indexes and run summary outputs
* submission-level parallelism with per-episode file locks
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Allow direct execution via `python scripts/...py` as well as `python -m scripts...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fetch_submission_logs import (
    detect_submission_agent_index,
    download_agent_log,
    download_replay,
    list_submission_episodes,
)
from ml.core.replay_io import deck_hash, extract_fast_header_from_file


ROOT = Path("data") / "kaggle_top100"
PRINT_LOCK = threading.Lock()
ALAKAZAM_DECK_CARD_IDS = {741, 742, 743, 245}


class EpisodeLockRegistry:
    """Provide stable per-episode locks for replay/log writes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._locks: dict[int, threading.Lock] = {}

    def get(self, episode_id: int) -> threading.Lock:
        with self._lock:
            lock = self._locks.get(episode_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[episode_id] = lock
            return lock


def thread_print(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def discover_input_csv(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Input CSV not found: {explicit}")
        return explicit

    preferred = ROOT / "latest" / "public_submissions_top100.csv"
    if preferred.exists():
        return preferred

    latest_candidates = sorted((ROOT / "latest").glob("public_submissions_top*.csv"))
    if latest_candidates:
        return latest_candidates[-1]

    candidates = sorted(ROOT.glob("*/public_submissions_top100.csv"))
    if candidates:
        return candidates[-1]

    top_n_candidates = sorted(ROOT.glob("*/public_submissions_top*.csv"))
    if top_n_candidates:
        return top_n_candidates[-1]

    raise FileNotFoundError(
        "Could not find public_submissions_top*.csv under data/kaggle_top100."
    )


def load_submission_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def dedupe_submission_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    deduped: dict[int, dict[str, Any]] = {}
    for row in rows:
        raw_submission_id = (
            row.get("public_submission_id")
            or row.get("representative_submission_id")
            or row.get("submission_id")
            or row.get("leaderboard_submission_id")
            or ""
        ).strip()
        if not raw_submission_id.isdigit():
            continue

        submission_id = int(raw_submission_id)
        normalized = dict(row)
        normalized["submission_id"] = str(submission_id)

        existing = deduped.get(submission_id)
        if existing is None:
            normalized["source_rows"] = [dict(row)]
            deduped[submission_id] = normalized
        else:
            existing.setdefault("source_rows", []).append(dict(row))

    return sorted(
        deduped.values(),
        key=lambda item: (
            int(item.get("rank") or "999999"),
            int(item["submission_id"]),
        ),
    )


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text.isdigit():
        return None
    return int(text)


def parse_iso_datetime(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def sort_episodes_oldest_first(episodes: list[Any]) -> list[Any]:
    def sort_key(episode: Any) -> tuple[datetime, datetime, int]:
        created_at = parse_iso_datetime(getattr(episode, "create_time", "")) or datetime.min.replace(
            tzinfo=timezone.utc
        )
        ended_at = parse_iso_datetime(getattr(episode, "end_time", "")) or created_at
        episode_id = parse_int(getattr(episode, "episode_id", None)) or -1
        return (created_at, ended_at, episode_id)

    return sorted(episodes, key=sort_key)


def extract_replay_episode_id(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None

    for key in ("episodeId", "episode_id"):
        value = parse_int(payload.get(key))
        if value is not None:
            return value

    info = payload.get("info")
    if isinstance(info, dict):
        for key in ("EpisodeId", "episodeId", "episode_id"):
            value = parse_int(info.get(key))
            if value is not None:
                return value

    return None


def validate_json_file(path: Path) -> tuple[bool, Any | None, str]:
    if not path.exists():
        return False, None, "missing"
    if path.stat().st_size == 0:
        return False, None, "empty_file"
    try:
        return True, read_json(path), ""
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def validate_replay_file(path: Path, episode_id: int) -> tuple[bool, Any | None, str]:
    ok, payload, error = validate_json_file(path)
    if not ok:
        return False, None, error
    actual_episode_id = extract_replay_episode_id(payload)
    if actual_episode_id != episode_id:
        return (
            False,
            payload,
            f"episode_id_mismatch expected={episode_id} actual={actual_episode_id}",
        )
    return True, payload, ""


def validate_log_file(
    path: Path,
    episode_id: int,
    agent_index: int,
) -> tuple[bool, str]:
    ok, payload, error = validate_json_file(path)
    if not ok:
        return False, error
    if not isinstance(payload, dict):
        return False, "log_payload_not_object"
    if parse_int(payload.get("episode_id")) != episode_id:
        return False, "log_episode_id_mismatch"
    if parse_int(payload.get("agent_index")) != agent_index:
        return False, "log_agent_index_mismatch"
    if not isinstance(payload.get("entries"), list):
        return False, "log_entries_not_list"
    return True, ""


def retry_call(label: str, attempts: int, delay: float, fn: Callable[[], Any]) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            sleep_seconds = delay * attempt
            thread_print(f"  {label}: retrying after {sleep_seconds:.1f}s ({exc})")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}")


def choose_submission_agent_index(
    episode: Any,
    submission_id: int,
    replay_path: Path,
) -> int | None:
    if getattr(episode, "agent_0_submission_id", "") == str(submission_id):
        return 0
    if getattr(episode, "agent_1_submission_id", "") == str(submission_id):
        return 1
    return detect_submission_agent_index(replay_path, submission_id)


def detect_opponent_name(payload: Any, self_index: int | None) -> str:
    if self_index is None or not isinstance(payload, dict):
        return ""
    info = payload.get("info")
    if not isinstance(info, dict):
        return ""
    team_names = info.get("TeamNames")
    if not isinstance(team_names, list) or len(team_names) < 2:
        return ""
    opponent_index = 1 - self_index
    if opponent_index >= len(team_names):
        return ""
    value = team_names[opponent_index]
    return str(value) if value is not None else ""


def detect_deck_match(
    replay_path: Path,
    seat_index: int | None,
    required_card_ids: set[int],
) -> dict[str, Any]:
    if not required_card_ids:
        return {
            "deck_filter_match": True,
            "deck_filter_reason": "",
            "deck_hash": "",
            "deck_cards": [],
        }
    if seat_index is None:
        return {
            "deck_filter_match": False,
            "deck_filter_reason": "seat_index_unknown",
            "deck_hash": "",
            "deck_cards": [],
        }

    try:
        header = extract_fast_header_from_file(replay_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "deck_filter_match": False,
            "deck_filter_reason": f"deck_header_error: {type(exc).__name__}: {exc}",
            "deck_hash": "",
            "deck_cards": [],
        }

    decks = header.get("decks")
    deck = decks[seat_index] if isinstance(decks, list) and seat_index < len(decks) else []
    if not isinstance(deck, list) or len(deck) != 60:
        return {
            "deck_filter_match": False,
            "deck_filter_reason": "deck_not_found",
            "deck_hash": "",
            "deck_cards": [],
        }

    cards = [int(card_id) for card_id in deck]
    matched_cards = sorted(required_card_ids & set(cards))
    return {
        "deck_filter_match": bool(matched_cards),
        "deck_filter_reason": "" if matched_cards else "required_cards_absent",
        "deck_hash": deck_hash(cards),
        "deck_cards": matched_cards,
    }


def aggregate_download_status(
    replay_status: str,
    log_statuses: list[str],
    error: str,
) -> str:
    if error:
        return "failed"
    successful = {
        "downloaded",
        "skipped_existing",
        "extracted_from_replay",
        "extracted_empty",
    }
    if replay_status not in successful:
        return "failed"
    if any(status not in successful for status in log_statuses):
        return "partial"
    if replay_status == "skipped_existing" and all(
        status == "skipped_existing" for status in log_statuses
    ):
        return "skipped_existing"
    return "success"


def safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass


def ensure_replay_for_episode(
    *,
    episode_id: int,
    replays_dir: Path,
    download_retries: int,
    retry_delay: float,
) -> tuple[str, bool, Any | None, str, Path]:
    canonical_replay_path = replays_dir / f"episode_{episode_id}.json"
    replay_ok, replay_payload, replay_error = validate_replay_file(
        canonical_replay_path,
        episode_id,
    )
    replay_status = "skipped_existing" if replay_ok else "not_attempted"

    if not replay_ok:
        safe_unlink(canonical_replay_path)
        replay_status, replay_path = retry_call(
            f"download replay for episode {episode_id}",
            download_retries,
            retry_delay,
            lambda: download_replay(
                episode_id,
                replays_dir,
                overwrite=False,
            ),
        )
        canonical_replay_path = replay_path or canonical_replay_path
        replay_ok, replay_payload, replay_error = validate_replay_file(
            canonical_replay_path,
            episode_id,
        )
        if not replay_ok:
            replay_status = "failed_validation"

    return replay_status, replay_ok, replay_payload, replay_error, canonical_replay_path


def ensure_log_for_episode(
    *,
    episode_id: int,
    agent_index: int,
    canonical_log_dir: Path,
    replay_path: Path | None,
    replay_ok: bool,
    download_retries: int,
    retry_delay: float,
) -> tuple[str, str]:
    log_path = canonical_log_dir / f"agent_{agent_index}_observation_logs.json"
    log_ok, log_error = validate_log_file(log_path, episode_id, agent_index)
    if log_ok:
        return "skipped_existing", ""

    safe_unlink(log_path)
    log_status = retry_call(
        f"extract agent {agent_index} log for episode {episode_id}",
        download_retries,
        retry_delay,
        lambda idx=agent_index: download_agent_log(
            episode_id,
            idx,
            canonical_log_dir,
            replay_path=replay_path if replay_ok else None,
            overwrite=False,
        ),
    )
    log_ok, log_error = validate_log_file(log_path, episode_id, agent_index)
    if not log_ok:
        return "failed_validation", log_error
    return log_status, ""


def process_submission(
    *,
    index: int,
    total: int,
    submission: dict[str, Any],
    input_csv: Path,
    output_root: Path,
    submissions_dir: Path,
    replays_dir: Path,
    logs_dir: Path,
    submission_team_lookup: dict[int, dict[str, Any]],
    max_episodes_per_submission: int,
    sleep_seconds: float,
    list_retries: int,
    download_retries: int,
    retry_delay: float,
    episode_locks: EpisodeLockRegistry,
    required_deck_card_ids: set[int],
) -> dict[str, Any]:
    submission_id = int(submission["submission_id"])
    submission_dir = ensure_dir(submissions_dir / str(submission_id))
    ensure_dir(submission_dir / "replays")
    ensure_dir(submission_dir / "logs")

    thread_print(
        f"\n[{index}/{total}] Submission {submission_id} "
        f"(rank={submission.get('rank') or '?'}, team={submission.get('team_name') or ''})"
    )

    submission_payload = {
        "submission_id": submission_id,
        "leaderboard_rank": submission.get("rank", ""),
        "team_id": submission.get("team_id", ""),
        "team_name": submission.get("team_name", ""),
        "leaderboard_score": submission.get("leaderboard_score", ""),
        "leaderboard_submission_id": submission.get("leaderboard_submission_id", ""),
        "public_submission_id": submission.get("public_submission_id", ""),
        "public_score": submission.get("public_score", ""),
        "submitted_at_utc": submission.get("submitted_at_utc", ""),
        "submitted_at_jst": submission.get("submitted_at_jst", ""),
        "source_csv": str(input_csv),
        "source_rows": submission.get("source_rows", []),
        "collected_at": utc_now(),
    }
    write_json(submission_dir / "submission.json", submission_payload)

    result: dict[str, Any] = {
        "submission_rows": [],
        "episode_rows": [],
        "replay_index_rows": [],
        "failures_rows": [],
        "unique_episode_ids": set(),
        "counts": {
            "submission_success": 0,
            "submission_no_data": 0,
            "submission_failed": 0,
            "replay_downloaded": 0,
            "replay_skipped_existing": 0,
            "replay_failures": 0,
            "log_downloaded": 0,
            "log_skipped_existing": 0,
            "log_failures": 0,
            "episode_filtered_by_deck": 0,
        },
    }

    try:
        episodes = retry_call(
            f"list episodes for submission {submission_id}",
            list_retries,
            retry_delay,
            lambda: list_submission_episodes(submission_id),
        )
    except Exception as exc:
        error_text = str(exc)
        result["counts"]["submission_failed"] += 1
        result["submission_rows"].append(
            {
                "leaderboard_rank": submission.get("rank", ""),
                "team_id": submission.get("team_id", ""),
                "team_name": submission.get("team_name", ""),
                "submission_id": submission_id,
                "submission_score": submission.get("public_score", ""),
                "leaderboard_submission_id": submission.get("leaderboard_submission_id", ""),
                "submitted_at_utc": submission.get("submitted_at_utc", ""),
                "submitted_at_jst": submission.get("submitted_at_jst", ""),
                "status": "failed",
                "episode_count": 0,
                "error": error_text,
                "downloaded_at": utc_now(),
            }
        )
        result["failures_rows"].append(
            {
                "scope": "submission",
                "submission_id": submission_id,
                "episode_id": "",
                "error": error_text,
                "recorded_at": utc_now(),
            }
        )
        write_json(submission_dir / "episodes.json", {"episodes": [], "error": error_text})
        thread_print(f"  FAILED: {error_text}")
        return result

    episodes = sort_episodes_oldest_first(episodes)
    if max_episodes_per_submission > 0:
        episodes = episodes[:max_episodes_per_submission]

    write_json(
        submission_dir / "episodes.json",
        {
            "submission_id": submission_id,
            "episode_count": len(episodes),
            "episodes": [asdict(episode) for episode in episodes],
        },
    )

    if not episodes:
        result["counts"]["submission_no_data"] += 1
        result["submission_rows"].append(
            {
                "leaderboard_rank": submission.get("rank", ""),
                "team_id": submission.get("team_id", ""),
                "team_name": submission.get("team_name", ""),
                "submission_id": submission_id,
                "submission_score": submission.get("public_score", ""),
                "leaderboard_submission_id": submission.get("leaderboard_submission_id", ""),
                "submitted_at_utc": submission.get("submitted_at_utc", ""),
                "submitted_at_jst": submission.get("submitted_at_jst", ""),
                "status": "no_data",
                "episode_count": 0,
                "error": "",
                "downloaded_at": utc_now(),
            }
        )
        thread_print("  No episodes returned.")
        return result

    per_submission_replay_refs: list[dict[str, Any]] = []
    per_submission_log_refs: list[dict[str, Any]] = []
    matched_episode_count = 0
    submission_deck_matches: bool | None = None
    submission_deck_hash = ""
    submission_matched_deck_cards: list[int] = []

    for episode in episodes:
        episode_id = episode.episode_id
        if required_deck_card_ids and submission_deck_matches is False:
            result["counts"]["episode_filtered_by_deck"] += 1
            continue

        canonical_log_dir = ensure_dir(logs_dir / str(episode_id))
        log_paths = [
            canonical_log_dir / "agent_0_observation_logs.json",
            canonical_log_dir / "agent_1_observation_logs.json",
        ]
        downloaded_at = utc_now()

        errors: list[str] = []
        replay_payload: Any | None = None
        replay_error = ""
        replay_status = "not_attempted"
        replay_ok = False
        canonical_replay_path = replays_dir / f"episode_{episode_id}.json"
        log_statuses: list[str] = []
        detected_index: int | None = None
        deck_filter = {
            "deck_filter_match": not required_deck_card_ids,
            "deck_filter_reason": "",
            "deck_hash": "",
            "deck_cards": [],
        }

        with episode_locks.get(episode_id):
            try:
                (
                    replay_status,
                    replay_ok,
                    replay_payload,
                    replay_error,
                    canonical_replay_path,
                ) = ensure_replay_for_episode(
                    episode_id=episode_id,
                    replays_dir=replays_dir,
                    download_retries=download_retries,
                    retry_delay=retry_delay,
                )
                if replay_status == "downloaded":
                    result["counts"]["replay_downloaded"] += 1
                elif replay_status == "skipped_existing":
                    result["counts"]["replay_skipped_existing"] += 1
                else:
                    result["counts"]["replay_failures"] += 1
            except Exception as exc:
                replay_status = "failed"
                replay_ok = False
                replay_payload = None
                replay_error = str(exc)
                result["counts"]["replay_failures"] += 1

            if replay_error:
                errors.append(f"replay: {replay_error}")

            detected_index = (
                choose_submission_agent_index(episode, submission_id, canonical_replay_path)
                if replay_ok
                else None
            )
            if replay_ok:
                if required_deck_card_ids and submission_deck_matches is True:
                    deck_filter = {
                        "deck_filter_match": True,
                        "deck_filter_reason": "",
                        "deck_hash": submission_deck_hash,
                        "deck_cards": submission_matched_deck_cards,
                    }
                else:
                    deck_filter = detect_deck_match(
                        canonical_replay_path,
                        detected_index,
                        required_deck_card_ids,
                    )
                    if required_deck_card_ids:
                        if deck_filter["deck_filter_match"]:
                            submission_deck_matches = True
                        elif deck_filter.get("deck_hash"):
                            submission_deck_matches = False
                        submission_deck_hash = str(deck_filter.get("deck_hash", ""))
                        submission_matched_deck_cards = list(deck_filter.get("deck_cards", []))

            if required_deck_card_ids and not deck_filter["deck_filter_match"]:
                result["counts"]["episode_filtered_by_deck"] += 1
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                continue

            for agent_index, _log_path in enumerate(log_paths):
                try:
                    log_status, log_error = ensure_log_for_episode(
                        episode_id=episode_id,
                        agent_index=agent_index,
                        canonical_log_dir=canonical_log_dir,
                        replay_path=canonical_replay_path,
                        replay_ok=replay_ok,
                        download_retries=download_retries,
                        retry_delay=retry_delay,
                    )
                    if log_status == "skipped_existing":
                        result["counts"]["log_skipped_existing"] += 1
                    elif log_status in {"extracted_from_replay", "extracted_empty"}:
                        result["counts"]["log_downloaded"] += 1
                    else:
                        result["counts"]["log_failures"] += 1
                except Exception as exc:
                    log_status = "failed"
                    log_error = str(exc)
                    result["counts"]["log_failures"] += 1

                if log_error:
                    errors.append(f"agent_{agent_index}: {log_error}")
                log_statuses.append(log_status)

        matched_episode_count += 1
        result["unique_episode_ids"].add(episode_id)
        opponent_submission_id = (
            episode.agent_1_submission_id
            if detected_index == 0
            else episode.agent_0_submission_id
            if detected_index == 1
            else ""
        )
        opponent_submission_int = parse_int(opponent_submission_id)
        opponent_row = submission_team_lookup.get(opponent_submission_int or -1)
        opponent_team_name = (
            str(opponent_row.get("team_name", ""))
            if opponent_row is not None
            else detect_opponent_name(replay_payload, detected_index)
        )

        error_text = " | ".join(errors)
        download_status = aggregate_download_status(
            replay_status,
            log_statuses,
            error_text,
        )

        per_submission_replay_refs.append(
            {
                "episode_id": episode_id,
                "replay_path": str(canonical_replay_path.relative_to(output_root)),
                "download_status": download_status,
            }
        )
        per_submission_log_refs.append(
            {
                "episode_id": episode_id,
                "agent_0_log_path": str(log_paths[0].relative_to(output_root)),
                "agent_1_log_path": str(log_paths[1].relative_to(output_root)),
                "download_status": download_status,
            }
        )

        row = {
            "leaderboard_rank": submission.get("rank", ""),
            "team_id": submission.get("team_id", ""),
            "team_name": submission.get("team_name", ""),
            "submission_id": submission_id,
            "submission_score": submission.get("public_score", ""),
            "leaderboard_submission_id": submission.get("leaderboard_submission_id", ""),
            "episode_id": episode_id,
            "episode_state": episode.state,
            "episode_type": episode.episode_type,
            "created_at": episode.create_time,
            "ended_at": episode.end_time,
            "seat_index": "" if detected_index is None else detected_index,
            "opponent_submission_id": opponent_submission_id,
            "opponent_team_name": opponent_team_name,
            "agent_0_submission_id": episode.agent_0_submission_id,
            "agent_1_submission_id": episode.agent_1_submission_id,
            "deck_hash": deck_filter.get("deck_hash", ""),
            "matched_deck_card_ids": " ".join(map(str, deck_filter.get("deck_cards", []))),
            "replay_path": str(canonical_replay_path.relative_to(output_root)),
            "log_paths": ";".join(
                str(path.relative_to(output_root)) for path in log_paths
            ),
            "download_status": download_status,
            "error": error_text,
            "downloaded_at": downloaded_at,
        }
        result["episode_rows"].append(row)
        result["replay_index_rows"].append(dict(row))
        if error_text:
            result["failures_rows"].append(
                {
                    "scope": "episode",
                    "submission_id": submission_id,
                    "episode_id": episode_id,
                    "error": error_text,
                    "recorded_at": downloaded_at,
                }
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    write_json(submission_dir / "replays" / "index.json", per_submission_replay_refs)
    write_json(submission_dir / "logs" / "index.json", per_submission_log_refs)

    result["counts"]["submission_success"] += 1
    submission_status = "success"
    if required_deck_card_ids and matched_episode_count == 0:
        submission_status = "no_matching_deck"
    result["submission_rows"].append(
        {
            "leaderboard_rank": submission.get("rank", ""),
            "team_id": submission.get("team_id", ""),
            "team_name": submission.get("team_name", ""),
            "submission_id": submission_id,
            "submission_score": submission.get("public_score", ""),
            "leaderboard_submission_id": submission.get("leaderboard_submission_id", ""),
            "submitted_at_utc": submission.get("submitted_at_utc", ""),
            "submitted_at_jst": submission.get("submitted_at_jst", ""),
            "status": submission_status,
            "episode_count": matched_episode_count,
            "error": "",
            "downloaded_at": utc_now(),
        }
    )
    thread_print(f"  Episodes discovered: {len(episodes)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect submission info, episodes, replays, and extracted logs "
            "for all top-100 public Kaggle submission IDs."
        )
    )
    parser.add_argument("--input", type=Path, help="Optional submission list CSV.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT,
        help="Collection root. Default: data/kaggle_top100",
    )
    parser.add_argument(
        "--max-submissions",
        type=int,
        default=0,
        help="Process only the first N deduped submission IDs. 0 means all.",
    )
    parser.add_argument(
        "--max-episodes-per-submission",
        type=int,
        default=0,
        help="Process only the oldest N episodes per submission by create_time. 0 means all.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to wait between episode downloads per worker.",
    )
    parser.add_argument(
        "--list-retries",
        type=int,
        default=4,
        help="Retry count for listing episodes.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=4,
        help="Retry count for replay/log downloads.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.5,
        help="Base delay in seconds between retries.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(2, (os.cpu_count() or 4))),
        help="Number of submission workers. Default: min(8, max(2, cpu_count)).",
    )
    parser.add_argument(
        "--alakazam-only",
        action="store_true",
        help="Keep only episodes where the target submission's deck contains Alakazam-line cards.",
    )
    parser.add_argument(
        "--require-deck-card-id",
        type=int,
        action="append",
        default=[],
        help="Require at least one of these card IDs in the target submission deck. Repeatable.",
    )
    args = parser.parse_args()

    required_deck_card_ids = set(args.require_deck_card_id)
    if args.alakazam_only:
        required_deck_card_ids.update(ALAKAZAM_DECK_CARD_IDS)

    input_csv = discover_input_csv(args.input)
    source_rows = load_submission_rows(input_csv)
    submissions = dedupe_submission_rows(source_rows)
    if args.max_submissions > 0:
        submissions = submissions[: args.max_submissions]

    output_root = args.output_root.resolve()
    submissions_dir = ensure_dir(output_root / "submissions")
    replays_dir = ensure_dir(output_root / "replays")
    logs_dir = ensure_dir(output_root / "logs")
    indexes_dir = ensure_dir(output_root / "indexes")

    thread_print(f"Input CSV: {input_csv}")
    thread_print(f"Submission IDs to process: {len(submissions)}")
    thread_print(f"Output root: {output_root}")
    thread_print(f"Workers: {max(1, args.workers)}")

    input_submission_ids = {int(item["submission_id"]) for item in submissions}
    submission_team_lookup = {int(item["submission_id"]): item for item in submissions}

    submission_rows_out: list[dict[str, Any]] = []
    episode_rows_out: list[dict[str, Any]] = []
    replay_index_rows: list[dict[str, Any]] = []
    failures_rows: list[dict[str, Any]] = []
    unique_episode_to_submission_ids: defaultdict[int, set[int]] = defaultdict(set)

    counts = {
        "submission_success": 0,
        "submission_no_data": 0,
        "submission_failed": 0,
        "replay_downloaded": 0,
        "replay_skipped_existing": 0,
        "replay_failures": 0,
        "log_downloaded": 0,
        "log_skipped_existing": 0,
        "log_failures": 0,
        "episode_filtered_by_deck": 0,
    }

    episode_locks = EpisodeLockRegistry()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                process_submission,
                index=index,
                total=len(submissions),
                submission=submission,
                input_csv=input_csv,
                output_root=output_root,
                submissions_dir=submissions_dir,
                replays_dir=replays_dir,
                logs_dir=logs_dir,
                submission_team_lookup=submission_team_lookup,
                max_episodes_per_submission=args.max_episodes_per_submission,
                sleep_seconds=args.sleep,
                list_retries=args.list_retries,
                download_retries=args.download_retries,
                retry_delay=args.retry_delay,
                episode_locks=episode_locks,
                required_deck_card_ids=required_deck_card_ids,
            )
            for index, submission in enumerate(submissions, start=1)
        ]

        for future in as_completed(futures):
            result = future.result()
            submission_rows_out.extend(result["submission_rows"])
            episode_rows_out.extend(result["episode_rows"])
            replay_index_rows.extend(result["replay_index_rows"])
            failures_rows.extend(result["failures_rows"])
            for row in result["episode_rows"]:
                unique_episode_to_submission_ids[int(row["episode_id"])].add(
                    int(row["submission_id"])
                )
            for key, value in result["counts"].items():
                counts[key] += value

    missing_submission_ids = sorted(
        input_submission_ids - {int(row["submission_id"]) for row in submission_rows_out}
    )

    for missing_submission_id in missing_submission_ids:
        failures_rows.append(
            {
                "scope": "submission",
                "submission_id": missing_submission_id,
                "episode_id": "",
                "error": "missing_from_final_index",
                "recorded_at": utc_now(),
            }
        )

    submission_rows_out.sort(
        key=lambda row: (int(row.get("leaderboard_rank") or 999999), int(row["submission_id"]))
    )
    episode_rows_out.sort(key=lambda row: (int(row["episode_id"]), int(row["submission_id"])))
    replay_index_rows.sort(
        key=lambda row: (int(row["episode_id"]), int(row["submission_id"]))
    )
    failures_rows.sort(
        key=lambda row: (
            row["scope"],
            int(row["submission_id"]) if str(row["submission_id"]).isdigit() else -1,
            int(row["episode_id"]) if str(row["episode_id"]).isdigit() else -1,
        )
    )

    submission_fieldnames = [
        "leaderboard_rank",
        "team_id",
        "team_name",
        "submission_id",
        "submission_score",
        "leaderboard_submission_id",
        "submitted_at_utc",
        "submitted_at_jst",
        "status",
        "episode_count",
        "error",
        "downloaded_at",
    ]
    episode_fieldnames = [
        "leaderboard_rank",
        "team_id",
        "team_name",
        "submission_id",
        "submission_score",
        "leaderboard_submission_id",
        "episode_id",
        "episode_state",
        "episode_type",
        "created_at",
        "ended_at",
        "seat_index",
        "opponent_submission_id",
        "opponent_team_name",
        "agent_0_submission_id",
        "agent_1_submission_id",
        "deck_hash",
        "matched_deck_card_ids",
        "replay_path",
        "log_paths",
        "download_status",
        "error",
        "downloaded_at",
    ]
    failures_fieldnames = ["scope", "submission_id", "episode_id", "error", "recorded_at"]

    write_csv(indexes_dir / "submissions.csv", submission_rows_out, submission_fieldnames)
    write_csv(indexes_dir / "episodes.csv", episode_rows_out, episode_fieldnames)
    write_csv(indexes_dir / "replay_index.csv", replay_index_rows, episode_fieldnames)
    write_csv(indexes_dir / "failures.csv", failures_rows, failures_fieldnames)

    unique_episode_count = len(unique_episode_to_submission_ids)
    if required_deck_card_ids:
        kept_episode_ids = set(unique_episode_to_submission_ids)
        for path in replays_dir.glob("episode_*.json"):
            episode_id = parse_int(path.stem.split("_")[-1])
            if episode_id is not None and episode_id not in kept_episode_ids:
                safe_unlink(path)
        for path in logs_dir.glob("*"):
            if path.is_dir():
                episode_id = parse_int(path.name)
                if episode_id is not None and episode_id not in kept_episode_ids:
                    for child in path.glob("*.json"):
                        safe_unlink(child)
                    try:
                        path.rmdir()
                    except OSError:
                        pass

    unique_replay_count = len(
        [
            path
            for path in replays_dir.glob("episode_*.json")
            if validate_replay_file(path, parse_int(path.stem.split("_")[-1]) or -1)[0]
        ]
    )
    unique_log_count = len([path for path in logs_dir.glob("*/*.json") if path.is_file()])
    duplicate_episode_relations = max(0, len(episode_rows_out) - unique_episode_count)

    run_summary = {
        "input_csv": str(input_csv),
        "output_root": str(output_root),
        "collected_at": utc_now(),
        "counts": {
            "input_submission_ids": len(input_submission_ids),
            "processed_submission_rows": len(submission_rows_out),
            "submission_success": counts["submission_success"],
            "submission_no_data": counts["submission_no_data"],
            "submission_failed": counts["submission_failed"],
            "submission_missing_from_final_index": len(missing_submission_ids),
            "episode_relations": len(episode_rows_out),
            "unique_episodes": unique_episode_count,
            "replays_present": unique_replay_count,
            "logs_present": unique_log_count,
            "duplicate_episode_relations": duplicate_episode_relations,
            "replay_downloaded": counts["replay_downloaded"],
            "replay_skipped_existing": counts["replay_skipped_existing"],
            "replay_failures": counts["replay_failures"],
            "log_downloaded": counts["log_downloaded"],
            "log_skipped_existing": counts["log_skipped_existing"],
            "log_failures": counts["log_failures"],
            "episode_filtered_by_deck": counts["episode_filtered_by_deck"],
            "failure_rows": len(failures_rows),
        },
        "deck_filter": {
            "required_deck_card_ids": sorted(required_deck_card_ids),
        },
        "missing_submission_ids": missing_submission_ids,
        "indexes": {
            "submissions_csv": str((indexes_dir / "submissions.csv").relative_to(output_root)),
            "episodes_csv": str((indexes_dir / "episodes.csv").relative_to(output_root)),
            "replay_index_csv": str((indexes_dir / "replay_index.csv").relative_to(output_root)),
            "failures_csv": str((indexes_dir / "failures.csv").relative_to(output_root)),
        },
    }
    write_json(output_root / "run_summary.json", run_summary)

    thread_print("\n=== Summary ===")
    thread_print(json.dumps(run_summary["counts"], ensure_ascii=False, indent=2))
    if missing_submission_ids:
        thread_print(f"Missing submission IDs in final index: {missing_submission_ids}")


if __name__ == "__main__":
    main()
