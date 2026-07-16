from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .common import episode_id_from_name, read_csv_text
from ml.core.replay_io import (
    extract_fast_header,
    legacy_replay_refs,
    replay_refs as _expanded_replay_refs,
    zip_metadata,
)


@dataclass(frozen=True)
class ReplayRef:
    zip_path: Path
    member: str
    episode_id: int
    target_seat: int | None = None
    created_at: str = ""
    ended_at: str = ""
    path_variant: str = ""


def find_member(names: list[str], suffix: str) -> str | None:
    return next((name for name in names if name.endswith(suffix)), None)


def bundle_metadata(path: Path) -> dict[str, Any]:
    return zip_metadata(path)


def episode_ids(path: Path) -> list[int]:
    """Discover episode IDs from observations or either full-replay layout."""
    ids: set[int] = set()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if (
                name.endswith("_observation_logs.json")
                or re.search(r"(?:^|/)(?:replay|replays)/episode_\d+\.json$", name)
            ):
                episode_id = episode_id_from_name(name)
                if episode_id is not None:
                    ids.add(episode_id)
    return sorted(ids)


def replay_refs(path: Path) -> list[ReplayRef]:
    """Return both replay/ and replays/ members while retaining source manifest seats."""
    seats: dict[int, int] = {}
    created: dict[int, tuple[str, str]] = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifest_name = find_member(names, "manifest.csv")
        if manifest_name:
            for row in read_csv_text(archive.read(manifest_name).decode("utf-8-sig")):
                try:
                    seats[int(row["episode_id"])] = int(row["detected_submission_agent_index"])
                except (KeyError, TypeError, ValueError):
                    pass
        episodes_name = find_member(names, "episodes.csv")
        if episodes_name:
            for row in read_csv_text(archive.read(episodes_name).decode("utf-8-sig")):
                try:
                    episode_id = int(row.get("id") or row.get("episode_id"))
                except (TypeError, ValueError):
                    continue
                created[episode_id] = (
                    row.get("create_time") or row.get("created_at") or "",
                    row.get("end_time") or row.get("ended_at") or "",
                )
    refs = []
    for ref in _expanded_replay_refs(path):
        timestamps = created.get(ref.episode_id, ("", ""))
        refs.append(
            ReplayRef(
                Path(ref.zip_path), ref.member, ref.episode_id,
                seats.get(ref.episode_id), *timestamps, ref.path_variant,
            )
        )
    return refs


def load_replay(ref: ReplayRef) -> dict[str, Any]:
    with zipfile.ZipFile(ref.zip_path) as archive:
        return json.loads(archive.read(ref.member))


def extract_initial_decks(replay: dict[str, Any]) -> list[list[int]]:
    try:
        frames = replay["steps"][0][0].get("visualize") or []
        for frame in frames:
            action = frame.get("action")
            if isinstance(action, list) and len(action) == 2:
                if all(isinstance(deck, list) and len(deck) == 60 for deck in action):
                    return [[int(card) for card in deck] for deck in action]
            players = (frame.get("current") or {}).get("players") or []
            if len(players) == 2:
                decks = [[int(card["id"]) for card in player.get("deck") or []] for player in players]
                if all(len(deck) == 60 for deck in decks):
                    return decks
    except (KeyError, TypeError, ValueError):
        pass
    return [[], []]


def legal_action(action: Any, select: dict[str, Any]) -> bool:
    if not isinstance(action, list) or not all(isinstance(index, int) for index in action):
        return False
    options = select.get("option") or []
    minimum = int(select.get("minCount") or 0)
    maximum = int(select.get("maxCount") if select.get("maxCount") is not None else len(options))
    return (
        minimum <= len(action) <= maximum
        and len(set(action)) == len(action)
        and all(0 <= index < len(options) for index in action)
    )


def alignment_counts(replay: dict[str, Any], seat: int, shift: int) -> dict[str, int]:
    steps = replay.get("steps") or []
    checked = legal = empty = wrong_seat = 0
    stop = len(steps) - max(0, shift)
    start = max(0, -shift)
    for index in range(start, stop):
        try:
            observation = steps[index][seat].get("observation") or {}
            select = observation.get("select") or {}
            if not select or not isinstance(select.get("option"), list):
                continue
            current = observation.get("current") or {}
            if current and current.get("yourIndex") not in (None, seat):
                wrong_seat += 1
            action = steps[index + shift][seat].get("action")
        except (IndexError, KeyError, TypeError):
            continue
        checked += 1
        if action in (None, []):
            empty += 1
        if legal_action(action, select):
            legal += 1
    return {"checked": checked, "legal": legal, "empty": empty, "wrong_seat": wrong_seat}


def aligned_decisions(
    replay: dict[str, Any], seat: int, shift: int = 1
) -> Iterator[tuple[int, dict[str, Any], list[int], dict[str, Any]]]:
    """Yield exact CABT option-index labels stored on the same seat at t+1."""
    steps = replay.get("steps") or []
    for index in range(0, len(steps) - shift):
        try:
            agent_state = steps[index][seat]
            observation = agent_state.get("observation") or {}
            select = observation.get("select") or {}
            action = steps[index + shift][seat].get("action")
        except (IndexError, KeyError, TypeError):
            continue
        if agent_state.get("status") not in (None, "ACTIVE"):
            continue
        if not select or not isinstance(select.get("option"), list):
            continue
        if not legal_action(action, select):
            continue
        yield index, observation, action, steps[index + shift][seat]


def replay_outcome(replay: dict[str, Any], seat: int) -> str:
    rewards = replay.get("rewards") or []
    try:
        reward = float(rewards[seat])
    except (IndexError, TypeError, ValueError):
        reward = 0.0
    return "win" if reward > 0 else "loss" if reward < 0 else "draw"


def replay_health(replay: dict[str, Any]) -> tuple[bool, bool, bool]:
    statuses = [str(value).upper() for value in replay.get("statuses") or []]
    timeout = any("TIME" in value for value in statuses)
    error = any(value not in {"DONE", "COMPLETED", "ACTIVE", "INACTIVE"} for value in statuses)
    normal = bool(statuses) and not timeout and not error
    return normal, timeout, error


def future_log_types(replay: dict[str, Any], seat: int, step: int, horizon: int = 4) -> set[int]:
    """Audit-only helper. Never use returned values as policy features."""
    result: set[int] = set()
    steps = replay.get("steps") or []
    for index in range(step + 1, min(len(steps), step + horizon + 1)):
        try:
            logs = (steps[index][seat].get("observation") or {}).get("logs") or []
        except (IndexError, AttributeError):
            continue
        for log in logs:
            if isinstance(log, dict) and isinstance(log.get("type"), int):
                result.add(log["type"])
    return result
