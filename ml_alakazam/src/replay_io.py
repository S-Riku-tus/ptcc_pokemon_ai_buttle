from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .common import episode_id_from_name, read_csv_text


@dataclass(frozen=True)
class ReplayRef:
    zip_path: Path
    member: str
    episode_id: int
    target_seat: int | None
    created_at: str = ""
    ended_at: str = ""


def find_member(names: list[str], suffix: str) -> str | None:
    return next((name for name in names if name.endswith(suffix)), None)


def bundle_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        member = find_member(names, "bundle_summary.json")
        if member:
            return json.loads(archive.read(member))
        member = find_member(names, "run_meta.json")
        if member:
            return json.loads(archive.read(member))
    return {}


def episode_ids(path: Path) -> list[int]:
    with zipfile.ZipFile(path) as archive:
        ids = {
            episode_id
            for name in archive.namelist()
            if name.endswith("_observation_logs.json") or "/replay/episode_" in name
            if (episode_id := episode_id_from_name(name)) is not None
        }
    return sorted(ids)


def replay_refs(path: Path) -> list[ReplayRef]:
    refs: list[ReplayRef] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        seats: dict[int, int] = {}
        created: dict[int, tuple[str, str]] = {}
        manifest_name = find_member(names, "manifest.csv")
        if manifest_name:
            for row in read_csv_text(archive.read(manifest_name).decode("utf-8")):
                try:
                    seats[int(row["episode_id"])] = int(row["detected_submission_agent_index"])
                except (KeyError, TypeError, ValueError):
                    continue
        episodes_name = find_member(names, "episodes.csv")
        if episodes_name:
            for row in read_csv_text(archive.read(episodes_name).decode("utf-8")):
                try:
                    episode_id = int(row["id"] if "id" in row else row["episode_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                created[episode_id] = (
                    row.get("create_time") or row.get("created_at") or "",
                    row.get("end_time") or row.get("ended_at") or "",
                )
        for name in names:
            if not re.search(r"/replay/episode_\d+\.json$", name):
                continue
            episode_id = episode_id_from_name(name)
            if episode_id is None:
                continue
            timestamps = created.get(episode_id, ("", ""))
            refs.append(ReplayRef(path, name, episode_id, seats.get(episode_id), *timestamps))
    return sorted(refs, key=lambda ref: ref.episode_id)


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
    steps = replay.get("steps") or []
    for index in range(0, len(steps) - shift):
        try:
            observation = steps[index][seat].get("observation") or {}
            select = observation.get("select") or {}
            action = steps[index + shift][seat].get("action")
        except (IndexError, KeyError, TypeError):
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
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def replay_health(replay: dict[str, Any]) -> tuple[bool, bool, bool]:
    statuses = [str(value).upper() for value in replay.get("statuses") or []]
    timeout = any("TIME" in value for value in statuses)
    error = any(value not in {"DONE", "COMPLETED", "ACTIVE", "INACTIVE"} for value in statuses)
    normal = bool(statuses) and not timeout and not error
    return normal, timeout, error


def future_log_types(replay: dict[str, Any], seat: int, step: int, horizon: int = 4) -> set[int]:
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

