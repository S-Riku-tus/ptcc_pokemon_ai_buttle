from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from zipfile import ZipFile

try:
    import orjson  # type: ignore
except ImportError:  # portable fallback
    class _OrjsonCompat:
        JSONDecodeError = json.JSONDecodeError
        @staticmethod
        def loads(data):
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            return json.loads(data)
    orjson = _OrjsonCompat()

# Accept both historical layouts and arbitrary leading directories.
_REPLAY_RE = re.compile(r"(?:^|/)(replay|replays)/episode_(\d+)\.json$")
_TEAM_RE = re.compile(rb'"TeamNames"\s*:\s*\[(.*?)\]')
_REWARD_RE = re.compile(rb'"rewards"\s*:\s*\[([^\]]*)\]')
_INITIAL_ACTION_RE = re.compile(
    rb'"visualize"\s*:\s*\[\s*\{\s*"action"\s*:\s*\[\s*\[([^\]]*)\]\s*,\s*\[([^\]]*)\]'
)
_RANK_RE = re.compile(r"(?:^|_)rank(\d+)(?:_|$)", re.I)
_SUB_RE = re.compile(r"(?:^|_)sub(\d+)(?:_|\.)", re.I)


@dataclass(frozen=True)
class ReplayRef:
    zip_path: Path
    member: str
    episode_id: int
    path_variant: str


def replay_refs(zip_path: str | Path) -> list[ReplayRef]:
    """Return all full replay members from both replay/ and replays/ layouts."""
    path = Path(zip_path)
    refs: list[ReplayRef] = []
    with ZipFile(path) as zf:
        for member in zf.namelist():
            match = _REPLAY_RE.search(member)
            if match:
                refs.append(ReplayRef(path, member, int(match.group(2)), match.group(1)))
    refs.sort(key=lambda r: (r.episode_id, r.member))
    return refs


def legacy_replay_refs(zip_path: str | Path) -> list[ReplayRef]:
    """Reference implementation of the old bug: only singular replay/."""
    return [ref for ref in replay_refs(zip_path) if ref.path_variant == "replay"]


def discover_zip_paths(roots: Iterable[str | Path]) -> list[Path]:
    out: set[Path] = set()
    for root in roots:
        p = Path(root)
        if p.is_file() and p.suffix.lower() == ".zip":
            out.add(p.resolve())
        elif p.exists():
            out.update(x.resolve() for x in p.rglob("*.zip"))
    return sorted(out)


def load_replay(ref: ReplayRef) -> dict[str, Any]:
    with ZipFile(ref.zip_path) as zf:
        return orjson.loads(zf.read(ref.member))


def read_replay_prefix(ref: ReplayRef, limit: int = 240_000) -> bytes:
    with ZipFile(ref.zip_path) as zf, zf.open(ref.member) as stream:
        return stream.read(limit)


def _int_list(raw: bytes) -> list[int]:
    return [int(x) for x in re.findall(rb"-?\d+", raw)]


def deck_hash(card_ids: Iterable[int]) -> str:
    counts = Counter(int(x) for x in card_ids)
    canonical = ";".join(f"{card_id}:{counts[card_id]}" for card_id in sorted(counts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def extract_fast_header_from_bytes(raw: bytes) -> dict[str, Any]:
    """Extract team names, rewards and both initial 60-card lists from replay bytes."""
    teams: list[Any] = [None, None]
    rewards: list[Any] = [None, None]
    decks: list[list[int]] = [[], []]
    match = _TEAM_RE.search(raw)
    if match:
        try:
            teams = json.loads(b"[" + match.group(1) + b"]")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    match = _REWARD_RE.search(raw)
    if match:
        try:
            rewards = json.loads(b"[" + match.group(1) + b"]")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    match = _INITIAL_ACTION_RE.search(raw)
    if match:
        decks = [_int_list(match.group(1)), _int_list(match.group(2))]
    return {
        "team_names": (teams + [None, None])[:2],
        "rewards": (rewards + [None, None])[:2],
        "decks": decks,
        "deck_hashes": [deck_hash(deck) if deck else "" for deck in decks],
    }


def extract_fast_header(ref: ReplayRef) -> dict[str, Any]:
    """Extract team names, rewards and both initial 60-card lists without parsing all steps."""
    return extract_fast_header_from_bytes(read_replay_prefix(ref))


def extract_fast_header_from_file(path: str | Path, limit: int = 240_000) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return extract_fast_header_from_bytes(handle.read(limit))


def replay_bundle_scope(member: str) -> str:
    """Return the archive prefix that owns a packaged episode replay."""
    parts = member.split("/")
    if "episodes" in parts:
        return "/".join(parts[:parts.index("episodes")])
    return ""


def zip_metadata(zip_path: str | Path, replay_member: str | None = None) -> dict[str, Any]:
    path = Path(zip_path)
    meta: dict[str, Any] = {}
    with ZipFile(path) as zf:
        all_members = zf.namelist()
        scope = replay_bundle_scope(replay_member) if replay_member is not None else None
        if scope:
            prefix = scope.rstrip("/") + "/"
            members = [name for name in all_members if name.startswith(prefix)]
        else:
            members = all_members

        def load_json_member(suffix: str) -> dict[str, Any] | None:
            member = next((name for name in members if name.endswith(suffix)), None)
            if member is None:
                return None
            try:
                value = orjson.loads(zf.read(member))
            except (orjson.JSONDecodeError, UnicodeDecodeError):
                return None
            return value if isinstance(value, dict) else None

        for member in members:
            if member.endswith("bundle_summary.json"):
                try:
                    meta.update(orjson.loads(zf.read(member)))
                except orjson.JSONDecodeError:
                    pass
                break

        # The full top-submission bundles use submission.json rather than the
        # older bundle_summary.json.  Read it before episodes.json so the
        # target submission can identify the exact replay seat without relying
        # on a display-name match.
        submission = load_json_member("submission.json") or {}
        if submission:
            for source_key, target_key in (
                ("submission_id", "submission_id"),
                ("team_name", "team_name"),
                ("team_id", "team_id"),
                ("leaderboard_score", "leaderboard_score"),
            ):
                if not meta.get(target_key) and submission.get(source_key) not in (None, ""):
                    meta[target_key] = submission[source_key]
            if not meta.get("rank") and submission.get("leaderboard_rank") not in (None, ""):
                meta["rank"] = int(submission["leaderboard_rank"])

        source_manifest: dict[int, dict[str, Any]] = {}
        episodes = load_json_member("episodes.json") or {}
        target_submission = str(meta.get("submission_id") or "")
        for episode in episodes.get("episodes", []):
            if not isinstance(episode, dict) or not str(episode.get("episode_id", "")).isdigit():
                continue
            matches = [
                seat for seat in (0, 1)
                if target_submission
                and str(episode.get(f"agent_{seat}_submission_id") or "") == target_submission
            ]
            if len(matches) == 1:
                source_manifest[int(episode["episode_id"])] = {
                    "episode_id": str(episode["episode_id"]),
                    "detected_submission_agent_index": str(matches[0]),
                    "seat_source": "episodes_submission_id",
                }

        for member in members:
            if member.endswith("manifest.csv"):
                text = zf.read(member).decode("utf-8-sig", "replace")
                # A collector-produced manifest may include richer diagnostics;
                # prefer those rows while retaining seats recovered from
                # episodes.json for any missing episode.
                for row in csv.DictReader(io.StringIO(text)):
                    if not row.get("episode_id", "").isdigit():
                        continue
                    episode_id = int(row["episode_id"])
                    recovered = source_manifest.get(episode_id, {})
                    merged = {**recovered, **row}
                    if not row.get("detected_submission_agent_index") and recovered.get(
                        "detected_submission_agent_index"
                    ):
                        merged["detected_submission_agent_index"] = recovered[
                            "detected_submission_agent_index"
                        ]
                        merged["seat_source"] = recovered.get(
                            "seat_source", "episodes_submission_id"
                        )
                    source_manifest[episode_id] = merged
                break
        if source_manifest:
            meta["source_manifest"] = source_manifest
    if not meta.get("rank"):
        match = _RANK_RE.search(path.name)
        if match:
            meta["rank"] = int(match.group(1))
    if not meta.get("submission_id"):
        match = _SUB_RE.search(path.name)
        if match:
            meta["submission_id"] = int(match.group(1))
        else:
            nums = re.findall(r"(?<!\d)(\d{8})(?!\d)", path.name)
            if nums:
                meta["submission_id"] = int(nums[-1])
    if not meta.get("team_name") and "Majkel1337" in path.name:
        meta["team_name"] = "Majkel1337"
    return meta
