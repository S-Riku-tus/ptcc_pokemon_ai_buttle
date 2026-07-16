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


def extract_fast_header(ref: ReplayRef) -> dict[str, Any]:
    """Extract team names, rewards and both initial 60-card lists without parsing all steps."""
    raw = read_replay_prefix(ref)
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


def zip_metadata(zip_path: str | Path) -> dict[str, Any]:
    path = Path(zip_path)
    meta: dict[str, Any] = {}
    with ZipFile(path) as zf:
        for member in zf.namelist():
            if member.endswith("bundle_summary.json"):
                try:
                    meta.update(orjson.loads(zf.read(member)))
                except orjson.JSONDecodeError:
                    pass
                break
        for member in zf.namelist():
            if member.endswith("manifest.csv"):
                text = zf.read(member).decode("utf-8-sig", "replace")
                meta["source_manifest"] = {
                    int(row["episode_id"]): row
                    for row in csv.DictReader(io.StringIO(text))
                    if row.get("episode_id", "").isdigit()
                }
                break
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
