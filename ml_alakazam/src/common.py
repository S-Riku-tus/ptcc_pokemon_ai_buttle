from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.json"

ALAKAZAM_LINE = {741, 742, 743}
MAJKEL_KEY_CARDS = {
    13, 19, 66, 140, 305, 343, 741, 742, 743, 1079, 1081, 1086,
    1097, 1129, 1152, 1182, 1184, 1197, 1225, 1231, 1266,
}
HIGH_IMPORTANCE_CARDS = {13, 19, 66, 1079, 1081, 1182, 1197, 1225, 1231}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["config_path"] = str(config_path.resolve())
    return config


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def stable_code(value: Any, buckets: int = 4093) -> int:
    raw = str(value if value is not None else "").encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=4).digest(), "little") % buckets + 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deck_counter(deck: Iterable[int]) -> Counter[int]:
    return Counter(int(card) for card in deck)


def canonical_deck(deck: Iterable[int]) -> str:
    counts = deck_counter(deck)
    return ";".join(f"{card}:{counts[card]}" for card in sorted(counts))


def deck_hash(deck: Iterable[int]) -> str:
    return sha256_bytes(canonical_deck(deck).encode("ascii"))


def deck_json(deck: Iterable[int]) -> str:
    return json.dumps(sorted(int(card) for card in deck), separators=(",", ":"))


def parse_bundle_name(path: Path) -> tuple[int | None, str, int | None]:
    name = path.stem
    rank_match = re.search(r"rank(\d+)", name, re.IGNORECASE)
    sub_match = re.search(r"(?:sub|top_)(\d{7,9})", name, re.IGNORECASE)
    rank = int(rank_match.group(1)) if rank_match else (1 if "latest_rank01" in name else None)
    submission = int(sub_match.group(1)) if sub_match else None
    team = re.sub(r"^latest_", "", name)
    team = re.sub(r"^rank\d+_", "", team)
    team = re.sub(r"_(?:sub|\d+_top_)\d+.*$", "", team)
    return rank, team, submission


def episode_id_from_name(name: str) -> int | None:
    parts = PurePosixPath(name).parts
    for part in reversed(parts):
        match = re.search(r"(?:episode_)?(\d{7,10})", part)
        if match:
            return int(match.group(1))
    return None


def read_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(text.lstrip("\ufeff").splitlines()))


def load_card_metadata() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    card_path = WORKSPACE / "vendor" / "cg" / "cards.json"
    attack_path = WORKSPACE / "vendor" / "cg" / "attacks.json"
    cards_raw = json.loads(card_path.read_text(encoding="utf-8"))
    attacks_raw = json.loads(attack_path.read_text(encoding="utf-8"))
    cards = {int(row.get("cardId", row.get("id"))): row for row in cards_raw}
    attacks = {int(row.get("attackId", row.get("id"))): row for row in attacks_raw}
    return cards, attacks


def load_deck_evidence(path: Path) -> dict[str, list[int]]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    if not rows:
        return {}
    teams = [key for key in rows[0] if key not in {"card_id", "card"}]
    result: dict[str, list[int]] = {team: [] for team in teams}
    for row in rows:
        card_id = int(row["card_id"])
        for team in teams:
            try:
                count = int(float(row.get(team) or 0))
            except ValueError:
                count = 0
            result[team].extend([card_id] * count)
    return result


def normalize_team(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def match_evidence_team(team: str, evidence: dict[str, list[int]]) -> str | None:
    needle = normalize_team(team)
    if not needle:
        return None
    for candidate in evidence:
        normalized = normalize_team(candidate)
        if needle == normalized or needle in normalized or normalized in needle:
            return candidate
    return None


def load_team_ratings(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    result: dict[str, float] = {}
    for row in csv.DictReader(path.open(encoding="utf-8-sig", newline="")):
        team = row.get("team") or row.get("team_name") or ""
        rating = row.get("rating") or row.get("Rating") or row.get("leaderboard_score")
        if team and rating:
            result[normalize_team(team)] = float(rating)
    return result


def classify_deck(deck: Iterable[int]) -> tuple[bool, str]:
    counts = deck_counter(deck)
    if not ALAKAZAM_LINE.issubset(counts):
        return False, "non_alakazam"
    tags = ["alakazam"]
    tags.append("enriching" if counts[13] else "no_enriching")
    tags.append("boss" if counts[1182] else "no_boss")
    tags.append("fez" if counts[140] else "no_fez")
    tags.append("shaymin" if counts[343] else "no_shaymin")
    tags.append(f"dunsparce{counts[305]}-{counts[66]}")
    tags.append(f"candy{counts[1079]}")
    tags.append(f"hammer{counts[1081]}")
    return True, "_".join(tags)


def describe_variant(deck: Iterable[int], majkel_deck: Iterable[int] | None = None) -> str:
    is_alakazam, deck_type = classify_deck(deck)
    if not is_alakazam:
        return deck_type
    if majkel_deck:
        left, right = deck_counter(deck), deck_counter(majkel_deck)
        distance = sum(abs(left[key] - right[key]) for key in set(left) | set(right)) // 2
        if distance == 0:
            return "majkel_exact"
        if distance <= 3:
            return f"majkel_near_{distance}"
    return deck_type
