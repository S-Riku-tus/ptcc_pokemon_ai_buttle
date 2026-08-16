"""Collect and verify an exact-deck multi-teacher replay corpus.

Unlike archetype filters, this script verifies the target submission's seat in
every replay and requires the full 60-card hash to match.  The resulting
``indexes/episodes.csv`` is directly consumable by
``build_grimmsnarl_v2_corpus.py`` (whose extractor is deck-parameterized despite
its historical name).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.core.replay_io import deck_hash, extract_fast_header_from_file  # noqa: E402
from scripts.fetch_submission_logs import (  # noqa: E402
    EpisodeRecord,
    download_replay,
    list_submission_episodes,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_int(row: dict[str, str], key: str) -> int:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"teacher row has no {key}: {row}")
    return int(value)


def list_with_retry(submission_id: int, retries: int, delay: float) -> list[EpisodeRecord]:
    error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return list_submission_episodes(submission_id)
        except Exception as exc:  # noqa: BLE001
            error = exc
            if attempt >= retries:
                break
            time.sleep(delay * (2 ** attempt))
    assert error is not None
    raise error


def target_seats(record: EpisodeRecord, submission_id: int) -> list[int]:
    seats = []
    if record.agent_0_submission_id == str(submission_id):
        seats.append(0)
    if record.agent_1_submission_id == str(submission_id):
        seats.append(1)
    return seats


def ensure_replay(episode_id: int, replay_root: Path, no_network: bool) -> tuple[str, str]:
    path = replay_root / f"episode_{episode_id}.json"
    if path.exists():
        try:
            header = extract_fast_header_from_file(str(path))
            if int(header.get("episode_id") or episode_id) == episode_id:
                return "skipped_existing", ""
        except Exception:  # corrupted cache is replaced when network is allowed
            pass
    if no_network:
        return "missing", "replay not cached and --no-network was set"
    try:
        status, downloaded = download_replay(episode_id, replay_root, overwrite=path.exists())
        if downloaded is None:
            return "failed", "download_replay returned no path"
        extract_fast_header_from_file(str(downloaded))
        return status, ""
    except Exception as exc:  # noqa: BLE001
        return "failed", f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--deck-hash", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--list-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=4.0)
    parser.add_argument("--max-episodes-per-teacher", type=int, default=0)
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument(
        "--refresh-lists", action="store_true",
        help="Ignore cached EpisodeService responses and list every submission again.",
    )
    args = parser.parse_args()

    teacher_rows = read_csv(args.teachers)
    required = {"team_id", "submission_id"}
    if not teacher_rows or any(required - set(row) for row in teacher_rows):
        parser.error("teacher CSV must contain team_id and submission_id")
    submission_ids = [parse_int(row, "submission_id") for row in teacher_rows]
    if len(submission_ids) != len(set(submission_ids)):
        parser.error("teacher CSV contains duplicate submission_id values")

    output_root = args.output_root.resolve()
    replay_root = output_root / "replays"
    submission_root = output_root / "submissions"
    replay_root.mkdir(parents=True, exist_ok=True)
    submission_root.mkdir(parents=True, exist_ok=True)

    relations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, teacher in enumerate(teacher_rows, 1):
        team_id = parse_int(teacher, "team_id")
        submission_id = parse_int(teacher, "submission_id")
        print(f"[{position}/{len(teacher_rows)}] list submission {submission_id}", flush=True)
        try:
            stored = submission_root / str(submission_id) / "episodes.json"
            if stored.exists() and not args.refresh_lists:
                payload = json.loads(stored.read_text(encoding="utf-8"))
                episodes = [EpisodeRecord(**row) for row in payload.get("episodes", [])]
            elif args.no_network:
                raise FileNotFoundError(f"no cached episode list: {stored}")
            else:
                episodes = list_with_retry(submission_id, args.list_retries, args.retry_delay)
                write_json(
                    submission_root / str(submission_id) / "episodes.json",
                    {"submission_id": submission_id, "episodes": [asdict(row) for row in episodes]},
                )
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "scope": "list", "team_id": team_id,
                "submission_id": submission_id, "episode_id": "",
                "seat_index": "", "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        completed = [row for row in episodes if row.state.upper() == "COMPLETED"]
        completed.sort(key=lambda row: row.episode_id)
        if args.max_episodes_per_teacher:
            completed = completed[-args.max_episodes_per_teacher:]
        for record in completed:
            seats = target_seats(record, submission_id)
            if not seats:
                failures.append({
                    "scope": "seat", "team_id": team_id,
                    "submission_id": submission_id, "episode_id": record.episode_id,
                    "seat_index": "", "error": "submission id absent from both seats",
                })
                continue
            for seat in seats:
                relations.append({
                    "team_id": team_id,
                    "team_name": teacher.get("team_name", ""),
                    "leaderboard_rank": teacher.get("rank", ""),
                    "leaderboard_score": teacher.get("score", ""),
                    "submission_id": submission_id,
                    "episode_id": record.episode_id,
                    "seat_index": seat,
                    "create_time": record.create_time,
                    "end_time": record.end_time,
                    "agent_0_submission_id": record.agent_0_submission_id,
                    "agent_1_submission_id": record.agent_1_submission_id,
                })

    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for relation in relations:
        by_episode[int(relation["episode_id"])].append(relation)
    print(
        f"listed relations={len(relations)} unique_episodes={len(by_episode)}; downloading",
        flush=True,
    )
    download_results: dict[int, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending = {
            executor.submit(ensure_replay, episode_id, replay_root, args.no_network): episode_id
            for episode_id in by_episode
        }
        for count, future in enumerate(as_completed(pending), 1):
            episode_id = pending[future]
            try:
                download_results[episode_id] = future.result()
            except Exception as exc:  # noqa: BLE001
                download_results[episode_id] = ("failed", f"{type(exc).__name__}: {exc}")
            if count % 50 == 0 or count == len(pending):
                print(f"  replays {count}/{len(pending)}", flush=True)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_relations: set[tuple[int, int]] = set()
    duplicate_relations = 0
    for relation in relations:
        episode_id = int(relation["episode_id"])
        seat = int(relation["seat_index"])
        key = (episode_id, seat)
        if key in seen_relations:
            duplicate_relations += 1
            continue
        seen_relations.add(key)
        status, error = download_results.get(episode_id, ("failed", "missing result"))
        row = dict(relation)
        row.update({
            "replay_path": f"replays/episode_{episode_id}.json",
            "download_status": status,
            "deck_hash": "",
            "deck_cards": 0,
            "error": error,
        })
        if status in {"failed", "missing"}:
            rejected.append(row)
            continue
        try:
            header = extract_fast_header_from_file(
                str(replay_root / f"episode_{episode_id}.json")
            )
            decks = header.get("decks") or [[], []]
            cards = list(decks[seat] or []) if seat < len(decks) else []
            row["deck_cards"] = len(cards)
            row["deck_hash"] = deck_hash(cards) if len(cards) == 60 else ""
            if len(cards) != 60:
                row["error"] = "target seat has no 60-card deck action"
                rejected.append(row)
            elif row["deck_hash"] != args.deck_hash:
                row["error"] = "exact deck hash mismatch"
                rejected.append(row)
            else:
                # The historical extractor accepts "success" by default.
                row["download_status"] = "success"
                accepted.append(row)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
            rejected.append(row)

    fields = [
        "team_id", "team_name", "leaderboard_rank", "leaderboard_score",
        "submission_id", "episode_id", "seat_index", "create_time", "end_time",
        "agent_0_submission_id", "agent_1_submission_id", "replay_path",
        "download_status", "deck_hash", "deck_cards", "error",
    ]
    accepted.sort(key=lambda row: (int(row["episode_id"]), int(row["seat_index"])))
    rejected.sort(key=lambda row: (int(row["episode_id"]), int(row["seat_index"])))
    write_csv(output_root / "indexes" / "episodes.csv", accepted, fields)
    write_csv(output_root / "indexes" / "rejected.csv", rejected, fields)
    write_csv(
        output_root / "indexes" / "failures.csv", failures,
        ["scope", "team_id", "submission_id", "episode_id", "seat_index", "error"],
    )

    per_teacher = Counter(int(row["team_id"]) for row in accepted)
    rejection_reasons = Counter(str(row.get("error", "")) for row in rejected)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teacher_source": str(args.teachers.resolve()),
        "target_deck_hash": args.deck_hash,
        "teachers_requested": len(teacher_rows),
        "teachers_accepted": len(per_teacher),
        "trajectories_accepted": len(accepted),
        "unique_episodes_accepted": len({int(row["episode_id"]) for row in accepted}),
        "per_teacher_trajectories": {str(k): v for k, v in sorted(per_teacher.items())},
        "relations_rejected": len(rejected),
        "rejection_reasons": dict(rejection_reasons),
        "list_failures": len(failures),
        "duplicate_episode_seat_relations_removed": duplicate_relations,
        "deck_mismatch_count": rejection_reasons.get("exact deck hash mismatch", 0),
        "seat_error_count": sum(row["scope"] == "seat" for row in failures),
        "data_gate": {
            "minimum_trajectories": 1000,
            "minimum_independent_teachers": 5,
            "trajectories_pass": len(accepted) >= 1000,
            "teachers_pass": len(per_teacher) >= 5,
            "integrity_pass": not rejected and not failures and duplicate_relations == 0,
        },
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
