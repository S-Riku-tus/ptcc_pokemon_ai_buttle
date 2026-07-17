from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv_to_zip(
    archive: zipfile.ZipFile,
    member: str,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    archive.writestr(member, buffer.getvalue())


def _detect_submission_agent_index(replay_path: Path, submission_id: int) -> str:
    from scripts.fetch_submission_logs import detect_submission_agent_index

    detected = detect_submission_agent_index(replay_path, submission_id)
    return "" if detected is None else str(detected)


def package_submission(
    *,
    source_root: Path,
    submission_id: int,
    output_zip: Path,
    rank: int | None,
    team_name: str | None,
    limit: int | None,
) -> dict[str, Any]:
    submission_dir = source_root / "submissions" / str(submission_id)
    replay_index_path = submission_dir / "replays" / "index.json"
    log_index_path = submission_dir / "logs" / "index.json"
    submission_json_path = submission_dir / "submission.json"
    if not replay_index_path.exists():
        raise FileNotFoundError(f"Replay index not found: {replay_index_path}")

    submission_meta = _read_json(submission_json_path) if submission_json_path.exists() else {}
    rank = rank if rank is not None else int(submission_meta.get("leaderboard_rank") or 0)
    team_name = team_name or str(submission_meta.get("team_name") or "")

    replay_rows = [row for row in _read_json(replay_index_path) if row.get("download_status") == "success"]
    if limit is not None:
        replay_rows = replay_rows[:limit]
    log_rows_by_episode: dict[int, dict[str, Any]] = {}
    if log_index_path.exists():
        for row in _read_json(log_index_path):
            if str(row.get("download_status")) == "success":
                log_rows_by_episode[int(row["episode_id"])] = row

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    added_replays = 0
    added_logs = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for row in replay_rows:
            episode_id = int(row["episode_id"])
            replay_rel = Path(str(row["replay_path"]))
            replay_path = source_root / replay_rel
            if not replay_path.exists():
                continue
            archive.write(replay_path, f"replays/episode_{episode_id}.json")
            added_replays += 1

            detected = _detect_submission_agent_index(replay_path, submission_id)
            log_statuses = ["", ""]
            log_row = log_rows_by_episode.get(episode_id, {})
            for seat in (0, 1):
                key = f"agent_{seat}_log_path"
                log_rel_raw = log_row.get(key)
                if not log_rel_raw:
                    continue
                log_path = source_root / Path(str(log_rel_raw))
                if log_path.exists():
                    archive.write(log_path, f"logs/{episode_id}/agent_{seat}_observation_logs.json")
                    log_statuses[seat] = "success"
                    added_logs += 1

            manifest_rows.append(
                {
                    "submission_id": submission_id,
                    "episode_id": episode_id,
                    "episode_state": "",
                    "detected_submission_agent_index": detected,
                    "replay_status": "success",
                    "agent_0_log_status": log_statuses[0],
                    "agent_1_log_status": log_statuses[1],
                    "error": "",
                }
            )

        _write_csv_to_zip(
            archive,
            "manifest.csv",
            manifest_rows,
            [
                "submission_id",
                "episode_id",
                "episode_state",
                "detected_submission_agent_index",
                "replay_status",
                "agent_0_log_status",
                "agent_1_log_status",
                "error",
            ],
        )
        archive.writestr("submission.json", json.dumps(submission_meta, ensure_ascii=False, indent=2) + "\n")
        archive.writestr(
            "bundle_summary.json",
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "source_root": str(source_root),
                    "rank": rank,
                    "submission_id": submission_id,
                    "team_name": team_name,
                    "replay_count": added_replays,
                    "log_file_count": added_logs,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

    return {
        "output_zip": str(output_zip),
        "rank": rank,
        "submission_id": submission_id,
        "team_name": team_name,
        "replay_count": added_replays,
        "log_file_count": added_logs,
        "bytes": output_zip.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package one collected Kaggle submission into an ML replay ZIP.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--submission", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--team-name")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = package_submission(
        source_root=args.source_root.resolve(),
        submission_id=args.submission,
        output_zip=args.output.resolve(),
        rank=args.rank,
        team_name=args.team_name,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
