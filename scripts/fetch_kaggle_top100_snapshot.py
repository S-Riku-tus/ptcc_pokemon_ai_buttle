from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener


JST = timezone(timedelta(hours=9))
UTC = timezone.utc
COMPETITION_ID = 116727
COMPETITION_NAME = "pokemon-tcg-ai-battle"
COMPETITION_LEADERBOARD_URL = (
    "https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard"
)
LEADERBOARD_API_URL = (
    "https://www.kaggle.com/api/i/competitions.LeaderboardService/GetLeaderboard"
)
TEAM_PUBLIC_SUBMISSIONS_API_URL = (
    "https://www.kaggle.com/api/i/competitions.SubmissionService/ListTeamPublicSubmissions"
)
TOP_N = 100
REQUEST_DELAY_SECONDS = 0.2
MAX_RETRIES = 3


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def kaggle_exe() -> Path:
    return repo_root() / ".venv" / "Scripts" / "kaggle.exe"


def now_jst() -> datetime:
    return datetime.now(JST)


def isoformat_z(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def safe_decimal(text: str | None) -> Decimal | None:
    if text is None:
        return None
    normalized = text.replace(",", "").strip()
    if not normalized:
        return None
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def parse_submission_time(value: str | None) -> datetime | None:
    """Parse a Kaggle timestamp, tolerating nanosecond fractional seconds.

    Kaggle mixes millisecond precision (``dateSubmitted``) with nanosecond
    precision (``lastSubmissionDate``) in the same API family, and
    ``datetime.fromisoformat`` accepts at most 6 fractional digits before
    Python 3.11, so the extra digits are truncated instead of raising.
    """
    if not value:
        return None

    text = value.strip().replace("Z", "+00:00").replace("z", "+00:00")
    match = re.match(r"^(?P<head>.*\.\d{6})\d+(?P<tail>.*)$", text)
    if match:
        text = match.group("head") + match.group("tail")

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def jst_string(value: str | None) -> str:
    dt = parse_submission_time(value)
    if dt is None:
        return ""
    return dt.astimezone(JST).isoformat()


def utc_string(value: str | None) -> str:
    dt = parse_submission_time(value)
    if dt is None:
        return ""
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_command(command: list[str]) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=repo_root(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return CommandResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def save_command_result(path: Path, name: str, result: CommandResult) -> None:
    mkdir(path)
    write_text(path / f"{name}.stdout.txt", result.stdout)
    write_text(path / f"{name}.stderr.txt", result.stderr)
    write_json(
        path / f"{name}.json",
        {
            "command": result.command,
            "exit_code": result.exit_code,
            "stdout_file": f"{name}.stdout.txt",
            "stderr_file": f"{name}.stderr.txt",
        },
    )


def build_opener_with_cookies() -> tuple[Any, CookieJar]:
    jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))
    return opener, jar


def find_cookie_value(jar: CookieJar, name: str) -> str:
    for cookie in jar:
        if cookie.name == name:
            return cookie.value
    raise RuntimeError(f"missing cookie: {name}")


def http_get(opener: Any, url: str, headers: dict[str, str] | None = None) -> str:
    request = Request(url, headers=headers or {}, method="GET")
    with opener.open(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def http_post_json(
    opener: Any,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=headers, method="POST")
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with opener.open(request, timeout=30) as response:
                raw = response.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(0.8 * attempt)
    raise RuntimeError(f"request failed after {MAX_RETRIES} attempts: {last_error}")


def build_api_headers(xsrf_token: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Referer": COMPETITION_LEADERBOARD_URL,
        "X-Requested-With": "XMLHttpRequest",
        "X-XSRF-TOKEN": xsrf_token,
    }


def prepare_directories(root: Path, timestamp_label: str) -> dict[str, Path]:
    root = root.resolve()
    snapshot = root / timestamp_label
    latest = root / "latest"
    raw_cli = snapshot / "raw" / "cli"
    raw_api = snapshot / "raw" / "api"
    raw_team_api = raw_api / "team_public_submissions"
    for path in (snapshot, latest, raw_cli, raw_api, raw_team_api):
        mkdir(path)
    return {
        "root": root,
        "snapshot": snapshot,
        "latest": latest,
        "raw_cli": raw_cli,
        "raw_api": raw_api,
        "raw_team_api": raw_team_api,
    }


def fetch_cli_diagnostics(raw_cli_dir: Path) -> list[dict[str, Any]]:
    commands = [
        ("version", [str(kaggle_exe()), "--version"]),
        ("competitions_help", [str(kaggle_exe()), "competitions", "--help"]),
        (
            "leaderboard_help",
            [str(kaggle_exe()), "competitions", "leaderboard", "--help"],
        ),
        (
            "team_submissions_help",
            [str(kaggle_exe()), "competitions", "team-submissions", "--help"],
        ),
        (
            "auth_check_leaderboard_show",
            [
                str(kaggle_exe()),
                "competitions",
                "leaderboard",
                COMPETITION_NAME,
                "--show",
                "--csv",
                "--quiet",
            ],
        ),
    ]

    diagnostics: list[dict[str, Any]] = []
    for name, command in commands:
        result = run_command(command)
        save_command_result(raw_cli_dir, name, result)
        diagnostics.append(
            {
                "name": name,
                "command": command,
                "exit_code": result.exit_code,
                "stdout_file": f"{name}.stdout.txt",
                "stderr_file": f"{name}.stderr.txt",
            }
        )
    return diagnostics


def fetch_public_session() -> tuple[Any, dict[str, str], str]:
    opener, jar = build_opener_with_cookies()
    html = http_get(opener, COMPETITION_LEADERBOARD_URL)
    xsrf_token = find_cookie_value(jar, "XSRF-TOKEN")
    return opener, build_api_headers(xsrf_token), html


def fetch_leaderboard(opener: Any, headers: dict[str, str]) -> dict[str, Any]:
    return http_post_json(
        opener,
        LEADERBOARD_API_URL,
        {"competitionId": COMPETITION_ID},
        headers,
    )


def fetch_team_public_submissions(
    opener: Any,
    headers: dict[str, str],
    team_id: int,
) -> dict[str, Any]:
    return http_post_json(
        opener,
        TEAM_PUBLIC_SUBMISSIONS_API_URL,
        {"teamId": team_id},
        headers,
    )


def choose_representative_submission(
    leaderboard_submission_id: int,
    leaderboard_score_text: str,
    submissions: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str, bool]:
    leaderboard_score = safe_decimal(leaderboard_score_text)
    score_matches: list[dict[str, Any]] = []
    for submission in submissions:
        submission_score = safe_decimal(submission.get("publicScoreFormatted"))
        if submission_score is not None and leaderboard_score is not None:
            if submission_score == leaderboard_score:
                score_matches.append(submission)

    if score_matches:
        for submission in score_matches:
            if submission.get("id") == leaderboard_submission_id:
                return submission, "matched_leaderboard_score_and_id", True
        score_matches.sort(
            key=lambda item: (
                parse_submission_time(item.get("dateSubmitted")) or datetime.min.replace(tzinfo=UTC),
                item.get("id", 0),
            ),
            reverse=True,
        )
        return score_matches[0], "matched_leaderboard_score", True

    if not submissions:
        return None, "no_public_submissions", False

    ranked = sorted(
        submissions,
        key=lambda item: (
            safe_decimal(item.get("publicScoreFormatted")) or Decimal("-Infinity"),
            parse_submission_time(item.get("dateSubmitted")) or datetime.min.replace(tzinfo=UTC),
            item.get("id", 0),
        ),
        reverse=True,
    )
    return ranked[0], "highest_public_score_fallback", False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a current Kaggle leaderboard snapshot and the public "
            "submissions for the top N teams."
        )
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N,
        help="Number of public leaderboard teams to fetch. Default: 100.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root() / "data" / "kaggle_top100",
        help="Snapshot root. Default: data/kaggle_top100.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be greater than 0")

    started_at = now_jst()
    timestamp_label = started_at.strftime("%Y%m%d_%H%M%S_JST")
    dirs = prepare_directories(args.output_root, timestamp_label)

    cli_diagnostics = fetch_cli_diagnostics(dirs["raw_cli"])

    opener, headers, leaderboard_page_html = fetch_public_session()
    write_text(dirs["raw_api"] / "leaderboard_page.html", leaderboard_page_html)

    leaderboard = fetch_leaderboard(opener, headers)
    write_json(dirs["raw_api"] / "leaderboard_full.json", leaderboard)

    public_leaderboard = list(leaderboard.get("publicLeaderboard") or [])
    teams = list(leaderboard.get("teams") or [])
    teams_by_id = {team.get("teamId"): team for team in teams}
    top_entries = public_leaderboard[: args.top_n]

    leaderboard_rows: list[dict[str, Any]] = []
    public_submission_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    team_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_public_submission_rows = 0

    for entry in top_entries:
        rank = int(entry["rank"])
        team_id = int(entry["teamId"])
        leaderboard_submission_id = int(entry["submissionId"])
        team_info = teams_by_id.get(team_id, {})
        team_name = team_info.get("teamName", "")
        leaderboard_score = entry.get("displayScore", "")
        medal = entry.get("medal", "")

        leaderboard_rows.append(
            {
                "rank": rank,
                "team_id": team_id,
                "team_name": team_name,
                "leaderboard_submission_id": leaderboard_submission_id,
                "leaderboard_score": leaderboard_score,
                "medal": medal,
            }
        )

        try:
            payload = fetch_team_public_submissions(opener, headers, team_id)
            write_json(dirs["raw_team_api"] / f"team_{team_id}.json", payload)
            submissions = list(payload.get("submissions") or [])
            representative, representative_reason, matched_score = (
                choose_representative_submission(
                    leaderboard_submission_id,
                    leaderboard_score,
                    submissions,
                )
            )

            normalized_submissions: list[dict[str, Any]] = []
            for submission in submissions:
                row = {
                    "id": int(submission["id"]),
                    "publicScore": submission.get("publicScoreFormatted", ""),
                    "submittedAtUtc": utc_string(submission.get("dateSubmitted")),
                    "submittedAtJst": jst_string(submission.get("dateSubmitted")),
                }
                normalized_submissions.append(row)
                public_submission_rows.append(
                    {
                        "rank": rank,
                        "team_id": team_id,
                        "team_name": team_name,
                        "leaderboard_score": leaderboard_score,
                        "leaderboard_submission_id": leaderboard_submission_id,
                        "public_submission_id": row["id"],
                        "public_score": row["publicScore"],
                        "submitted_at_utc": row["submittedAtUtc"],
                        "submitted_at_jst": row["submittedAtJst"],
                        "is_representative": representative is not None
                        and row["id"] == representative.get("id"),
                        "representative_reason": representative_reason
                        if representative is not None and row["id"] == representative.get("id")
                        else "",
                        "leaderboard_score_match_found": matched_score,
                    }
                )

            total_public_submission_rows += len(normalized_submissions)

            representative_row = {
                "rank": rank,
                "team_id": team_id,
                "team_name": team_name,
                "leaderboard_score": leaderboard_score,
                "leaderboard_submission_id": leaderboard_submission_id,
                "representative_submission_id": representative.get("id")
                if representative
                else "",
                "representative_public_score": representative.get("publicScoreFormatted", "")
                if representative
                else "",
                "representative_submitted_at_utc": utc_string(
                    representative.get("dateSubmitted") if representative else None
                ),
                "representative_submitted_at_jst": jst_string(
                    representative.get("dateSubmitted") if representative else None
                ),
                "representative_reason": representative_reason,
                "leaderboard_score_match_found": matched_score,
                "public_submission_count": len(normalized_submissions),
            }
            representative_rows.append(representative_row)

            team_results.append(
                {
                    "rank": rank,
                    "teamId": team_id,
                    "teamName": team_name,
                    "leaderboardSubmissionId": leaderboard_submission_id,
                    "leaderboardScore": leaderboard_score,
                    "medal": medal,
                    "representativeSubmissionId": representative.get("id")
                    if representative
                    else None,
                    "representativePublicScore": representative.get("publicScoreFormatted")
                    if representative
                    else None,
                    "representativeSubmittedAtUtc": utc_string(
                        representative.get("dateSubmitted") if representative else None
                    ),
                    "representativeSubmittedAtJst": jst_string(
                        representative.get("dateSubmitted") if representative else None
                    ),
                    "representativeReason": representative_reason,
                    "leaderboardScoreMatchFound": matched_score,
                    "publicSubmissions": normalized_submissions,
                }
            )

            time.sleep(REQUEST_DELAY_SECONDS)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "rank": rank,
                    "teamId": team_id,
                    "teamName": team_name,
                    "reason": str(exc),
                }
            )

    completed_at = now_jst()
    manifest = {
        "competition": {
            "id": COMPETITION_ID,
            "name": COMPETITION_NAME,
            "leaderboardUrl": COMPETITION_LEADERBOARD_URL,
        },
        "retrievedAtJst": completed_at.isoformat(),
        "retrievedAtUtc": isoformat_z(completed_at),
        "snapshotDirectory": str(dirs["snapshot"].relative_to(repo_root())),
        "latestDirectory": str(dirs["latest"].relative_to(repo_root())),
        "topN": args.top_n,
        "counts": {
            "leaderboardRowsFetched": len(public_leaderboard),
            "teamsRequested": len(top_entries),
            "teamsSucceeded": len(team_results),
            "teamsFailed": len(failures),
            "publicSubmissionRows": total_public_submission_rows,
        },
        "cli": {
            "versionCommandFile": "raw/cli/version.json",
            "teamSubmissionsSupported": False,
            "teamSubmissionsNote": (
                "Kaggle API 1.7.4.5 in this environment does not expose "
                "`kaggle competitions team-submissions`; equivalent public web "
                "API endpoints were used for per-team public submissions."
            ),
            "diagnostics": cli_diagnostics,
        },
    }

    write_json(dirs["snapshot"] / "manifest.json", manifest)
    write_json(
        dirs["snapshot"] / f"top{args.top_n}_results.json",
        {
            "metadata": manifest,
            "teams": team_results,
            "failures": failures,
        },
    )
    write_json(dirs["snapshot"] / "failures.json", failures)
    write_csv(
        dirs["snapshot"] / "failures.csv",
        failures,
        ["rank", "teamId", "teamName", "reason"],
    )
    write_csv(
        dirs["snapshot"] / f"leaderboard_top{args.top_n}.csv",
        leaderboard_rows,
        [
            "rank",
            "team_id",
            "team_name",
            "leaderboard_submission_id",
            "leaderboard_score",
            "medal",
        ],
    )
    write_csv(
        dirs["snapshot"] / f"public_submissions_top{args.top_n}.csv",
        public_submission_rows,
        [
            "rank",
            "team_id",
            "team_name",
            "leaderboard_score",
            "leaderboard_submission_id",
            "public_submission_id",
            "public_score",
            "submitted_at_utc",
            "submitted_at_jst",
            "is_representative",
            "representative_reason",
            "leaderboard_score_match_found",
        ],
    )
    write_csv(
        dirs["snapshot"] / f"representative_submissions_top{args.top_n}.csv",
        representative_rows,
        [
            "rank",
            "team_id",
            "team_name",
            "leaderboard_score",
            "leaderboard_submission_id",
            "representative_submission_id",
            "representative_public_score",
            "representative_submitted_at_utc",
            "representative_submitted_at_jst",
            "representative_reason",
            "leaderboard_score_match_found",
            "public_submission_count",
        ],
    )

    if dirs["latest"].exists():
        shutil.rmtree(dirs["latest"])
    shutil.copytree(dirs["snapshot"], dirs["latest"])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
