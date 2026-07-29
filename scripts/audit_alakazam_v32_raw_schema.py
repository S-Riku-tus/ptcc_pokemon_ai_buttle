"""Audit raw observation fields available to the v32 imitation policy.

The report intentionally records schema paths, value kinds, and bounded example
values.  It does not persist complete observations or hidden card contents.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_replay(
    row: dict[str, str],
    archives: dict[str, zipfile.ZipFile],
) -> dict[str, Any]:
    storage_path = row["storage_path"]
    if row["storage_type"] == "zip":
        archive = archives.get(storage_path)
        if archive is None:
            archive = zipfile.ZipFile(storage_path)
            archives[storage_path] = archive
        return json.loads(archive.read(row["replay_path"]))
    return json.loads(
        (Path(storage_path) / row["replay_path"]).read_text(
            encoding="utf-8",
        )
    )


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _example(value: Any) -> Any:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return value[:80]
    if isinstance(value, list):
        return {"length": len(value)}
    if isinstance(value, dict):
        return {"keys": sorted(value)[:20]}
    return str(value)[:80]


def _record(
    value: Any,
    path: str,
    counts: Counter[str],
    kinds: dict[str, Counter[str]],
    examples: dict[str, list[Any]],
) -> None:
    counts[path] += 1
    kinds[path][_kind(value)] += 1
    rendered = _example(value)
    if rendered not in examples[path] and len(examples[path]) < 8:
        examples[path].append(rendered)
    if isinstance(value, dict):
        for key, child in value.items():
            _record(
                child,
                f"{path}.{key}",
                counts,
                kinds,
                examples,
            )
    elif isinstance(value, list):
        for child in value:
            _record(
                child,
                f"{path}[]",
                counts,
                kinds,
                examples,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.index.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if row["team_name"] == "Majkel1337"
    ][: args.max_episodes]

    counts: Counter[str] = Counter()
    kinds: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[Any]] = defaultdict(list)
    select_types: Counter[int] = Counter()
    option_types: Counter[int] = Counter()
    decisions = 0
    archives: dict[str, zipfile.ZipFile] = {}
    try:
        for row in rows:
            replay = _read_replay(row, archives)
            seat = int(row["seat_index"])
            steps = replay.get("steps") or []
            for step_index, step in enumerate(steps[:-1]):
                if seat >= len(step) or seat >= len(steps[step_index + 1]):
                    continue
                record = step[seat] or {}
                observation = record.get("observation") or {}
                select = observation.get("select") or {}
                options = list(select.get("option") or [])
                recorded = (steps[step_index + 1][seat] or {}).get("action")
                if (
                    record.get("status") != "ACTIVE"
                    or int(select.get("context", -1)) != 0
                    or int(select.get("minCount") or 0) != 1
                    or int(select.get("maxCount") or 0) != 1
                    or len(options) < 2
                    or not isinstance(recorded, list)
                    or len(recorded) != 1
                    or not isinstance(recorded[0], int)
                    or not 0 <= recorded[0] < len(options)
                ):
                    continue
                decisions += 1
                select_types[int(select.get("type", -1))] += 1
                for option in options:
                    option_types[int(option.get("type", -1))] += 1
                _record(
                    observation,
                    "observation",
                    counts,
                    kinds,
                    examples,
                )
    finally:
        for archive in archives.values():
            archive.close()

    paths = []
    for path in sorted(counts):
        paths.append({
            "path": path,
            "observations": counts[path],
            "kinds": dict(kinds[path]),
            "examples": examples[path],
        })
    report = {
        "episodes": len(rows),
        "decisions": decisions,
        "select_types": dict(select_types),
        "option_types": dict(option_types),
        "paths": paths,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "episodes": len(rows),
        "decisions": decisions,
        "paths": len(paths),
        "select_types": dict(select_types),
        "option_types": dict(option_types),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
