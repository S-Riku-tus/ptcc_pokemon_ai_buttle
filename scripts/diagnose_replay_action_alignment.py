"""Print and quantify same-step versus next-step replay action alignment."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def _valid(action: object, option_count: int) -> bool:
    return (
        isinstance(action, list)
        and len(action) == 1
        and isinstance(action[0], int)
        and 0 <= action[0] < option_count
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    counts = {
        "decisions": 0,
        "current_valid": 0,
        "next_valid": 0,
        "both_valid": 0,
        "same_value": 0,
    }
    examples = []
    with zipfile.ZipFile(args.archive) as archive:
        names = [
            name
            for name in archive.namelist()
            if "/replay/episode_" in name
        ][:args.episodes]
        for name in names:
            replay = json.loads(archive.read(name))
            steps = replay.get("steps") or []
            for step_index, step in enumerate(steps[:-1]):
                for seat, record in enumerate(step):
                    if not record:
                        continue
                    observation = record.get("observation") or {}
                    select = observation.get("select") or {}
                    options = select.get("option") or []
                    if (
                        record.get("status") != "ACTIVE"
                        or int(select.get("context", -1)) != 0
                        or int(select.get("minCount") or 0) != 1
                        or int(select.get("maxCount") or 0) != 1
                        or len(options) < 2
                    ):
                        continue
                    current = record.get("action")
                    following = (
                        (steps[step_index + 1][seat] or {}).get("action")
                        if seat < len(steps[step_index + 1])
                        else None
                    )
                    current_valid = _valid(current, len(options))
                    next_valid = _valid(following, len(options))
                    counts["decisions"] += 1
                    counts["current_valid"] += int(current_valid)
                    counts["next_valid"] += int(next_valid)
                    counts["both_valid"] += int(current_valid and next_valid)
                    counts["same_value"] += int(
                        current_valid and next_valid and current == following
                    )
                    if len(examples) < 30:
                        examples.append({
                            "episode": name,
                            "step": step_index,
                            "seat": seat,
                            "options": len(options),
                            "current": current,
                            "next": following,
                            "current_valid": current_valid,
                            "next_valid": next_valid,
                        })
    print(json.dumps({
        "counts": counts,
        "rates": {
            key: value / max(counts["decisions"], 1)
            for key, value in counts.items()
            if key != "decisions"
        },
        "examples": examples,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
