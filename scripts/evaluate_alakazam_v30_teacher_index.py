"""Evaluate an agent on the unified v30 teacher-forced problem set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(
    0,
    str(ROOT / "agents" / "alakazam" / "alakazam_ml_v30"),
)

from agent_loader import load_dir_agent  # noqa: E402
from teacher_memory import semantic_action_key  # noqa: E402


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


def _rates(counter: Counter[str]) -> dict[str, int | float | None]:
    decisions = counter["decisions"]
    return {
        "decisions": decisions,
        "exact": (
            counter["exact"] / decisions if decisions else None
        ),
        "semantic": (
            counter["semantic"] / decisions if decisions else None
        ),
        "errors": counter["errors"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher_index", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.teacher_index.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    if args.max_trajectories is not None:
        rows = rows[: args.max_trajectories]

    agent, _, main_module = load_dir_agent(args.agent_dir.resolve())
    aggregate: Counter[str] = Counter()
    by_context: dict[str, Counter[str]] = defaultdict(Counter)
    by_cohort: dict[str, Counter[str]] = defaultdict(Counter)
    archives: dict[str, zipfile.ZipFile] = {}
    for row in rows:
        agent({"select": None})
        replay = _read_replay(row, archives)
        seat = int(row["seat_index"])
        steps = replay.get("steps") or []
        for step_index, step in enumerate(steps[:-1]):
            if seat >= len(step) or seat >= len(steps[step_index + 1]):
                continue
            record = step[seat] or {}
            observation = record.get("observation") or {}
            select = observation.get("select")
            if select is None or record.get("status") != "ACTIVE":
                continue
            recorded = (
                steps[step_index + 1][seat] or {}
            ).get("action")
            if (
                not isinstance(recorded, list)
                or len(recorded) == 60
            ):
                continue
            context = str(int(select.get("context", -1)))
            cohort = row["source_cohort"]
            aggregate["decisions"] += 1
            by_context[context]["decisions"] += 1
            by_cohort[cohort]["decisions"] += 1
            try:
                predicted = list(agent(observation))
            except Exception:
                aggregate["errors"] += 1
                by_context[context]["errors"] += 1
                by_cohort[cohort]["errors"] += 1
                continue
            exact = predicted == recorded
            semantic = (
                semantic_action_key(
                    observation,
                    predicted,
                )
                == semantic_action_key(
                    observation,
                    recorded,
                )
            )
            for counter in (
                aggregate,
                by_context[context],
                by_cohort[cohort],
            ):
                counter["exact"] += int(exact)
                counter["semantic"] += int(semantic)

    for archive in archives.values():
        archive.close()
    report = {
        "teacher_index": str(args.teacher_index.resolve()),
        "agent_dir": str(args.agent_dir.resolve()),
        "trajectories": len(rows),
        "metrics": _rates(aggregate),
        "by_context": {
            context: _rates(counter)
            for context, counter in sorted(
                by_context.items(),
                key=lambda item: int(item[0]),
            )
        },
        "by_cohort": {
            cohort: _rates(counter)
            for cohort, counter in sorted(by_cohort.items())
        },
        "diagnostics": main_module.diag_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "trajectories": len(rows),
        "metrics": report["metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
