"""Build the lossless replay-question memory for Alakazam v30."""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from audit_alakazam_v30_teacher_policy import _iter_decisions


def _resolved(
    groups: dict[str, Counter[str]],
    *,
    minimum_support: int,
) -> tuple[dict[str, str], int]:
    result = {}
    conflicts = 0
    for key, labels in groups.items():
        support = sum(labels.values())
        if support < minimum_support:
            continue
        if len(labels) != 1:
            conflicts += 1
            continue
        result[key] = labels.most_common(1)[0][0]
    return result, conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher_index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with args.teacher_index.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    exact_groups: dict[str, Counter[str]] = defaultdict(Counter)
    canonical_groups: dict[str, Counter[str]] = defaultdict(Counter)
    decisions = 0
    for decision in _iter_decisions(rows):
        decisions += 1
        exact_groups[decision["exact_key"]][decision["semantic"]] += 1
        canonical_groups[
            decision["canonical_key"]
        ][decision["semantic"]] += 1

    exact, exact_conflicts = _resolved(
        exact_groups,
        minimum_support=1,
    )
    canonical, canonical_conflicts = _resolved(
        canonical_groups,
        minimum_support=2,
    )
    payload = {
        "format": "v30_teacher_memory_v1",
        "exact": exact,
        "canonical_repeated": canonical,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    compressed = zlib.compress(encoded, level=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(compressed)

    report = {
        "teacher_index": str(args.teacher_index.resolve()),
        "trajectories": len(rows),
        "decisions": decisions,
        "exact_keys": len(exact),
        "exact_conflicting_keys_excluded": exact_conflicts,
        "canonical_repeated_keys": len(canonical),
        "canonical_conflicting_keys_excluded": canonical_conflicts,
        "uncompressed_bytes": len(encoded),
        "compressed_bytes": len(compressed),
        "compression_ratio": len(compressed) / max(1, len(encoded)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
