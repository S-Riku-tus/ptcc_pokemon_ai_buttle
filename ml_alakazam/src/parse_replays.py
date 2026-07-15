from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .common import resolve_workspace_path
from .replay_io import ReplayRef, legal_action, load_replay


def build_alignment_report(manifest: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    totals = {"same": Counter(), "next": Counter()}
    context_counts = Counter()
    for row in manifest[manifest["usable"] == True].to_dict("records"):
        ref = ReplayRef(
            resolve_workspace_path(row["source_zip"]), row["source_member"],
            int(row["episode_id"]), int(row["target_seat"]),
        )
        replay = load_replay(ref)
        steps = replay.get("steps") or []
        seat = ref.target_seat or 0
        for index in range(len(steps)):
            try:
                observation = steps[index][seat].get("observation") or {}
                select = observation.get("select") or {}
            except (IndexError, TypeError, AttributeError):
                continue
            if not select or not isinstance(select.get("option"), list):
                continue
            context_counts[str(select.get("context"))] += 1
            for name, shift in (("same", 0), ("next", 1)):
                if index + shift >= len(steps):
                    continue
                action = steps[index + shift][seat].get("action")
                totals[name]["observations"] += 1
                if isinstance(action, list):
                    totals[name]["action_present"] += 1
                    is_legal = legal_action(action, select)
                    if action:
                        totals[name]["nonempty_action"] += 1
                        if is_legal:
                            totals[name]["legal_nonempty_action"] += 1
                    if is_legal:
                        totals[name]["legal"] += 1
                    else:
                        totals[name]["illegal"] += 1
                    if not action:
                        totals[name]["empty_action"] += 1
                else:
                    totals[name]["missing_action"] += 1
    report: dict[str, Any] = {"methods": {}, "context_counts": dict(context_counts)}
    for name, counts in totals.items():
        present = counts["action_present"]
        report["methods"][name] = {
            **dict(counts),
            "legal_rate_given_action": counts["legal"] / present if present else 0.0,
            "legal_rate_given_nonempty_action": (
                counts["legal_nonempty_action"] / counts["nonempty_action"]
                if counts["nonempty_action"] else 0.0
            ),
        }
    next_rate = report["methods"]["next"]["legal_rate_given_action"]
    same_rate = report["methods"]["same"]["legal_rate_given_action"]
    report["selected_method"] = "observation[t] -> action[t+1]" if next_rate > same_rate else "observation[t] -> action[t]"
    report["selection_reason"] = (
        f"next-step legal rate {next_rate:.6f} exceeds same-step {same_rate:.6f} on normal usable episodes"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1]
    parser.add_argument("--manifest", default=str(base / "data_processed" / "episode_manifest.csv"))
    parser.add_argument("--output", default=str(base / "reports" / "alignment_report.json"))
    args = parser.parse_args()
    report = build_alignment_report(pd.read_csv(args.manifest), Path(args.output))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
