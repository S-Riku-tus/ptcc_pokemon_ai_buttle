"""Formally promote a reviewed Challenger to a NEW agent directory.

This is the ONLY script that copies a Challenger's files, and it is intentionally
separate from evaluation. Safety guarantees:

  * Default mode is --dry-run: it prints the plan and changes nothing.
  * --apply copies the Challenger into a NEW agents/<new-agent-name>/ directory.
  * It NEVER overwrites an existing Champion or agent directory.
  * It NEVER edits ranker_model.json in place, updates champion config,
    runs git (commit/tag/push), deletes old agents, or submits to Kaggle.

Examples:
  python scripts/promote_challenger.py \
    --report artifacts/champion_challenger/<run>/promotion_report.json \
    --new-agent-name alakazam_ml_v4 --dry-run

  python scripts/promote_challenger.py \
    --report artifacts/champion_challenger/<run>/promotion_report.json \
    --new-agent-name alakazam_ml_v4 --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import cc_core  # noqa: E402
from cc_core import ROOT  # noqa: E402

VALID_NAME = __import__("re").compile(r"^[A-Za-z0-9_]+$")


def load_report(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan_promotion(report: dict, new_agent_name: str, allow_non_promote: bool) -> dict:
    """Validate the promotion request and return a plan dict (no side effects)."""
    meta = report.get("meta", {})
    judgement = report.get("judgement", {})
    verdict = judgement.get("verdict")
    challenger = meta.get("challenger")

    problems: list[str] = []
    if not challenger:
        problems.append("report has no challenger name")
    if not VALID_NAME.match(new_agent_name):
        problems.append(f"invalid --new-agent-name {new_agent_name!r} (use [A-Za-z0-9_])")
    if verdict != cc_core.VERDICT_PROMOTE and not allow_non_promote:
        problems.append(
            f"verdict is {verdict!r}, not {cc_core.VERDICT_PROMOTE!r}; "
            "re-run with --allow-non-promote to override after human review"
        )

    source_dir = None
    if challenger:
        try:
            source_dir = cc_core.resolve_agent_dir(challenger)
        except FileNotFoundError:
            problems.append(f"challenger agent directory not found: {challenger}")

    destination = ROOT / "agents" / new_agent_name
    if destination.exists():
        problems.append(f"destination already exists (refusing to overwrite): {destination}")

    return {
        "challenger": challenger,
        "verdict": verdict,
        "source_dir": str(source_dir) if source_dir else None,
        "destination": str(destination),
        "problems": problems,
    }


def apply_promotion(plan: dict) -> None:
    source = Path(plan["source_dir"])
    destination = Path(plan["destination"])
    if destination.exists():
        raise FileExistsError(destination)  # defensive; plan already checked
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    metadata_path = destination / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        metadata["name"] = destination.name
        metadata["role"] = "champion"
        metadata["promoted_from"] = plan["challenger"]
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--new-agent-name", required=True)
    parser.add_argument("--apply", action="store_true", help="Actually copy files (default is dry-run).")
    parser.add_argument("--dry-run", action="store_true", help="Explicit dry-run (default behaviour).")
    parser.add_argument(
        "--allow-non-promote",
        action="store_true",
        help="Permit promoting a report whose verdict is not PROMOTE_RECOMMENDED.",
    )
    args = parser.parse_args(argv)

    if not args.report.exists():
        print(f"Report not found: {args.report}", file=sys.stderr)
        return 2

    report = load_report(args.report)
    plan = plan_promotion(report, args.new_agent_name, args.allow_non_promote)

    print("== Promotion plan ==")
    print(f"challenger:  {plan['challenger']}")
    print(f"verdict:     {plan['verdict']}")
    print(f"source:      {plan['source_dir']}")
    print(f"destination: {plan['destination']}")

    if plan["problems"]:
        print("\nBlocked:")
        for problem in plan["problems"]:
            print(f"  - {problem}")
        return 1

    apply = args.apply and not args.dry_run
    if not apply:
        print("\nDRY-RUN: no files changed. Re-run with --apply to copy the Challenger.")
        return 0

    apply_promotion(plan)
    print(f"\nAPPLIED: copied Challenger into {plan['destination']}")
    print("Champion left unchanged. No git/tag/push/Kaggle actions were performed.")
    print("Next (human, manual): review the new agent, run validate_agent.py, then git as you see fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
