"""Copy an existing versioned agent to a new agent directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _resolve_source(spec: str) -> Path:
    """Find an existing agent dir by bare name or <pokemon>/<agent> spec."""
    direct = ROOT / "agents" / spec
    if direct.is_dir():
        return direct
    agents_root = ROOT / "agents"
    for group in sorted(agents_root.iterdir(), key=lambda p: p.name):
        nested = group / spec
        if group.is_dir() and nested.is_dir():
            return nested
    return direct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()

    # Agents are grouped one level deep by main Pokemon
    # (agents/<pokemon>/<agent>). Accept either a bare name or a
    # "<pokemon>/<agent>" spec for both source and destination.
    source = _resolve_source(args.source)
    destination = ROOT / "agents" / args.destination

    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    dest_name = destination.name

    metadata_path = destination / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["name"] = dest_name
        metadata["version"] = "0.1.0"
        metadata["status"] = "development"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    strategy_path = destination / "STRATEGY.md"
    if strategy_path.exists():
        text = strategy_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if lines and lines[0].startswith("# "):
            lines[0] = f"# {dest_name}"
        strategy_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Created: {destination}")
    print("Next:")
    print(f"  git switch -c feature/{dest_name}")
    print(f"  edit agents/{args.destination}/main.py")
    print(f"  python scripts/validate_agent.py --agent {dest_name}")


if __name__ == "__main__":
    main()
