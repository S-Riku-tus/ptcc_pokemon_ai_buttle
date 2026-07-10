"""Copy an existing versioned agent to a new agent directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()

    source = ROOT / "agents" / args.source
    destination = ROOT / "agents" / args.destination

    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)

    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    metadata_path = destination / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["name"] = args.destination
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
            lines[0] = f"# {args.destination}"
        strategy_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Created: {destination}")
    print("Next:")
    print(f"  git switch -c feature/{args.destination}")
    print(f"  edit agents/{args.destination}/main.py")
    print(f"  python scripts/validate_agent.py --agent {args.destination}")


if __name__ == "__main__":
    main()
