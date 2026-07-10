"""Statically validate an agent directory without importing cg."""

from __future__ import annotations

import argparse
import ast
import collections
import json
import py_compile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASIC_ENERGY_IDS = set(range(1, 9))


def read_deck(path: Path) -> list[int]:
    values = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(values) != 60:
        raise ValueError(f"{path}: deck must contain 60 IDs, got {len(values)}")
    return values


def validate_main(path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ptcg_validate_") as tmp:
        cfile = Path(tmp) / f"{path.stem}.pyc"
        py_compile.compile(str(path), cfile=str(cfile), doraise=True)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "agent" not in names:
        raise ValueError(f"{path}: top-level agent() function was not found")


def validate_agent(agent_dir: Path) -> dict:
    main_path = agent_dir / "main.py"
    deck_path = agent_dir / "deck.csv"

    if not main_path.exists():
        raise FileNotFoundError(main_path)
    if not deck_path.exists():
        raise FileNotFoundError(deck_path)

    validate_main(main_path)
    deck = read_deck(deck_path)
    counts = collections.Counter(deck)

    warnings: list[str] = []
    for card_id, count in sorted(counts.items()):
        if count > 4 and card_id not in BASIC_ENERGY_IDS:
            warnings.append(
                f"card ID {card_id} appears {count} times; verify deck legality"
            )

    metadata_path = agent_dir / "metadata.json"
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "agent_dir": str(agent_dir),
        "deck_size": len(deck),
        "unique_cards": len(counts),
        "metadata": metadata,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="alakazam741_v2")
    args = parser.parse_args()

    agent_dir = (ROOT / "agents" / args.agent).resolve()
    result = validate_agent(agent_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(" -", warning)
    else:
        print("\nValidation passed.")


if __name__ == "__main__":
    main()
