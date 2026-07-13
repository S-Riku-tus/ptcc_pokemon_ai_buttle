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


def resolve_agent_dir(spec: str) -> Path:
    direct = Path(spec)
    if direct.is_dir():
        return direct.resolve()

    for base in (
        ROOT / "agents",
        ROOT / "archive" / "agents",
        ROOT / "data" / "runs",
    ):
        candidate = base / spec
        if candidate.is_dir():
            return candidate.resolve()

    runs_dir = ROOT / "data" / "runs"
    if runs_dir.is_dir():
        for candidate in runs_dir.iterdir():
            if not candidate.is_dir():
                continue
            if candidate.name == spec or candidate.name.endswith(f"_{spec}"):
                return candidate.resolve()
            metadata_path = candidate / "metadata.json"
            if not metadata_path.exists():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if metadata.get("name") == spec:
                return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve agent directory from {spec!r}. "
        "Checked direct path, agents/, archive/agents/, and data/runs/ "
        "(including metadata.json name aliases)."
    )


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
    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assigned_names = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)
    if "agent" not in function_names and "agent" not in assigned_names:
        raise ValueError(f"{path}: top-level agent function or callable assignment was not found")


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

    agent_dir = resolve_agent_dir(args.agent)
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
