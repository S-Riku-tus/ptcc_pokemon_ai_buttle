"""Build a Kaggle submission.tar.gz from a versioned agent directory."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

try:
    from scripts.validate_agent import resolve_agent_dir, validate_agent
except ModuleNotFoundError:
    from validate_agent import resolve_agent_dir, validate_agent

ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_NAMES = {
    "__pycache__",
    "STRATEGY.md",
    "STRATEGY_V7.md",
    "CHANGELOG.md",
    "metadata.json",
    "README.md",
}
FORBIDDEN_ARCHIVE_PARTS = {
    "data_processed",
    "processed",
    "reports",
    "models",
    "tests",
}
FORBIDDEN_ARCHIVE_SUFFIXES = {
    ".joblib",
    ".parquet",
    ".csv",
    ".gz",
    ".zip",
}


def locate_cg(explicit: str | None) -> Path:
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit))

    candidates.extend([
        ROOT / "cg",
        ROOT / "vendor" / "cg",
        Path("/kaggle/input/pokemon-tcg-ai-battle/sample_submission/sample_submission/cg"),
        Path("/kaggle/input/pokemon-tcg-ai-battle/sample_submission/cg"),
    ])

    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.glob("**/sample_submission/cg"))
        candidates.extend(
            path for path in input_root.rglob("cg")
            if path.is_dir() and (path / "api.py").exists()
        )

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir() and (resolved / "api.py").exists():
            api_text = (resolved / "api.py").read_text(encoding="utf-8", errors="ignore")
            if "compatibility shim" in api_text:
                # vendor/cg is the local-testing shim, never the official cg.
                continue
            return resolved

    checked = "\n".join(f"  - {path}" for path in list(seen)[:30])
    raise FileNotFoundError(
        "Official cg/ was not found. Add Simulation competition data, "
        "copy cg to vendor/cg, or pass --cg-source.\n"
        f"Checked:\n{checked}"
    )


def should_copy(path: Path) -> bool:
    if any(part in EXCLUDED_NAMES for part in path.parts):
        return False
    if path.name.startswith("."):
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    return True


def copy_runtime_files(agent_dir: Path, stage: Path) -> None:
    for source in agent_dir.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(agent_dir)
        if not should_copy(relative):
            continue
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build(agent_dir: Path, output: Path, cg_source: Path) -> None:
    validate_agent(agent_dir)

    with tempfile.TemporaryDirectory(prefix="ptcg_submission_") as tmp:
        stage = Path(tmp)
        copy_runtime_files(agent_dir, stage)
        shutil.copytree(cg_source, stage / "cg")

        if output.exists():
            output.unlink()
        output.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(output, "w:gz") as archive:
            for source in sorted(stage.rglob("*")):
                archive.add(source, arcname=source.relative_to(stage), recursive=False)

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())

    required = {"main.py", "deck.csv", "cg/api.py"}
    missing = required.difference(names)
    if missing:
        raise RuntimeError(f"Archive validation failed; missing {sorted(missing)}")
    forbidden = sorted(
        name for name in names
        if name != "deck.csv"
        and (
            any(part in FORBIDDEN_ARCHIVE_PARTS for part in Path(name).parts)
            or Path(name).suffix.lower() in FORBIDDEN_ARCHIVE_SUFFIXES
        )
    )
    if forbidden:
        preview = forbidden[:20]
        raise RuntimeError(f"Archive validation failed; forbidden training artifacts: {preview}")

    print(f"Created: {output}")
    print(f"Agent: {agent_dir.name}")
    print(f"Official cg: {cg_source}")
    print(f"Archive entries: {len(names)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        default="alakazam_ml_v2_expanded",
        help="Agent dir name or path. Checks direct path, agents/, archive/agents/, and data/runs/.",
    )
    parser.add_argument("--cg-source")
    parser.add_argument(
        "--output",
        default=str(ROOT / "artifacts" / "submission.tar.gz"),
    )
    args = parser.parse_args()

    agent_dir = resolve_agent_dir(args.agent)

    cg_source = locate_cg(args.cg_source)
    build(agent_dir, Path(args.output).resolve(), cg_source)


if __name__ == "__main__":
    main()
