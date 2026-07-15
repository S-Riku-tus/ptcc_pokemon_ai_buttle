from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import zipfile
from collections import Counter
from pathlib import Path


RUNTIME_FILES = [
    "main.py", "fallback_v12.py", "policy_base.py", "ml_runtime.py", "ml_features.py",
    "common_runtime.py", "deck.csv", "ranker_model.json", "metadata.json", "README.md",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_agent(agent_dir: Path) -> None:
    for name in ("main.py", "deck.csv", "ranker_model.json"):
        if not (agent_dir / name).exists():
            raise FileNotFoundError(agent_dir / name)
    for path in agent_dir.glob("*.py"):
        py_compile.compile(str(path), doraise=True)
    deck = [int(value) for value in (agent_dir / "deck.csv").read_text(encoding="utf-8-sig").split()]
    if len(deck) != 60:
        raise ValueError("deck must contain 60 cards")
    counts = Counter(deck)
    if any(count > 4 for card, count in counts.items() if card not in range(1, 9)):
        raise ValueError("deck exceeds four-copy limit")
    model = json.loads((agent_dir / "ranker_model.json").read_text(encoding="utf-8"))
    if model.get("format") != "lightgbm_tree_v1" or not model.get("trees"):
        raise ValueError("invalid distilled model")


def export(base: Path) -> dict[str, str | int | bool]:
    agent_dir = base / "agents" / "alakazam_ml_v1"
    output_dir = base / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_agent(agent_dir)
    payload = output_dir / "alakazam_ml_v1_payload.zip"
    package_manifest = {
        "agent": "alakazam_ml_v1",
        "official_cg_included": False,
        "official_cg_required": True,
        "reason": "The repository contains only a local compatibility shim; Kaggle must inject competition-provided official cg/.",
        "runtime_dependencies": ["Python standard library", "competition cg"],
    }
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in RUNTIME_FILES:
            archive.write(agent_dir / name, name)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(package_manifest, indent=2))
    with zipfile.ZipFile(payload) as archive:
        names = set(archive.namelist())
        if not {"main.py", "deck.csv", "ranker_model.json", "PACKAGE_MANIFEST.json"}.issubset(names):
            raise RuntimeError("payload structure validation failed")
        if any("__pycache__" in name or name.endswith((".pyc", ".parquet")) for name in names):
            raise RuntimeError("payload contains forbidden generated files")

    complete = output_dir / "ml_alakazam_complete.zip"
    allowed_roots = {"agents", "configs", "data_processed", "models", "reports", "src", "tests"}
    with zipfile.ZipFile(complete, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(base)
            if relative.parts[0] not in allowed_roots:
                continue
            if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            archive.write(path, Path("ml_alakazam") / relative)
        archive.write(base / "README.md", "ml_alakazam/README.md")
    checksums = output_dir / "SHA256SUMS.txt"
    checksums.write_text(
        f"{_sha256(payload)}  {payload.name}\n{_sha256(complete)}  {complete.name}\n",
        encoding="ascii",
    )
    result = {
        "payload": str(payload), "payload_sha256": _sha256(payload),
        "complete": str(complete), "complete_sha256": _sha256(complete),
        "official_cg_included": False,
        "payload_entries": len(zipfile.ZipFile(payload).namelist()),
        "complete_entries": len(zipfile.ZipFile(complete).namelist()),
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(export(args.base), indent=2))


if __name__ == "__main__":
    main()

