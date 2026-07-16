from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_MARKERS = ("pyproject.toml", "scripts", "agents")


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root without relying on the current directory."""
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in REPO_MARKERS):
            return candidate
    raise FileNotFoundError(
        f"Could not find repo root from {current}. Expected markers: {', '.join(REPO_MARKERS)}"
    )


def resolve_config_path(
    value: str | Path | None = None,
    *,
    archetype: str = "alakazam",
    repo_root: str | Path | None = None,
) -> Path:
    root = Path(repo_root).resolve() if repo_root else find_repo_root()
    if value is None:
        path = root / "ml" / "configs" / f"{archetype}.json"
    else:
        raw = Path(value)
        path = raw if raw.is_absolute() else (Path.cwd() / raw)
        if not path.exists():
            path = root / raw
    if not path.exists():
        raise FileNotFoundError(f"ML config was not found: {path}")
    return path.resolve()


def load_json_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_data_path(
    value: str | Path,
    *,
    config_path: str | Path | None = None,
    repo_root: str | Path | None = None,
    must_exist: bool = False,
) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        path = raw
    else:
        root = Path(repo_root).resolve() if repo_root else find_repo_root()
        config_base = Path(config_path).resolve().parent if config_path else None
        config_candidate = config_base / raw if config_base else None
        path = config_candidate if config_candidate and config_candidate.exists() else root / raw
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Configured data path does not exist: {path}")
    return path


def resolve_agent_path(
    value: str | Path,
    *,
    repo_root: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    raw = Path(value)
    root = Path(repo_root).resolve() if repo_root else find_repo_root()
    candidates = [raw if raw.is_absolute() else root / raw]
    if not raw.is_absolute():
        candidates.append(root / "agents" / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    path = candidates[-1].resolve()
    if must_exist:
        raise FileNotFoundError(f"Agent directory was not found for {value!r}; checked {candidates}")
    return path


def locate_card_metadata(
    *,
    explicit: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Path]:
    """Locate training-time card metadata, separate from official Kaggle cg bundling."""
    root = Path(repo_root).resolve() if repo_root else find_repo_root()
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([root / "vendor" / "cg", root / "cg"])

    checked: list[Path] = []
    for base in candidates:
        path = base if base.is_absolute() else root / base
        path = path.resolve()
        checked.append(path)
        cards = path / "cards.json"
        attacks = path / "attacks.json"
        if cards.exists() and attacks.exists():
            return {"root": path, "cards": cards, "attacks": attacks}

    formatted = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Card metadata for ML training was not found. Expected cards.json and attacks.json. "
        "This is training metadata and is intentionally separate from the official cg/ copied "
        f"into Kaggle submissions.\nChecked:\n{formatted}"
    )

