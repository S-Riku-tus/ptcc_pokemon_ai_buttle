from __future__ import annotations

from pathlib import Path

import pytest

from ml.core.paths import (
    find_repo_root,
    locate_card_metadata,
    resolve_agent_path,
    resolve_config_path,
    resolve_data_path,
)


ROOT = Path(__file__).resolve().parents[1]


def test_find_repo_root_from_nested_ml_file() -> None:
    assert find_repo_root(ROOT / "ml" / "core" / "paths.py") == ROOT


def test_resolve_config_and_data_paths() -> None:
    config = resolve_config_path(archetype="alakazam", repo_root=ROOT)
    assert config == ROOT / "ml" / "configs" / "alakazam.json"
    assert resolve_data_path("data/ml/alakazam/processed", config_path=config, repo_root=ROOT) == (
        ROOT / "data" / "ml" / "alakazam" / "processed"
    )


def test_resolve_runtime_agent_path() -> None:
    assert resolve_agent_path("alakazam_ml_v2_expanded", repo_root=ROOT) == (
        ROOT / "agents" / "alakazam_ml_v2_expanded"
    )


def test_missing_card_metadata_error_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cards.json and attacks.json"):
        locate_card_metadata(explicit=tmp_path, repo_root=tmp_path)

