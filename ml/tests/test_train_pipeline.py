from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from ml.pipelines.train_archetype import _discover_zip_paths


def test_explicit_missing_replay_root_refuses_partial_retrain(tmp_path: Path):
    present = tmp_path / "present.zip"
    with ZipFile(present, "w"):
        pass
    missing = tmp_path / "moved_top21_40"

    with pytest.raises(FileNotFoundError, match="refusing a partial-corpus retrain") as exc:
        _discover_zip_paths(
            {"replay_roots": [str(present), str(missing)]},
            tmp_path / "config.json",
            tmp_path,
        )

    assert str(missing) in str(exc.value)


def test_explicit_existing_replay_roots_are_discovered(tmp_path: Path):
    bundle = tmp_path / "bundle.zip"
    with ZipFile(bundle, "w"):
        pass
    paths = _discover_zip_paths(
        {"replay_roots": [str(bundle)]},
        tmp_path / "config.json",
        tmp_path,
    )
    assert paths == [bundle.resolve()]
