"""Kaggle-importable script that builds a submission directly from GitHub.

Before use, replace REPO_URL with your GitHub repository URL.
Add the Pokemon TCG AI Battle Simulation competition data to Notebook Inputs.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/YOUR_GITHUB_USERNAME/ptcg-ai-battle.git"
BRANCH = "main"
AGENT = "mega_lucario_v1"

WORKING = Path("/kaggle/working")
REPO_DIR = WORKING / "ptcg-ai-battle"
OUTPUT = WORKING / "submission.tar.gz"


def locate_cg() -> Path:
    candidates = [
        path
        for path in Path("/kaggle/input").rglob("cg")
        if path.is_dir() and (path / "api.py").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            "Official cg/ was not found. Add Simulation competition data "
            "to the Notebook Inputs."
        )
    candidates.sort(
        key=lambda path: (
            "sample_submission" not in str(path).lower(),
            len(str(path)),
        )
    )
    return candidates[0]


def main() -> None:
    if "YOUR_GITHUB_USERNAME" in REPO_URL:
        raise ValueError(
            "Edit REPO_URL at the top of this file before running it."
        )

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)

    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            BRANCH,
            REPO_URL,
            str(REPO_DIR),
        ],
        check=True,
    )

    cg_source = locate_cg()
    subprocess.run(
        [
            sys.executable,
            str(REPO_DIR / "scripts" / "build_submission.py"),
            "--agent",
            AGENT,
            "--cg-source",
            str(cg_source),
            "--output",
            str(OUTPUT),
        ],
        check=True,
    )

    print("\nSubmission ready:")
    print(OUTPUT)
    print("Size:", OUTPUT.stat().st_size, "bytes")


main()
