"""Kaggle-importable script that builds submissions directly from GitHub.

Before use, replace REPO_URL with your GitHub repository URL.
Add the Pokemon TCG AI Battle Simulation competition data to Notebook Inputs.

Builds one submission_<agent>.tar.gz per active agent in AGENTS.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/S-Riku-tus/ptcc_pokemon_ai_buttle.git"
BRANCH = "main"
AGENTS = ["alakazam741_v5"]

WORKING = Path("/kaggle/working")
REPO_DIR = WORKING / "ptcc_pokemon_ai_buttle"


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


def clone_url() -> str:
    """Private repos: store a GitHub token as a Kaggle Secret named GITHUB_TOKEN
    (Notebook: Add-ons -> Secrets -> Attach), and it is injected here."""
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        if token:
            return REPO_URL.replace("https://", f"https://{token}@", 1)
    except Exception:
        pass
    return REPO_URL


def main() -> None:
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
            clone_url(),
            str(REPO_DIR),
        ],
        check=True,
    )

    cg_source = locate_cg()
    for agent in AGENTS:
        output = WORKING / f"submission_{agent}.tar.gz"
        subprocess.run(
            [
                sys.executable,
                str(REPO_DIR / "scripts" / "build_submission.py"),
                "--agent",
                agent,
                "--cg-source",
                str(cg_source),
                "--output",
                str(output),
            ],
            check=True,
        )
        print(f"\nSubmission ready: {output} ({output.stat().st_size} bytes)")


main()
