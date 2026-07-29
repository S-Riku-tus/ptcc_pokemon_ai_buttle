from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_expected_repository_structure():
    expected = [
        ROOT / "agents" / "alakazam" / "alakazam_ml_v31" / "main.py",
        ROOT / "agents" / "alakazam" / "alakazam_ml_v31" / "deck.csv",
        ROOT / "scripts" / "build_submission.py",
        ROOT / "scripts" / "validate_agent.py",
        ROOT / "kaggle" / "create_submission_from_git.py",
    ]
    for path in expected:
        assert path.exists(), path
