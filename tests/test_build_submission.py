import json
import tarfile
from pathlib import Path

from scripts.build_submission import build

ROOT = Path(__file__).resolve().parents[1]


def test_build_with_fake_cg(tmp_path):
    fake_cg = tmp_path / "cg"
    fake_cg.mkdir()
    (fake_cg / "api.py").write_text("# fake cg api\n", encoding="utf-8")

    output = tmp_path / "submission.tar.gz"
    build(
        ROOT / "agents" / "alakazam" / "alakazam_ml_v2",
        output,
        fake_cg,
    )

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())

    assert "main.py" in names
    assert "deck.csv" in names
    assert "cg/api.py" in names


def test_ml_agent_submission_excludes_training_artifacts(tmp_path):
    fake_cg = tmp_path / "cg"
    fake_cg.mkdir()
    (fake_cg / "api.py").write_text("# fake cg api\n", encoding="utf-8")

    output = tmp_path / "submission_ml.tar.gz"
    build(
        ROOT / "agents" / "alakazam" / "alakazam_ml_v2",
        output,
        fake_cg,
    )

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())

    assert {"main.py", "deck.csv", "cg/api.py", "ranker_model.json"} <= names
    forbidden_fragments = ("dataset", "joblib", "reports/", "data_processed", "README.md")
    assert not any(any(fragment in name for fragment in forbidden_fragments) for name in names)


def test_v32_submission_excludes_inherited_tests_and_reports(tmp_path):
    fake_cg = tmp_path / "cg"
    fake_cg.mkdir()
    (fake_cg / "api.py").write_text("# fake cg api\n", encoding="utf-8")

    output = tmp_path / "submission_v32.tar.gz"
    build(
        ROOT / "agents" / "alakazam" / "alakazam_ml_v32",
        output,
        fake_cg,
    )

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())

    assert {
        "main.py",
        "deck.csv",
        "cg/api.py",
        "ranker_model.json",
        "teacher_memory.bin",
    } <= names
    assert not any(
        Path(name).name.startswith(
            ("test_", "ANALYSIS_", "CHANGELOG_", "VALIDATION_REPORT_")
        )
        for name in names
    )


def test_v33_submission_respects_selector_adoption_gate(tmp_path):
    fake_cg = tmp_path / "cg"
    fake_cg.mkdir()
    (fake_cg / "api.py").write_text("# fake cg api\n", encoding="utf-8")

    output = tmp_path / "submission_v33.tar.gz"
    build(
        ROOT / "agents" / "alakazam" / "alakazam_ml_v33",
        output,
        fake_cg,
    )

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())

    selector_path = (
        ROOT / "agents" / "alakazam" / "alakazam_ml_v33"
        / "selector_model.json"
    )
    if selector_path.exists():
        selector_enabled = bool(
            json.loads(
                selector_path.read_text(encoding="utf-8")
            ).get("enabled")
        )
        base_members = {
            name for name in names
            if Path(name).name.startswith("selector_base_")
        }
        assert bool(base_members) == selector_enabled


def test_shared_policy_is_bundled_for_compact_agent(tmp_path):
    fake_cg = tmp_path / "cg"
    fake_cg.mkdir()
    (fake_cg / "api.py").write_text("# fake cg api\n", encoding="utf-8")

    output = tmp_path / "grimmsnarl_submission.tar.gz"
    build(
        ROOT / "agents" / "grimmsnarl" / "marnies_grimmsnarl_ex_v1",
        output,
        fake_cg,
    )

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())

    assert {"main.py", "policy_base.py", "deck.csv", "cg/api.py"} <= names
