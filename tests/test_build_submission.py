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
        ROOT / "agents" / "alakazam741_v2",
        output,
        fake_cg,
    )

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())

    assert "main.py" in names
    assert "deck.csv" in names
    assert "cg/api.py" in names
