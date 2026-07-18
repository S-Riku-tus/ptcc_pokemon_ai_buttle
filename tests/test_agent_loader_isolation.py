from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import agent_loader  # noqa: E402


def _write_agent(root: Path, name: str, deck_id: int) -> Path:
    agent_dir = root / name
    agent_dir.mkdir()
    (agent_dir / "policy_base.py").write_text("", encoding="ascii")
    (agent_dir / "helper.py").write_text(
        f"DECK = [{deck_id}] * 60\n",
        encoding="ascii",
    )
    (agent_dir / "main.py").write_text(
        "import helper\n\n"
        "def agent(observation):\n"
        "    return list(helper.DECK)\n",
        encoding="ascii",
    )
    return agent_dir


def test_bare_local_modules_are_isolated_between_agents(tmp_path):
    first_dir = _write_agent(tmp_path, "first_agent", 101)
    second_dir = _write_agent(tmp_path, "second_agent", 202)

    try:
        first, _, _ = agent_loader.load_dir_agent(first_dir)
        second, _, _ = agent_loader.load_dir_agent(second_dir)

        assert first({"select": None}) == [101] * 60
        assert second({"select": None}) == [202] * 60
        assert Path(sys.modules["helper"].__file__).parent == second_dir
    finally:
        sys.modules.pop("helper", None)
