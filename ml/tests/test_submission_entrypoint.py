from __future__ import annotations

import ast
from pathlib import Path


def _base() -> Path:
    return Path(__file__).resolve().parents[2]


def test_main_agent_is_final_top_level_statement() -> None:
    path = _base() / "agents" / "alakazam_ml_v2_expanded" / "main.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    statements = [node for node in tree.body if not (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )]
    assert isinstance(statements[-1], ast.FunctionDef)
    assert statements[-1].name == "agent"


def test_submission_runtime_files_exist() -> None:
    agent = _base() / "agents" / "alakazam_ml_v2_expanded"
    required = {
        "main.py", "fallback_v12.py", "policy_base.py", "ml_runtime.py",
        "ml_features.py", "common_runtime.py", "ranker_model.json", "deck.csv",
    }
    assert required <= {path.name for path in agent.iterdir() if path.is_file()}
