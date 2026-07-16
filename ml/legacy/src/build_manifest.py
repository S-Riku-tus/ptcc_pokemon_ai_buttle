from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import load_config, resolve_workspace_path
from ml.core.manifest import build_deck_clusters, build_manifest as build_expanded_manifest


def _zip_paths(config: dict[str, Any]) -> list[Path]:
    configured = config.get("replay_roots")
    if configured:
        paths: set[Path] = set()
        for value in configured:
            root = resolve_workspace_path(value)
            if root.is_file() and root.suffix.lower() == ".zip":
                paths.add(root.resolve())
            elif root.exists():
                paths.update(path.resolve() for path in root.rglob("*.zip"))
        return sorted(paths)
    replay_glob = resolve_workspace_path(config["replay_glob"])
    return sorted(path.resolve() for path in replay_glob.parent.glob(replay_glob.name))


def _original_compatibility_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    aliases = {
        "team": "target_team",
        "source_zip": "zip_path",
        "source_member": "replay_path",
        "target_seat": "target_seat",
        "deck_60": "initial_deck_json",
        "is_alakazam": "alakazam_line",
        "usable": "usable_manifest",
        "outcome": None,
    }
    for destination, source in aliases.items():
        if source is None:
            result[destination] = result.apply(
                lambda row: "win" if bool(row.get("target_win")) else "loss" if bool(row.get("target_loss")) else "draw",
                axis=1,
            )
        elif destination not in result:
            result[destination] = result[source]
    result["replay_available"] = result["replay_path"].astype(str).str.len().gt(0)
    result["normal_end"] = result["target_reward"].notna() & result["usable_manifest"].fillna(False)
    result["timeout"] = False
    result["error"] = False
    result["duplicate"] = result.duplicated(["submission_id", "episode_id", "target_seat"], keep="first")
    result.loc[result["duplicate"], "usable_manifest"] = False
    result.loc[result["duplicate"], "usable"] = False
    result.loc[result["duplicate"], "exclusion_reason"] = "duplicate_trajectory"
    if "decision_count" not in result:
        result["decision_count"] = 0
    if "legal_candidate_count" not in result:
        result["legal_candidate_count"] = 0
    result["deck_evidence"] = "replay_initial_deck"
    return result


def build_manifest(
    config: dict[str, Any], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    zip_paths = _zip_paths(config)
    if not zip_paths:
        raise FileNotFoundError(
            "No replay ZIPs found. Place them under data/runs/kaggle_top50 or set replay_roots."
        )
    expanded, stats, _reference = build_expanded_manifest(zip_paths, output_dir)
    manifest = _original_compatibility_columns(expanded)
    clusters = build_deck_clusters(manifest, output_dir / "deck_clusters.csv")

    stats = dict(stats)
    stats.update({
        "episode_count": int(manifest["episode_id"].nunique()),
        "alakazam_episode_count": int(manifest["alakazam_line"].fillna(False).sum()),
        "usable_episode_count": int(manifest.loc[manifest["usable_manifest"] == True, "episode_id"].nunique()),
        "usable_trajectory_count": int(manifest["usable_manifest"].fillna(False).sum()),
        "usable_decision_count": 0,
        "missing_rank_21_50": False,
        "seat_ambiguity_is_excluded": True,
        "replay_layouts_supported": ["replay", "replays"],
    })
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    manifest.to_csv(output_dir / "episode_manifest.csv", index=False)
    (output_dir / "manifest_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest, clusters, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument(
        "--output", default=str(Path(__file__).resolve().parents[1] / "data_processed")
    )
    args = parser.parse_args()
    _, _, stats = build_manifest(load_config(args.config), Path(args.output))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
