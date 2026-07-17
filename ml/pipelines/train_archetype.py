from __future__ import annotations

import argparse
import zipfile
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.archetypes import get_archetype
from ml.core.paths import (
    find_repo_root,
    load_json_config,
    resolve_agent_path,
    resolve_config_path,
    resolve_data_path,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_with_archive(source: Path, destination: Path, archive_dir: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    before = _sha256(destination) if destination.exists() else None
    after = _sha256(source)
    archived = None
    changed = before != after
    if destination.exists() and changed:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = archive_dir / f"{destination.stem}_{stamp}{destination.suffix}"
        shutil.copy2(destination, archived)
    shutil.copy2(source, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "changed": changed,
        "previous_sha256": before,
        "new_sha256": after,
        "archived_previous": str(archived) if archived else None,
    }


def _archive_existing_models(models_dir: Path) -> list[str]:
    archive_dir = models_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived: list[str] = []
    for name in ("ranker_model.json", "ranker.txt", "ranker.joblib", "model_schema.json"):
        source = models_dir / name
        if not source.exists():
            continue
        destination = archive_dir / f"{source.stem}_{stamp}{source.suffix}"
        shutil.copy2(source, destination)
        archived.append(str(destination))
    return archived


def _discover_zip_paths(config: dict[str, Any], config_path: Path, repo_root: Path) -> list[Path]:
    from ml.core.replay_io import discover_zip_paths

    roots: list[Path] = []
    for value in config.get("replay_roots", []):
        roots.append(resolve_data_path(value, config_path=config_path, repo_root=repo_root))
    replay_glob = config.get("replay_glob")
    if replay_glob:
        pattern = resolve_data_path(replay_glob, config_path=config_path, repo_root=repo_root)
        roots.extend(pattern.parent.glob(pattern.name))
    return discover_zip_paths(roots)


def _required_processed_files(processed: Path) -> list[Path]:
    return [
        processed / "manifest.csv",
        processed / "manifest_stats.json",
        processed / "decisions.csv",
        processed / "dataset_stats.json",
        processed / "dataset_rows.csv.gz",
    ]


def _assert_processed_ready(processed: Path) -> None:
    archive = processed.with_suffix(".zip")
    if not processed.exists() and archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(processed.parent)
    missing = [path for path in _required_processed_files(processed) if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Processed ML data is incomplete:\n{formatted}")


def _archive_processed(processed: Path, *, remove_original: bool) -> dict[str, Any] | None:
    if not processed.exists():
        return None
    archive = processed.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    file_count = sum(1 for path in processed.rglob("*") if path.is_file())
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=processed.parent, base_dir=processed.name)
    if remove_original:
        shutil.rmtree(processed)
    return {
        "archive": str(archive),
        "file_count": file_count,
        "archive_bytes": archive.stat().st_size,
        "removed_original": remove_original,
    }


def _validate_runtime(agent_dir: Path) -> dict[str, Any]:
    from scripts.validate_agent import validate_agent

    return validate_agent(agent_dir)


def _update_runtime_metadata(
    agent_dir: Path,
    *,
    processed_dir: Path,
    model_sha256: str,
    mode: str,
    replay_zip_count: int,
) -> dict[str, Any] | None:
    metadata_path = agent_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stats_path = processed_dir / "dataset_stats.json"
    manifest_stats_path = processed_dir / "manifest_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        metadata["model_training_decisions"] = int(stats.get("usable_decision_count", 0))
        metadata["model_candidate_rows"] = int(stats.get("candidate_row_count", 0))
        metadata["training_teams"] = int(stats.get("team_count", metadata.get("training_teams", 0)))
        metadata["training_submissions"] = int(stats.get("submission_count", metadata.get("training_submissions", 0)))
        metadata["training_decks"] = int(stats.get("deck_count", metadata.get("training_decks", 0)))
    if manifest_stats_path.exists():
        manifest_stats = json.loads(manifest_stats_path.read_text(encoding="utf-8"))
        metadata["training_replay_files"] = int(
            manifest_stats.get("full_replay_count", manifest_stats.get("replay_file_count", 0))
        )
        metadata["training_trajectories"] = int(manifest_stats.get("usable_trajectory_count", 0))
    metadata["last_training_mode"] = mode
    metadata["last_replay_zip_count"] = replay_zip_count
    metadata["ranker_model_sha256"] = model_sha256
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = report_path.with_suffix(".md")
    lines = [
        "# ML pipeline report",
        "",
        f"- archetype: `{payload['archetype']}`",
        f"- mode: `{payload['mode']}`",
        f"- processed_dir: `{payload['processed_dir']}`",
        f"- models_dir: `{payload['models_dir']}`",
        f"- reports_dir: `{payload['reports_dir']}`",
        f"- runtime_agent_dir: `{payload['runtime_agent_dir']}`",
        f"- training_ran: `{payload['training_ran']}`",
        f"- runtime_validated: `{payload['runtime_validated']}`",
    ]
    if payload.get("model_copy"):
        copy = payload["model_copy"]
        lines.extend([
            "",
            "## Model Copy",
            "",
            f"- changed: `{copy['changed']}`",
            f"- archived_previous: `{copy.get('archived_previous')}`",
        ])
    if payload.get("notes"):
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in payload["notes"])
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = find_repo_root()
    archetype = get_archetype(args.archetype)
    config_path = resolve_config_path(args.config, archetype=archetype.archetype_name, repo_root=repo_root)
    config = load_json_config(config_path)

    processed = resolve_data_path(config["processed_dir"], config_path=config_path, repo_root=repo_root)
    models = resolve_data_path(config["models_dir"], config_path=config_path, repo_root=repo_root)
    reports = resolve_data_path(config["reports_dir"], config_path=config_path, repo_root=repo_root)
    runtime_agent = resolve_agent_path(config.get("runtime_agent_dir", archetype.runtime_agent_dir), repo_root=repo_root)

    notes: list[str] = []
    mode = "reuse_processed" if args.reuse_processed else "raw_replay"
    zip_paths: list[Path] = []

    if not args.reuse_processed:
        zip_paths = _discover_zip_paths(config, config_path, repo_root)
        if zip_paths:
            import pandas as pd  # noqa: F401 - imported here to fail only when rebuilding data
            from ml.core.dataset import build_dataset
            from ml.core.manifest import build_deck_clusters, build_manifest

            manifest, manifest_stats, _reference = build_manifest(zip_paths, processed)
            build_deck_clusters(manifest, processed / "deck_clusters.csv")
            _rows, _decisions, dataset_stats = build_dataset(
                manifest,
                processed,
                load_rows=False,
                workers=int(config.get("dataset_workers", 4)),
            )
            matrix_dir = processed / "matrix"
            if matrix_dir.exists():
                shutil.rmtree(matrix_dir)
            notes.append(
                f"rebuilt processed data from {len(zip_paths)} replay ZIPs; "
                f"{dataset_stats.get('usable_decision_count')} decisions"
            )
        else:
            mode = "reuse_processed_no_raw_replays"
            notes.append("raw replay ZIPs were not found; reused processed data instead")
            _assert_processed_ready(processed)
    else:
        _assert_processed_ready(processed)

    training_summary: dict[str, Any] | None = None
    archived_model_files: list[str] = []
    training_ran = False
    if not args.skip_training:
        from ml.core.ranker_training import train_all

        archived_model_files = _archive_existing_models(models)
        training_summary = train_all(processed, models, reports)
        training_ran = True
    else:
        notes.append("training skipped by --skip-training")

    model_path = models / "ranker_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Distilled runtime model was not found: {model_path}")

    copy_result = _copy_with_archive(
        model_path,
        runtime_agent / "ranker_model.json",
        models / "archive",
    )
    metadata = _update_runtime_metadata(
        runtime_agent,
        processed_dir=processed,
        model_sha256=copy_result["new_sha256"],
        mode=mode,
        replay_zip_count=len(zip_paths),
    )

    runtime_validation: dict[str, Any] | None = None
    runtime_validated = False
    if not args.skip_runtime_validation:
        runtime_validation = _validate_runtime(runtime_agent)
        runtime_validated = True
    else:
        notes.append("runtime validation skipped by --skip-runtime-validation")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "archetype": archetype.archetype_name,
        "mode": mode,
        "config_path": str(config_path),
        "processed_dir": str(processed),
        "models_dir": str(models),
        "reports_dir": str(reports),
        "runtime_agent_dir": str(runtime_agent),
        "replay_zip_count": len(zip_paths),
        "training_ran": training_ran,
        "archived_model_files": archived_model_files,
        "training_summary": training_summary,
        "model_copy": copy_result,
        "runtime_metadata": metadata,
        "runtime_validated": runtime_validated,
        "runtime_validation": runtime_validation,
        "notes": notes,
        "submission_generation": "Use scripts/build_submission.py; the ML pipeline does not create Kaggle archives.",
    }
    archive_cfg = config.get("archive_processed_after_training", {})
    if archive_cfg and training_ran:
        payload["processed_archive"] = _archive_processed(
            processed,
            remove_original=bool(archive_cfg.get("remove_original", False)),
        )
    _write_report(reports / "pipeline_report.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archetype", default="alakazam")
    parser.add_argument("--config")
    parser.add_argument("--reuse-processed", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-runtime-validation", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
