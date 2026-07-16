from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def generate_reports(base: Path) -> dict[str, Path]:
    processed, reports, models = base / "data_processed", base / "reports", base / "models"
    reports.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(processed / "episode_manifest.csv")
    clusters = pd.read_csv(processed / "deck_clusters.csv")
    manifest_stats = _load_json(processed / "manifest_stats.json", {})
    dataset_stats = _load_json(processed / "dataset_stats.json", {})
    offline = _load_json(reports / "offline_evaluation.json", {})
    ablations = _load_json(reports / "ablation_metrics.json", {})
    legacy = _load_json(reports / "legacy_vs_expanded.json", {})
    schema = _load_json(models / "model_schema.json", {})
    smoke = _load_json(reports / "replay_policy_smoke.json", {})

    audit_lines = [
        "# Dataset audit", "",
        "## Recovered replay layouts", "",
        f"- ZIPs: {manifest_stats.get('zip_count', 0)}",
        f"- Full replay files: {manifest_stats.get('full_replay_count', 0)}",
        f"- Singular `replay/`: {manifest_stats.get('legacy_singular_replay_count', 0)}",
        f"- Newly recovered plural `replays/`: {manifest_stats.get('plural_replays_recovered', 0)}",
        f"- Usable trajectories: {manifest_stats.get('usable_trajectory_count', 0)}",
        f"- Decisions: {dataset_stats.get('usable_decision_count', dataset_stats.get('decision_count', 0))}",
        f"- Legal candidates: {dataset_stats.get('candidate_row_count', dataset_stats.get('candidate_count', 0))}",
        "", "## Seat resolution", "",
    ]
    audit_lines.extend(
        f"- `{method}`: {count}" for method, count in (manifest_stats.get("seat_methods") or {}).items()
    )
    audit_lines.extend([
        "", "Ambiguous seats are excluded rather than guessed.",
        "", "## Leakage controls", "",
        "- Policy features use only the acting observation and a supplied legal candidate.",
        "- Opponent private hand identities, initial full decks, outcome, future logs, and visualize data are not policy features.",
        "- Initial decks and outcome are used only for deck clustering, sample weights, and audit reports.",
        "- Labels are the exact legal-option indices serialized on the same seat at replay step t+1.",
    ])
    audit_path = reports / "dataset_audit.md"
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    feature_lines = [
        "# Feature specification", "",
        f"The legal-candidate ranker uses {len(dataset_stats.get('feature_columns', []))} features.",
        "The major additions over v1 are route readiness, hand-to-damage interactions, KO preservation, "
        "target energy/HP, low-deck risk, and explicit Boss/Hammer/Xerosic/Retreat/Energy interactions.",
        "", "## Feature names", "",
    ]
    feature_lines.extend(f"- `{name}`" for name in dataset_stats.get("feature_columns", []))
    feature_path = reports / "feature_spec.md"
    feature_path.write_text("\n".join(feature_lines) + "\n", encoding="utf-8")

    lines = [
        "# Final report — expanded Alakazam imitation ranker", "",
        "## Conclusion", "",
        "The original project design was retained: exact legal-option imitation learning, episode-level leakage control, "
        "a distilled dependency-free tree runtime, and a deterministic fallback. The critical parser bug was fixed by "
        "supporting both `replay/episode_*.json` and `replays/episode_*.json`.", "",
        "The expanded corpus improves broad generalization, but Boss, Retreat, Xerosic, and Hammer remain unsafe for "
        "direct ML control. They are therefore hard-routed to the v12 fallback; Energy is ML-controlled only at high confidence.", "",
        "## Data recovery", "",
        f"- Full replay count: {manifest_stats.get('full_replay_count', 0)}",
        f"- Recovered plural-path replays: {manifest_stats.get('plural_replays_recovered', 0)}",
        f"- Usable trajectories: {manifest_stats.get('usable_trajectory_count', 0)}",
        f"- Decisions: {dataset_stats.get('usable_decision_count', 0)}",
        f"- Candidate rows: {dataset_stats.get('candidate_row_count', 0)}",
        f"- Teams / submissions / decks: {dataset_stats.get('team_count', 0)} / "
        f"{dataset_stats.get('submission_count', 0)} / {dataset_stats.get('deck_count', 0)}", "",
        "## Deck clusters", "",
    ]
    for row in clusters.to_dict("records"):
        lines.append(
            f"- `{row.get('deck_type')}` distance {row.get('majkel_distance')}: "
            f"{row.get('trajectory_count', row.get('episode_count'))} trajectories, "
            f"{row.get('teams', '')}"
        )
    lines.extend(["", "## Holdout evaluation", "",
                  "| Holdout | Top 1 | Top 3 | MRR | ECE | Fallback |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, metric in offline.items():
        lines.append(
            f"| {name} | {_pct(metric.get('top1'))} | {_pct(metric.get('top3'))} | "
            f"{float(metric.get('mrr', 0)):.3f} | {float(metric.get('ece', 0)):.3f} | "
            f"{_pct(metric.get('fallback_rate'))} |"
        )
    lines.extend(["", "## Weight ablation", ""])
    for name, metric in ablations.items():
        lines.append(
            f"- `{name}`: Top1 {_pct(metric.get('top1'))}, Top3 {_pct(metric.get('top3'))}, "
            f"MRR {float(metric.get('mrr', 0)):.3f}"
        )
    if legacy:
        lines.extend(["", "## Singular-only versus expanded corpus", ""])
        for name, metric in legacy.items():
            lines.append(
                f"- `{name}`: Top1 {_pct(metric.get('top1'))}, Top3 {_pct(metric.get('top3'))}, "
                f"MRR {float(metric.get('mrr', 0)):.3f}"
            )
    lines.extend([
        "", "## Runtime safety policy", "",
        f"- Default probability threshold: {schema.get('fallback_probability', 0.55)}",
        f"- Margin threshold: {schema.get('fallback_margin', 0.12)}",
        "- Boss / Retreat / Xerosic / Hammer: always fallback",
        "- Energy: ML only when probability is at least 0.85 and the margin gate also passes",
        "- A fallback-confirmed immediate KO is never overridden",
        "- Dudunsparce self-removal is blocked when it would leave no body or a critically low deck",
        "- Nested target/search selections and multi-select decisions remain fallback-controlled",
        "", "## Validation", "",
        f"- Replay-policy smoke illegal actions: {smoke.get('illegal_action_count', 'not rerun')}",
        f"- Replay-policy smoke exceptions: {smoke.get('exception_count', 'not rerun')}",
        "- Distilled runtime supports both numeric and LightGBM categorical tree splits.",
        "- Actual Kaggle Rating improvement is not claimed without official-engine ladder evaluation.",
        "", "## Remaining risks", "",
        "- The rank49 Jack replay bundle was not included.",
        "- Focus-action expert weighting improved rare actions but materially reduced global Top1, so it was rejected.",
        "- Deck/rank/outcome weighting was only weakly supported by ablation; weights are intentionally mild.",
        "- Battle smoke requires the repository's official `vendor/cg` and opponent agents.",
    ])
    final_path = reports / "final_report.md"
    final_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reproducibility = {
        "python": sys.version,
        "platform": platform.platform(),
        "decision_count": dataset_stats.get("usable_decision_count"),
        "candidate_count": dataset_stats.get("candidate_row_count"),
        "feature_count": len(dataset_stats.get("feature_columns", [])),
        "model_estimators": schema.get("n_estimators"),
        "label_provenance": dataset_stats.get("label_provenance"),
    }
    repro_path = reports / "reproducibility.json"
    repro_path.write_text(json.dumps(reproducibility, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dataset_audit": audit_path,
        "feature_spec": feature_path,
        "final_report": final_path,
        "reproducibility": repro_path,
    }
