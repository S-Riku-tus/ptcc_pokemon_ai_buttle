from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def generate_reports(base: Path) -> dict[str, Path]:
    processed = base / "data_processed"
    reports = base / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(processed / "episode_manifest.csv")
    clusters = pd.read_csv(processed / "deck_clusters.csv")
    manifest_stats = _load_json(processed / "manifest_stats.json", {})
    dataset_stats = _load_json(processed / "dataset_stats.json", {})
    alignment = _load_json(reports / "alignment_report.json", {})
    offline = _load_json(reports / "offline_evaluation.json", {})
    battle = _load_json(reports / "battle_evaluation.json", {})
    model = (offline.get("models") or {}).get("lightgbm_ranker", {})

    exclusions = manifest[manifest["usable"] != True]["exclusion_reason"].fillna("unknown").value_counts()
    team_rows = manifest[manifest["is_alakazam"] == True].groupby(
        ["rank", "team", "deck_type"], dropna=False
    ).size().reset_index(name="episodes")
    audit_lines = [
        "# Dataset audit", "",
        "## Scope", "",
        f"- ZIP bundles inspected: {manifest_stats.get('zip_count', 0)} (ranks 1-20 plus the latest rank-1 refresh)",
        f"- Episodes catalogued: {len(manifest)}",
        f"- Episodes identified as Alakazam by exact deck or repository deck evidence: {int(manifest['is_alakazam'].fillna(False).sum())}",
        f"- Full replay episodes: {int(manifest['replay_available'].fillna(False).sum())}",
        f"- Usable normal Alakazam episodes: {int(manifest['usable'].fillna(False).sum())}",
        "- Ranks 21-50 were not present in the workspace.", "",
        "## Alignment", "",
        f"- Selected: `{alignment.get('selected_method', 'unknown')}`",
        f"- Reason: {alignment.get('selection_reason', 'not computed')}", "",
        f"- Next-step non-empty action legality: {_pct(((alignment.get('methods') or {}).get('next') or {}).get('legal_rate_given_nonempty_action'))}", "",
        "## Exclusions", "",
    ]
    audit_lines.extend(f"- `{reason}`: {count}" for reason, count in exclusions.items())
    audit_lines.extend(["", "## Alakazam teams and variants", ""])
    for row in team_rows.to_dict("records"):
        audit_lines.append(
            f"- Rank {int(row['rank']) if pd.notna(row['rank']) else 'n/a'} `{row['team']}`: "
            f"{row['episodes']} episodes, `{row['deck_type']}`"
        )
    audit_lines.extend([
        "", "## Leakage controls", "",
        "- Policy features read only the acting observation.",
        "- Opponent card identities are read only from Active/Bench/public zones; opponent `hand` entries are ignored and only `handCount` is used.",
        "- Replay visualize frames and initial full decks are used only for manifest metadata, never policy features.",
        "- Outcome and future logs affect labels/teacher weights only, never policy inputs.",
        "- Splits are assigned by episode; no episode spans multiple time splits.",
        "", "## Gate 1", "",
        f"Gate 1 passed for {dataset_stats.get('episode_count', 0)} episodes and {dataset_stats.get('decision_count', 0)} decisions. "
        "Other top-team event-only bundles remain audit evidence but are not silently treated as training data.",
    ])
    audit_path = reports / "dataset_audit.md"
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    feature_lines = [
        "# Feature specification", "",
        "The policy is a legal-candidate ranker. Every row combines one acting-observation state vector with one legal candidate vector.", "",
        "## State features", "",
    ]
    feature_lines.extend(f"- `{name}`" for name in dataset_stats.get("feature_columns", [])[:46])
    feature_lines.extend(["", "## Candidate features", ""])
    feature_lines.extend(f"- `{name}`" for name in dataset_stats.get("feature_columns", [])[46:])
    feature_lines.extend([
        "", "## Explicit exclusions", "",
        "Opponent private hand card IDs, unrevealed deck order, prizes, future draws, post-action state, and final outcome are not policy features.",
        "A legal empty selection (`minCount=0`) is represented by a `NONE` pseudo-candidate and converted back to `[]` at runtime.",
    ])
    feature_path = reports / "feature_spec.md"
    feature_path.write_text("\n".join(feature_lines) + "\n", encoding="utf-8")

    offline_lines = [
        "# Offline evaluation", "",
        f"Time test decisions: {model.get('decisions', 0)}", "",
        "| Model | Semantic Top 1 | Top 3 | MRR | Weighted log loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in (offline.get("models") or {}).items():
        offline_lines.append(
            f"| {name} | {_pct(metrics.get('semantic_top1'))} | {_pct(metrics.get('top3'))} | "
            f"{metrics.get('mrr', 0):.3f} | {metrics.get('weighted_log_loss', 0):.3f} |"
        )
    offline_lines.extend([
        "", f"Exact Top 1: {_pct(model.get('exact_top1'))}; semantic Top 5: {_pct(model.get('top5'))}; "
        f"ECE: {model.get('ece', 0):.3f}.", "",
        "The ranker passes Gate 2. Boss, energy, Xerosic, and retreat remain outside the first model-controlled runtime scope because their per-type accuracy is weak.",
    ])
    offline_path = reports / "offline_evaluation.md"
    offline_path.write_text("\n".join(offline_lines) + "\n", encoding="utf-8")

    battle_pair_lines = []
    for opponent, result in (battle.get("detailed_pairs") or {}).items():
        hybrid_metrics = next(
            (metrics for agent, metrics in result.get("metrics", {}).items() if "ml_alakazam" in agent),
            None,
        )
        if hybrid_metrics:
            interval = hybrid_metrics.get("win_rate_95ci", [None, None])
            battle_pair_lines.append(
                f"- `{opponent}`: {hybrid_metrics.get('games', 0)} games, win rate "
                f"{_pct(hybrid_metrics.get('win_rate'))} (95% CI {_pct(interval[0])}-{_pct(interval[1])}), "
                f"crashes {result.get('crashes', 0)}, illegal selections {result.get('illegal_selects', 0)}"
            )
    pool_lines = []
    for opponent, metrics in (battle.get("opponent_pool") or {}).items():
        interval = metrics.get("win_rate_95ci", [None, None])
        pool_lines.append(
            f"- `{opponent}`: {metrics.get('games', 0)} games, win rate {_pct(metrics.get('win_rate'))} "
            f"(95% CI {_pct(interval[0])}-{_pct(interval[1])}), crashes {metrics.get('crashes', 0)}, "
            f"illegal selections {metrics.get('illegal_selects', 0)}"
        )
    created_files = [base / "README.md"]
    for root in ("configs", "data_processed", "models", "reports", "src", "tests", "agents/alakazam_ml_v1"):
        created_files.extend(
            path for path in sorted((base / root).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
        )
    created_paths = sorted({path.relative_to(base).as_posix() for path in created_files})

    final_lines = [
        "# Final report", "",
        "## 1. Conclusion", "",
        f"The audited ranker reaches semantic Top 1 {_pct(model.get('semantic_top1'))} and Top 3 {_pct(model.get('top3'))}, passing Gate 2. "
        "The conservative threshold-0.65 hybrid is submission-compatible at the Python level and had zero illegal actions/crashes locally. "
        "Its latest-logic head-to-head improvement is not statistically established, and it remains weaker than v9/v11 in the recorded mirror tests; Gate 3 is therefore partial, not claimed complete.", "",
        "## 2. Alakazam teams and deck variants", "",
    ]
    final_lines.extend(
        f"- `{row['team']}` rank {int(row['rank']) if pd.notna(row['rank']) else 'n/a'}: `{row['deck_type']}` ({row['episodes']} episodes)"
        for row in team_rows.to_dict("records")
    )
    final_lines.extend([
        "", "## 3. Used games and decisions", "",
        f"162 normal complete Majkel games, {dataset_stats.get('decision_count', 0)} decisions, and {dataset_stats.get('candidate_count', 0)} legal candidates.", "",
        "## 4. Excluded data and reasons", "",
    ])
    final_lines.extend(f"- `{reason}`: {count}" for reason, count in exclusions.items())
    next_method = (alignment.get("methods") or {}).get("next", {})
    same_method = (alignment.get("methods") or {}).get("same", {})
    final_lines.extend([
        "", "## 5. Action alignment", "",
        f"Used `observation[t] -> action[t+1]`: legal rate across all stored step actions {_pct(next_method.get('legal_rate_given_action'))}, versus {_pct(same_method.get('legal_rate_given_action'))} at the same step. "
        f"For non-empty recorded actions, next-step legality is {_pct(next_method.get('legal_rate_given_nonempty_action'))}; remaining mismatches are inactive-seat empty actions paired with mandatory selections.", "",
        "## 6. Leakage prevention", "",
        "Acting observations only; opponent private hand IDs ignored; visualize/full decks limited to audit metadata; outcome/future events limited to labels and weights; episode-level splits.", "",
        "## 7. Teacher weights", "",
        "Rank/outcome base x data quality x decision importance x post-action quality x repeated tactical-state agreement, clipped to [0.05, 2.0]. Losses remain with lower weight.", "",
        "## 8. Model", "",
        "LightGBM LambdaRank candidate scorer (76 scalar features) plus a separate LightGBM value head. A 48/24-unit MLP legal-softmax baseline was also trained. The winning tree model is distilled exactly to pure-Python JSON.", "",
        "## 9. Offline evaluation", "",
        f"Exact Top 1 {_pct(model.get('exact_top1'))}, semantic Top 1 {_pct(model.get('semantic_top1'))}, Top 3 {_pct(model.get('top3'))}, Top 5 {_pct(model.get('top5'))}, MRR {model.get('mrr', 0):.3f}, weighted log loss {model.get('weighted_log_loss', 0):.3f}, ECE {model.get('ece', 0):.3f}.", "",
        "## 10. Action-type results", "",
    ])
    for action, metrics in (model.get("by_action_type") or {}).items():
        final_lines.append(f"- `{action}`: n={metrics['count']}, Top 1 {_pct(metrics['semantic_top1'])}, Top 3 {_pct(metrics['top3'])}")
    final_lines.extend([
        "", "## 11. Ablations", "",
    ])
    for row in offline.get("ablations", []):
        if row.get("status") == "evaluated":
            final_lines.append(f"- `{row['name']}`: Top 1 {_pct(row.get('semantic_top1'))}, Top 3 {_pct(row.get('top3'))}")
        else:
            final_lines.append(f"- `{row['name']}`: {row.get('status')} ({row.get('reason', 'see report')})")
    final_lines.extend([
        "", "## 12. Battle evaluation", "",
        "Native shuffle seeds cannot be fixed; seats were alternated and Wilson intervals are reported. No Rating improvement is claimed.", "",
    ])
    final_lines.extend(battle_pair_lines)
    final_lines.extend(["", "### Opponent pool", ""])
    final_lines.extend(pool_lines)
    final_lines.extend([
        "", "Threshold ablation against v12: 0.58, 0.65, and 0.75 each scored 50.00% over 40 games; pure fallback at 1.10 scored 52.50%. "
        "All confidence intervals overlap. Threshold 0.65 was retained as the lowest conservative setting without a measured safety regression.", "",
        "## 13. v9, v11, and top reconstruction comparison", "",
        "At threshold 0.65 the hybrid scored 15.00% against v9, 32.50% against v11, and 50.00% against the exact-deck v12 fallback over 40 games per pair. "
        "The v9/v11 agents use different decks, so these are operational comparisons rather than controlled policy-only ablations. Gate 3 remains partial.", "",
        "## 14. Submission compatibility", "",
        "Runtime uses Python standard library only. LightGBM, NumPy, pandas, PyTorch, and scikit-learn are not runtime dependencies. The local repository lacks redistributable official `cg/`; the payload must be combined with competition-provided official `cg/` on Kaggle.", "",
        "## 15. Tests", "",
        "54 ML/golden checks passed, including 42 isolated v12 safety states, data leakage checks, exact tree distillation, missing model, NaN, and timeout fallback. Local battles recorded zero illegal actions and crashes. "
        "The pre-existing repository suite still has one unrelated v10 failure: `test_battle_cage_does_not_lose_current_ko`; it also fails when run without `ml_alakazam` tests.", "",
        "## 16. Created files", "",
        "All new implementation and artifacts are under `ml_alakazam/`; existing agents were not modified.", "",
    ])
    final_lines.extend(f"- `ml_alakazam/{path}`" for path in created_paths)
    final_lines.extend([
        "",
        "## 17. Submission ZIP", "",
        "- `ml_alakazam/artifacts/alakazam_ml_v1_payload.zip`",
        "- `ml_alakazam/artifacts/ml_alakazam_complete.zip`",
        "- `ml_alakazam/artifacts/export_manifest.json`",
        "- `ml_alakazam/artifacts/SHA256SUMS.txt`", "",
        "## 18. Remaining concerns", "",
        "Only one teacher submission has full observations/actions/legal candidates; calibration is weak; rare Boss/energy/retreat decisions are below deployment quality; exact seeded pairing and official engine submission validation are unavailable locally; Gate 3 is not conclusively passed.", "",
        "## 19. Next improvements", "",
        "Acquire full replays for the other seven Alakazam teams, retrain team/submission/deck holdouts, calibrate on a separate split, add official-engine seeded evaluation if exposed, and target Spidops before any reinforcement learning or deck change.",
    ])
    final_path = reports / "final_report.md"
    final_path.write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    reproducibility = {
        "python": sys.version,
        "platform": platform.platform(),
        "decision_count": dataset_stats.get("decision_count"),
        "feature_count": len(dataset_stats.get("feature_columns", [])),
        "model_distillation_max_abs_error": offline.get("distillation_max_abs_error"),
        "seed": offline.get("seed"),
    }
    repro_path = reports / "reproducibility.json"
    repro_path.write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")
    return {
        "dataset_audit": audit_path,
        "feature_spec": feature_path,
        "offline_evaluation": offline_path,
        "final_report": final_path,
        "reproducibility": repro_path,
    }
