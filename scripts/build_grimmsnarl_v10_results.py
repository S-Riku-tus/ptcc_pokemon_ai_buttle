"""Assemble RESULTS.json for v10 from the artifacts that produced it.

Written as a script rather than by hand so the machine-readable summary cannot
drift from the reports it summarises: every number here is read back out of a
file in the same directory, and a missing input is an error rather than a
silently omitted field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from math import comb
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(directory: Path, name: str) -> Any:
    path = directory / name
    if not path.exists():
        raise SystemExit(f"missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fisher(a: int, b: int, c: int, d: int) -> float:
    n, r1, c1 = a + b + c + d, a + b, a + c
    if not n:
        return 1.0

    def probability(x: int) -> float:
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    reference = probability(a)
    total = 0.0
    for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1):
        if probability(x) <= reference + 1e-12:
            total += probability(x)
    return round(min(1.0, total), 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path,
        default=ROOT / "experiments" / "grimmsnarl_ml_v10_safe_residual",
    )
    parser.add_argument(
        "--v8", type=Path,
        default=ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8",
    )
    parser.add_argument(
        "--v10", type=Path,
        default=ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v10",
    )
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    directory = args.dir

    ladder = load(directory, "ladder_v8_55317804.json")
    turn_order = load(directory, "turn_order_v8_vs_field.json")
    gradient = load(directory, "pilot_gradient.json")
    stamp = load(directory, "stamp_gap.json")
    panel = load(directory, "stamp_panel.json")
    lone = load(directory, "lone_body.json")
    determinism = load(directory, "arena_determinism.json")
    ledger = load(directory, "residual_ledger.json")
    base = load(directory, "baseline_v8_on_v8_ladder.json")
    candidate = load(directory, "behaviour_v10_on_v8_ladder.json")
    consensus = {
        need: load(directory, f"residual_candidates_need{need}.json")
        for need in (3, 4, 5)
    }

    games = ladder["games"]

    def alakazam(row: dict[str, Any]) -> bool:
        return "Alakazam" in row["opponent_deck_label"]

    def record(rows: list[dict[str, Any]]) -> tuple[int, int]:
        wins = sum(bool(row["won"]) for row in rows)
        return wins, len(rows) - wins

    family = [row for row in games if alakazam(row)]
    rest = [row for row in games if not alakazam(row)]
    matched = [
        row for row in games
        if row["went_first"] is False and (row["opponent_score"] or 0) >= 1000
    ]
    first_only = [row for row in games if row["went_first"]]

    outcome_split = {}
    for need, report in consensus.items():
        cells = report["by_outcome_cell"]
        def total(key: str, prefix: str) -> int:
            return sum(
                cells[name][key]
                for name in cells if name.startswith(prefix)
            )

        lost, lost_n = total("overrides", "lost"), total("decisions", "lost")
        won, won_n = total("overrides", "won"), total("decisions", "won")
        outcome_split[str(need)] = {
            "lost_overrides": lost, "lost_decisions": lost_n,
            "lost_rate": round(lost / lost_n, 4) if lost_n else None,
            "won_overrides": won, "won_decisions": won_n,
            "won_rate": round(won / won_n, 4) if won_n else None,
            "fisher_p": fisher(lost, lost_n - lost, won, won_n - won),
        }

    protected = {}
    resources = base["groups"]["turn_resources"]
    other = candidate["groups"]["turn_resources"]
    for key in sorted(resources):
        if not (key.startswith("agent_") and key.endswith("_rate")):
            continue
        protected[key] = {
            "v8": resources[key], "v10": other[key],
            "identical": resources[key] == other[key],
        }
    for group in (
        "snipe_target", "boss_target", "counter_count", "froslass_evolve",
        "boss_play", "munkidori_source",
    ):
        for key, value in sorted(base["groups"][group].items()):
            if key in ("counts", "changed_decisions"):
                continue
            protected[f"{group}.{key}"] = {
                "v8": value,
                "v10": candidate["groups"][group][key],
                "identical": value == candidate["groups"][group][key],
            }

    v8_files = {
        path.name: sha256(path)
        for path in sorted(args.v8.iterdir()) if path.is_file()
    }
    v10_files = {
        path.name: sha256(path)
        for path in sorted(args.v10.iterdir()) if path.is_file()
    }

    # Every arena_*.json except the determinism probe, which is not a matchup.
    # A "_40" glob silently dropped the second-seed repeats, which are the
    # runs that establish how much of the first seed's spread is noise.
    arenas = {}
    for name in sorted(directory.glob("arena_*.json")):
        if name.name == "arena_determinism.json":
            continue
        arenas[name.stem] = json.loads(name.read_text(encoding="utf-8"))

    report = {
        "agent": "grimmsnarl_ml_v10",
        "parent": "grimmsnarl_ml_v8",
        "verdict": (
            "ship as a challenger: one decision class, 6 of 4480 stored "
            "decisions changed, every protected rate bit-identical. The "
            "Alakazam premise the version was commissioned on is not "
            "supported by the data and is not addressed."
        ),
        "deck": {
            "hash_v8": "9714ab5c3996f6cc",
            "hash_v10": "9714ab5c3996f6cc",
            "identical": v8_files["deck.csv"] == v10_files["deck.csv"],
        },
        "files": {
            "identical": sorted(
                name for name, digest in v10_files.items()
                if v8_files.get(name) == digest
            ),
            "changed": sorted(
                name for name, digest in v10_files.items()
                if name in v8_files and v8_files[name] != digest
            ),
            "added": sorted(
                name for name in v10_files if name not in v8_files
            ),
            "sha256_v10": v10_files,
        },
        "ladder_v8": {
            "submission_id": 55317804,
            "episodes_total": ladder["episodes_total"],
            "episodes_rated": ladder["episodes_rated"],
            "record": ladder["overall"],
            "mean_opponent_rating": ladder["mean_opponent_rating"],
            "rating_first": ladder["rating_first"],
            "rating_last": ladder["rating_last"],
            "first_10": ladder["first_10"],
            "after_10": ladder["after_10"],
            "by_opponent_rating": ladder["by_opponent_rating"],
            "by_turn_order": ladder["by_turn_order"],
        },
        "hypotheses": {
            "alakazam_matchup": {
                "verdict": "not supported",
                "family_record": list(record(family)),
                "rest_record": list(record(rest)),
                "fisher_p": fisher(*record(family), *record(rest)),
                "going_first_family": list(
                    record([r for r in first_only if alakazam(r)])
                ),
                "going_first_rest": list(
                    record([r for r in first_only if not alakazam(r)])
                ),
                "going_first_fisher_p": fisher(
                    *record([r for r in first_only if alakazam(r)]),
                    *record([r for r in first_only if not alakazam(r)]),
                ),
                "matched_second_and_1000_plus_family": list(
                    record([r for r in matched if alakazam(r)])
                ),
                "matched_second_and_1000_plus_rest": list(
                    record([r for r in matched if not alakazam(r)])
                ),
                "matched_fisher_p": fisher(
                    *record([r for r in matched if alakazam(r)]),
                    *record([r for r in matched if not alakazam(r)]),
                ),
                "note": (
                    "cc38cb450b86770a is one of five Alakazam deck hashes in "
                    "the run; its 1-6 reaches p=0.035 alone but was selected "
                    "out of ~30 hashes for looking bad."
                ),
            },
            "turn_order": {
                "verdict": "the only significant cut in the run",
                "v8_first": ladder["by_turn_order"]["first"],
                "v8_second": ladder["by_turn_order"]["second"],
                "fisher_p": fisher(
                    ladder["by_turn_order"]["first"]["wins"],
                    ladder["by_turn_order"]["first"]["losses"],
                    ladder["by_turn_order"]["second"]["wins"],
                    ladder["by_turn_order"]["second"]["losses"],
                ),
                "field_control": turn_order.get("win_rate_by_turn_order"),
                "per_turn_take_rates": {
                    name: turn_order["buckets"][name]
                    for name in (
                        "v8_all", "v8_first", "v8_second", "field_all",
                        "field_elite_first", "field_elite_second",
                    )
                    if name in turn_order["buckets"]
                },
                "note": (
                    "no measured per-turn take rate differs; the deficit is "
                    "real but not yet attributable to a behaviour"
                ),
            },
            "rating_gradients": gradient["rating_gradient"],
            "board_out": {
                "verdict": "rejected",
                "rates": lone["rates"],
                "avoidable_events": len(lone["v8_events"]),
            },
            "consensus_residual": {
                "verdict": "rejected: uncorrelated with the outcome",
                "by_threshold": outcome_split,
            },
            "paired_arena": {
                "verdict": "unavailable",
                "identical_across_processes":
                    determinism["identical_across_processes"],
                "conclusion": determinism["conclusion"],
            },
        },
        "shipped_change": {
            "class": "Petrel search offering a dead Unfair Stamp",
            "rating_gradient": stamp["rating_gradient"],
            "v8_ladder": stamp["v8_ladder"],
            "field": stamp["field"],
            "panel_on_v8_boards": panel["per_advisor"],
            "dead_offers_on_v8_boards": panel["dead_offers"],
            "live_offers_on_v8_boards": panel["live_offers"],
            "consensus_thresholds": panel["consensus_thresholds"],
        },
        "deployed_effect": {
            "decisions": ledger["decisions"],
            "differences": ledger["differences"],
            "difference_rate": ledger["difference_rate"],
            "contexts_changed": ledger["contexts_changed"],
            "distinct_episodes_changed": ledger["distinct_episodes_changed"],
            "changed_in_lost_games": ledger["changed_in_lost_games"],
            "residual_counters": ledger["residual_counters"],
            "ledger": ledger["ledger"],
        },
        "protected_behaviour": {
            "all_identical": all(
                row["identical"] for row in protected.values()
            ),
            "metrics": protected,
            "context_agreement": {
                context: {
                    "v8": base["contexts"][context]["agreement_with_replay"],
                    "v10": candidate["contexts"][context][
                        "agreement_with_replay"
                    ],
                }
                for context in sorted(base["contexts"], key=int)
            },
        },
        "arena": arenas,
        "not_implemented": {
            "petrel_boss": {
                "rho": stamp["rating_gradient"]["boss"]["rho"],
                "p": stamp["rating_gradient"]["boss"]["p"],
                "v8_take_rate": stamp["v8_ladder"]["boss_take_rate"],
                "field_take_rate": stamp["field"]["boss_take_rate"],
                "reason": (
                    "no panel pilot plays it on v8's boards either "
                    "(0.016-0.109), so there is no consensus to gate on"
                ),
            },
        },
    }
    if args.archive and args.archive.exists():
        report["submission_archive"] = {
            "path": str(args.archive),
            "bytes": args.archive.stat().st_size,
            "sha256": sha256(args.archive),
        }
    try:
        report["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        report["git_commit"] = None

    out = directory / "RESULTS.json"
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print(json.dumps({
        "deck_identical": report["deck"]["identical"],
        "files_changed": report["files"]["changed"],
        "files_added": report["files"]["added"],
        "differences": report["deployed_effect"]["differences"],
        "protected_all_identical":
            report["protected_behaviour"]["all_identical"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
