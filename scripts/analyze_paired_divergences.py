"""Replay discordant CRN pairs and expose their first policy divergence.

The gauntlet intentionally stores compact hashes by default.  This tool reads
one completed report, selects only seed/seat pairs where champion and
challenger produced opposite outcomes, reruns those exact games, and stores a
bounded public-information trace for the evaluated agent.  It never uses the
opponent's hidden hand or a future state as an input to either policy.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import Any

from paired_gauntlet import Job, _run_job, _worker_init


def _pair_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["opponent"]), int(row["seed"]), int(row["evaluated_seat"])


def _first_divergence(
    champion: list[dict[str, Any]], challenger: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for index, (left, right) in enumerate(zip(champion, challenger)):
        if left.get("observation_hash") != right.get("observation_hash"):
            return {
                "index": index,
                "kind": "state_before_evaluated_action",
                "champion": left,
                "challenger": right,
            }
        if left.get("action") != right.get("action"):
            return {
                "index": index,
                "kind": "evaluated_action",
                "champion": left,
                "challenger": right,
            }
    if len(champion) != len(challenger):
        return {
            "index": min(len(champion), len(challenger)),
            "kind": "trace_length",
            "champion": champion[-1] if champion else None,
            "challenger": challenger[-1] if challenger else None,
        }
    return None


def _choice_label(decision: dict[str, Any] | None) -> str:
    if not decision:
        return "none"
    chosen = decision.get("chosen") or []
    if not chosen:
        return "decline"
    return "+".join(
        f"t{int(item.get('type', -1))}:c{int(item.get('card_id', -1))}:"
        f"a{int(item.get('attack_id', -1))}"
        for item in chosen
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-limit", type=int, default=160)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = json.loads(args.report.read_text(encoding="utf-8"))
    config = source["config"]
    rows = [
        row for row in source.get("rows", [])
        if row.get("repeat") == 0 and not row.get("error")
    ]
    paired: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        paired.setdefault(_pair_key(row), {})[str(row["treatment"])] = row
    discordant = [
        (key, values)
        for key, values in paired.items()
        if set(values) == {"champion", "challenger"}
        and values["champion"].get("evaluated_win")
        != values["challenger"].get("evaluated_win")
    ]
    if args.limit > 0:
        discordant = discordant[: args.limit]

    jobs: list[Job] = []
    originals_by_job: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for (opponent, seed, seat), originals in discordant:
        for treatment in ("champion", "challenger"):
            job = Job(
                block_id=int(originals[treatment]["block_id"]),
                opponent=opponent,
                seed=seed,
                evaluated_seat=seat,
                treatment=treatment,
            )
            jobs.append(job)
            originals_by_job[(opponent, seed, seat, treatment)] = originals[treatment]

    context = mp.get_context("spawn")
    with context.Pool(
        max(1, args.workers),
        initializer=_worker_init,
        initargs=(config["champion"], config["challenger"], config["opponents"]),
    ) as pool:
        rerun_rows = list(
            pool.imap_unordered(
                _run_job,
                (
                    (job, int(config.get("max_steps", 8000)), max(1, args.trace_limit))
                    for job in jobs
                ),
                chunksize=1,
            )
        )

    reruns_by_pair: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for row in rerun_rows:
        key = _pair_key(row)
        reruns_by_pair.setdefault(key, {})[str(row["treatment"])] = row
        original = originals_by_job[(*key, str(row["treatment"]))]
        expected = original.get("evaluated_win")
        actual = row.get("evaluated_win")
        if actual != expected or row.get("error"):
            mismatches.append(
                {
                    "opponent": key[0],
                    "seed": key[1],
                    "seat": key[2],
                    "treatment": row["treatment"],
                    "expected": expected,
                    "actual": actual,
                    "error": row.get("error"),
                }
            )

    for (opponent, seed, seat), originals in discordant:
        reruns = reruns_by_pair.get((opponent, seed, seat), {})
        if set(reruns) != {"champion", "challenger"}:
            continue
        first = _first_divergence(
            reruns["champion"].get("trace", []),
            reruns["challenger"].get("trace", []),
        )
        details.append(
            {
                "opponent": opponent,
                "seed": seed,
                "seat": seat,
                "direction": (
                    "challenger_only"
                    if originals["challenger"]["evaluated_win"]
                    else "champion_only"
                ),
                "first_divergence": first,
                "setup_funnel": {
                    treatment: reruns[treatment].get("setup_funnel")
                    for treatment in ("champion", "challenger")
                },
                "results": {
                    treatment: {
                        "evaluated_win": reruns[treatment].get("evaluated_win"),
                        "moves": reruns[treatment].get("moves"),
                    }
                    for treatment in ("champion", "challenger")
                },
            }
        )

    signatures = Counter()
    by_direction = Counter()
    by_own_turn = Counter()
    for item in details:
        first = item.get("first_divergence") or {}
        left = first.get("champion")
        right = first.get("challenger")
        signature = f"{_choice_label(left)} -> {_choice_label(right)}"
        signatures[signature] += 1
        by_direction[f"{item['direction']} | {signature}"] += 1
        turn = (left or right or {}).get("own_turn")
        by_own_turn[f"{item['direction']} | own_turn={turn}"] += 1

    report = {
        "valid": not mismatches and len(details) == len(discordant),
        "source": str(args.report),
        "config": {
            "champion": config["champion"],
            "challenger": config["challenger"],
            "trace_limit": args.trace_limit,
            "workers": args.workers,
            "selected_discordant_pairs": len(discordant),
        },
        "reproduction_mismatches": mismatches,
        "summary": {
            "first_divergence_signatures": dict(signatures.most_common()),
            "by_direction_and_signature": dict(by_direction.most_common()),
            "by_direction_and_own_turn": dict(by_own_turn.most_common()),
        },
        "details": details,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))
    print(f"report: {args.out.resolve()}")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
