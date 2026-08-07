"""Build a deployable rank-fusion ensemble on the current top-pilot corpus.

Every member is a fixed policy at inference time: either a teacher-conditioned
ranker with its dense pin baked in, or a pooled/single-teacher ranker without a
teacher column.  Candidate ranks, rather than raw LightGBM margins, are summed
so models trained on different objectives and tree counts remain comparable.

The member subset is selected only on the chronological validation block.  The
test block is evaluated once after selection and the report records the exact
members and weights required by the runtime.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_grimmsnarl_v2_teacher import (
    Corpus,
    error_taxonomy,
    select_decisions,
    top1,
    topk,
    wilson,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMBERS = [
    ("old_pin9", "ranker_v5_data_refresh_base.txt", 9, 2000),
    ("old_pin12", "ranker_v5_data_refresh_base.txt", 12, 2000),
    ("old_pin16", "ranker_v5_data_refresh_base.txt", 16, 2000),
    ("old_pin20", "ranker_v5_data_refresh_base.txt", 20, 2000),
    ("fresh_pin0", "ranker_v8_current_top4.txt", 0, 0),
    ("fresh_pin2", "ranker_v8_current_top4.txt", 2, 0),
    ("fresh_pin3", "ranker_v8_current_top4.txt", 3, 0),
    ("current_pooled", "ranker_v11_win1.txt", None, 0),
    ("team16561259", "ranker_v10_team16561259.txt", None, 0),
    ("team16422241", "ranker_v10_team16422241.txt", None, 0),
    ("team16561259_win2", "ranker_v11_team16561259_win2.txt", None, 0),
]


def _rows(corpus: Corpus, decisions: np.ndarray) -> np.ndarray:
    return corpus.rows_for(decisions)


def _matrix(corpus: Corpus, decisions: np.ndarray, pin: int | None) -> np.ndarray:
    rows = _rows(corpus, decisions)
    base = corpus.features[rows]
    if pin is None:
        return base
    matrix = np.empty((len(rows), base.shape[1] + 1), dtype=np.float32)
    matrix[:, :-1] = base
    matrix[:, -1] = float(pin)
    return matrix


def _rank_scores(scores: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    fused = np.empty_like(scores, dtype=np.float32)
    cursor = 0
    for size_value in sizes:
        size = int(size_value)
        window = scores[cursor:cursor + size]
        order = np.argsort(-window, kind="stable")
        values = np.empty(size, dtype=np.float32)
        values[order] = (size - np.arange(size, dtype=np.float32)) / size
        fused[cursor:cursor + size] = values
        cursor += size
    return fused


def _positions(scores: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    positions = np.empty(len(sizes), dtype=np.int16)
    cursor = 0
    for slot, size_value in enumerate(sizes):
        size = int(size_value)
        positions[slot] = int(np.argmax(scores[cursor:cursor + size]))
        cursor += size
    return positions


def _forced_scores(
    baseline: np.ndarray,
    sizes: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Keep baseline ordering below Top-1 while forcing gated selections."""
    output = baseline.copy()
    cursor = 0
    for position, size_value in zip(positions, sizes):
        size = int(size_value)
        output[cursor + int(position)] = 3.0
        cursor += size
    return output


def _metrics(
    corpus: Corpus,
    decisions: np.ndarray,
    scores: np.ndarray,
) -> dict:
    sizes = corpus.groups[decisions]
    offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
    hits = top1(scores, corpus, decisions, offsets)
    low, high = wilson(int(hits.sum()), len(hits))
    return {
        "decisions": int(len(decisions)),
        "hits": int(hits.sum()),
        "top1": round(float(hits.mean()), 4),
        "top1_wilson95": [low, high],
        "top2": topk(corpus, decisions, offsets, scores, 2),
        "top3": topk(corpus, decisions, offsets, scores, 3),
        "taxonomy": error_taxonomy(
            corpus, decisions, offsets, scores, corpus.names
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data/ml/grimmsnarl/processed/corpus_v8_current_top4.npz",
    )
    parser.add_argument(
        "--model-root", type=Path,
        default=ROOT / "data/ml/grimmsnarl/models",
    )
    parser.add_argument("--maximum-members", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    corpus = Corpus(args.corpus)
    boundaries = corpus.resplit_per_team(0.12, 0.12)
    decisions = {
        split: select_decisions(corpus, split, None, None)
        for split in ("validation", "test")
    }
    predictions: dict[str, dict[str, np.ndarray]] = {
        split: {} for split in decisions
    }
    positions: dict[str, dict[str, np.ndarray]] = {
        split: {} for split in decisions
    }
    member_metadata: dict[str, dict] = {}
    for name, filename, pin, iterations in DEFAULT_MEMBERS:
        path = args.model_root / filename
        if not path.exists():
            continue
        booster = lgb.Booster(model_file=str(path))
        expected = corpus.names + (["teacher_team_id"] if pin is not None else [])
        if booster.feature_name() != expected:
            raise SystemExit(f"feature mismatch for {name}")
        member_metadata[name] = {
            "model": str(path.resolve()),
            "filename": filename,
            "teacher_code": pin,
            "iterations": int(iterations or booster.num_trees()),
        }
        for split, selected in decisions.items():
            matrix = _matrix(corpus, selected, pin)
            scores = booster.predict(
                matrix, num_iteration=(iterations or None)
            )
            del matrix
            predictions[split][name] = _rank_scores(
                scores, corpus.groups[selected]
            )
            positions[split][name] = _positions(
                scores, corpus.groups[selected]
            )

    names = sorted(member_metadata)
    if not names:
        raise SystemExit("no ensemble members were available")
    chosen_positions = {
        split: np.asarray([
            int(np.flatnonzero(
                corpus.labels[corpus.starts[d]:corpus.ends[d]] == 1
            )[0])
            for d in selected
        ], dtype=np.int16)
        for split, selected in decisions.items()
    }
    validation_rows: list[dict] = []
    best: tuple[float, int, tuple[str, ...], np.ndarray] | None = None
    for count in range(1, min(args.maximum_members, len(names)) + 1):
        for members in itertools.combinations(names, count):
            fused = np.sum(
                [predictions["validation"][name] for name in members], axis=0
            )
            fused_positions = _positions(
                fused, corpus.groups[decisions["validation"]]
            )
            hits = fused_positions == chosen_positions["validation"]
            top = round(float(hits.mean()), 6)
            validation_rows.append({
                "members": list(members),
                "top1": top,
                "hits": int(hits.sum()),
            })
            key = (top, -count, members, fused)
            if best is None or key[:3] > best[:3]:
                best = key
    assert best is not None
    selected_members = best[2]
    validation_fused = np.sum(
        [predictions["validation"][name] for name in selected_members], axis=0
    )
    test_fused = np.sum(
        [predictions["test"][name] for name in selected_members], axis=0
    )
    validation_rows.sort(key=lambda row: (-row["top1"], len(row["members"])))

    # Conservative alternative: keep one baseline unless a voter subset has a
    # sufficiently large exact Top-1 majority.  This is the deployable version
    # of "use current teachers only where they agree" and avoids averaging a
    # confident old-policy answer away on every decision.
    gate_rows: list[dict] = []
    best_gate: tuple[float, int, int, str, tuple[str, ...], int] | None = None
    baseline_names = [
        name for name in ("old_pin16", "old_pin20", "current_pooled")
        if name in names
    ]
    for baseline_name in baseline_names:
        # Limit the gate search to policies that are independently meaningful
        # at deployment.  Exhaustively mixing every deliberately weak ablation
        # creates thousands of validation-only combinations and no new signal.
        voter_pool = {
            "current_pooled", "fresh_pin0", "fresh_pin3", "old_pin20",
            "team16561259",
        }
        voters_available = [
            name for name in names
            if name != baseline_name and name in voter_pool
        ]
        for count in range(2, min(5, len(voters_available)) + 1):
            for voters in itertools.combinations(voters_available, count):
                for threshold in sorted({count, count // 2 + 1}):
                    vote_matrix = np.stack([
                        positions["validation"][name] for name in voters
                    ])
                    baseline_position = positions["validation"][baseline_name]
                    gated = baseline_position.copy()
                    overrides = 0
                    for slot in range(vote_matrix.shape[1]):
                        values, counts = np.unique(
                            vote_matrix[:, slot], return_counts=True
                        )
                        winner_slot = int(np.argmax(counts))
                        if int(counts[winner_slot]) >= threshold:
                            candidate = int(values[winner_slot])
                            overrides += int(candidate != gated[slot])
                            gated[slot] = candidate
                    hits = gated == chosen_positions["validation"]
                    top = round(float(hits.mean()), 6)
                    row = {
                        "baseline": baseline_name,
                        "voters": list(voters),
                        "threshold": threshold,
                        "validation_top1": top,
                        "validation_hits": int(hits.sum()),
                        "overrides": overrides,
                    }
                    gate_rows.append(row)
                    # Ties prefer fewer overrides, fewer voters, then stable
                    # lexical identifiers for exact reproducibility.
                    key = (
                        top, -overrides, -count, baseline_name, voters,
                        threshold,
                    )
                    if best_gate is None or key > best_gate:
                        best_gate = key
    assert best_gate is not None
    _, _, _, gate_baseline, gate_voters, gate_threshold = best_gate

    def apply_gate(split: str) -> tuple[np.ndarray, int]:
        vote_matrix = np.stack([
            positions[split][name] for name in gate_voters
        ])
        gated = positions[split][gate_baseline].copy()
        overrides = 0
        for slot in range(vote_matrix.shape[1]):
            values, counts = np.unique(vote_matrix[:, slot], return_counts=True)
            winner_slot = int(np.argmax(counts))
            if int(counts[winner_slot]) >= gate_threshold:
                candidate = int(values[winner_slot])
                overrides += int(candidate != gated[slot])
                gated[slot] = candidate
        scores = _forced_scores(
            predictions[split][gate_baseline],
            corpus.groups[decisions[split]], gated,
        )
        return scores, overrides

    gate_validation_scores, gate_validation_overrides = apply_gate("validation")
    gate_test_scores, gate_test_overrides = apply_gate("test")
    gate_rows.sort(key=lambda row: (
        -row["validation_top1"], row["overrides"], len(row["voters"])
    ))
    result = {
        "method": "equal-weight per-decision candidate-rank fusion",
        "corpus": str(args.corpus.resolve()),
        "split_boundaries": boundaries,
        "selection_rule": (
            "maximum chronological validation strict Top-1; ties prefer fewer members"
        ),
        "available_members": member_metadata,
        "selected_members": list(selected_members),
        "member_weight": 1.0,
        "validation": _metrics(
            corpus, decisions["validation"], validation_fused
        ),
        "test": _metrics(corpus, decisions["test"], test_fused),
        "top_validation_combinations": validation_rows[:30],
        "single_member_test": {
            name: _metrics(corpus, decisions["test"], predictions["test"][name])
            for name in names
        },
        "conservative_gate": {
            "baseline": gate_baseline,
            "voters": list(gate_voters),
            "threshold": gate_threshold,
            "selection_rule": (
                "maximum validation Top-1; ties prefer fewer overrides and voters"
            ),
            "validation_overrides": gate_validation_overrides,
            "test_overrides": gate_test_overrides,
            "validation": _metrics(
                corpus, decisions["validation"], gate_validation_scores
            ),
            "test": _metrics(corpus, decisions["test"], gate_test_scores),
            "top_validation_gates": gate_rows[:30],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_members": result["selected_members"],
        "validation": {
            "top1": result["validation"]["top1"],
            "order_insensitive_top1": result["validation"]["taxonomy"]["order_insensitive_top1"],
        },
        "test": {
            "top1": result["test"]["top1"],
            "top2": result["test"]["top2"],
            "top3": result["test"]["top3"],
            "order_insensitive_top1": result["test"]["taxonomy"]["order_insensitive_top1"],
        },
        "conservative_gate": {
            "baseline": result["conservative_gate"]["baseline"],
            "voters": result["conservative_gate"]["voters"],
            "threshold": result["conservative_gate"]["threshold"],
            "validation_top1": result["conservative_gate"]["validation"]["top1"],
            "test_top1": result["conservative_gate"]["test"]["top1"],
            "test_order_insensitive_top1": result["conservative_gate"]["test"]["taxonomy"]["order_insensitive_top1"],
            "test_overrides": result["conservative_gate"]["test_overrides"],
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
