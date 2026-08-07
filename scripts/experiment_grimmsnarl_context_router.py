"""Select an old Grimmsnarl teacher pin by observable decision context.

The deployed ranker is conditioned on a dense teacher code.  v6 routes one
hand-written class (Froslass evolution) to another code; this experiment makes
that idea measurable.  A router is fitted on one target pilot's chronological
validation games only, then read once on the untouched test block.

Two routers are reported:

* context: one pin for each select context;
* menu: context plus the offered action-type set and a coarse turn band.

Both shrink bucket accuracy toward the native target pin, so a tiny bucket
cannot win because of one lucky decision.  The resulting JSON contains the
exact routing tables needed by a standard-library runtime.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

from train_grimmsnarl_v2_teacher import Corpus, select_decisions, wilson


OLD_TEAM_CODES = {
    16371703: 0,
    16375320: 1,
    16376653: 2,
    16381904: 3,
    16385817: 4,
    16388364: 5,
    16388654: 6,
    16407282: 7,
    16421840: 8,
    16422241: 9,
    16430670: 10,
    16431331: 11,
    16452116: 12,
    16461850: 13,
    16462035: 14,
    16463316: 15,
    16494330: 16,
    16514272: 17,
    16531269: 18,
    16556346: 19,
    16561259: 20,
}


def _offsets(groups: np.ndarray) -> np.ndarray:
    return np.r_[0, np.cumsum(groups)[:-1]].astype(np.int64)


def _predicted_positions(scores: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    offsets = _offsets(sizes)
    return np.asarray([
        int(np.argmax(scores[offset:offset + int(size)]))
        for offset, size in zip(offsets, sizes)
    ], dtype=np.int16)


def _chosen_positions(corpus: Corpus, decisions: np.ndarray) -> np.ndarray:
    return np.asarray([
        int(np.flatnonzero(
            corpus.labels[corpus.starts[d]:corpus.ends[d]] == 1
        )[0])
        for d in decisions
    ], dtype=np.int16)


def _menu_keys(corpus: Corpus, decisions: np.ndarray) -> list[str]:
    action_col = corpus.names.index("action_type_id")
    keys: list[str] = []
    for decision in decisions:
        rows = corpus.features[corpus.starts[decision]:corpus.ends[decision]]
        actions = sorted(set(int(value) for value in rows[:, action_col]))
        turn = int(corpus.turns[decision])
        band = 0 if turn <= 4 else 1 if turn <= 8 else 2
        keys.append(
            f"{int(corpus.contexts[decision])}|{band}|"
            + ",".join(map(str, actions))
        )
    return keys


def _context_keys(corpus: Corpus, decisions: np.ndarray) -> list[str]:
    return [str(int(corpus.contexts[d])) for d in decisions]


def _fit_router(
    keys: list[str],
    hits_by_code: dict[int, np.ndarray],
    baseline_code: int,
    prior_strength: float,
    minimum_support: int,
) -> dict[str, int]:
    slots: dict[str, list[int]] = defaultdict(list)
    for slot, key in enumerate(keys):
        slots[key].append(slot)
    baseline = hits_by_code[baseline_code]
    global_prior = float(baseline.mean())
    route: dict[str, int] = {}
    for key, indexes_list in slots.items():
        indexes = np.asarray(indexes_list, dtype=np.int64)
        if len(indexes) < minimum_support:
            continue
        best_code = baseline_code
        best_score = (
            float(baseline[indexes].sum()) + prior_strength * global_prior
        ) / (len(indexes) + prior_strength)
        for code, hits in hits_by_code.items():
            score = (
                float(hits[indexes].sum()) + prior_strength * global_prior
            ) / (len(indexes) + prior_strength)
            if score > best_score + 1e-12:
                best_code, best_score = code, score
        if best_code != baseline_code:
            route[key] = best_code
    return route


def _apply_router(
    keys: list[str],
    predictions_by_code: dict[int, np.ndarray],
    chosen: np.ndarray,
    route: dict[str, int],
    baseline_code: int,
) -> tuple[np.ndarray, np.ndarray]:
    codes = np.asarray(
        [route.get(key, baseline_code) for key in keys], dtype=np.int16
    )
    predicted = np.asarray([
        predictions_by_code[int(code)][slot]
        for slot, code in enumerate(codes)
    ], dtype=np.int16)
    return predicted == chosen, codes


def _summary(hits: np.ndarray, codes: np.ndarray | None = None) -> dict:
    successes = int(hits.sum())
    low, high = wilson(successes, len(hits))
    result = {
        "decisions": int(len(hits)),
        "hits": successes,
        "top1": round(float(hits.mean()), 4),
        "wilson95": [low, high],
    }
    if codes is not None:
        unique, counts = np.unique(codes, return_counts=True)
        result["decisions_by_code"] = {
            str(int(code)): int(count)
            for code, count in zip(unique, counts)
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target-team", type=int, required=True)
    parser.add_argument("--baseline-code", type=int, required=True)
    parser.add_argument("--codes", default=",".join(map(str, range(21))))
    parser.add_argument("--num-iteration", type=int, default=2000)
    parser.add_argument("--prior-strength", type=float, default=80.0)
    parser.add_argument("--minimum-support", type=int, default=30)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    codes = [int(value) for value in args.codes.split(",") if value.strip()]
    if args.baseline_code not in codes:
        raise SystemExit("baseline code must be present in --codes")
    inverse = {code: team for team, code in OLD_TEAM_CODES.items()}
    missing = sorted(set(codes) - set(inverse))
    if missing:
        raise SystemExit(f"unknown old teacher codes: {missing}")

    corpus = Corpus(args.corpus)
    boundaries = corpus.resplit_per_team(0.12, 0.12)
    corpus.add_team_feature()
    # The evaluated corpus can omit most old teachers.  Add their known dense
    # codes explicitly; only the numeric value reaches LightGBM.
    corpus.team_codes.update(OLD_TEAM_CODES)
    splits = {
        name: select_decisions(corpus, name, {args.target_team}, None)
        for name in ("validation", "test")
    }
    if any(not len(value) for value in splits.values()):
        raise SystemExit("target team has an empty validation or test split")

    booster = lgb.Booster(model_file=str(args.model))
    if booster.feature_name() != corpus.names:
        raise SystemExit("model/corpus feature names differ")

    predictions: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in splits
    }
    hits: dict[str, dict[int, np.ndarray]] = {name: {} for name in splits}
    chosen: dict[str, np.ndarray] = {}
    for split, decisions in splits.items():
        chosen[split] = _chosen_positions(corpus, decisions)
    for code in codes:
        team = inverse[code]
        for split, decisions in splits.items():
            matrix = corpus.matrix(decisions, pin_team=team)
            scores = booster.predict(matrix, num_iteration=args.num_iteration)
            del matrix
            predicted = _predicted_positions(scores, corpus.groups[decisions])
            predictions[split][code] = predicted
            hits[split][code] = predicted == chosen[split]

    key_functions = {
        "context": _context_keys,
        "menu": _menu_keys,
    }
    routers: dict[str, dict] = {}
    for name, key_function in key_functions.items():
        validation_keys = key_function(corpus, splits["validation"])
        test_keys = key_function(corpus, splits["test"])
        route = _fit_router(
            validation_keys,
            hits["validation"],
            args.baseline_code,
            args.prior_strength,
            args.minimum_support,
        )
        validation_hits, validation_codes = _apply_router(
            validation_keys, predictions["validation"], chosen["validation"],
            route, args.baseline_code,
        )
        test_hits, test_codes = _apply_router(
            test_keys, predictions["test"], chosen["test"], route,
            args.baseline_code,
        )
        routers[name] = {
            "route": route,
            "validation": _summary(validation_hits, validation_codes),
            "test": _summary(test_hits, test_codes),
        }

    result = {
        "method": "validation-fitted teacher-code router",
        "corpus": str(args.corpus.resolve()),
        "model": str(args.model.resolve()),
        "model_iterations": args.num_iteration,
        "target_team": args.target_team,
        "baseline_code": args.baseline_code,
        "candidate_codes": codes,
        "split_boundaries": boundaries,
        "prior_strength": args.prior_strength,
        "minimum_support": args.minimum_support,
        "per_code": {
            str(code): {
                "team": inverse[code],
                "validation": _summary(hits["validation"][code]),
                "test": _summary(hits["test"][code]),
            }
            for code in codes
        },
        "routers": routers,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "target_team": args.target_team,
        "baseline": result["per_code"][str(args.baseline_code)],
        "context": routers["context"],
        "menu": routers["menu"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
