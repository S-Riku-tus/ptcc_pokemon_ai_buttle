"""Train the state-value head used by ``grimmsnarl_ml_v7``.

One candidate row is retained per decision from the frozen current-top-four
corpus.  Only public columns selected by ``value_features`` are used, so the
target action, offered option and private hand identity cannot leak into the
value estimate.
Each episode receives equal total weight; otherwise long games would count as
many independent outcome labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.distill import compact_booster  # noqa: E402


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _state_names(agent_dir: Path, manifest: Path, data_root: Path) -> list[str]:
    # ml_features is an absolute import inside value_features, matching the
    # Kaggle flat-directory runtime.  Install that exact module name here too.
    features = _load_module(agent_dir / "ml_features.py", "ml_features")
    value = _load_module(agent_dir / "value_features.py", "grim_v7_value_features")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        replay = data_root / row["replay_path"]
        if not replay.exists():
            continue
        payload = json.loads(replay.read_text(encoding="utf-8"))
        seat = int(row["seat_index"])
        for step in payload.get("steps") or []:
            entry = step[seat] if seat < len(step) else {}
            observation = entry.get("observation") or {}
            current = observation.get("current") or {}
            if current and int(current.get("turn", 0) or 0) >= 1:
                names = list(value.value_features(observation, seat))
                # perspective_turn is a runtime assertion, constant one in
                # this own-turn corpus, and intentionally not fitted.
                return [name for name in names if name != "perspective_turn"]
    raise RuntimeError("could not find a usable observation in manifest")


def _episode_weights(episode_ids: np.ndarray) -> np.ndarray:
    counts = Counter(int(value) for value in episode_ids)
    mean_count = len(episode_ids) / max(len(counts), 1)
    return np.asarray(
        [mean_count / counts[int(value)] for value in episode_ids],
        dtype=np.float32,
    )


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    episode_ids: np.ndarray,
) -> dict:
    result = {
        "states": int(len(labels)),
        "episodes": int(len(np.unique(episode_ids))),
        "win_rate": round(float(np.average(labels, weights=weights)), 4),
        "auc": round(float(roc_auc_score(labels, probabilities,
                                           sample_weight=weights)), 4),
        "logloss": round(float(log_loss(labels, probabilities,
                                         sample_weight=weights)), 4),
        "accuracy": round(float(accuracy_score(
            labels, probabilities >= 0.5, sample_weight=weights
        )), 4),
    }
    episode_order = np.unique(episode_ids)
    episode_prob = np.asarray([
        float(np.mean(probabilities[episode_ids == episode]))
        for episode in episode_order
    ])
    episode_label = np.asarray([
        int(labels[np.flatnonzero(episode_ids == episode)[0]])
        for episode in episode_order
    ])
    if len(np.unique(episode_label)) > 1:
        result["episode_mean_auc"] = round(float(
            roc_auc_score(episode_label, episode_prob)
        ), 4)
    return result


def _calibration(labels: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    bins = np.linspace(0.0, 1.0, 6)
    output = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= low) & (
            probabilities <= high if high == 1.0 else probabilities < high
        )
        if not np.any(mask):
            continue
        output.append({
            "range": [round(float(low), 1), round(float(high), 1)],
            "states": int(mask.sum()),
            "mean_prediction": round(float(probabilities[mask].mean()), 4),
            "win_rate": round(float(labels[mask].mean()), 4),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", type=Path,
        default=ROOT / "data/ml/grimmsnarl/processed/corpus_v8_current_top4.npz",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v8_imitation/current_top4_selection.csv",
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data/kaggle_grimmsnarl_v8",
    )
    parser.add_argument(
        "--agent-dir", type=Path,
        default=ROOT / "agents/grimmsnarl/grimmsnarl_ml_v7",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "agents/grimmsnarl/grimmsnarl_ml_v7/value_model.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "experiments/grimmsnarl_ml_v7_value/training_report.json",
    )
    parser.add_argument("--seed", type=int, default=7007)
    parser.add_argument("--trees", type=int, default=900)
    args = parser.parse_args()

    wanted = _state_names(args.agent_dir, args.manifest, args.data_root)
    with np.load(args.corpus, allow_pickle=False) as corpus:
        names = [str(value) for value in corpus["feature_names"].tolist()]
        name_to_index = {name: index for index, name in enumerate(names)}
        missing = sorted(set(wanted).difference(name_to_index))
        if missing:
            raise ValueError(f"state features absent from corpus: {missing[:20]}")
        selected_names = [name for name in names if name in set(wanted)]
        selected_columns = np.asarray(
            [name_to_index[name] for name in selected_names], dtype=np.int64
        )
        groups = corpus["groups"].astype(np.int64)
        starts = np.empty(len(groups), dtype=np.int64)
        starts[0] = 0
        if len(groups) > 1:
            starts[1:] = np.cumsum(groups[:-1])
        # Advanced indexing materialises only the 427 state columns and lets
        # the 1.2 GB decompressed candidate matrix be released immediately.
        matrix = corpus["features"][starts][:, selected_columns].astype(
            np.float32, copy=False
        )
        labels = corpus["won"].astype(np.int8)
        splits = corpus["splits"].astype(str)
        episode_ids = corpus["episode_ids"].astype(np.int64)
        turns = corpus["turns"].astype(np.int16)
        teams = corpus["team_ids"].astype(np.int64)

    if matrix.shape[0] != len(labels):
        raise ValueError("one state row per decision invariant failed")
    weights = _episode_weights(episode_ids)
    train = splits == "train"
    validation = splits == "validation"
    test = splits == "test"
    categorical_names = [
        name for name in ("opp_active_id", "self_active_id", "stadium_id")
        if name in selected_names
    ]
    categorical_indices = [selected_names.index(name) for name in categorical_names]

    params = dict(
        objective="binary",
        n_estimators=args.trees,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=8,
        min_child_samples=180,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.25,
        reg_lambda=3.0,
        random_state=args.seed,
        n_jobs=16,
        verbosity=-1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(
        matrix[train], labels[train], sample_weight=weights[train],
        categorical_feature=categorical_indices,
        eval_set=[(matrix[validation], labels[validation])],
        eval_sample_weight=[weights[validation]],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    best_iteration = int(model.best_iteration_ or args.trees)
    validation_probability = model.predict_proba(
        matrix[validation], num_iteration=best_iteration
    )[:, 1]
    test_probability = model.predict_proba(
        matrix[test], num_iteration=best_iteration
    )[:, 1]

    refit_mask = train | validation
    refit_params = dict(params)
    refit_params["n_estimators"] = best_iteration
    final = lgb.LGBMClassifier(**refit_params)
    final.fit(
        matrix[refit_mask], labels[refit_mask],
        sample_weight=weights[refit_mask],
        categorical_feature=categorical_indices,
    )
    compact = compact_booster(final.booster_, "grimmsnarl_state_value")
    compact.update({
        "target": "eventual_win",
        "scope": "public_next_turn",
        "source_corpus": args.corpus.name,
        "episodes": int(len(np.unique(episode_ids))),
        "best_iteration": best_iteration,
        "categorical_features": categorical_names,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(compact, separators=(",", ":")), encoding="utf-8"
    )

    importance = sorted(
        zip(selected_names, final.booster_.feature_importance(
            importance_type="gain"
        )),
        key=lambda pair: float(pair[1]), reverse=True,
    )
    report = {
        "corpus": str(args.corpus.resolve()),
        "model": str(args.output.resolve()),
        "model_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "state_features": len(selected_names),
        "categorical_features": categorical_names,
        "decisions": int(len(labels)),
        "episodes": int(len(np.unique(episode_ids))),
        "teams": sorted(int(value) for value in np.unique(teams)),
        "split_states": {
            name: int(np.sum(splits == name))
            for name in ("train", "validation", "test")
        },
        "split_episodes": {
            name: int(len(np.unique(episode_ids[splits == name])))
            for name in ("train", "validation", "test")
        },
        "params": params,
        "best_iteration": best_iteration,
        "validation": _metrics(
            labels[validation], validation_probability, weights[validation],
            episode_ids[validation],
        ),
        "test": _metrics(
            labels[test], test_probability, weights[test], episode_ids[test],
        ),
        "test_by_turn": {},
        "test_calibration": _calibration(labels[test], test_probability),
        "top_gain_features": [
            {"name": name, "gain": round(float(gain), 2)}
            for name, gain in importance[:30]
        ],
    }
    for label, low, high in (
        ("turn_1_4", 1, 4), ("turn_5_8", 5, 8), ("turn_9_plus", 9, 32767)
    ):
        mask = test & (turns >= low) & (turns <= high)
        if np.sum(mask) and len(np.unique(labels[mask])) > 1:
            probability = model.predict_proba(
                matrix[mask], num_iteration=best_iteration
            )[:, 1]
            report["test_by_turn"][label] = _metrics(
                labels[mask], probability, weights[mask], episode_ids[mask]
            )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
