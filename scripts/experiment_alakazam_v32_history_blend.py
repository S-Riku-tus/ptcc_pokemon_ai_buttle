"""Blend the public-history DeepSets challenger with the current v32 trio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v31_ranker_ensemble as ensemble  # noqa: E402
import experiment_alakazam_v32_deepset as deepset  # noqa: E402
import experiment_alakazam_v32_history_deepset as history_model  # noqa: E402


def _history_scores(
    checkpoint_path: Path,
    features: np.ndarray,
    names: list[str],
    groups: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    decisions: np.ndarray,
) -> np.ndarray:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    continuous_columns = [
        names.index(name)
        for name in checkpoint["continuous_feature_names"]
    ]
    categorical_columns = [
        names.index(name)
        for name in checkpoint["categorical_feature_names"]
    ]
    continuous = np.empty(
        (len(features), len(continuous_columns)),
        dtype=np.float16,
    )
    for start in range(0, len(features), 50_000):
        end = min(start + 50_000, len(features))
        values = (
            features[start:end, continuous_columns].astype(np.float32)
            - checkpoint["continuous_mean"]
        ) / checkpoint["continuous_std"]
        continuous[start:end] = np.clip(
            values, -8.0, 8.0
        ).astype(np.float16)
    categorical = np.zeros(
        (len(features), len(categorical_columns)),
        dtype=np.int16,
    )
    for output_column, (source_column, vocabulary) in enumerate(zip(
        categorical_columns,
        checkpoint["categorical_vocabularies"],
    )):
        raw = features[:, source_column]
        positions = np.searchsorted(vocabulary, raw)
        clipped = np.minimum(positions, len(vocabulary) - 1)
        known = (
            (positions < len(vocabulary))
            & (vocabulary[clipped] == raw)
        )
        categorical[:, output_column] = np.where(
            known,
            positions + 1,
            len(vocabulary) + 1,
        ).astype(np.int16)
    starts, _ = deepset._ranges(groups)
    history_columns = [
        names.index(name)
        for name in checkpoint["history_feature_names"]
    ]
    history = features[starts][:, history_columns].astype(np.float32)
    history = np.clip(
        (
            history - checkpoint["history_mean"]
        ) / checkpoint["history_std"],
        -8.0,
        8.0,
    ).astype(np.float16)
    store = history_model.HistoryStore(
        continuous,
        categorical,
        groups,
        labels,
        weights,
        history=history,
    )
    model = deepset.DeepSetPolicy(
        len(continuous_columns),
        checkpoint["categorical_sizes"],
        checkpoint["hidden"],
        checkpoint["dropout"],
        state_features=len(history_columns),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(decisions), 128):
            batch_decisions = decisions[start:start + 128]
            (
                continuous_batch,
                categorical_batch,
                mask,
                _,
                _,
                state,
            ) = store.history_batch(batch_decisions, torch.device("cpu"))
            scores = model(
                continuous_batch,
                categorical_batch,
                mask,
                state,
            ).numpy()
            for row, decision in enumerate(batch_decisions):
                output.append(scores[row, :groups[decision]].copy())
    return np.concatenate(output).astype(np.float32)


def _oracle(
    score_sets: list[np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
) -> float:
    starts, ends = ensemble._ranges(groups)
    return float(np.mean([
        any(
            labels[start + int(np.argmax(scores[start:end]))] == 1
            for scores in score_sets
        )
        for start, end in zip(starts, ends)
    ]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(12, max(1, torch.get_num_threads())))
    with np.load(args.scores, allow_pickle=False) as saved:
        val_scores = [
            saved["validation_large"],
            saved["validation_numeric"],
            saved["validation_deep"],
        ]
        test_scores = [
            saved["test_large"],
            saved["test_numeric"],
            saved["test_deep"],
        ]
        val_labels = saved["validation_labels"]
        test_labels = saved["test_labels"]
        val_groups = saved["validation_groups"]
        test_groups = saved["test_groups"]
    with np.load(args.cache, allow_pickle=False) as cached:
        features = cached["features"]
        names = cached["feature_names"].astype(str).tolist()
        groups = cached["groups"]
        labels = cached["labels"]
        weights = cached["weights"]
        splits = cached["splits"].astype(str)
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    combined = _history_scores(
        args.checkpoint,
        features,
        names,
        groups,
        labels,
        weights,
        np.concatenate([validation, test]),
    )
    validation_rows = int(val_groups.sum())
    val_history = ensemble._normalize(
        combined[:validation_rows], val_groups.tolist()
    )
    test_history = ensemble._normalize(
        combined[validation_rows:], test_groups.tolist()
    )
    val_scores.append(val_history)
    test_scores.append(test_history)
    history_metrics = {
        "validation_top1": ensemble._accuracy(
            val_history, val_labels, val_groups.tolist()
        ),
        "test_top1": ensemble._accuracy(
            test_history, test_labels, test_groups.tolist()
        ),
    }
    print(history_metrics, flush=True)

    selected = None
    for numeric_weight in np.arange(0.3, 1.51, 0.1):
        for deep_weight in np.arange(0.0, 1.51, 0.1):
            for history_weight in np.arange(0.0, 1.51, 0.1):
                validation_scores = (
                    val_scores[0]
                    + float(numeric_weight) * val_scores[1]
                    + float(deep_weight) * val_scores[2]
                    + float(history_weight) * val_scores[3]
                )
                accuracy = ensemble._accuracy(
                    validation_scores,
                    val_labels,
                    val_groups.tolist(),
                )
                row = {
                    "numeric_weight": float(numeric_weight),
                    "deep_weight": float(deep_weight),
                    "history_weight": float(history_weight),
                    "validation_top1": accuracy,
                }
                if selected is None or (
                    accuracy,
                    -history_weight,
                    -deep_weight,
                ) > (
                    selected["validation_top1"],
                    -selected["history_weight"],
                    -selected["deep_weight"],
                ):
                    selected = row
    assert selected is not None
    test_blend = (
        test_scores[0]
        + selected["numeric_weight"] * test_scores[1]
        + selected["deep_weight"] * test_scores[2]
        + selected["history_weight"] * test_scores[3]
    )
    test_top1 = ensemble._accuracy(
        test_blend, test_labels, test_groups.tolist()
    )
    report = {
        "history_model": history_metrics,
        "selected": selected,
        "test_top1": test_top1,
        "validation_oracle_any_model": _oracle(
            val_scores, val_labels, val_groups
        ),
        "test_oracle_any_model": _oracle(
            test_scores, test_labels, test_groups
        ),
        "previous_v32_blend_test_top1": 0.7699600798403193,
        "target_top1": 0.9,
        "target_met": test_top1 >= 0.9,
    }
    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.scores_output,
        validation_large=val_scores[0],
        validation_numeric=val_scores[1],
        validation_deep=val_scores[2],
        validation_history=val_scores[3],
        validation_labels=val_labels,
        validation_groups=val_groups,
        test_large=test_scores[0],
        test_numeric=test_scores[1],
        test_deep=test_scores[2],
        test_history=test_scores[3],
        test_labels=test_labels,
        test_groups=test_groups,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
