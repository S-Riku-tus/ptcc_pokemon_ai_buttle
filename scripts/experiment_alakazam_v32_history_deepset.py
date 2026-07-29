"""Add structured public-history state to the v32 DeepSets candidate model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import experiment_alakazam_v32_deepset as base  # noqa: E402


HISTORY_PREFIXES = (
    "opp_discard_slot_",
    "long_recent_log_",
    "turn_open_log_",
)


class HistoryStore(base.GroupStore):
    def __init__(self, *args: Any, history: np.ndarray, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.history = history

    def history_batch(
        self,
        decisions: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        regular = super().batch(decisions, device)
        state = torch.from_numpy(
            self.history[decisions].astype(np.float32)
        ).to(device)
        return (*regular, state)


def _evaluate(
    model: nn.Module,
    store: HistoryStore,
    decisions: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    top1 = top2 = top3 = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(decisions), batch_size):
            batch_decisions = decisions[start:start + batch_size]
            (
                continuous,
                categorical,
                mask,
                targets,
                _,
                history,
            ) = store.history_batch(batch_decisions, device)
            scores = model(
                continuous,
                categorical,
                mask,
                history,
            )
            loss_sum += float(nn.functional.cross_entropy(
                scores, targets, reduction="sum"
            ).item())
            order = scores.topk(min(3, scores.shape[1]), dim=1).indices
            top1 += int((order[:, 0] == targets).sum().item())
            top2 += int(
                (order[:, :min(2, order.shape[1])] == targets[:, None])
                .any(dim=1).sum().item()
            )
            top3 += int(
                (order == targets[:, None]).any(dim=1).sum().item()
            )
    count = len(decisions)
    return {
        "loss": loss_sum / count,
        "semantic_top1": top1 / count,
        "semantic_top2": top2 / count,
        "semantic_top3": top3 / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("schema_cache", type=Path)
    parser.add_argument("v31_agent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-limit", type=int, default=180)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1197)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(12, max(1, torch.get_num_threads())))
    device = torch.device("cpu")

    with np.load(args.schema_cache, allow_pickle=False) as schema:
        schema_names = schema["feature_names"].astype(str).tolist()
    selected_schema_columns = base._select_features(
        schema_names,
        (
            args.v31_agent / "ranker_model.json",
            args.v31_agent / "ranker_numeric_model.json",
        ),
        args.feature_limit,
    )
    selected_names = [
        schema_names[index] for index in selected_schema_columns
    ]
    with np.load(args.cache, allow_pickle=False) as cached:
        source_names = cached["feature_names"].astype(str).tolist()
        source = cached["features"]
        selected_columns = [
            source_names.index(name) for name in selected_names
        ]
        groups = cached["groups"].copy()
        starts, _ = base._ranges(groups)
        history_columns = [
            index
            for index, name in enumerate(source_names)
            if name.startswith(HISTORY_PREFIXES)
        ]
        history_names = [
            source_names[index] for index in history_columns
        ]
        features = source[:, selected_columns].astype(
            np.float32, copy=True
        )
        history = source[starts][:, history_columns].astype(
            np.float32, copy=True
        )
        labels = cached["labels"].copy()
        weights = cached["weights"].copy()
        splits = cached["splits"].astype(str)

    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    test = np.flatnonzero(splits == "test")
    row_starts, row_ends = base._ranges(groups)
    train_rows = np.concatenate([
        np.arange(row_starts[index], row_ends[index], dtype=np.int64)
        for index in train
    ])
    categorical_local = [
        index for index, name in enumerate(selected_names)
        if base._categorical(name)
    ]
    categorical_set = set(categorical_local)
    continuous_local = [
        index for index in range(len(selected_names))
        if index not in categorical_set
    ]
    continuous, mean, std = base._normalise_continuous(
        features, continuous_local, train_rows
    )
    categorical, vocabularies, categorical_sizes = (
        base._encode_categorical(
            features, categorical_local, train_rows
        )
    )
    del features

    history_mean = history[train].mean(axis=0, dtype=np.float64)
    history_std = history[train].std(axis=0, dtype=np.float64)
    history_std[history_std < 1e-5] = 1.0
    history = np.clip(
        (
            history
            - history_mean.astype(np.float32)
        ) / history_std.astype(np.float32),
        -8.0,
        8.0,
    ).astype(np.float16)
    store = HistoryStore(
        continuous,
        categorical,
        groups,
        labels,
        weights,
        history=history,
    )
    model = base.DeepSetPolicy(
        len(continuous_local),
        categorical_sizes,
        args.hidden,
        args.dropout,
        state_features=len(history_names),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=7e-4,
        weight_decay=4e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=7e-5,
    )
    rng = np.random.default_rng(args.seed)
    best_top1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = rng.permutation(train)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(shuffled), args.batch_size):
            decisions = shuffled[start:start + args.batch_size]
            (
                continuous_batch,
                categorical_batch,
                mask,
                targets,
                batch_weights,
                state,
            ) = store.history_batch(decisions, device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(
                continuous_batch,
                categorical_batch,
                mask,
                state,
            )
            losses = nn.functional.cross_entropy(
                scores, targets, reduction="none"
            )
            loss = (
                losses * batch_weights
            ).sum() / batch_weights.sum().clamp_min(1e-6)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(decisions)
            seen += len(decisions)
        scheduler.step()
        metrics = _evaluate(
            model, store, validation, args.batch_size, device
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": metrics,
        }
        history_rows.append(row)
        print(json.dumps(row), flush=True)
        improved = (
            metrics["semantic_top1"] > best_top1
            or (
                metrics["semantic_top1"] == best_top1
                and metrics["loss"] < best_loss
            )
        )
        if improved:
            best_top1 = metrics["semantic_top1"]
            best_loss = metrics["loss"]
            best_epoch = epoch
            best_state = base._state_dict_cpu(model)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    test_metrics = _evaluate(
        model, store, test, args.batch_size, device
    )
    checkpoint = {
        "format": "alakazam_v32_history_deepset_v1",
        "state_dict": best_state,
        "selected_feature_names": selected_names,
        "continuous_feature_names": [
            selected_names[index] for index in continuous_local
        ],
        "categorical_feature_names": [
            selected_names[index] for index in categorical_local
        ],
        "continuous_mean": mean,
        "continuous_std": std,
        "categorical_vocabularies": vocabularies,
        "categorical_sizes": categorical_sizes,
        "history_feature_names": history_names,
        "history_mean": history_mean.astype(np.float32),
        "history_std": history_std.astype(np.float32),
        "hidden": args.hidden,
        "dropout": args.dropout,
        "best_epoch": best_epoch,
        "seed": args.seed,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    report = {
        "selected_candidate_features": len(selected_names),
        "history_features": len(history_names),
        "parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "best_epoch": best_epoch,
        "best_validation": {
            "loss": best_loss,
            "semantic_top1": best_top1,
        },
        "test": test_metrics,
        "v32_weighted_blend_reference": 0.7699600798403193,
        "target_top1": 0.9,
        "target_met": test_metrics["semantic_top1"] >= 0.9,
        "history": history_rows,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
