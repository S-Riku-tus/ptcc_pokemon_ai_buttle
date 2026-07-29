"""Train a leakage-free DeepSets policy over complete legal-option groups.

v31's tree rankers score each candidate independently.  This challenger
encodes every candidate, pools the complete legal set, and then scores each
candidate conditioned on the pooled alternatives.  Feature/model selection
uses only the frozen chronological training and validation episodes; the
chronological test episodes are reported once after early stopping.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


MANDATORY_FEATURES = {
    "turn",
    "turn_action_count",
    "self_hand_count",
    "self_deck_count",
    "self_prize_count",
    "opp_hand_count",
    "opp_deck_count",
    "opp_prize_count",
    "self_active_id",
    "opp_active_id",
    "option_type",
    "candidate_option_position",
    "candidate_option_reverse_position",
    "candidate_raw_index",
    "candidate_raw_inplay_index",
    "candidate_raw_player_relative",
    "candidate_same_action_preceding",
    "candidate_same_card_preceding",
    "candidate_is_first_action_copy",
    "candidate_is_first_card_copy",
    "candidate_card_id",
    "candidate_attack_id",
    "candidate_target_id",
    "candidate_target_hp",
    "candidate_target_energy",
    "action_type",
    "fallback_selected",
    "fallback_action_type",
    "fallback_card_id",
    "fallback_policy_score",
    "fallback_policy_score_gap",
    "fallback_policy_rank",
    "legacy_ranker_score",
    "legacy_ranker_score_gap",
    "legacy_ranker_rank",
    "v29_selected",
    "v29_ranker_score",
    "v29_ranker_score_gap",
    "v29_ranker_rank",
    "v29_ranker_raw_selected",
}


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


def _count_splits(node: dict[str, Any], counts: Counter[int]) -> None:
    feature = node.get("f")
    if feature is None:
        return
    counts[int(feature)] += 1
    _count_splits(node["l"], counts)
    _count_splits(node["r"], counts)


def _select_features(
    names: list[str],
    model_paths: Iterable[Path],
    limit: int,
) -> list[int]:
    counts: Counter[int] = Counter()
    for path in model_paths:
        model = json.loads(path.read_text(encoding="utf-8"))
        if model["feature_names"] != names:
            raise RuntimeError(f"Feature schema mismatch: {path}")
        for tree in model["trees"]:
            _count_splits(tree, counts)
    selected = {
        index
        for index, _ in counts.most_common(limit)
    }
    selected.update(
        index for index, name in enumerate(names)
        if name in MANDATORY_FEATURES
    )
    return sorted(selected)


def _categorical(name: str) -> bool:
    return (
        name.endswith("_id")
        or name in {
            "action_type",
            "option_type",
            "select_type",
            "select_context",
            "fallback_action_type",
        }
    )


def _normalise_continuous(
    features: np.ndarray,
    columns: list[int],
    train_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = np.zeros(len(columns), dtype=np.float64)
    squared = np.zeros(len(columns), dtype=np.float64)
    chunk_size = 50_000
    for start in range(0, len(train_rows), chunk_size):
        rows = train_rows[start:start + chunk_size]
        values = features[rows][:, columns].astype(np.float64)
        total += values.sum(axis=0)
        squared += np.square(values).sum(axis=0)
    mean = total / len(train_rows)
    variance = np.maximum(
        squared / len(train_rows) - np.square(mean),
        0.0,
    )
    std = np.sqrt(variance)
    std[std < 1e-5] = 1.0
    output = np.empty((len(features), len(columns)), dtype=np.float16)
    for start in range(0, len(features), chunk_size):
        end = min(start + chunk_size, len(features))
        values = (
            features[start:end, columns].astype(np.float32)
            - mean.astype(np.float32)
        ) / std.astype(np.float32)
        output[start:end] = np.clip(values, -8.0, 8.0).astype(np.float16)
    return output, mean.astype(np.float32), std.astype(np.float32)


def _encode_categorical(
    features: np.ndarray,
    columns: list[int],
    train_rows: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    output = np.zeros((len(features), len(columns)), dtype=np.int16)
    vocabularies: list[np.ndarray] = []
    sizes: list[int] = []
    for output_column, source_column in enumerate(columns):
        vocabulary = np.unique(features[train_rows, source_column])
        vocabulary.sort()
        raw = features[:, source_column]
        positions = np.searchsorted(vocabulary, raw)
        clipped = np.minimum(positions, len(vocabulary) - 1)
        known = (
            (positions < len(vocabulary))
            & (vocabulary[clipped] == raw)
        )
        # 0 is padding, 1..N are train values, N+1 is unknown.
        output[:, output_column] = np.where(
            known,
            positions + 1,
            len(vocabulary) + 1,
        ).astype(np.int16)
        vocabularies.append(vocabulary.astype(np.float32))
        sizes.append(len(vocabulary) + 2)
    return output, vocabularies, sizes


class GroupStore:
    def __init__(
        self,
        continuous: np.ndarray,
        categorical: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        self.continuous = continuous
        self.categorical = categorical
        self.groups = groups
        self.starts, self.ends = _ranges(groups)
        self.targets = np.empty(len(groups), dtype=np.int64)
        self.weights = np.empty(len(groups), dtype=np.float32)
        for decision, (start, end) in enumerate(
            zip(self.starts, self.ends)
        ):
            positives = np.flatnonzero(labels[start:end] == 1)
            if len(positives) != 1:
                raise RuntimeError(
                    f"Decision {decision} has {len(positives)} positives"
                )
            self.targets[decision] = int(positives[0])
            self.weights[decision] = float(weights[start])

    def batch(
        self,
        decisions: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        counts = self.groups[decisions].astype(int)
        width = int(counts.max())
        continuous = np.zeros(
            (len(decisions), width, self.continuous.shape[1]),
            dtype=np.float32,
        )
        categorical = np.zeros(
            (len(decisions), width, self.categorical.shape[1]),
            dtype=np.int64,
        )
        mask = np.zeros((len(decisions), width), dtype=bool)
        for row, decision in enumerate(decisions):
            start, end = self.starts[decision], self.ends[decision]
            count = end - start
            continuous[row, :count] = self.continuous[start:end]
            categorical[row, :count] = self.categorical[start:end]
            mask[row, :count] = True
        return (
            torch.from_numpy(continuous).to(device),
            torch.from_numpy(categorical).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(self.targets[decisions]).to(device),
            torch.from_numpy(self.weights[decisions]).to(device),
        )


class DeepSetPolicy(nn.Module):
    def __init__(
        self,
        continuous_features: int,
        categorical_sizes: list[int],
        hidden: int,
        dropout: float,
        state_features: int = 0,
    ) -> None:
        super().__init__()
        embedding_dims = [
            min(16, max(3, int(math.ceil(math.sqrt(size)))))
            for size in categorical_sizes
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, dimension, padding_idx=0)
            for size, dimension in zip(categorical_sizes, embedding_dims)
        ])
        input_features = continuous_features + sum(embedding_dims)
        self.candidate = nn.Sequential(
            nn.Linear(input_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.state = (
            nn.Sequential(
                nn.Linear(state_features, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            if state_features
            else None
        )
        self.score = nn.Sequential(
            nn.Linear(hidden * (5 if state_features else 4), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        mask: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded = [
            embedding(categorical[..., index])
            for index, embedding in enumerate(self.embeddings)
        ]
        inputs = torch.cat([continuous, *embedded], dim=-1)
        candidates = self.candidate(inputs)
        mask_float = mask.unsqueeze(-1).to(candidates.dtype)
        mean = (
            (candidates * mask_float).sum(dim=1)
            / mask_float.sum(dim=1).clamp_min(1.0)
        )
        maximum = candidates.masked_fill(
            ~mask.unsqueeze(-1), -1e4
        ).amax(dim=1)
        context = self.context(torch.cat([mean, maximum], dim=-1))
        mean_expanded = mean.unsqueeze(1).expand_as(candidates)
        context_expanded = context.unsqueeze(1).expand_as(candidates)
        score_parts = [
            candidates,
            context_expanded,
            candidates - mean_expanded,
            candidates * mean_expanded,
        ]
        if self.state is not None:
            if state is None:
                raise ValueError("state tensor is required")
            state_encoded = self.state(state)
            score_parts.append(
                state_encoded.unsqueeze(1).expand_as(candidates)
            )
        scores = self.score(torch.cat(
            score_parts, dim=-1
        )).squeeze(-1)
        return scores.masked_fill(~mask, -1e9)


class SetAttentionPolicy(DeepSetPolicy):
    def __init__(
        self,
        continuous_features: int,
        categorical_sizes: list[int],
        hidden: int,
        dropout: float,
        layers: int = 2,
    ) -> None:
        super().__init__(
            continuous_features,
            categorical_sizes,
            hidden,
            dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.attention_score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        mask: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state is not None:
            raise ValueError("SetAttentionPolicy does not consume state")
        embedded = [
            embedding(categorical[..., index])
            for index, embedding in enumerate(self.embeddings)
        ]
        inputs = torch.cat([continuous, *embedded], dim=-1)
        candidates = self.candidate(inputs)
        attended = self.attention(
            candidates,
            src_key_padding_mask=~mask,
        )
        scores = self.attention_score(attended).squeeze(-1)
        return scores.masked_fill(~mask, -1e9)


def _evaluate(
    model: nn.Module,
    store: GroupStore,
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
            continuous, categorical, mask, targets, _ = store.batch(
                batch_decisions, device
            )
            scores = model(continuous, categorical, mask)
            loss_sum += float(
                nn.functional.cross_entropy(
                    scores, targets, reduction="sum"
                ).item()
            )
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


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
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
    parser.add_argument(
        "--architecture",
        choices=("deepset", "attention"),
        default="deepset",
    )
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=743)
    parser.add_argument("--recency-floor", type=float)
    parser.add_argument("--recency-power", type=float)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(12, max(1, torch.get_num_threads())))
    device = torch.device("cpu")

    with np.load(args.schema_cache, allow_pickle=False) as schema:
        schema_names = schema["feature_names"].astype(str).tolist()
    selected_schema_columns = _select_features(
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
        cache_names = cached["feature_names"].astype(str).tolist()
        cache_columns = [cache_names.index(name) for name in selected_names]
        features = cached["features"][:, cache_columns].astype(
            np.float32, copy=True
        )
        labels = cached["labels"].copy()
        weights = cached["weights"].copy()
        groups = cached["groups"].copy()
        splits = cached["splits"].astype(str)
        episode_ids = cached["episode_ids"].copy()

    train_decisions = np.flatnonzero(splits == "train")
    validation_decisions = np.flatnonzero(splits == "validation")
    test_decisions = np.flatnonzero(splits == "test")
    starts, ends = _ranges(groups)
    if (args.recency_floor is None) != (args.recency_power is None):
        parser.error("--recency-floor and --recency-power must be used together")
    if args.recency_floor is not None and args.recency_power is not None:
        ordered = np.unique(episode_ids[train_decisions])
        ordered.sort()
        positions = {
            int(episode): index / max(len(ordered) - 1, 1)
            for index, episode in enumerate(ordered)
        }
        for decision in train_decisions:
            multiplier = (
                args.recency_floor
                + (1.0 - args.recency_floor)
                * positions[int(episode_ids[decision])] ** args.recency_power
            )
            weights[starts[decision]:ends[decision]] *= multiplier
    train_rows = np.concatenate([
        np.arange(starts[decision], ends[decision], dtype=np.int64)
        for decision in train_decisions
    ])

    categorical_local = [
        index for index, name in enumerate(selected_names)
        if _categorical(name)
    ]
    continuous_local = [
        index for index in range(len(selected_names))
        if index not in set(categorical_local)
    ]
    continuous, mean, std = _normalise_continuous(
        features, continuous_local, train_rows
    )
    categorical, vocabularies, categorical_sizes = _encode_categorical(
        features, categorical_local, train_rows
    )
    del features
    store = GroupStore(
        continuous, categorical, groups, labels, weights
    )
    model = (
        SetAttentionPolicy(
            len(continuous_local),
            categorical_sizes,
            args.hidden,
            args.dropout,
            args.attention_layers,
        )
        if args.architecture == "attention"
        else DeepSetPolicy(
            len(continuous_local),
            categorical_sizes,
            args.hidden,
            args.dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        weight_decay=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=8e-5,
    )

    rng = np.random.default_rng(args.seed)
    history = []
    best_top1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = rng.permutation(train_decisions)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(shuffled), args.batch_size):
            decisions = shuffled[start:start + args.batch_size]
            continuous_batch, categorical_batch, mask, targets, batch_weights = (
                store.batch(decisions, device)
            )
            optimizer.zero_grad(set_to_none=True)
            scores = model(continuous_batch, categorical_batch, mask)
            losses = nn.functional.cross_entropy(
                scores, targets, reduction="none"
            )
            loss = (
                losses * batch_weights
            ).sum() / batch_weights.sum().clamp_min(1e-6)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(decisions)
            seen += len(decisions)
        scheduler.step()
        validation = _evaluate(
            model,
            store,
            validation_decisions,
            args.batch_size,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = (
            validation["semantic_top1"] > best_top1
            or (
                validation["semantic_top1"] == best_top1
                and validation["loss"] < best_loss
            )
        )
        if improved:
            best_top1 = validation["semantic_top1"]
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = _state_dict_cpu(model)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    test = _evaluate(
        model,
        store,
        test_decisions,
        args.batch_size,
        device,
    )
    checkpoint = {
        "format": "alakazam_v32_deepset_v1",
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
        "hidden": args.hidden,
        "dropout": args.dropout,
        "architecture": args.architecture,
        "attention_layers": args.attention_layers,
        "best_epoch": best_epoch,
        "seed": args.seed,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    report = {
        "cache": str(args.cache.resolve()),
        "schema_cache": str(args.schema_cache.resolve()),
        "selected_features": len(selected_names),
        "continuous_features": len(continuous_local),
        "categorical_features": len(categorical_local),
        "parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "architecture": args.architecture,
        "split_decisions": {
            "train": len(train_decisions),
            "validation": len(validation_decisions),
            "test": len(test_decisions),
        },
        "selection_rule": (
            "best validation Top-1, validation loss tie-break; "
            "chronological test evaluated after selection"
        ),
        "recency_weight": (
            {
                "floor": args.recency_floor,
                "power": args.recency_power,
                "episode_order": "ascending_episode_id",
            }
            if args.recency_floor is not None
            else None
        ),
        "best_epoch": best_epoch,
        "best_validation": {
            "loss": best_loss,
            "semantic_top1": best_top1,
        },
        "test": test,
        "v31_reference_top1": 0.7630988023952096,
        "target_top1": 0.9,
        "target_met": test["semantic_top1"] >= 0.9,
        "history": history,
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
