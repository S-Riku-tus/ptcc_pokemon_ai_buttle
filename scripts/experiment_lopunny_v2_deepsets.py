"""Validation-only DeepSets listwise challenger for Lopunny v2.

Unlike LambdaRank, this model scores a candidate after pooling the complete
legal option set.  It can therefore learn comparisons such as "attach before
supporter only when both action families and a ready target are present"
without forcing a tree to rediscover the other candidates from count fields.

The neural model is trained only on train MAIN decisions.  Epoch, blend weight,
and confidence gate are selected on validation.  Test is never loaded here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import train_lopunny_top1_teacher as v1  # noqa: E402


def _standardiser(
    matrix: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = matrix[rows]
    minimum = np.nanmin(subset, axis=0)
    maximum = np.nanmax(subset, axis=0)
    keep = np.flatnonzero(minimum != maximum)
    mean = np.nanmean(subset[:, keep], axis=0).astype(np.float32)
    scale = np.nanstd(subset[:, keep], axis=0).astype(np.float32)
    scale[scale < 1e-5] = 1.0
    return keep, mean, scale


def _transform(
    matrix: np.ndarray,
    keep: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    result = (matrix[:, keep] - mean) / scale
    return np.clip(np.nan_to_num(result), -8.0, 8.0).astype(np.float32)


class DecisionDataset(torch.utils.data.Dataset):
    def __init__(self, decisions: np.ndarray):
        self.decisions = decisions.astype(np.int64)

    def __len__(self) -> int:
        return len(self.decisions)

    def __getitem__(self, index: int) -> int:
        return int(self.decisions[index])


class Collator:
    def __init__(
        self,
        states: np.ndarray,
        candidates: np.ndarray,
        labels: np.ndarray,
        semantics: np.ndarray,
        groups: np.ndarray,
        weights: np.ndarray,
    ):
        self.states = states
        self.candidates = candidates
        self.labels = labels
        self.semantics = semantics
        self.groups = groups
        self.weights = weights
        self.starts, self.ends = v1._group_ranges(groups)

    def __call__(self, decisions: list[int]):
        batch = len(decisions)
        width = max(int(self.groups[d]) for d in decisions)
        candidate = np.zeros(
            (batch, width, self.candidates.shape[1]), dtype=np.float32
        )
        valid = np.zeros((batch, width), dtype=bool)
        positive = np.zeros((batch, width), dtype=bool)
        state = self.states[np.asarray(decisions)]
        weight = self.weights[np.asarray(decisions)]
        for local, decision in enumerate(decisions):
            start, end = int(self.starts[decision]), int(self.ends[decision])
            size = end - start
            candidate[local, :size] = self.candidates[start:end]
            valid[local, :size] = True
            chosen = np.flatnonzero(self.labels[start:end] == 1)
            teacher_semantics = {
                tuple(self.semantics[start + row].tolist()) for row in chosen
            }
            positive[local, :size] = np.asarray([
                tuple(row.tolist()) in teacher_semantics
                for row in self.semantics[start:end]
            ])
        return (
            torch.from_numpy(state),
            torch.from_numpy(candidate),
            torch.from_numpy(valid),
            torch.from_numpy(positive),
            torch.from_numpy(weight),
            torch.as_tensor(decisions, dtype=torch.int64),
        )


class DeepSetRanker(nn.Module):
    def __init__(self, state_width: int, candidate_width: int):
        super().__init__()
        self.state = nn.Sequential(
            nn.Linear(state_width, 96), nn.GELU(), nn.Dropout(0.08),
            nn.Linear(96, 48), nn.GELU(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(candidate_width, 64), nn.GELU(),
            nn.Linear(64, 48), nn.GELU(),
        )
        self.score = nn.Sequential(
            nn.Linear(48 + 48 + 48 + 48, 96), nn.GELU(),
            nn.Dropout(0.08), nn.Linear(96, 48), nn.GELU(),
            nn.Linear(48, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        candidate: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        state_embedding = self.state(state)
        candidate_embedding = self.candidate(candidate)
        valid_float = valid.unsqueeze(-1).to(candidate_embedding.dtype)
        mean = (candidate_embedding * valid_float).sum(1) / valid_float.sum(1).clamp_min(1)
        masked = candidate_embedding.masked_fill(~valid.unsqueeze(-1), -1e4)
        maximum = masked.max(1).values
        context = torch.cat((state_embedding, mean, maximum), dim=1)
        context = context.unsqueeze(1).expand(-1, candidate.shape[1], -1)
        scores = self.score(torch.cat((candidate_embedding, context), dim=2)).squeeze(-1)
        return scores.masked_fill(~valid, -1e9)


def _loss(
    scores: torch.Tensor,
    valid: torch.Tensor,
    positive: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    all_scores = scores.masked_fill(~valid, -1e9)
    positive_scores = scores.masked_fill(~positive, -1e9)
    per_decision = (
        torch.logsumexp(all_scores, dim=1)
        - torch.logsumexp(positive_scores, dim=1)
    )
    return (per_decision * weights).sum() / weights.sum().clamp_min(1e-6)


@torch.no_grad()
def _predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], float]:
    model.eval()
    predictions: dict[int, np.ndarray] = {}
    correct = total = 0
    for state, candidate, valid, positive, _, decisions in loader:
        state, candidate, valid = (
            state.to(device), candidate.to(device), valid.to(device)
        )
        scores = model(state, candidate, valid).cpu()
        for row, decision in enumerate(decisions.tolist()):
            size = int(valid[row].sum().item())
            block = scores[row, :size].numpy().astype(np.float32)
            predictions[int(decision)] = block
            picked = int(np.argmax(block))
            correct += int(bool(positive[row, picked]))
            total += 1
    return predictions, correct / max(1, total)


def _fit_base(
    arrays: dict[str, np.ndarray],
    names: list[str],
    train: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    rankable = train[
        (arrays["chosen_counts"][train] > 0)
        & (arrays["chosen_counts"][train] < arrays["groups"][train])
        & (arrays["forced"][train] == 0)
    ]
    rows = v1._rows_for(arrays["groups"], rankable)
    varying = v1._varying_columns(arrays["features"], rows)
    selected_names = [names[index] for index in varying]
    model = lgb.LGBMRanker(**v1._ranker_params(55137818, 900, False))
    group_sizes = arrays["groups"][rankable].astype(int)
    model.fit(
        arrays["features"][rows][:, varying], arrays["labels"][rows],
        group=group_sizes,
        sample_weight=np.repeat(
            v1._episode_recency(arrays["episode_ids"][rankable], 0.40, 2.0),
            group_sizes,
        ),
        feature_name=selected_names,
        categorical_feature=v1._categorical_columns(selected_names),
    )
    validation = np.flatnonzero(arrays["splits"].astype(str) == "validation")
    validation_rows = v1._rows_for(arrays["groups"], validation)
    scores = model.predict(
        arrays["features"][validation_rows][:, varying], num_iteration=900
    ).astype(np.float32)

    count_names = arrays["count_feature_names"].astype(str).tolist()
    variable = train[arrays["minimums"][train] < arrays["maximums"][train]]
    count = lgb.LGBMRegressor(**v1._count_params(55137818, 200))
    count.fit(
        arrays["count_features"][variable], arrays["chosen_counts"][variable],
        sample_weight=v1._episode_recency(
            arrays["episode_ids"][variable], 0.40, 2.0
        ),
        feature_name=count_names,
        categorical_feature=v1._categorical_columns(count_names),
    )
    counts = v1._predict_counts(
        count, arrays["count_features"], validation,
        arrays["minimums"], arrays["maximums"], num_iteration=200,
    )
    return scores, counts


def _blend(
    base: np.ndarray,
    deep: dict[int, np.ndarray],
    validation: np.ndarray,
    groups: np.ndarray,
    base_weight: float,
    confidence: float,
) -> tuple[np.ndarray, int]:
    starts, ends = v1._group_ranges(groups[validation])
    result = base.copy()
    applied = 0
    for local, (start, end) in enumerate(zip(starts, ends)):
        decision = int(validation[local])
        if decision not in deep:
            continue
        neural = deep[decision].astype(np.float64)
        shifted = np.exp(np.clip(neural - neural.max(), -50, 0))
        probability = float(shifted.max() / shifted.sum())
        if probability < confidence:
            continue
        baseline = result[start:end].astype(np.float64)
        baseline = (baseline - baseline.mean()) / max(baseline.std(), 1e-5)
        neural = (neural - neural.mean()) / max(neural.std(), 1e-5)
        result[start:end] = neural + base_weight * baseline
        applied += 1
    return result, applied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    torch.manual_seed(55137818)
    np.random.seed(55137818)
    torch.set_num_threads(min(16, torch.get_num_threads()))
    device = torch.device("cpu")
    with np.load(args.cache, allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files}
    names = arrays["feature_names"].astype(str).tolist()
    splits = arrays["splits"].astype(str)
    train = np.flatnonzero(splits == "train")
    validation = np.flatnonzero(splits == "validation")
    main_train = train[
        (arrays["select_contexts"][train] == 0)
        & (arrays["forced"][train] == 0)
        & (arrays["chosen_counts"][train] == 1)
    ]
    main_validation = validation[
        (arrays["select_contexts"][validation] == 0)
        & (arrays["forced"][validation] == 0)
        & (arrays["chosen_counts"][validation] == 1)
    ]
    starts, ends = v1._group_ranges(arrays["groups"])
    candidate_start = names.index("option_type")
    candidate_rows = v1._rows_for(arrays["groups"], main_train)
    state_keep, state_mean, state_scale = _standardiser(
        arrays["count_features"], main_train
    )
    candidate_raw = arrays["features"][:, candidate_start:]
    candidate_keep, candidate_mean, candidate_scale = _standardiser(
        candidate_raw, candidate_rows
    )
    states = _transform(
        arrays["count_features"], state_keep, state_mean, state_scale
    )
    candidates = _transform(
        candidate_raw, candidate_keep, candidate_mean, candidate_scale
    )
    decision_weights = np.ones(len(arrays["groups"]), dtype=np.float32)
    decision_weights[main_train] = v1._episode_recency(
        arrays["episode_ids"][main_train], 0.40, 2.0
    )
    collator = Collator(
        states, candidates, arrays["labels"], arrays["semantics"],
        arrays["groups"], decision_weights,
    )
    generator = torch.Generator().manual_seed(55137818)
    train_loader = torch.utils.data.DataLoader(
        DecisionDataset(main_train), batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, generator=generator, num_workers=0,
    )
    validation_loader = torch.utils.data.DataLoader(
        DecisionDataset(main_validation), batch_size=args.batch_size,
        shuffle=False, collate_fn=collator, num_workers=0,
    )
    model = DeepSetRanker(states.shape[1], candidates.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=2e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    curve = []
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    patience = 10
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_batches = 0
        for state, candidate, valid, positive, weight, _ in train_loader:
            state, candidate, valid, positive, weight = (
                state.to(device), candidate.to(device), valid.to(device),
                positive.to(device), weight.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(state, candidate, valid), valid, positive, weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1
        scheduler.step()
        _, accuracy = _predict(model, validation_loader, device)
        curve.append({
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_batches),
            "validation_main_semantic_top1": accuracy,
        })
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        print(json.dumps(curve[-1]), flush=True)
        if epoch - best_epoch >= patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    deep_scores, deep_accuracy = _predict(model, validation_loader, device)

    group_starts, _ = v1._group_ranges(arrays["groups"])
    arrays["decision_turns"] = np.rint(
        arrays["features"][group_starts, names.index("turn")]
    ).astype(np.int16)
    arrays["turn_pick_sets"] = v1._turn_pick_sets(arrays)
    base_scores, counts = _fit_base(arrays, names, train)
    base_metrics = v1.evaluate(base_scores, validation, arrays, counts)
    blends = []
    for confidence in (0.0, 0.20, 0.30, 0.40, 0.50, 0.60):
        for base_weight in (0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0):
            scores, applied = _blend(
                base_scores, deep_scores, validation, arrays["groups"],
                base_weight, confidence,
            )
            metrics = v1.evaluate(scores, validation, arrays, counts)
            blends.append({
                "base_weight": base_weight,
                "confidence": confidence,
                "applied": applied,
                "nonforced_semantic_exact": metrics["nonforced_semantic_exact"],
                "single_top1": metrics["single_choice_semantic_top1"],
                "main_top1": metrics["main_single_choice_semantic_top1"],
            })
    selected = max(
        blends,
        key=lambda row: (
            row["nonforced_semantic_exact"], row["main_top1"],
            -row["base_weight"], -row["confidence"],
        ),
    )
    report: dict[str, Any] = {
        "cache": str(args.cache.resolve()),
        "test_read": False,
        "train_main_decisions": int(len(main_train)),
        "validation_main_decisions": int(len(main_validation)),
        "state_features": int(states.shape[1]),
        "candidate_features": int(candidates.shape[1]),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "fit_seconds": time.perf_counter() - started,
        "best_epoch": best_epoch,
        "deep_main_semantic_top1": deep_accuracy,
        "base_validation": base_metrics,
        "selected_blend": selected,
        "curve": curve,
        "blends": blends,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "deep_main_top1": deep_accuracy,
        "base_exact": base_metrics["nonforced_semantic_exact"],
        "selected": selected,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
