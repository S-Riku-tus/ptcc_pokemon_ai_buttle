from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def train_neural_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    seed: int = 741,
    epochs: int = 7,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    train_x = train[feature_columns].fillna(0).to_numpy(dtype=np.float32)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-5] = 1.0

    class Ranker(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(len(feature_columns), 48), nn.ReLU(),
                nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1),
            )

        def forward(self, value):
            return self.layers(value).squeeze(-1)

    model = Ranker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    groups: list[tuple[np.ndarray, np.ndarray, float]] = []
    for _, group in train.groupby("decision_id", sort=False):
        indices = group.index.to_numpy()
        local_x = (group[feature_columns].fillna(0).to_numpy(dtype=np.float32) - mean) / scale
        target = group["selected"].to_numpy(dtype=np.float32)
        target /= max(target.sum(), 1.0)
        groups.append((local_x, target, float(group.iloc[0]["teacher_weight"])))

    epoch_losses = []
    model.train()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(groups)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for index, (local_x, target, weight) in enumerate(groups, 1):
            x_tensor = torch.from_numpy(local_x)
            target_tensor = torch.from_numpy(target)
            scores = model(x_tensor)
            loss = -(target_tensor * torch.log_softmax(scores, dim=0)).sum() * weight
            (loss / 64.0).backward()
            running += float(loss.detach())
            if index % 64 == 0 or index == len(groups):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        epoch_losses.append(running / max(len(groups), 1))

    model.eval()
    with torch.no_grad():
        test_x = (test[feature_columns].fillna(0).to_numpy(dtype=np.float32) - mean) / scale
        scores = model(torch.from_numpy(test_x)).numpy()
    payload = {
        "format": "small_mlp_v1",
        "feature_names": feature_columns,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "layers": [
            {"weight": layer.weight.detach().numpy().tolist(), "bias": layer.bias.detach().numpy().tolist()}
            for layer in model.layers if isinstance(layer, nn.Linear)
        ],
        "epoch_losses": epoch_losses,
        "seed": seed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return scores, {"epoch_losses": epoch_losses, "parameter_count": sum(p.numel() for p in model.parameters())}

