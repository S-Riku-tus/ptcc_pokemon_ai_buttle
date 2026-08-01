"""Train, honestly evaluate, and export the frozen Lopunny v2 policy.

Architecture and hyperparameters were selected without reading test:

* binary LambdaRank candidate model, 900 trees;
* DeepSets MAIN listwise residual, 11 epochs;
* decision-local blend ``deep_z + 2 * base_z`` at confidence >= 0.20;
* L1 variable pick-count model, 200 trees.

The first fit uses train only and scores validation plus test exactly once.
Exported artifacts are then refit on all frozen episodes at those fixed sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.core.distill import compact_booster  # noqa: E402
from scripts import experiment_lopunny_v2_deepsets as deep  # noqa: E402
from scripts import train_lopunny_top1_teacher as v1  # noqa: E402

RANKER_TREES = 900
COUNT_TREES = 200
DEEP_EPOCHS = 11
DEEP_SCHEDULE_EPOCHS = 50
BASE_WEIGHT = 2.0
CONFIDENCE = 0.20
SEED = 55137818


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_deep(
    arrays: dict[str, np.ndarray],
    names: list[str],
    fit_main: np.ndarray,
) -> dict[str, Any]:
    candidate_start = names.index("option_type")
    candidate_rows = v1._rows_for(arrays["groups"], fit_main)
    state_keep, state_mean, state_scale = deep._standardiser(
        arrays["count_features"], fit_main
    )
    candidate_raw = arrays["features"][:, candidate_start:]
    candidate_keep, candidate_mean, candidate_scale = deep._standardiser(
        candidate_raw, candidate_rows
    )
    return {
        "candidate_start": candidate_start,
        "state_keep": state_keep,
        "state_mean": state_mean,
        "state_scale": state_scale,
        "candidate_keep": candidate_keep,
        "candidate_mean": candidate_mean,
        "candidate_scale": candidate_scale,
        "states": deep._transform(
            arrays["count_features"], state_keep, state_mean, state_scale
        ),
        "candidates": deep._transform(
            candidate_raw, candidate_keep, candidate_mean, candidate_scale
        ),
    }


def _fit_deep(
    arrays: dict[str, np.ndarray],
    prepared: dict[str, Any],
    fit_main: np.ndarray,
) -> torch.nn.Module:
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    weights = np.ones(len(arrays["groups"]), dtype=np.float32)
    weights[fit_main] = v1._episode_recency(
        arrays["episode_ids"][fit_main], 0.40, 2.0
    )
    collator = deep.Collator(
        prepared["states"], prepared["candidates"], arrays["labels"],
        arrays["semantics"], arrays["groups"], weights,
    )
    loader = torch.utils.data.DataLoader(
        deep.DecisionDataset(fit_main), batch_size=128, shuffle=True,
        collate_fn=collator, generator=generator, num_workers=0,
    )
    model = deep.DeepSetRanker(
        prepared["states"].shape[1], prepared["candidates"].shape[1]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=2e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=DEEP_SCHEDULE_EPOCHS, eta_min=1e-5
    )
    for epoch in range(1, DEEP_EPOCHS + 1):
        model.train()
        loss_total = 0.0
        for state, candidate, valid, positive, weight, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = deep._loss(
                model(state, candidate, valid), valid, positive, weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.item())
        scheduler.step()
        print(json.dumps({
            "deep_epoch": epoch,
            "mean_batch_loss": loss_total / max(1, len(loader)),
        }), flush=True)
    return model.eval()


def _deep_predictions(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    prepared: dict[str, Any],
    decisions: np.ndarray,
) -> dict[int, np.ndarray]:
    main = decisions[
        (arrays["select_contexts"][decisions] == 0)
        & (arrays["forced"][decisions] == 0)
        & (arrays["chosen_counts"][decisions] == 1)
    ]
    weights = np.ones(len(arrays["groups"]), dtype=np.float32)
    collator = deep.Collator(
        prepared["states"], prepared["candidates"], arrays["labels"],
        arrays["semantics"], arrays["groups"], weights,
    )
    loader = torch.utils.data.DataLoader(
        deep.DecisionDataset(main), batch_size=128, shuffle=False,
        collate_fn=collator, num_workers=0,
    )
    predictions, _ = deep._predict(model, loader, torch.device("cpu"))
    return predictions


def _fit_ranker(
    arrays: dict[str, np.ndarray],
    names: list[str],
    fit_decisions: np.ndarray,
    varying: np.ndarray,
) -> lgb.LGBMRanker:
    rankable = fit_decisions[
        (arrays["chosen_counts"][fit_decisions] > 0)
        & (arrays["chosen_counts"][fit_decisions] < arrays["groups"][fit_decisions])
        & (arrays["forced"][fit_decisions] == 0)
    ]
    rows = v1._rows_for(arrays["groups"], rankable)
    groups = arrays["groups"][rankable].astype(int)
    selected_names = [names[index] for index in varying]
    model = lgb.LGBMRanker(**v1._ranker_params(SEED, RANKER_TREES, False))
    model.fit(
        arrays["features"][rows][:, varying], arrays["labels"][rows],
        group=groups,
        sample_weight=np.repeat(
            v1._episode_recency(arrays["episode_ids"][rankable], 0.40, 2.0),
            groups,
        ),
        feature_name=selected_names,
        categorical_feature=v1._categorical_columns(selected_names),
    )
    return model


def _fit_count(
    arrays: dict[str, np.ndarray],
    names: list[str],
    fit_decisions: np.ndarray,
) -> lgb.LGBMRegressor:
    variable = fit_decisions[
        arrays["minimums"][fit_decisions] < arrays["maximums"][fit_decisions]
    ]
    model = lgb.LGBMRegressor(**v1._count_params(SEED, COUNT_TREES))
    model.fit(
        arrays["count_features"][variable], arrays["chosen_counts"][variable],
        sample_weight=v1._episode_recency(
            arrays["episode_ids"][variable], 0.40, 2.0
        ),
        feature_name=names,
        categorical_feature=v1._categorical_columns(names),
    )
    return model


def _evaluate_split(
    ranker: lgb.LGBMRanker,
    count: lgb.LGBMRegressor,
    network: torch.nn.Module,
    prepared: dict[str, Any],
    arrays: dict[str, np.ndarray],
    varying: np.ndarray,
    decisions: np.ndarray,
) -> dict[str, Any]:
    rows = v1._rows_for(arrays["groups"], decisions)
    base_scores = ranker.predict(
        arrays["features"][rows][:, varying], num_iteration=RANKER_TREES
    ).astype(np.float32)
    counts = v1._predict_counts(
        count, arrays["count_features"], decisions,
        arrays["minimums"], arrays["maximums"], num_iteration=COUNT_TREES,
    )
    predictions = _deep_predictions(
        network, arrays, prepared, decisions
    )
    blended, applied = deep._blend(
        base_scores, predictions, decisions, arrays["groups"],
        BASE_WEIGHT, CONFIDENCE,
    )
    return {
        "base": v1.evaluate(base_scores, decisions, arrays, counts),
        "v2": v1.evaluate(blended, decisions, arrays, counts),
        "deep_applied": applied,
    }


def _compact_network(
    model: torch.nn.Module,
    prepared: dict[str, Any],
    count_names: list[str],
    feature_names: list[str],
) -> dict[str, Any]:
    state = model.state_dict()
    tensors = {
        name: tensor.detach().cpu().numpy().astype(np.float32).tolist()
        for name, tensor in state.items()
    }
    candidate_start = int(prepared["candidate_start"])
    return {
        "format": "lopunny_deepset_v1",
        "architecture": "state_96_48__candidate_64_48__pool_mean_max__score_96_48_1",
        "activation": "gelu_exact",
        "epochs": DEEP_EPOCHS,
        "base_weight": BASE_WEIGHT,
        "confidence_threshold": CONFIDENCE,
        "runtime_scope": "single_choice_main_context_0",
        "state_feature_names": [
            count_names[index] for index in prepared["state_keep"]
        ],
        "state_mean": prepared["state_mean"].tolist(),
        "state_scale": prepared["state_scale"].tolist(),
        "candidate_feature_names": [
            feature_names[candidate_start + index]
            for index in prepared["candidate_keep"]
        ],
        "candidate_mean": prepared["candidate_mean"].tolist(),
        "candidate_scale": prepared["candidate_scale"].tolist(),
        "tensors": tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    with np.load(args.cache, allow_pickle=False) as cached:
        arrays = {key: cached[key] for key in cached.files}
    names = arrays["feature_names"].astype(str).tolist()
    count_names = arrays["count_feature_names"].astype(str).tolist()
    splits = arrays["splits"].astype(str)
    decisions = {
        split: np.flatnonzero(splits == split)
        for split in ("train", "validation", "test")
    }
    starts, _ = v1._group_ranges(arrays["groups"])
    arrays["decision_turns"] = np.rint(
        arrays["features"][starts, names.index("turn")]
    ).astype(np.int16)
    arrays["turn_pick_sets"] = v1._turn_pick_sets(arrays)
    train_rankable = decisions["train"][
        (arrays["chosen_counts"][decisions["train"]] > 0)
        & (arrays["chosen_counts"][decisions["train"]]
           < arrays["groups"][decisions["train"]])
        & (arrays["forced"][decisions["train"]] == 0)
    ]
    varying = v1._varying_columns(
        arrays["features"], v1._rows_for(arrays["groups"], train_rankable)
    )
    train_main = decisions["train"][
        (arrays["select_contexts"][decisions["train"]] == 0)
        & (arrays["forced"][decisions["train"]] == 0)
        & (arrays["chosen_counts"][decisions["train"]] == 1)
    ]

    prepared = _prepare_deep(arrays, names, train_main)
    ranker = _fit_ranker(arrays, names, decisions["train"], varying)
    count = _fit_count(arrays, count_names, decisions["train"])
    network = _fit_deep(arrays, prepared, train_main)
    honest = {
        split: _evaluate_split(
            ranker, count, network, prepared, arrays, varying,
            decisions[split],
        )
        for split in ("validation", "test")
    }
    print(json.dumps({
        split: {
            "base": values["base"]["nonforced_semantic_exact"],
            "v2": values["v2"]["nonforced_semantic_exact"],
        }
        for split, values in honest.items()
    }), flush=True)

    all_decisions = np.arange(len(arrays["groups"]), dtype=np.int64)
    all_rankable = all_decisions[
        (arrays["chosen_counts"] > 0)
        & (arrays["chosen_counts"] < arrays["groups"])
        & (arrays["forced"] == 0)
    ]
    final_varying = v1._varying_columns(
        arrays["features"], v1._rows_for(arrays["groups"], all_rankable)
    )
    all_main = all_decisions[
        (arrays["select_contexts"] == 0)
        & (arrays["forced"] == 0)
        & (arrays["chosen_counts"] == 1)
    ]
    final_prepared = _prepare_deep(arrays, names, all_main)
    final_ranker = _fit_ranker(
        arrays, names, all_decisions, final_varying
    )
    final_count = _fit_count(arrays, count_names, all_decisions)
    final_network = _fit_deep(arrays, final_prepared, all_main)

    args.agent_dir.mkdir(parents=True, exist_ok=True)
    ranker_payload = compact_booster(final_ranker.booster_, "ranker")
    ranker_payload.update({
        "tree_count": RANKER_TREES,
        "tree_count_selected_by": "v1_validation_then_v2_frozen",
        "runtime_scope": "all_select_contexts_base",
        "teacher_team": "Majkel1337",
        "teacher_submission_id": 55137818,
        "teacher_trajectories": int(len(np.unique(arrays["episode_ids"]))),
        "training_decisions": int(len(all_rankable)),
        "label_definition": "binary_chosen",
        "recency_floor": 0.40,
        "recency_power": 2.0,
    })
    count_payload = compact_booster(final_count.booster_, "regressor")
    count_payload.update({
        "tree_count": COUNT_TREES,
        "runtime_scope": "variable_pick_count",
        "training_decisions": int(np.sum(
            arrays["minimums"] < arrays["maximums"]
        )),
    })
    network_payload = _compact_network(
        final_network, final_prepared, count_names, names
    )
    paths = {
        "ranker_model.json": ranker_payload,
        "count_model.json": count_payload,
        "deepset_model.json": network_payload,
    }
    exported = []
    for filename, payload in paths.items():
        path = args.agent_dir / filename
        path.write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        exported.append({
            "file": filename,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    report = {
        "cache": str(args.cache.resolve()),
        "architecture_frozen_before_test": True,
        "selection_source": "validation only",
        "hyperparameters": {
            "ranker_trees": RANKER_TREES,
            "count_trees": COUNT_TREES,
            "deep_epochs": DEEP_EPOCHS,
            "deep_schedule_epochs": DEEP_SCHEDULE_EPOCHS,
            "base_weight": BASE_WEIGHT,
            "confidence": CONFIDENCE,
        },
        "features": {
            "ranker": int(len(final_varying)),
            "deep_state": int(len(final_prepared["state_keep"])),
            "deep_candidate": int(len(final_prepared["candidate_keep"])),
        },
        "honest": honest,
        "target": {
            "metric": "test_nonforced_semantic_exact",
            "value": 0.85,
            "met": bool(
                honest["test"]["v2"]["nonforced_semantic_exact"] >= 0.85
            ),
        },
        "exported": exported,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "target": report["target"], "exported": exported
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
