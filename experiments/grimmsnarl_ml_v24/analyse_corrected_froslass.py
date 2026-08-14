"""Separate true Froslass evolutions from the legacy broad card-action count.

``analyze_grimmsnarl_v20_ladder.py`` names a column ``froslass_evolves`` but
increments it for every selected Froslass card whose action is not an attack,
ability, end or retreat.  Deck-search selections therefore enter the same
column.  V24 must distinguish the measured line-investment proxy from a real
``action_type == evolve`` before choosing the intervention.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts", ROOT / "agents/grimmsnarl/grimmsnarl_ml_v22"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ml_features as mf  # noqa: E402

GAMES = ROOT / "experiments/grimmsnarl_ml_v23/ladder_v22_v23_games.csv"
RUNS = ROOT / "data/runs/grimmsnarl"
OUT = ROOT / "experiments/grimmsnarl_ml_v24/corrected_froslass.json"
FOOTPRINT = ROOT / "experiments/grimmsnarl_ml_v24/footprint_v22_v24.json"
EXACT_DECK_HASH = "9714ab5c3996f6cc"
ELO = 400.0 / math.log(10.0)


def _single(action: Any) -> int | None:
    if isinstance(action, list) and len(action) == 1 and isinstance(action[0], int):
        return int(action[0])
    return None


def _public_ids(player: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for area in ("active", "bench", "discard"):
        for card in mf._cards(player, area):
            result.add(int(card.get("id", -1)))
            for previous in card.get("preEvolution") or []:
                if isinstance(previous, dict):
                    result.add(int(previous.get("id", -1)))
    result.discard(-1)
    return result


def walk(replay: dict[str, Any], seat: int) -> dict[str, int]:
    result = Counter({
        "legacy_froslass_actions": 0,
        "true_froslass_evolutions": 0,
        "froslass_to_hand": 0,
        "snorunt_to_hand": 0,
        "snorunt_to_bench": 0,
        "bindable_froslass_evolutions": 0,
    })
    mirror_visible = False
    line = {mf.IMPIDIMP_ID, mf.MORGREM_ID, mf.GRIMMSNARL_EX_ID}
    steps = replay.get("steps") or []
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        obs = record.get("observation") or {}
        select = obs.get("select")
        current = obs.get("current")
        if not isinstance(select, dict) or not isinstance(current, dict):
            continue
        players = current.get("players") or []
        if len(players) < 2:
            continue
        mirror_visible = mirror_visible or bool(_public_ids(players[1 - seat]) & line)
        options = list(select.get("option") or [])
        chosen = _single((steps[index + 1][seat] or {}).get("action"))
        if chosen is None or not 0 <= chosen < len(options):
            continue
        option = options[chosen]
        try:
            kind = mf.action_type(current, option, select)
            card = mf.candidate_card(current, option, select) or {}
        except Exception:  # noqa: BLE001
            continue
        card_id = int(card.get("id", -1))
        context = int(select.get("context", -1))

        if card_id == mf.FROSLASS_ID:
            if kind not in {"attack", "ability", "end", "retreat"}:
                result["legacy_froslass_actions"] += 1
            if context == mf.CTX_TO_HAND:
                result["froslass_to_hand"] += 1
            if kind == "evolve":
                result["true_froslass_evolutions"] += 1
                result["bindable_froslass_evolutions"] += int(mirror_visible)
        elif card_id == mf.SNORUNT_ID:
            result["snorunt_to_hand"] += int(context == mf.CTX_TO_HAND)
            result["snorunt_to_bench"] += int(
                kind == "bench" or context == 5
            )
    return dict(result)


def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for row in rows if row["won"])
    return {
        "games": len(rows),
        "wins": wins,
        "losses": len(rows) - wins,
        "win_rate": round(wins / len(rows), 4) if rows else None,
    }


def fit(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    X, y = [], []
    for row in rows:
        if row["opponent_rating"] is None or row["went_first"] is None:
            continue
        X.append([
            row["opponent_rating"] / 400.0,
            float(row["went_first"]),
            float(row[key] > 0),
        ])
        y.append(int(row["won"]))
    matrix = np.asarray(X, float)
    target = np.asarray(y, int)
    if (
        len(target) < 12
        or len(set(target.tolist())) < 2
        or len(set(matrix[:, 2].tolist())) < 2
    ):
        return {"n": len(target), "error": "insufficient variation"}
    model = LogisticRegression(penalty=None, max_iter=8000).fit(matrix, target)
    probabilities = model.predict_proba(matrix)[:, 1]
    design = np.hstack([matrix, np.ones((len(matrix), 1))])
    try:
        covariance = np.linalg.inv(
            design.T @ np.diag(probabilities * (1 - probabilities)) @ design
        )
    except np.linalg.LinAlgError:
        return {"n": len(target), "error": "singular covariance"}
    se = float(np.sqrt(np.diag(covariance))[2])
    coefficient = float(model.coef_[0][2])
    z = coefficient / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {
        "n": len(target),
        "elo": round(coefficient * ELO, 1),
        "z": round(z, 2),
        "p": round(p, 4),
    }


def contrast(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    none = [row for row in rows if row[key] == 0]
    some = [row for row in rows if row[key] > 0]
    table = [
        [sum(row["won"] for row in some), sum(not row["won"] for row in some)],
        [sum(row["won"] for row in none), sum(not row["won"] for row in none)],
    ]
    return {
        "none": block(none),
        "one_or_more": block(some),
        "events": sum(row[key] for row in rows),
        "fisher_p": round(float(fisher_exact(table).pvalue), 4),
        "controlled": fit(rows, key),
    }


def describe_option(
    current: dict[str, Any], select: dict[str, Any], option: dict[str, Any]
) -> str:
    try:
        kind = mf.action_type(current, option, select)
        card = mf.candidate_card(current, option, select) or {}
        card_id = int(card.get("id", -1))
        if card_id >= 0:
            return f"{kind}:{card_id}"
        attack_id = int(option.get("attackId", -1))
        return f"{kind}:{attack_id}" if attack_id >= 0 else kind
    except Exception:  # noqa: BLE001
        return "unresolved"


def main() -> int:
    replay_index: dict[str, tuple[Path, int]] = {}
    for run_dir in sorted(RUNS.iterdir()):
        manifest = run_dir / "manifest.csv"
        if not run_dir.is_dir() or not manifest.exists():
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            if row.get("detected_submission_agent_index") in {"0", "1"}:
                replay_index[row["episode_id"]] = (
                    run_dir, int(row["detected_submission_agent_index"])
                )

    rows: list[dict[str, Any]] = []
    legacy_mismatches = []
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        if not raw["version"].startswith("v22"):
            continue
        entry = replay_index.get(raw["episode_id"])
        if entry is None:
            continue
        run_dir, seat = entry
        path = (
            run_dir
            / "episodes"
            / raw["episode_id"]
            / "replay"
            / f"episode_{raw['episode_id']}.json"
        )
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        own = walk(replay, seat)
        other = walk(replay, 1 - seat)
        if own["legacy_froslass_actions"] != int(raw["froslass_evolves"]):
            legacy_mismatches.append(raw["episode_id"])
        rows.append({
            "episode_id": raw["episode_id"],
            "version": raw["version"],
            "won": raw["won"] == "True",
            "went_first": {"True": True, "False": False}.get(raw["went_first"]),
            "opponent_rating": float(raw["opponent_rating"]) if raw["opponent_rating"] else None,
            "family": raw["opponent_family"],
            "opponent_deck_hash": raw["opponent_deck_hash"],
            **own,
            "opponent_true_froslass_evolutions": other["true_froslass_evolutions"],
        })

    mirrors = [row for row in rows if row["family"] == "Grimmsnarl (mirror)"]
    exact = [row for row in mirrors if row["opponent_deck_hash"] == EXACT_DECK_HASH]
    footprint_audit: dict[str, Any] = {"available": False}
    if FOOTPRINT.exists():
        footprint = json.loads(FOOTPRINT.read_text(encoding="utf-8"))
        replacements: Counter[str] = Counter()
        for episode_id, diffs in (footprint.get("diffs") or {}).items():
            entry = replay_index.get(str(episode_id))
            if entry is None:
                continue
            run_dir, seat = entry
            path = run_dir / "episodes" / str(episode_id) / "replay" / f"episode_{episode_id}.json"
            replay = json.loads(path.read_text(encoding="utf-8"))
            for diff in diffs:
                record = (replay["steps"][int(diff["step"])][seat] or {})
                obs = record.get("observation") or {}
                select = obs.get("select") or {}
                current = obs.get("current") or {}
                options = list(select.get("option") or [])
                before = int(diff["v22"])
                after = int(diff["v23"])
                if 0 <= before < len(options) and 0 <= after < len(options):
                    key = (
                        f"{describe_option(current, select, options[before])} -> "
                        f"{describe_option(current, select, options[after])}"
                    )
                    replacements[key] += 1
        footprint_audit = {
            "available": True,
            "valid": bool(footprint.get("valid")),
            "games": footprint.get("games"),
            "decisions": (footprint.get("totals") or {}).get("decisions"),
            "changed": (footprint.get("totals") or {}).get("changed"),
            "games_touched": (footprint.get("totals") or {}).get("games_touched"),
            "replacements": dict(sorted(replacements.items())),
        }

    mirror_mismatches = [
        episode_id for episode_id in legacy_mismatches
        if any(row["episode_id"] == episode_id for row in mirrors)
    ]
    report = {
        "valid_for_mirror_question": not mirror_mismatches,
        "legacy_mirror_mismatches": mirror_mismatches,
        "valid_for_all_matchups": not legacy_mismatches,
        "legacy_mismatches": legacy_mismatches,
        "games": len(rows),
        "mirrors": len(mirrors),
        "legacy_column_definition": (
            "Selected Froslass card actions excluding attack, ability, end and retreat; "
            "includes searches and true evolutions."
        ),
        "mirror_contrasts": {
            "legacy_broad_action": contrast(mirrors, "legacy_froslass_actions"),
            "true_evolution": contrast(mirrors, "true_froslass_evolutions"),
            "froslass_to_hand": contrast(mirrors, "froslass_to_hand"),
        },
        "binding": {
            "true_evolutions": sum(row["true_froslass_evolutions"] for row in mirrors),
            "publicly_bindable_evolutions": sum(
                row["bindable_froslass_evolutions"] for row in mirrors
            ),
            "games_with_true_evolution": sum(
                row["true_froslass_evolutions"] > 0 for row in mirrors
            ),
            "games_publicly_bindable": sum(
                row["bindable_froslass_evolutions"] > 0 for row in mirrors
            ),
        },
        "exact_60_current_symmetry": {
            "games": len(exact),
            "winner_evolutions_per_game": round(sum(
                row["true_froslass_evolutions"] if row["won"]
                else row["opponent_true_froslass_evolutions"] for row in exact
            ) / max(1, len(exact)), 3),
            "loser_evolutions_per_game": round(sum(
                row["opponent_true_froslass_evolutions"] if row["won"]
                else row["true_froslass_evolutions"] for row in exact
            ) / max(1, len(exact)), 3),
            "ours_in_wins_per_game": round(sum(
                row["true_froslass_evolutions"] for row in exact if row["won"]
            ) / max(1, sum(row["won"] for row in exact)), 3),
            "ours_in_losses_per_game": round(sum(
                row["true_froslass_evolutions"] for row in exact if not row["won"]
            ) / max(1, sum(not row["won"] for row in exact)), 3),
            "opponent_in_our_losses_per_game": round(sum(
                row["opponent_true_froslass_evolutions"] for row in exact if not row["won"]
            ) / max(1, sum(not row["won"] for row in exact)), 3),
        },
        "v22_v24_footprint_audit": footprint_audit,
        "by_run_true_evolution": {
            version: contrast(
                [row for row in mirrors if row["version"] == version],
                "true_froslass_evolutions",
            )
            for version in sorted({row["version"] for row in mirrors})
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
