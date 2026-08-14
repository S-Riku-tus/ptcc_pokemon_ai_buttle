"""Did the v24 mirror guard actually fire on the ladder, and did it help?

`analyse_corrected_froslass.py` measured the Froslass association on v22's
194-game pool.  V24 shipped a veto for exactly that action in the visible
mirror.  The offline footprint said it would bind 18 times in 194 games.  This
replays v24's own 87 ladder games with the same walker and asks:

1. how many true Froslass evolutions survive in v24 versus v22, per mirror
   game - if the guard bound, this rate must fall towards zero;
2. whether the win-rate contrast that motivated v24 is still present *inside*
   v24.  If the association was causal the contrast should collapse with the
   behaviour; if it persists, the column was a symptom of a losing board and
   the whole lever is confounded.

Same walker, same columns, two pools.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts", ROOT / "experiments/grimmsnarl_ml_v24",
             ROOT / "agents/grimmsnarl/grimmsnarl_ml_v22"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyse_corrected_froslass import block, contrast, walk  # noqa: E402

GAMES = ROOT / "experiments/grimmsnarl_ml_v24/ladder_v24_games.csv"
RUNS = ROOT / "data/runs/grimmsnarl"
OUT = ROOT / "experiments/grimmsnarl_ml_v24/v24_guard_binding.json"
EXACT_DECK_HASH = "9714ab5c3996f6cc"
MIRROR_FAMILY = "Grimmsnarl (mirror)"


def replay_index() -> dict[str, tuple[Path, int]]:
    index: dict[str, tuple[Path, int]] = {}
    for run_dir in sorted(RUNS.iterdir()):
        manifest = run_dir / "manifest.csv"
        if not run_dir.is_dir() or not manifest.exists():
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8-sig")):
            if row.get("detected_submission_agent_index") in {"0", "1"}:
                index[row["episode_id"]] = (
                    run_dir, int(row["detected_submission_agent_index"]))
    return index


def load() -> list[dict[str, Any]]:
    index = replay_index()
    rows: list[dict[str, Any]] = []
    for raw in csv.DictReader(GAMES.open(encoding="utf-8-sig")):
        version = raw["version"]
        if not (version.startswith("v22") or version.startswith("v24")):
            continue
        entry = index.get(raw["episode_id"])
        if entry is None:
            continue
        run_dir, seat = entry
        path = (run_dir / "episodes" / raw["episode_id"] / "replay"
                / f"episode_{raw['episode_id']}.json")
        if not path.exists():
            continue
        replay = json.loads(path.read_text(encoding="utf-8"))
        own = walk(replay, seat)
        other = walk(replay, 1 - seat)
        rows.append({
            "episode_id": raw["episode_id"],
            "pool": "v24" if version.startswith("v24") else "v22",
            "version": version,
            "won": raw["won"] == "True",
            "went_first": {"True": True, "False": False}.get(raw["went_first"]),
            "opponent_rating": float(raw["opponent_rating"]) if raw["opponent_rating"] else None,
            "family": raw["opponent_family"],
            "opponent_deck_hash": raw["opponent_deck_hash"],
            **own,
            "opponent_true_froslass_evolutions": other["true_froslass_evolutions"],
        })
    return rows


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return round(sum(r[key] for r in rows) / len(rows), 4) if rows else None


def summarise(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    mirrors = [r for r in rows if r["family"] == MIRROR_FAMILY]
    exact = [r for r in rows if r["opponent_deck_hash"] == EXACT_DECK_HASH]
    return {
        "pool": label,
        "all_games": {
            **block(rows),
            "true_evo_per_game": rate(rows, "true_froslass_evolutions"),
            "legacy_actions_per_game": rate(rows, "legacy_froslass_actions"),
        },
        "mirror_family": {
            **block(mirrors),
            "true_evo_per_game": rate(mirrors, "true_froslass_evolutions"),
            "bindable_evo_per_game": rate(mirrors, "bindable_froslass_evolutions"),
            "games_with_true_evo": sum(
                1 for r in mirrors if r["true_froslass_evolutions"] > 0),
            "games_with_bindable_evo": sum(
                1 for r in mirrors if r["bindable_froslass_evolutions"] > 0),
            "opponent_true_evo_per_game": rate(
                mirrors, "opponent_true_froslass_evolutions"),
        },
        "exact_list_mirror": {
            **block(exact),
            "true_evo_per_game": rate(exact, "true_froslass_evolutions"),
            "bindable_evo_per_game": rate(exact, "bindable_froslass_evolutions"),
        },
        "non_mirror": {
            **block([r for r in rows if r["family"] != MIRROR_FAMILY]),
            "true_evo_per_game": rate(
                [r for r in rows if r["family"] != MIRROR_FAMILY],
                "true_froslass_evolutions"),
        },
        "legacy_proxy_contrast": contrast(rows, "legacy_froslass_actions"),
        "true_evo_contrast": contrast(rows, "true_froslass_evolutions"),
    }


def main() -> int:
    rows = load()
    v22 = [r for r in rows if r["pool"] == "v22"]
    v24 = [r for r in rows if r["pool"] == "v24"]
    payload = {
        "loaded": {"v22": len(v22), "v24": len(v24)},
        "v22": summarise(v22, "v22"),
        "v24": summarise(v24, "v24"),
        "v24_mirror_games_with_surviving_evolution": [
            {
                "episode_id": r["episode_id"],
                "version": r["version"],
                "won": r["won"],
                "true": r["true_froslass_evolutions"],
                "bindable": r["bindable_froslass_evolutions"],
                "deck_hash": r["opponent_deck_hash"],
            }
            for r in v24
            if r["family"] == MIRROR_FAMILY and r["true_froslass_evolutions"] > 0
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nReport: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
