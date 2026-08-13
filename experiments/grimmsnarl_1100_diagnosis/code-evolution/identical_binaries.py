"""How much of the v1..v21 rating spread is inside the same-binary noise?

Six pairs of ladder submissions in this project shipped byte-identical code
(``deck_snapshot`` md5 over *.py + ranker_model.json).  Their rating gaps are
pure noise by construction, so they measure the instrument.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "data" / "runs" / "grimmsnarl"
TRAJ = json.loads(
    (ROOT / "experiments" / "grimmsnarl_1100_diagnosis" / "code-evolution"
     / "trajectory.json").read_text(encoding="utf-8")
)

LADDER_DIR = {
    "v1": "20260803_grimmsnarl_ml_v1_sub55185513",
    "v2": "20260803_grimmsnarl_ml_v2_sub55205556",
    "v3a": "20260805_grimmsnarl_ml_v3_sub55216787",
    "v3b": "20260805_grimmsnarl_ml_v3_sub55217233",
    "v4": "20260805_grimmsnarl_ml_v4_sub55253296",
    "v4.5": "20260806_grimmsnarl_ml_v4_5_sub55275464",
    "v5": "20260806_grimmsnarl_ml_v5_sub55275642",
    "v6": "20260806_grimmsnarl_ml_v6_sub55290882",
    "v7": "20260806_grimmsnarl_ml_v7_sub_55302846",
    "v8": "20260807_grimmsnarl_ml_v8_sub_55317804",
    "v11": "20260809_grimmsnarl_ml_v11_sub55353978",
    "v12a": "20260809_grimmsnarl_ml_v12_a_sub55373676",
    "v12b": "20260809_grimmsnarl_ml_v12_b_sub55374240",
    "v13a": "20260810_grimmsnarl_ml_v13_a_sub55380882",
    "v13b": "20260810_grimmsnarl_ml_v13_b_sub55380958",
    "v14": "20260810_grimmsnarl_ml_v14_sub55395386",
    "v15": "20260810_grimmsnarl_ml_v15_sub55404196",
    "v15b": "20260811_grimmsnarl_ml_v15_b_sub55409394",
    "v16": "20260811_grimmsnarl_ml_v16_sub55422280",
    "v17": "20260811_grimmsnarl_ml_v17_sub55423572",
    "v18": "20260811_grimmsnarl_ml_v18_sub55428191",
    "v19a": "20260811_grimmsnarl_ml_v19_sub55428196",
    "v19b": "20260812_grimmsnarl_ml_v19_sub55445763",
    "v20": "20260812_grimmsnarl_ml_v20_sub55445769",
    "v21": "20260813_grimmsnarl_ml_v21_sub55456713",
}


def digest(label: str) -> str | None:
    snap = RUNS / LADDER_DIR[label] / "deck_snapshot"
    if not snap.exists():
        return None
    h = hashlib.md5()
    for p in sorted(snap.glob("*.py")) + sorted(snap.glob("ranker_model.json")):
        h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    rows = {r["version"]: r for r in TRAJ["rows"]}
    groups: dict[str, list[str]] = {}
    for label in LADDER_DIR:
        d = digest(label)
        if d:
            groups.setdefault(d, []).append(label)

    pairs = []
    for d, labels in groups.items():
        if len(labels) < 2:
            continue
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                ra = rows[a]["reported_rating"]
                rb = rows[b]["reported_rating"]
                pairs.append({
                    "digest": d[:12], "a": a, "b": b,
                    "rating_a": ra, "rating_b": rb,
                    "abs_rating_gap": round(abs(ra - rb), 1),
                    "win_a": rows[a]["win_rate"], "win_b": rows[b]["win_rate"],
                    "abs_win_gap": round(
                        abs(rows[a]["win_rate"] - rows[b]["win_rate"]), 4
                    ),
                    "games_a": rows[a]["games"], "games_b": rows[b]["games"],
                    "opp_a": rows[a]["mean_opponent"],
                    "opp_b": rows[b]["mean_opponent"],
                })
    gaps = [p["abs_rating_gap"] for p in pairs]
    rms = math.sqrt(sum(g * g for g in gaps) / len(gaps))
    sd_single = rms / math.sqrt(2)
    ratings = [r["reported_rating"] for r in TRAJ["rows"]]
    payload = {
        "identical_code_groups": {
            d[:12]: labels for d, labels in groups.items() if len(labels) > 1
        },
        "pairs": pairs,
        "n_pairs": len(pairs),
        "mean_abs_rating_gap": round(sum(gaps) / len(gaps), 1),
        "max_abs_rating_gap": max(gaps),
        "rms_rating_gap": round(rms, 1),
        "implied_sd_single_run": round(sd_single, 1),
        "implied_95pct_band_single_run": round(1.96 * sd_single, 1),
        "all_versions_rating_min": min(ratings),
        "all_versions_rating_max": max(ratings),
        "all_versions_rating_range": round(max(ratings) - min(ratings), 1),
        "mean_abs_win_rate_gap": round(
            sum(p["abs_win_gap"] for p in pairs) / len(pairs), 4
        ),
    }
    out = ROOT / "experiments" / "grimmsnarl_1100_diagnosis" / \
        "code-evolution" / "identical_binaries.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
