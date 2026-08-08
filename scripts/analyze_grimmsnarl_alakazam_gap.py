"""Our Alakazam record against the field's, on the same deck and same matchup.

The v10 report tested the Alakazam matchup the weak way - "is it worse than our
*other* matchups" - and got p = 0.18, because our own 50-game run has no power
to rank matchups. The sharp test is against the field playing the identical 60:
they win it 242-84 and we win it 4-7. This measures that comparison properly and
then asks the only question that decides whether it is fixable: **is the field's
Alakazam win rate turn-order dependent the way ours is?**

If the field also collapses on the draw, our record is turn order and there is
nothing matchup-specific to fix. If it does not, the matchup is a real defect
and it is worth 8.9% of the meta.

Restricted to the Alakazam family so a full parse of every replay stays cheap;
the mirror is carried as the control because it is the one matchup we are
*above* the field in, which is what rules out "we are simply weaker".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_grimmsnarl_matchup_ceiling import family  # noqa: E402
from ml.core.replay_io import deck_hash  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"


def fisher(a: int, b: int, c: int, d: int) -> float:
    n, r1, c1 = a + b + c + d, a + b, a + c
    if not n:
        return 1.0

    def probability(x: int) -> float:
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    reference = probability(a)
    return round(min(1.0, sum(
        probability(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1)
        if probability(x) <= reference + 1e-12
    )), 4)


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963985
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) * z
    return [round((centre - margin) / denominator, 4),
            round((centre + margin) / denominator, 4)]


def read(path: Path, seat: int) -> dict[str, Any] | None:
    try:
        replay = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    steps = replay.get("steps") or []
    decks: list[list[int] | None] = [None, None]
    if len(steps) > 1:
        for s in (0, 1):
            action = (steps[1][s] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[s] = [int(v) for v in action]
    if decks[seat] is None or deck_hash(decks[seat]) != OUR_DECK_HASH:
        return None
    went_first = None
    for step in reversed(steps):
        if seat >= len(step):
            continue
        current = ((step[seat] or {}).get("observation") or {}).get("current")
        if isinstance(current, dict) and current.get("players"):
            first = int(current.get("firstPlayer", -1))
            went_first = (first == seat) if first >= 0 else None
            break
    rewards = replay.get("rewards") or [None, None]
    if rewards[seat] is None:
        return None
    other = rewards[1 - seat]
    # Prizes taken: the opening six minus what is left on the last board seen.
    taken = given = None
    for step in reversed(steps):
        if seat >= len(step):
            continue
        current = ((step[seat] or {}).get("observation") or {}).get("current")
        if isinstance(current, dict) and current.get("players"):
            players = current["players"]
            taken = 6 - len(players[seat].get("prize") or [])
            given = 6 - len(players[1 - seat].get("prize") or [])
            break
    return {
        "won": bool(rewards[seat] > (other if other is not None else 0)),
        "went_first": went_first,
        "family": family(decks[1 - seat]),
        "prizes_taken": taken,
        "prizes_given": given,
        "turns": len(steps),
    }


def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(int(r["won"]) for r in rows)
    prizes = [r["prizes_taken"] for r in rows if r["prizes_taken"] is not None]
    return {
        "games": len(rows), "wins": wins, "losses": len(rows) - wins,
        "win_rate": round(wins / len(rows), 4) if rows else None,
        "wilson95": wilson(wins, len(rows)),
        "mean_prizes_taken": (
            round(sum(prizes) / len(prizes), 3) if prizes else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "data" / "submissions" / "submission_55317804",
    )
    parser.add_argument("--submission", default="55317804")
    parser.add_argument(
        "--families", default="Alakazam,Grimmsnarl (mirror)",
        help="Comma separated; the mirror is the control.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    wanted = {name.strip() for name in args.families.split(",")}

    field: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    scanned = 0
    for raw in csv.DictReader(
        (args.data_root / "indexes" / "episodes.csv").open(
            encoding="utf-8-sig"
        )
    ):
        if raw.get("download_status") != "success":
            continue
        if raw.get("deck_hash") != OUR_DECK_HASH:
            continue
        if raw.get("episode_type") != "EPISODE_TYPE_PUBLIC":
            continue
        episode_id, seat = int(raw["episode_id"]), int(raw["seat_index"])
        if (episode_id, seat) in seen:
            continue
        seen.add((episode_id, seat))
        path = args.data_root / "replays" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        row = read(path, seat)
        scanned += 1
        if row is None or row["family"] not in wanted:
            continue
        row["team"] = int(raw["team_id"])
        field.append(row)

    ours: list[dict[str, Any]] = []
    for raw in csv.DictReader(
        (args.run_dir / "episodes.csv").open(encoding="utf-8-sig")
    ):
        a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
        if raw["episode_type"] != "EPISODE_TYPE_PUBLIC" or a0 == a1:
            continue
        episode_id = int(raw["episode_id"])
        path = (
            args.run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not path.exists():
            continue
        row = read(path, 0 if a0 == args.submission else 1)
        if row is None or row["family"] not in wanted:
            continue
        ours.append(row)

    report: dict[str, Any] = {"scanned": scanned, "families": {}}
    for name in sorted(wanted):
        f = [r for r in field if r["family"] == name]
        o = [r for r in ours if r["family"] == name]
        entry: dict[str, Any] = {"field": block(f), "v8": block(o)}
        for label, predicate in (
            ("first", lambda r: r["went_first"] is True),
            ("second", lambda r: r["went_first"] is False),
        ):
            fs = [r for r in f if predicate(r)]
            os_ = [r for r in o if predicate(r)]
            entry[f"field_{label}"] = block(fs)
            entry[f"v8_{label}"] = block(os_)
            entry[f"fisher_{label}"] = fisher(
                block(os_)["wins"], block(os_)["losses"],
                block(fs)["wins"], block(fs)["losses"],
            )
        entry["fisher_overall"] = fisher(
            entry["v8"]["wins"], entry["v8"]["losses"],
            entry["field"]["wins"], entry["field"]["losses"],
        )
        entry["field_turn_order_fisher"] = fisher(
            entry["field_first"]["wins"], entry["field_first"]["losses"],
            entry["field_second"]["wins"], entry["field_second"]["losses"],
        )
        report["families"][name] = entry

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"scanned {scanned} same-deck replays")
    for name, entry in report["families"].items():
        print(f"\n=== {name} ===")
        for who in ("field", "v8"):
            for label in ("", "_first", "_second"):
                row = entry[f"{who}{label}"]
                if not row["games"]:
                    continue
                print(f"  {who + label:14s} {row['wins']:4d}-{row['losses']:<4d} "
                      f"{row['win_rate']} {row['wilson95']} "
                      f"prizes={row['mean_prizes_taken']}")
        print(f"  ours vs field: overall p={entry['fisher_overall']}, "
              f"first p={entry['fisher_first']}, "
              f"second p={entry['fisher_second']}")
        print(f"  field's own turn-order split p="
              f"{entry['field_turn_order_fisher']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
