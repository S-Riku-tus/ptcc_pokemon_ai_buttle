"""Per-own-turn resource take rates, split by turn order, for us and the field.

The one deficit in the v8 ladder run that survives a significance test is going
second: 20-8 first against 8-14 second, p = 0.021 by Fisher, where the Alakazam
matchup people point at is p = 0.18 once its own turn-order split is taken out.
So the question this answers is not "what does v8 do wrong" in general but
"what does v8 do differently *on the turns it goes second* from what the field
does on the same turns".

Denominators are per own turn, never per decision: MAIN is re-asked after every
intermediate action, and the same data reads 18.9% or 88.6% depending on which
denominator is used.

``firstPlayer`` is read from a late step. It is -1 until the flip resolves, and
latching it from the first observation once labelled 99% of games "second".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8"))

import ml_features as mf  # noqa: E402

FROSLASS_ID = mf.FROSLASS_ID
GRIMMSNARL_EX_ID = mf.GRIMMSNARL_EX_ID
OUR_DECK_HASH = "9714ab5c3996f6cc"


def wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963985
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) * z
    return [
        round((centre - margin) / denominator, 4),
        round((centre + margin) / denominator, 4),
    ]


def spearman(points: list[tuple[float, float, int]]) -> dict[str, Any]:
    """Rank correlation of a per-pilot rate against pilot rating.

    Reported with its n and its two-sided p, because this line has already
    justified a change with "monotone in pilot rating" on a gradient that did
    not survive a test. A t approximation is enough at n ~ 20 and needs no
    scipy inside a repo whose runtime is standard library only.
    """
    if len(points) < 4:
        return {"n": len(points), "rho": None, "p": None}
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while (stop + 1 < len(order)
                   and values[order[stop + 1]] == values[order[index]]):
                stop += 1
            average = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                out[order[position]] = average
            index = stop + 1
        return out

    x = ranks([p[0] for p in points])
    y = ranks([p[1] for p in points])
    n = len(points)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    var_x = sum((a - mean_x) ** 2 for a in x)
    var_y = sum((b - mean_y) ** 2 for b in y)
    if var_x <= 0 or var_y <= 0:
        return {"n": n, "rho": None, "p": None}
    rho = cov / math.sqrt(var_x * var_y)
    if abs(rho) >= 1.0:
        return {"n": n, "rho": round(rho, 4), "p": 0.0}
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    # Two-sided p from Student's t with n-2 df, via the incomplete beta
    # identity that only needs lgamma.
    df = n - 2
    xbeta = df / (df + t * t)
    p = _betainc(df / 2.0, 0.5, xbeta)
    return {"n": n, "rho": round(rho, 4), "p": round(p, 5)}


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta, continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    ) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -(
                ((a + m) * (a + b + m) * x)
                / ((a + 2 * m) * (a + 2 * m + 1))
            )
        d = 1.0 + numerator * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + numerator / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return min(1.0, max(0.0, front * (f - 1.0)))


def deck_hash(card_ids: list[int]) -> str:
    counts = Counter(int(x) for x in card_ids)
    canonical = ";".join(f"{cid}:{counts[cid]}" for cid in sorted(counts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class TurnCounter:
    """One row per own turn: was it offered at all, and was it taken."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.turn: int | None = None
        self.flags: set[str] = set()
        self.bucket = ""

    def start(self, bucket: str) -> None:
        self.flush()
        self.bucket = bucket
        self.turn = None
        # Counted here rather than derived from ``turns``: the decisive control
        # for "is going second a policy problem or a deck problem" is the
        # field's own win rate by turn order on this exact 60, and that needs a
        # per-bucket *game* count, not a turn count.
        self.counts[f"{bucket}|games"] += 1

    def note(self, turn: int, current: dict[str, Any],
             select: dict[str, Any], options: list[dict[str, Any]],
             played: int) -> None:
        if turn != self.turn:
            self.flush()
            self.turn = turn
        actions = [mf.action_type(current, o, select) for o in options]
        cards = [
            int((mf.candidate_card(current, o, select) or {}).get("id", -1))
            for o in options
        ]

        def slots(predicate) -> list[int]:
            return [s for s in range(len(options)) if predicate(s)]

        classes = {
            "energy": slots(
                lambda s: actions[s] == "energy"
                and cards[s] == mf.DARK_ENERGY_ID
            ),
            "boss": slots(lambda s: actions[s] == "boss"),
            "froslass": slots(
                lambda s: actions[s] == "evolve" and cards[s] == FROSLASS_ID
            ),
            "grimmsnarl": slots(
                lambda s: actions[s] == "evolve"
                and cards[s] == GRIMMSNARL_EX_ID
            ),
            "attack": slots(lambda s: actions[s] == "attack"),
            "bench": slots(lambda s: actions[s] == "bench"),
        }
        for name, group in classes.items():
            if not group:
                continue
            self.flags.add(f"offer_{name}")
            if played in group:
                self.flags.add(f"take_{name}")

        for slot in classes["energy"]:
            target = mf.candidate_target(current, options[slot]) or {}
            target_id = int(target.get("id", -1))
            dark = mf._dark_energy_count(target)
            if (
                (target_id == mf.MUNKIDORI_ID and dark == 0)
                or (target_id == GRIMMSNARL_EX_ID
                    and dark == mf.SHADOW_BULLET_COST - 1)
            ):
                self.flags.add("offer_enabling_energy")
                if played == slot:
                    self.flags.add("take_enabling_energy")

    def flush(self) -> None:
        if self.turn is None:
            return
        self.counts[f"{self.bucket}|turns"] += 1
        for flag in sorted(self.flags):
            self.counts[f"{self.bucket}|{flag}"] += 1
        self.flags = set()
        self.turn = None


KINDS = (
    "energy", "enabling_energy", "froslass", "grimmsnarl", "boss",
    "attack", "bench",
)


def summarise(counts: Counter[str]) -> dict[str, Any]:
    buckets = sorted({key.split("|", 1)[0] for key in counts})
    out: dict[str, Any] = {}
    for bucket in buckets:
        row: dict[str, Any] = {
            "games": counts[f"{bucket}|games"],
            "turns": counts[f"{bucket}|turns"],
        }
        for kind in KINDS:
            offered = counts[f"{bucket}|offer_{kind}"]
            taken = counts[f"{bucket}|take_{kind}"]
            row[f"{kind}_offered"] = offered
            row[f"{kind}_taken"] = taken
            row[f"{kind}_rate"] = (
                round(taken / offered, 4) if offered else None
            )
            row[f"{kind}_wilson95"] = wilson(taken, offered)
        out[bucket] = row
    return out


def walk(replay: dict[str, Any], seat: int, counter: TurnCounter) -> None:
    steps = replay.get("steps") or []
    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        if int(select.get("context", -1)) != mf.MAIN_CONTEXT:
            continue
        options = list(select.get("option") or [])
        action = (steps[index + 1][seat] or {}).get("action")
        if not options or not isinstance(action, list) or len(action) != 1:
            continue
        if not isinstance(action[0], int) or not 0 <= action[0] < len(options):
            continue
        current = observation.get("current") or {}
        if not (current.get("players") or []):
            continue
        counter.note(
            int(current.get("turn", -1)), current, select, options, action[0]
        )
    counter.flush()


def turn_order(replay: dict[str, Any], seat: int) -> bool | None:
    for step in reversed(replay.get("steps") or []):
        if seat >= len(step):
            continue
        current = ((step[seat] or {}).get("observation") or {}).get("current")
        if isinstance(current, dict) and current.get("players"):
            first = int(current.get("firstPlayer", -1))
            return (first == seat) if first >= 0 else None
    return None


def our_run(run_dir: Path, submission: str, counter: TurnCounter) -> int:
    played = 0
    for raw in csv.DictReader(
        (run_dir / "episodes.csv").open(encoding="utf-8-sig")
    ):
        a0, a1 = raw["agent_0_submission_id"], raw["agent_1_submission_id"]
        if raw["episode_type"] != "EPISODE_TYPE_PUBLIC" or a0 == a1:
            continue
        episode_id = int(raw["episode_id"])
        path = (
            run_dir / "episodes" / str(episode_id) / "replay"
            / f"episode_{episode_id}.json"
        )
        if not path.exists():
            continue
        seat = 0 if a0 == submission else 1
        replay = json.loads(path.read_text(encoding="utf-8"))
        first = turn_order(replay, seat)
        if first is None:
            continue
        rewards = replay.get("rewards") or [None, None]
        other = rewards[1 - seat]
        won = bool(rewards[seat] > (other if other is not None else 0))
        counter.start(
            f"v8_{'first' if first else 'second'}_"
            f"{'won' if won else 'lost'}"
        )
        walk(replay, seat, counter)
        counter.flush()
        played += 1
    return played


def field(data_root: Path, counter: TurnCounter, limit: int,
          ratings: dict[int, float], by_team: bool = False) -> dict[str, int]:
    index_path = data_root / "indexes" / "episodes.csv"
    seen: set[tuple[int, int]] = set()
    stats = Counter()
    rows = list(csv.DictReader(index_path.open(encoding="utf-8-sig")))
    for raw in rows:
        if raw.get("download_status") != "success":
            continue
        if raw.get("deck_hash") != OUR_DECK_HASH:
            continue
        if raw.get("episode_type") != "EPISODE_TYPE_PUBLIC":
            continue
        episode_id = int(raw["episode_id"])
        seat = int(raw["seat_index"])
        if (episode_id, seat) in seen:
            continue
        seen.add((episode_id, seat))
        path = data_root / "replays" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        if limit and stats["games"] >= limit:
            break
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            stats["unreadable"] += 1
            continue
        decks: list[list[int] | None] = [None, None]
        steps = replay.get("steps") or []
        if len(steps) > 1:
            for s in (0, 1):
                action = (steps[1][s] or {}).get("action")
                if isinstance(action, list) and len(action) == 60:
                    decks[s] = [int(v) for v in action]
        # The index says which seat holds our 60, but a mirror has two.
        if decks[seat] is None or deck_hash(decks[seat]) != OUR_DECK_HASH:
            stats["deck_mismatch"] += 1
            continue
        first = turn_order(replay, seat)
        if first is None:
            stats["no_turn_order"] += 1
            continue
        rewards = replay.get("rewards") or [None, None]
        other = rewards[1 - seat]
        won = bool((rewards[seat] or 0) > (other if other is not None else 0))
        team = int(raw["team_id"])
        rating = ratings.get(team)
        band = (
            "elite" if rating is not None and rating >= 1100
            else "mid" if rating is not None and rating >= 1050
            else "rest"
        )
        counter.start(
            f"team{team}_{'first' if first else 'second'}_"
            f"{'won' if won else 'lost'}"
            if by_team else
            f"field_{band}_{'first' if first else 'second'}_"
            f"{'won' if won else 'lost'}"
        )
        walk(replay, seat, counter)
        counter.flush()
        stats["games"] += 1
    return dict(stats)


def load_ratings(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    if not path.exists():
        return out
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        try:
            out[int(row["team_id"])] = float(row["leaderboard_score"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def merge(counts: Counter[str], sources: set[str], prefixes: list[str],
          name: str) -> None:
    """Roll leaf buckets up into a named one.

    Only the buckets that existed *before* any rollup are summed. Matching on a
    prefix over the live counter instead once folded ``v8_first`` and
    ``v8_second`` into ``v8_all`` a second time and printed 602 turns for the
    301 that were played.
    """
    for key in list(counts):
        bucket, rest = key.split("|", 1)
        if bucket in sources and any(bucket.startswith(p) for p in prefixes):
            counts[f"{name}|{rest}"] += counts[key]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "data" / "submissions" / "submission_55317804",
    )
    parser.add_argument("--submission", default="55317804")
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument(
        "--ratings", type=Path,
        default=ROOT / "data" / "kaggle_top50_meta" / "analysis"
        / "submissions_grimmsnarl.csv",
    )
    parser.add_argument("--field-limit", type=int, default=0)
    parser.add_argument(
        "--by-team", action="store_true",
        help="Bucket the field per pilot instead of per rating band, so a "
             "claimed rating gradient can be tested rather than assumed.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ratings = load_ratings(args.ratings)
    counter = TurnCounter()
    ours = our_run(args.run_dir, args.submission, counter)
    field_stats = field(
        args.data_root, counter, args.field_limit, ratings, args.by_team
    )
    counter.flush()

    leaves = {key.split("|", 1)[0] for key in counter.counts}
    if args.by_team:
        for team in sorted({
            int(leaf[4:].split("_", 1)[0])
            for leaf in leaves if leaf.startswith("team")
        }):
            merge(counter.counts, leaves, [f"team{team}_"], f"team{team}_all")
            merge(counter.counts, leaves, [f"team{team}_first"],
                  f"team{team}_first_all")
            merge(counter.counts, leaves, [f"team{team}_second"],
                  f"team{team}_second_all")
    for prefixes, name in (
        (["v8_first"], "v8_first"),
        (["v8_second"], "v8_second"),
        (["v8_"], "v8_all"),
        (["field_elite_first", "field_mid_first", "field_rest_first"],
         "field_first"),
        (["field_elite_second", "field_mid_second", "field_rest_second"],
         "field_second"),
        (["field_elite_first"], "field_elite_first"),
        (["field_elite_second"], "field_elite_second"),
        (["field_"], "field_all"),
    ):
        merge(counter.counts, leaves, prefixes, name)

    report = {
        "our_games": ours,
        "field_stats": field_stats,
        "buckets": summarise(counter.counts),
    }
    if args.by_team:
        report["ratings"] = {
            str(team): ratings.get(team) for team in sorted(ratings)
        }
        report["rating_gradient"] = {
            kind: spearman([
                (ratings[team], row[f"{kind}_rate"], row[f"{kind}_offered"])
                for team, row in (
                    (int(name[4:-4]), report["buckets"][name])
                    for name in report["buckets"]
                    if name.startswith("team") and name.endswith("_all")
                    and name.count("_") == 1
                )
                if team in ratings and row[f"{kind}_rate"] is not None
                and row[f"{kind}_offered"] >= 25
            ])
            for kind in KINDS
        }
    win_rates: dict[str, dict[str, Any]] = {}
    for leaf in sorted(leaves):
        if not leaf.endswith(("_won", "_lost")):
            continue
        stem, result = leaf.rsplit("_", 1)
        entry = win_rates.setdefault(stem, {"won": 0, "lost": 0})
        entry[result] += counter.counts[f"{leaf}|games"]
    for stem, entry in win_rates.items():
        total = entry["won"] + entry["lost"]
        entry["games"] = total
        entry["win_rate"] = round(entry["won"] / total, 4) if total else None
        entry["wilson95"] = wilson(entry["won"], total)
    report["win_rate_by_turn_order"] = win_rates

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"our games={ours} field={field_stats}")
    print("win rate by bucket:")
    for stem, entry in sorted(win_rates.items()):
        print(f"  {stem:26s} {entry['won']:4d}-{entry['lost']:<4d} "
              f"{entry['win_rate']} {entry['wilson95']}")
    header = ["bucket", "games", "turns"] + [f"{k}" for k in KINDS]
    print(" ".join(f"{h:>18s}" for h in header))
    names = (
        [
            "v8_all", "v8_first", "v8_second",
        ] + sorted(
            (n for n in report["buckets"]
             if n.startswith("team") and n.endswith("_all")
             and n.count("_") == 1),
            key=lambda n: -(ratings.get(int(n[4:-4])) or 0),
        )
        if args.by_team else
        [
            "v8_all", "v8_first", "v8_second",
            "field_all", "field_first", "field_second",
            "field_elite_first", "field_elite_second",
        ]
    )
    for name in names:
        row = report["buckets"].get(name)
        if not row:
            continue
        cells = [name, str(row["games"]), str(row["turns"])]
        for kind in KINDS:
            rate = row[f"{kind}_rate"]
            cells.append(
                f"{rate:.3f}({row[f'{kind}_offered']})"
                if rate is not None else "-"
            )
        print(" ".join(f"{c:>18s}" for c in cells))
    if args.by_team:
        print()
        print("rate vs pilot rating (Spearman, pilots with >= 25 offers):")
        for kind, value in report["rating_gradient"].items():
            print(f"  {kind:18s} n={value['n']:2d} rho={value['rho']} "
                  f"p={value['p']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
