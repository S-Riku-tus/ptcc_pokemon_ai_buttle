"""What actually separates a 1220 pilot from a 1050 one, on the same 60 cards.

Every Grimmsnarl version so far picked a behaviour first and then measured it.
Seven of those a-priori picks have now been measured against pilot rating and
none of them correlates (attach p=0.46, Froslass p=0.11, Grimmsnarl evolve
p=0.23, Boss p=0.35): the per-turn resource rates are saturated. But the three
statistics that *did* correlate in that sweep - the dead Unfair Stamp, Petrel's
Boss, and the attack rate - are all about *which card* or *whether to commit*,
not about taking an offered resource.

So this inverts the method: compute a wide panel of statistics per pilot, and
let the rating gradient choose the target instead of choosing it first.
Multiplicity is controlled with Benjamini-Hochberg, because scanning ~40
statistics at alpha 0.05 buys two false positives for free and this line has
already shipped one gradient that did not survive a test.

Three families, all per own turn or per offer, never per decision:

* **sequencing** - what the turn is spent on, how many actions it contains,
  what the board looks like when it ends;
* **selection** - which card a search takes, how many a multi-pick takes;
* **race** - prizes taken and conceded per own turn, and how early.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v8"))

import ml_features as mf  # noqa: E402
from analyze_grimmsnarl_v10_turn_order import spearman  # noqa: E402
from analyze_grimmsnarl_v10_stamp import nested_id  # noqa: E402

OUR_DECK_HASH = "9714ab5c3996f6cc"
FROSLASS = mf.FROSLASS_ID
GRIMM = mf.GRIMMSNARL_EX_ID
MORGREM = mf.MORGREM_ID
IMPIDIMP = mf.IMPIDIMP_ID
MUNKIDORI = mf.MUNKIDORI_ID
SNORUNT = mf.SNORUNT_ID
DARK = mf.DARK_ENERGY_ID
STAMP = mf.UNFAIR_STAMP_ID
PETREL = mf.PETREL_ID
BOSS = mf.BOSS_ID
LILLIE = mf.LILLIE_ID
DAWN = mf.DAWN_ID
CANDY = mf.RARE_CANDY_ID
POFFIN = mf.POFFIN_ID
STRETCHER = mf.NIGHT_STRETCHER_ID
POKE_PAD = mf.POKE_PAD_ID
GYM = mf.SPIKEMUTH_GYM_ID
SCRAPPER = mf.TOOL_SCRAPPER_ID

# name -> (numerator key, denominator key). Everything the scan reports.
RATES = {
    # --- sequencing, per own turn -------------------------------------------
    "attack_taken": ("take_attack", "offer_attack"),
    "boss_taken": ("take_boss", "offer_boss"),
    "energy_taken": ("take_energy", "offer_energy"),
    "froslass_evolve": ("take_froslass", "offer_froslass"),
    "grimmsnarl_evolve": ("take_grimmsnarl", "offer_grimmsnarl"),
    "morgrem_evolve": ("take_morgrem", "offer_morgrem"),
    "bench_taken": ("take_bench", "offer_bench"),
    "candy_taken": ("take_candy", "offer_candy"),
    "stadium_taken": ("take_stadium", "offer_stadium"),
    "retreat_taken": ("take_retreat", "offer_retreat"),
    "munkidori_ability": ("take_munkidori_ability", "offer_munkidori_ability"),
    "punk_up_ability": ("take_punk_up", "offer_punk_up"),
    "supporter_taken": ("take_supporter", "offer_supporter"),
    "scrapper_taken": ("take_scrapper", "offer_scrapper"),
    "pokepad_taken": ("take_pokepad", "offer_pokepad"),
    "stretcher_taken": ("take_stretcher", "offer_stretcher"),
    "poffin_taken": ("take_poffin", "offer_poffin"),
    "stamp_played_when_live": ("take_stamp_play", "offer_stamp_play"),
    # --- which supporter, given that one is played --------------------------
    "supporter_is_petrel": ("sup_petrel", "sup_total"),
    "supporter_is_lillie": ("sup_lillie", "sup_total"),
    "supporter_is_dawn": ("sup_dawn", "sup_total"),
    "supporter_is_boss": ("sup_boss", "sup_total"),
    # --- selection ----------------------------------------------------------
    "petrel_dead_stamp": ("petrel_dead_take", "petrel_dead_offer"),
    "petrel_live_stamp": ("petrel_live_take", "petrel_live_offer"),
    "petrel_boss": ("petrel_boss_take", "petrel_boss_offer"),
    "petrel_takes_grimmsnarl_line": ("petrel_line_take", "petrel_any"),
    "punk_up_takes_max": ("punk_max", "punk_activations"),
    "poffin_takes_max": ("poffin_max", "poffin_activations"),
    # --- race, per own turn / per game --------------------------------------
    "prizes_per_own_turn": ("prizes_taken", "own_turns"),
    "prizes_conceded_per_own_turn": ("prizes_given", "own_turns"),
    "turns_with_a_prize": ("turn_scored", "own_turns"),
    "actions_per_own_turn": ("main_actions", "own_turns"),
    "bench_full_at_turn_end": ("bench_ge3", "closed_turns"),
    "attacked_this_turn": ("turn_attacked", "own_turns"),
    "win_rate": ("wins", "games"),
    "own_turns_per_game": ("own_turns", "games"),
}
# Denominators below this are not reported: a rate over 8 offers is noise.
MIN_DENOMINATOR = 40


def deck_hash(card_ids: list[int]) -> str:
    counts = Counter(int(x) for x in card_ids)
    canonical = ";".join(f"{cid}:{counts[cid]}" for cid in sorted(counts))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def turn_order(replay: dict[str, Any], seat: int) -> bool | None:
    for step in reversed(replay.get("steps") or []):
        if seat >= len(step):
            continue
        current = ((step[seat] or {}).get("observation") or {}).get("current")
        if isinstance(current, dict) and current.get("players"):
            first = int(current.get("firstPlayer", -1))
            return (first == seat) if first >= 0 else None
    return None


def scan(replay: dict[str, Any], seat: int) -> Counter:
    counts: Counter = Counter()
    steps = replay.get("steps") or []
    opponent_prize_by_turn: dict[int, int] = {}
    turn = None
    flags: set[str] = set()
    my_prize = opponent_prize = None
    last_board: dict[str, Any] | None = None
    actions_this_turn = 0

    def flush() -> None:
        nonlocal flags, actions_this_turn, last_board
        if turn is None:
            return
        counts["own_turns"] += 1
        counts["main_actions"] += actions_this_turn
        for flag in flags:
            counts[flag] += 1
        if last_board is not None:
            counts["closed_turns"] += 1
            counts["bench_ge3"] += int(last_board["bench"] >= 3)
        flags = set()
        actions_this_turn = 0
        last_board = None

    for index, step in enumerate(steps[:-1]):
        if seat >= len(step) or seat >= len(steps[index + 1]):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        current = observation.get("current") or {}
        players = current.get("players") or []
        your = int(current.get("yourIndex", seat))
        if not select or len(players) < 2 or your >= len(players):
            continue
        me, opponent = players[your], players[1 - your]
        action = (steps[index + 1][seat] or {}).get("action")
        if not isinstance(action, list) or not action:
            continue
        options = list(select.get("option") or [])
        if not options:
            continue
        context = int(select.get("context", -1))
        this_turn = int(current.get("turn", -1))
        opponent_prize_by_turn.setdefault(
            this_turn, len(opponent.get("prize") or [])
        )

        # Prize deltas are read on every observation, so a knockout that lands
        # on the opponent's turn is still attributed to the turn it happened.
        mine = len(me.get("prize") or [])
        theirs = len(opponent.get("prize") or [])
        if my_prize is not None and mine < my_prize:
            counts["prizes_taken"] += my_prize - mine
            if turn is not None:
                flags.add("turn_scored")
        if opponent_prize is not None and theirs < opponent_prize:
            counts["prizes_given"] += opponent_prize - theirs
        my_prize, opponent_prize = mine, theirs

        if context == mf.MAIN_CONTEXT:
            if this_turn != turn:
                flush()
                turn = this_turn
            if not (len(action) == 1 and isinstance(action[0], int)
                    and 0 <= action[0] < len(options)):
                continue
            played = action[0]
            actions_this_turn += 1
            kinds = [mf.action_type(current, o, select) for o in options]
            cards = [
                int((mf.candidate_card(current, o, select) or {})
                    .get("id", -1))
                for o in options
            ]

            def offer(name: str, predicate) -> None:
                group = [s for s in range(len(options)) if predicate(s)]
                if not group:
                    return
                flags.add(f"offer_{name}")
                if played in group:
                    flags.add(f"take_{name}")

            offer("attack", lambda s: kinds[s] == "attack")
            offer("boss", lambda s: kinds[s] == "boss")
            offer("energy",
                  lambda s: kinds[s] == "energy" and cards[s] == DARK)
            offer("froslass",
                  lambda s: kinds[s] == "evolve" and cards[s] == FROSLASS)
            offer("grimmsnarl",
                  lambda s: kinds[s] == "evolve" and cards[s] == GRIMM)
            offer("morgrem",
                  lambda s: kinds[s] == "evolve" and cards[s] == MORGREM)
            offer("bench", lambda s: kinds[s] == "bench")
            offer("candy", lambda s: cards[s] == CANDY)
            offer("stadium", lambda s: cards[s] == GYM)
            offer("retreat", lambda s: kinds[s] == "retreat")
            offer("scrapper", lambda s: cards[s] == SCRAPPER)
            offer("pokepad", lambda s: cards[s] == POKE_PAD)
            offer("stretcher", lambda s: cards[s] == STRETCHER)
            offer("poffin", lambda s: cards[s] == POFFIN)
            offer("stamp_play", lambda s: cards[s] == STAMP)
            offer("munkidori_ability",
                  lambda s: kinds[s] == "ability" and cards[s] == MUNKIDORI)
            offer("punk_up",
                  lambda s: kinds[s] == "ability" and cards[s] == GRIMM)
            offer("supporter",
                  lambda s: kinds[s] in ("supporter", "boss")
                  or cards[s] in (PETREL, LILLIE, DAWN, BOSS))
            if kinds[played] == "attack":
                flags.add("turn_attacked")
            if cards[played] in (PETREL, LILLIE, DAWN, BOSS):
                counts["sup_total"] += 1
                counts["sup_petrel"] += int(cards[played] == PETREL)
                counts["sup_lillie"] += int(cards[played] == LILLIE)
                counts["sup_dawn"] += int(cards[played] == DAWN)
                counts["sup_boss"] += int(cards[played] == BOSS)
            last_board = {"bench": len(me.get("bench") or [])}
            continue

        # ----- selection contexts -------------------------------------------
        resolved = [
            int((mf.resolve_option(current, select, option)[0] or {})
                .get("id", -1))
            for option in options
        ]
        picked = {
            resolved[slot] for slot in action
            if isinstance(slot, int) and 0 <= slot < len(resolved)
        }
        effect_id = nested_id(select.get("effect"))
        if context == mf.CTX_TO_HAND and effect_id == PETREL:
            counts["petrel_any"] += 1
            counts["petrel_line_take"] += int(
                bool(picked & {IMPIDIMP, MORGREM, GRIMM, CANDY})
            )
            in_hand = Counter(
                int(card.get("id", -1))
                for card in (me.get("hand") or []) if isinstance(card, dict)
            )
            if STAMP in resolved and not in_hand[STAMP]:
                earlier = [t for t in opponent_prize_by_turn if t < this_turn]
                prior = (
                    opponent_prize_by_turn[max(earlier)] if earlier else 6
                )
                key = "live" if theirs < prior else "dead"
                counts[f"petrel_{key}_offer"] += 1
                counts[f"petrel_{key}_take"] += int(STAMP in picked)
            if BOSS in resolved:
                counts["petrel_boss_offer"] += 1
                counts["petrel_boss_take"] += int(BOSS in picked)
        elif int(select.get("maxCount") or 0) > 1:
            # The multi-pick searches no ranker ever scores.
            if all(card == DARK for card in resolved if card >= 0):
                counts["punk_activations"] += 1
                counts["punk_max"] += int(
                    len(action) >= int(select.get("maxCount") or 0)
                )
            elif effect_id == POFFIN:
                counts["poffin_activations"] += 1
                counts["poffin_max"] += int(
                    len(action) >= int(select.get("maxCount") or 0)
                )
    flush()
    counts["games"] += 1
    return counts


def benjamini_hochberg(rows: list[dict[str, Any]], alpha: float = 0.05):
    tested = [row for row in rows if row["p"] is not None]
    tested.sort(key=lambda row: row["p"])
    total = len(tested)
    threshold = 0.0
    for rank, row in enumerate(tested, start=1):
        row["bh_critical"] = round(alpha * rank / total, 5) if total else None
        if row["p"] <= alpha * rank / total:
            threshold = row["p"]
    for row in tested:
        row["bh_significant"] = row["p"] <= threshold
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50",
    )
    parser.add_argument(
        "--ratings", type=Path,
        default=ROOT / "data" / "kaggle_grimmsnarl_top50" / "indexes"
        / "submissions.csv",
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "data" / "submissions" / "submission_55317804",
    )
    parser.add_argument("--submission", default="55317804")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    ratings: dict[int, float] = {}
    for row in csv.DictReader(args.ratings.open(encoding="utf-8-sig")):
        try:
            ratings[int(row["team_id"])] = float(row["submission_score"])
        except (KeyError, TypeError, ValueError):
            continue

    per_team: dict[int, Counter] = defaultdict(Counter)
    read = 0
    for raw in csv.DictReader(
        (args.data_root / "indexes" / "episodes.csv").open(
            encoding="utf-8-sig"
        )
    ):
        if args.limit and read >= args.limit:
            break
        if raw.get("download_status") != "success":
            continue
        if raw.get("deck_hash") != OUR_DECK_HASH:
            continue
        if raw.get("episode_type") != "EPISODE_TYPE_PUBLIC":
            continue
        episode_id = int(raw["episode_id"])
        seat = int(raw["seat_index"])
        path = args.data_root / "replays" / f"episode_{episode_id}.json"
        if not path.exists():
            continue
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        steps = replay.get("steps") or []
        deck = None
        if len(steps) > 1:
            action = (steps[1][seat] or {}).get("action")
            if isinstance(action, list) and len(action) == 60:
                deck = [int(v) for v in action]
        if deck is None or deck_hash(deck) != OUR_DECK_HASH:
            continue
        counts = scan(replay, seat)
        rewards = replay.get("rewards") or [None, None]
        other = rewards[1 - seat]
        counts["wins"] = int(
            (rewards[seat] or 0) > (other if other is not None else 0)
        )
        per_team[int(raw["team_id"])] += counts
        read += 1

    ours = Counter()
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
        seat = 0 if a0 == args.submission else 1
        replay = json.loads(path.read_text(encoding="utf-8"))
        counts = scan(replay, seat)
        rewards = replay.get("rewards") or [None, None]
        other = rewards[1 - seat]
        counts["wins"] = int(
            (rewards[seat] or 0) > (other if other is not None else 0)
        )
        ours += counts

    def rate(counts: Counter, name: str):
        numerator, denominator = RATES[name]
        total = counts[denominator]
        if total < MIN_DENOMINATOR:
            return None, total
        return round(counts[numerator] / total, 4), total

    teams = {}
    for team, counts in per_team.items():
        teams[str(team)] = {
            "rating": ratings.get(team),
            "games": counts["games"],
            **{
                name: {"rate": rate(counts, name)[0],
                       "n": rate(counts, name)[1]}
                for name in RATES
            },
        }
    v8 = {
        "rating": None, "games": ours["games"],
        **{
            name: {"rate": rate(ours, name)[0], "n": rate(ours, name)[1]}
            for name in RATES
        },
    }

    gradients = []
    for name in RATES:
        points = [
            (row["rating"], row[name]["rate"], row[name]["n"])
            for row in teams.values()
            if row["rating"] is not None and row[name]["rate"] is not None
        ]
        result = spearman(points)
        values = [p[1] for p in points]
        elite = [
            row[name]["rate"] for row in teams.values()
            if row["rating"] is not None and row["rating"] >= 1100
            and row[name]["rate"] is not None
        ]
        rest = [
            row[name]["rate"] for row in teams.values()
            if row["rating"] is not None and row["rating"] < 1100
            and row[name]["rate"] is not None
        ]
        gradients.append({
            "statistic": name,
            "n_pilots": result["n"],
            "rho": result["rho"],
            "p": result["p"],
            "field_min": round(min(values), 4) if values else None,
            "field_max": round(max(values), 4) if values else None,
            "elite_mean": (
                round(sum(elite) / len(elite), 4) if elite else None
            ),
            "rest_mean": round(sum(rest) / len(rest), 4) if rest else None,
            "v8": v8[name]["rate"],
            "v8_n": v8[name]["n"],
        })
    benjamini_hochberg(gradients)
    gradients.sort(key=lambda row: (row["p"] is None, row["p"]))

    report = {
        "field_replays_read": read,
        "pilots": len(teams),
        "statistics": len(RATES),
        "min_denominator": MIN_DENOMINATOR,
        "gradients": gradients,
        "per_team": teams,
        "v8": v8,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"replays={read} pilots={len(teams)} statistics={len(RATES)}")
    print(f"{'statistic':30s} {'n':>3} {'rho':>7} {'p':>8} {'BH':>4} "
          f"{'elite':>7} {'rest':>7} {'v8':>7} {'v8 n':>6}")
    for row in gradients:
        if row["p"] is None:
            continue
        print(
            f"{row['statistic']:30s} {row['n_pilots']:3d} "
            f"{row['rho']:7.3f} {row['p']:8.4f} "
            f"{'*' if row.get('bh_significant') else '':>4} "
            f"{_fmt(row['elite_mean'])} {_fmt(row['rest_mean'])} "
            f"{_fmt(row['v8'])} {row['v8_n']:6d}"
        )
    return 0


def _fmt(value) -> str:
    return f"{value:7.3f}" if value is not None else "      -"


if __name__ == "__main__":
    raise SystemExit(main())
