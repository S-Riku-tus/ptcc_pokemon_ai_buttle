"""Split the Alakazam-vs-Grimmsnarl gap into board management vs execution.

Alakazam is a consumable: Shadow Bullet (180 x2 Darkness weakness = 360)
one-shots it every turn, so the matchup is a rebuild race. At the start of
each of our turns we are in one of a few "readiness" states, and each state
has a different chance of producing a Powerful Hand that turn.

This measures, for our agent and for the field's agents playing the
IDENTICAL 60-card list:

  * how often we start a turn in each readiness state (board management)
  * how often each state converts into a Powerful Hand (execution)

Usage:
  python scripts/analyze_alakazam_rebuild_readiness.py
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "runs" / "leaderboard_top50" / "grimmsnarl"

CARDS: dict[int, dict[str, Any]] = {
    c["cardId"]: c
    for c in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}

ALAKAZAM, KADABRA, ABRA = 743, 742, 741
RARE_CANDY = 1079
# Powerful Hand costs {P}; Enriching Energy (13) only provides {C}.
PSYCHIC_ENERGY = {5, 19}
MAIN_SELECT, ATTACK_OPTION = 0, 13

OUR_RUNS = [
    ("data/runs/alakazam/20260730_v31_sub55076863", 55076863),
    ("data/runs/alakazam/20260729_v28_sub55059233", 55059233),
    ("data/runs/alakazam/20260728_v26_sub55045289", 55045289),
    ("data/runs/alakazam/20260728_v25_sub55031311", 55031311),
    ("data/runs/alakazam/20260727_v24_rerun_sub55021266", 55021266),
]

STATES = [
    "A. Alakazam ACTIVE with {P} - can attack now",
    "B. Alakazam BENCHED with {P} - needs a switch",
    "C. Alakazam in play, no {P} on it - needs the attachment",
    "D. Kadabra ready + Alakazam in hand - one evolve away",
    "E. Abra ready + Rare Candy + Alakazam in hand",
    "F. Alakazam in hand, no body ready to evolve",
    "G1. Kadabra ready, Alakazam NOT in hand - must search",
    "G2. Abra ready only, Alakazam NOT in hand - must search",
    "G3. nothing: no Alakazam line in play or hand",
]


def archetype(deck: list[int]) -> str:
    pokes = Counter(
        cid for cid in deck
        if CARDS.get(cid) and CARDS[cid]["cardType"] == 0
    )
    if not pokes:
        return "unknown"

    def key(item: tuple[int, int]) -> tuple:
        cid, count = item
        card = CARDS[cid]
        return (
            card["stage2"], card["megaEx"] or card["ex"],
            card["stage1"], count, card["hp"],
        )

    return CARDS[max(pokes.items(), key=key)[0]]["name"]


def has_psychic(pokemon: dict[str, Any]) -> bool:
    return any(
        card.get("id") in PSYCHIC_ENERGY
        for card in (pokemon.get("energyCards") or [])
    )


def classify(state: dict[str, Any], seat: int) -> str:
    me = state["players"][seat]
    active_list = me.get("active") or []
    active = active_list[0] if active_list else {}
    bench = me.get("bench") or []
    hand = [c.get("id") for c in (me.get("hand") or [])]
    board = ([active] if active else []) + list(bench)

    zam = [p for p in board if p.get("id") == ALAKAZAM]
    ready_kad = [
        p for p in board
        if p.get("id") == KADABRA and not p.get("appearThisTurn")
    ]
    ready_abra = [
        p for p in board
        if p.get("id") == ABRA and not p.get("appearThisTurn")
    ]

    if active.get("id") == ALAKAZAM and has_psychic(active):
        return STATES[0]
    if any(has_psychic(p) for p in zam):
        return STATES[1]
    if zam:
        return STATES[2]
    if ready_kad and ALAKAZAM in hand:
        return STATES[3]
    if ready_abra and RARE_CANDY in hand and ALAKAZAM in hand:
        return STATES[4]
    if ALAKAZAM in hand:
        return STATES[5]
    if ready_kad:
        return STATES[6]
    if ready_abra:
        return STATES[7]
    return STATES[8]


def scan_game(steps: list, seat: int) -> Iterator[tuple[str, bool]]:
    """Yield (readiness state at the start of our turn, attacked?) pairs."""
    first_state: dict[int, dict[str, Any]] = {}
    attacked: set[int] = set()
    for step in steps:
        obs = step[seat].get("observation") or {}
        sel = obs.get("select") or {}
        state = obs.get("current")
        if sel.get("type") != MAIN_SELECT or not isinstance(state, dict):
            continue
        turn = state.get("turn") or 0
        first_state.setdefault(turn, state)
        me = state["players"][seat]
        active_list = me.get("active") or []
        active = active_list[0] if active_list else {}
        if active.get("id") == ALAKAZAM and any(
            o.get("type") == ATTACK_OPTION for o in (sel.get("option") or [])
        ):
            attacked.add(turn)

    if not first_state:
        return
    last = max(first_state)
    for turn, state in sorted(first_state.items()):
        # skip the opening turns (no Stage 2 is possible) and the final
        # partial turn, which the game may have ended during
        if turn < 4 or turn == last:
            continue
        yield classify(state, seat), turn in attacked


def our_games() -> Iterator[tuple[str, bool]]:
    for run_dir, sub in OUR_RUNS:
        run = ROOT / run_dir
        seats: dict[str, int] = {}
        for row in csv.DictReader(
            (run / "episodes.csv").read_text(encoding="utf-8-sig").splitlines()
        ):
            for seat in (0, 1):
                if str(row.get(f"agent_{seat}_submission_id")) == str(sub):
                    seats[str(row["episode_id"])] = seat
        for ep_dir in sorted((run / "episodes").iterdir()):
            seat = seats.get(ep_dir.name)
            if seat is None:
                continue
            replay = json.loads(
                next((ep_dir / "replay").glob("*.json"))
                .read_text(encoding="utf-8")
            )
            steps = replay.get("steps") or []
            if len(steps) < 2:
                continue
            opp_deck = steps[1][1 - seat].get("action")
            if not (isinstance(opp_deck, list) and len(opp_deck) == 60):
                continue
            if archetype(opp_deck) != "Marnie's Grimmsnarl ex":
                continue
            yield from scan_game(steps, seat)


def field_games(our_deck: Counter) -> Iterator[tuple[str, bool]]:
    for zip_path in sorted(ARCHIVE.glob("*.zip")):
        sub = int(zip_path.stem.split("_")[-1])
        with ZipFile(zip_path) as zf:
            seats: dict[str, int] = {}
            for entry in zf.namelist():
                if entry.endswith("episodes.csv"):
                    for row in csv.DictReader(io.StringIO(
                        zf.read(entry).decode("utf-8-sig")
                    )):
                        for seat in (0, 1):
                            if str(
                                row.get(f"agent_{seat}_submission_id")
                            ) == str(sub):
                                seats[str(row["episode_id"])] = seat
            for entry in zf.namelist():
                if "/replay/" not in entry or not entry.endswith(".json"):
                    continue
                episode = Path(entry).stem.replace("episode_", "")
                grimm_seat = seats.get(episode)
                if grimm_seat is None:
                    continue
                replay = json.loads(zf.read(entry).decode("utf-8"))
                steps = replay.get("steps") or []
                if len(steps) < 2:
                    continue
                zam_seat = 1 - grimm_seat
                deck = steps[1][zam_seat].get("action")
                if not (isinstance(deck, list) and len(deck) == 60):
                    continue
                if Counter(deck) != our_deck:
                    continue
                yield from scan_game(steps, zam_seat)


def report(label: str, pairs: list[tuple[str, bool]]) -> dict[str, Any]:
    total = len(pairs)
    by_state: defaultdict[str, list[bool]] = defaultdict(list)
    for state, attacked in pairs:
        by_state[state].append(attacked)
    print(f"\n=== {label}: {total} of our turns (turn>=4) ===")
    print(f"{'readiness at start of our turn':52s} {'share':>7} "
          f"{'->attacked':>11}")
    for state in STATES:
        rows = by_state.get(state, [])
        if not rows:
            continue
        print(f"{state:52s} {len(rows) / total * 100:6.1f}% "
              f"{sum(rows)}/{len(rows):<4} "
              f"({sum(rows) / len(rows) * 100:5.1f}%)")
    landed = sum(a for _, a in pairs)
    print(f"{'TOTAL':52s} {100.0:6.1f}% {landed}/{total:<4} "
          f"({landed / total * 100:5.1f}%)")
    return {"total": total, "by_state": by_state}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--our-deck",
        type=Path,
        default=ROOT / "agents" / "alakazam" / "alakazam_ml_v31" / "deck.csv",
    )
    args = parser.parse_args()
    deck = Counter(
        int(x) for x in args.our_deck.read_text(encoding="utf-8").split()
    )

    mine = list(our_games())
    field = list(field_games(deck))
    ours = report("OUR agents (v24r-v31)", mine)
    theirs = report("FIELD agents, identical 60-card list", field)

    print("\n=== where the gap comes from ===")
    print(f"{'readiness state':52s} {'our share':>10} {'field':>7}   "
          f"{'our conv':>9} {'field':>7}")
    for state in STATES:
        a = ours["by_state"].get(state, [])
        b = theirs["by_state"].get(state, [])
        if not a and not b:
            continue
        share_a = len(a) / ours["total"] * 100
        share_b = len(b) / theirs["total"] * 100
        conv_a = f"{sum(a) / len(a) * 100:.0f}%" if a else "-"
        conv_b = f"{sum(b) / len(b) * 100:.0f}%" if b else "-"
        print(f"{state:52s} {share_a:9.1f}% {share_b:6.1f}%   "
              f"{conv_a:>9} {conv_b:>7}")

    # Counterfactual: if we reached each state as often as the field does
    # but kept our own conversion rates, how many attacks would we land?
    gain_share = 0.0
    gain_conv = 0.0
    for state in STATES:
        a = ours["by_state"].get(state, [])
        b = theirs["by_state"].get(state, [])
        if not b:
            continue
        conv_a = (sum(a) / len(a)) if a else 0.0
        conv_b = sum(b) / len(b)
        share_a = len(a) / ours["total"]
        share_b = len(b) / theirs["total"]
        gain_share += (share_b - share_a) * conv_a
        gain_conv += share_a * (conv_b - conv_a)
    base = sum(a for _, a in mine) / ours["total"]
    print(f"\nour Powerful Hands per turn: {base * 100:.1f}%")
    print(f"  + field's readiness mix, our conversion : "
          f"{(base + gain_share) * 100:.1f}% "
          f"({gain_share * 100:+.1f}pp  <- board management)")
    print(f"  + our readiness mix, field's conversion : "
          f"{(base + gain_conv) * 100:.1f}% "
          f"({gain_conv * 100:+.1f}pp  <- execution)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
