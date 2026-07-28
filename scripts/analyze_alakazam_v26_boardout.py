"""Quantify Alakazam board-out risk: turns we END holding a single body.

For every MAIN decision where we picked END while our board had at most one
Pokemon, records which options were still on the table (Basic searches, plays,
abilities). Also flags Run Away Draw uses that shrank the board to one body.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END = 7, 8, 9, 10, 12, 13, 14
MAIN = 0
DUDUNSPARCE = 66

CARDS = {
    c["cardId"]: c
    for c in json.loads(Path("vendor/cg/cards.json").read_text(encoding="utf-8"))
}

# Cards that can put a Basic Pokemon onto the bench or into hand.
BASIC_FETCHERS = {
    1086: "Buddy-Buddy Poffin",
    1231: "Dawn",
    1152: "Poke Pad",
    19: "Telepath Psychic Energy",
    1097: "Night Stretcher",
    1184: "Lana's Aid",
}


def _name(cid: int | None) -> str:
    if cid is None:
        return "?"
    c = CARDS.get(cid)
    return c["name"] if c else str(cid)


def _iter_replays(source: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    if source.is_dir():
        for path in sorted(source.glob("episodes/*/replay/*.json")):
            yield path.stem, json.loads(path.read_text(encoding="utf-8"))
        return
    with ZipFile(source) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if "/replay/" in info.filename and info.filename.endswith(".json"):
                with zf.open(info) as fh:
                    yield Path(info.filename).stem, json.loads(fh.read().decode("utf-8"))


def _seats(source: Path, submission_id: int) -> dict[str, int]:
    rows: list[dict[str, str]] = []
    if source.is_dir():
        p = source / "episodes.csv"
        if p.exists():
            rows = list(csv.DictReader(p.read_text(encoding="utf-8-sig").splitlines()))
    else:
        with ZipFile(source) as zf:
            names = [n for n in zf.namelist() if n.endswith("episodes.csv")]
            if names:
                rows = list(
                    csv.DictReader(zf.read(names[0]).decode("utf-8-sig").splitlines())
                )
    out: dict[str, int] = {}
    for row in rows:
        for seat in (0, 1):
            if str(row.get(f"agent_{seat}_submission_id")) == str(submission_id):
                out[str(row["episode_id"])] = seat
    return out


def _bodies(player: dict[str, Any]) -> int:
    n = 0
    for zone in ("active", "bench"):
        n += sum(1 for c in (player.get(zone) or []) if isinstance(c, dict))
    return n


def _describe(option: dict[str, Any], hand: list[dict[str, Any]]) -> str:
    t = option.get("type")
    idx = option.get("index")
    cid = option.get("cardId")
    if cid is None and isinstance(idx, int) and t in (PLAY, ATTACH) and idx < len(hand):
        cid = hand[idx]["id"]
    label = {
        PLAY: "PLAY", ATTACH: "ATTACH", EVOLVE: "EVOLVE", ABILITY: "ABILITY",
        RETREAT: "RETREAT", ATTACK: "ATTACK", END: "END",
    }.get(t, str(t))
    return f"{label}:{_name(cid)}" if cid else label


def analyse(eid: str, replay: dict[str, Any], seat: int) -> dict[str, Any]:
    steps = replay["steps"]
    won = bool((steps[-1][seat].get("reward") or 0) > 0)
    lone_ends: list[dict[str, Any]] = []
    runaway_to_lone: list[dict[str, Any]] = []
    min_bodies_after_our_turn = 99
    prev_bodies = None

    for i, step in enumerate(steps):
        if seat >= len(step):
            continue
        obs = step[seat].get("observation") or {}
        state = obs.get("current")
        select = obs.get("select")
        if not isinstance(state, dict) or not select:
            continue
        if select.get("type") != MAIN:
            continue
        if i + 1 >= len(steps):
            continue
        action = steps[i + 1][seat].get("action")
        options = select.get("option") or []
        if not (isinstance(action, list) and len(action) == 1):
            continue
        pick = action[0]
        if not (isinstance(pick, int) and 0 <= pick < len(options)):
            continue
        chosen = options[pick]
        us = state["players"][seat]
        hand = [c for c in (us.get("hand") or []) if isinstance(c, dict)]
        bodies = _bodies(us)
        prev_bodies = bodies

        if chosen.get("type") == END:
            min_bodies_after_our_turn = min(min_bodies_after_our_turn, bodies)
            if bodies <= 1:
                hand_ids = Counter(c["id"] for c in hand)
                lone_ends.append(
                    {
                        "turn": state.get("turn"),
                        "bodies": bodies,
                        "hand": [_name(c["id"]) for c in hand],
                        "options": [_describe(o, hand) for o in options],
                        "basic_fetchers_in_hand": [
                            BASIC_FETCHERS[cid]
                            for cid in hand_ids
                            if cid in BASIC_FETCHERS
                        ],
                        "playable_non_end_options": sum(
                            1 for o in options if o.get("type") != END
                        ),
                    }
                )
        if chosen.get("type") == ABILITY:
            # Run Away Draw shrinks the board by one body
            idx = chosen.get("index")
            area = chosen.get("area")
            cid = chosen.get("cardId")
            if cid is None:
                zone = "active" if area == 4 else "bench"
                cards = [c for c in (us.get(zone) or []) if isinstance(c, dict)]
                if isinstance(idx, int) and idx < len(cards):
                    cid = cards[idx]["id"]
            if cid == DUDUNSPARCE and bodies <= 2:
                runaway_to_lone.append(
                    {"turn": state.get("turn"), "bodies_before": bodies}
                )

    del prev_bodies
    return {
        "episode_id": eid,
        "won": won,
        "lone_end_turns": lone_ends,
        "lone_end_count": len(lone_ends),
        "runaway_to_lone": runaway_to_lone,
        "min_bodies_at_our_end": (
            None if min_bodies_after_our_turn == 99 else min_bodies_after_our_turn
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--submission-id", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    seats = _seats(args.source, args.submission_id)
    out = []
    for stem, replay in _iter_replays(args.source):
        eid = stem.replace("episode_", "")
        seat = seats.get(eid)
        if seat is None:
            continue
        out.append(analyse(eid, replay, seat))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    lone = [e for e in out if e["lone_end_count"]]
    lost = [e for e in lone if not e["won"]]
    print(f"{len(out)} games | games with a lone-body END: {len(lone)} "
          f"(lost {len(lost)}) | total lone ENDs "
          f"{sum(e['lone_end_count'] for e in out)}")
    print(f"runaway-to-lone events: {sum(len(e['runaway_to_lone']) for e in out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
