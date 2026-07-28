"""Turn-level board-count audit: every one of our turns spent at <= 1 body.

For each such turn it dumps the full ordered list of MAIN options we saw and
what we picked, so a missed "add a body" line (bench a Basic, Poffin, Dawn,
Telepath attach onto a {P} Pokemon) is visible instead of inferred.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

PLAY, ATTACH, EVOLVE, ABILITY, RETREAT, ATTACK, END = 7, 8, 9, 10, 12, 13, 14
MAIN = 0

CARDS = {
    c["cardId"]: c
    for c in json.loads(Path("vendor/cg/cards.json").read_text(encoding="utf-8"))
}
PSYCHIC = 5
TELEPATH = 19
# Items/Supporters that can put a Basic Pokemon on the bench (directly or via hand).
BODY_SOURCES = {1086: "Poffin", 1231: "Dawn", 1152: "PokePad", 1097: "NightStretcher",
                1184: "LanasAid", 19: "Telepath"}


def _name(cid):
    c = CARDS.get(cid)
    return c["name"] if c else str(cid)


def _is_basic_pokemon(cid):
    c = CARDS.get(cid)
    return bool(c and c["cardType"] == 0 and not c["stage1"] and not c["stage2"])


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


def _bodies(player):
    return sum(
        1
        for zone in ("active", "bench")
        for c in (player.get(zone) or [])
        if isinstance(c, dict)
    )


def _label(option, hand):
    t = option.get("type")
    idx = option.get("index")
    cid = option.get("cardId")
    if cid is None and isinstance(idx, int) and t in (PLAY, ATTACH) and idx < len(hand):
        cid = hand[idx]["id"]
    tag = {PLAY: "PLAY", ATTACH: "ATTACH", EVOLVE: "EVOLVE", ABILITY: "ABILITY",
           RETREAT: "RETREAT", ATTACK: "ATTACK", END: "END"}.get(t, str(t))
    return (f"{tag}:{_name(cid)}" if cid else tag), cid, t


def analyse(eid, replay, seat):
    steps = replay["steps"]
    won = bool((steps[-1][seat].get("reward") or 0) > 0)
    turns: dict[int, dict[str, Any]] = defaultdict(
        lambda: {"min_bodies": 99, "decisions": []}
    )

    for i, step in enumerate(steps):
        if seat >= len(step):
            continue
        obs = step[seat].get("observation") or {}
        state = obs.get("current")
        select = obs.get("select")
        if not isinstance(state, dict) or not select or select.get("type") != MAIN:
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
        us = state["players"][seat]
        hand = [c for c in (us.get("hand") or []) if isinstance(c, dict)]
        turn = state.get("turn")
        bodies = _bodies(us)
        active = [c for c in (us.get("active") or []) if isinstance(c, dict)]
        active_is_psychic = bool(
            active and (CARDS.get(active[0]["id"]) or {}).get("pokemonType") == PSYCHIC
        )
        labels = []
        adds_body = []
        for o in options:
            lab, cid, t = _label(o, hand)
            labels.append(lab)
            if t == PLAY and cid is not None and _is_basic_pokemon(cid):
                adds_body.append(lab)
            elif t == PLAY and cid in (1086, 1231, 1152, 1097, 1184):
                adds_body.append(lab)
            elif t == ATTACH and cid == TELEPATH and active_is_psychic:
                adds_body.append(lab)
        entry = turns[turn]
        entry["min_bodies"] = min(entry["min_bodies"], bodies)
        entry["decisions"].append(
            {
                "bodies": bodies,
                "chose": labels[pick],
                "adds_body_options": adds_body,
                "options": labels,
                "hand": [_name(c["id"]) for c in hand],
            }
        )

    risky = []
    for turn, entry in sorted(turns.items()):
        if entry["min_bodies"] > 1:
            continue
        missed = [d for d in entry["decisions"] if d["adds_body_options"]]
        risky.append(
            {
                "turn": turn,
                "min_bodies": entry["min_bodies"],
                "had_body_option": bool(missed),
                "took_body_option": any(
                    d["chose"] in d["adds_body_options"] for d in entry["decisions"]
                ),
                "decisions": entry["decisions"],
            }
        )
    return {
        "episode_id": eid,
        "won": won,
        "risky_turns": risky,
        "risky_turn_count": len(risky),
        "missed_body_turns": sum(
            1 for r in risky if r["had_body_option"] and not r["took_body_option"]
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

    risky_games = [e for e in out if e["risky_turn_count"]]
    missed_games = [e for e in out if e["missed_body_turns"]]
    print(f"{len(out)} games")
    print(f"  games with a <=1-body turn : {len(risky_games)} "
          f"(lost {sum(1 for e in risky_games if not e['won'])})")
    print(f"  games that MISSED a body-add: {len(missed_games)} "
          f"(lost {sum(1 for e in missed_games if not e['won'])})")
    counts = Counter()
    for e in out:
        for r in e["risky_turns"]:
            if r["had_body_option"] and not r["took_body_option"]:
                for d in r["decisions"]:
                    for lab in d["adds_body_options"]:
                        counts[lab] += 1
    print("  missed body-add options:", counts.most_common(12))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
