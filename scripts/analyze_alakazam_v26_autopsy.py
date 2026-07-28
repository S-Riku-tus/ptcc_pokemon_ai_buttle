"""Autopsy Alakazam ladder replays: how we actually lose, per matchup.

Reconstructs our seat's board timeline from ``current.players`` snapshots and
reports, per episode:

* prizes taken by each side and the turn we first attacked
* every one of our Pokemon that got Knocked Out, split into Active KOs and
  Bench KOs (bench snipe damage) with the damage source turn
* Shaymin (Flower Curtain) availability at the time each bench KO happened
* Powerful Hand size at each of our attacks (Alakazam damage = 20 x hand)
* idle MAIN turns after the first attack
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

ABRA, KADABRA, ALAKAZAM = 741, 742, 743
DUNSPARCE, DUDUNSPARCE = 305, 66
SHAYMIN, FEZANDIPITI = 343, 140
POKE_PAD, DAWN, POFFIN = 1152, 1231, 1086
ATTACK_TYPE = 13
END_TYPE = 14
MAIN_SELECT_TYPE = 0


def _load_cards() -> dict[int, dict[str, Any]]:
    path = Path("vendor/cg/cards.json")
    return {c["cardId"]: c for c in json.loads(path.read_text(encoding="utf-8"))}


CARDS = _load_cards()


def _name(card_id: int | None) -> str:
    if card_id is None:
        return "?"
    card = CARDS.get(card_id)
    return card["name"] if card else str(card_id)


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


def _our_seat(replay: dict[str, Any], submission_id: int) -> int | None:
    agents = ((replay.get("info") or {}).get("TeamNames")) or []
    for seat, step in enumerate(replay["steps"][0]):
        del step
    subs = (replay.get("info") or {}).get("SubmissionIds")
    if isinstance(subs, list) and submission_id in subs:
        return subs.index(submission_id)
    del agents
    return None


def _seat_from_manifest(source: Path, submission_id: int) -> dict[str, int]:
    """episode_id -> our seat, using episodes.csv agent_N_submission_id columns."""
    rows: list[dict[str, str]] = []
    if source.is_dir():
        csv_path = source / "episodes.csv"
        if csv_path.exists():
            rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8-sig").splitlines()))
    else:
        with ZipFile(source) as zf:
            names = [n for n in zf.namelist() if n.endswith("episodes.csv")]
            if names:
                text = zf.read(names[0]).decode("utf-8-sig")
                rows = list(csv.DictReader(text.splitlines()))
    mapping: dict[str, int] = {}
    for row in rows:
        for seat in (0, 1):
            if str(row.get(f"agent_{seat}_submission_id")) == str(submission_id):
                mapping[str(row["episode_id"])] = seat
    return mapping


def _board(player: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for slot, area in (("active", "active"), ("bench", "bench")):
        for card in player.get(slot) or []:
            if isinstance(card, dict) and isinstance(card.get("serial"), int):
                out[card["serial"]] = {**card, "area": area}
    return out


def _zone_ids(player: dict[str, Any], zone: str) -> Counter:
    return Counter(c["id"] for c in (player.get(zone) or []) if isinstance(c, dict))


def _discard_serials(player: dict[str, Any]) -> set[int]:
    return {
        c["serial"]
        for c in (player.get("discard") or [])
        if isinstance(c, dict) and isinstance(c.get("serial"), int)
    }


def analyse_episode(episode_id: str, replay: dict[str, Any], seat: int) -> dict[str, Any]:
    steps = replay["steps"]
    reward = steps[-1][seat].get("reward")
    won = bool(reward and reward > 0)

    prev_board: dict[int, dict[str, Any]] | None = None
    prev_state: dict[str, Any] | None = None
    prev_turn = 0

    our_kos: list[dict[str, Any]] = []
    attacks: list[dict[str, Any]] = []
    bench_damage_events: list[dict[str, Any]] = []
    shaymin_ever_played = False
    shaymin_zone_when_bench_ko: list[str] = []
    max_our_prizes_taken = 0
    max_opp_prizes_taken = 0
    last_state: dict[str, Any] | None = None

    for index, step in enumerate(steps):
        if seat >= len(step):
            continue
        obs = step[seat].get("observation") or {}
        state = obs.get("current")
        if not isinstance(state, dict):
            continue
        players = state.get("players") or []
        if len(players) < 2:
            continue
        us = players[seat]
        opp = players[1 - seat]
        turn = state.get("turn") or 0
        board = _board(us)
        last_state = state

        if turn >= 1:
            max_our_prizes_taken = max(
                max_our_prizes_taken, 6 - len(us.get("prize") or [])
            )
            max_opp_prizes_taken = max(
                max_opp_prizes_taken, 6 - len(opp.get("prize") or [])
            )

        if any(c.get("id") == SHAYMIN for c in board.values()):
            shaymin_ever_played = True

        if prev_board is not None:
            # damage on surviving bench mons
            for serial, card in board.items():
                old = prev_board.get(serial)
                if not old:
                    continue
                delta = (old.get("hp") or 0) - (card.get("hp") or 0)
                if delta > 0 and old["area"] == "bench" and card["area"] == "bench":
                    bench_damage_events.append(
                        {"turn": turn, "id": card["id"], "name": _name(card["id"]), "damage": delta}
                    )
            # KOs: the body left the board AND landed in our discard pile.
            # Evolution (pre-evo serial absorbed) and Run Away Draw
            # (Dudunsparce shuffles itself back into the deck) are not KOs.
            discard_now = _discard_serials(us)
            for serial, old in prev_board.items():
                if serial in board:
                    continue
                evolved = any(
                    any(
                        pre.get("serial") == serial
                        for pre in (c.get("preEvolution") or [])
                    )
                    for c in board.values()
                )
                if evolved:
                    continue
                if serial not in discard_now:
                    continue  # shuffled back / moved, not knocked out
                ko = {
                    "turn": turn,
                    "id": old["id"],
                    "name": _name(old["id"]),
                    "area": old["area"],
                    "hp_before": old.get("hp"),
                    "max_hp": old.get("maxHp"),
                }
                our_kos.append(ko)
                if old["area"] == "bench":
                    if shaymin_ever_played:
                        zone = "in_play"
                    elif prev_state is not None:
                        hand = _zone_ids(players[seat], "hand")
                        discard = _zone_ids(players[seat], "discard")
                        if hand.get(SHAYMIN):
                            zone = "hand"
                        elif discard.get(SHAYMIN):
                            zone = "discard"
                        else:
                            zone = "deck_or_prize"
                    else:
                        zone = "unknown"
                    shaymin_zone_when_bench_ko.append(zone)

        # our attack selections
        select = obs.get("select") or {}
        if select.get("type") == MAIN_SELECT_TYPE and index + 1 < len(steps):
            action = steps[index + 1][seat].get("action")
            options = select.get("option") or []
            if isinstance(action, list) and len(action) == 1:
                pick = action[0]
                if isinstance(pick, int) and 0 <= pick < len(options):
                    chosen = options[pick]
                    if chosen.get("type") == ATTACK_TYPE:
                        active = (us.get("active") or [{}])[0] if us.get("active") else {}
                        attacks.append(
                            {
                                "turn": turn,
                                "attacker": _name(active.get("id")),
                                "attacker_id": active.get("id"),
                                "hand": us.get("handCount") or 0,
                                "deck": us.get("deckCount") or 0,
                            }
                        )

        prev_board = board
        prev_state = state
        prev_turn = turn

    del prev_turn
    opp_ids: Counter = Counter()
    if last_state:
        opp = last_state["players"][1 - seat]
        for zone in ("active", "bench", "discard", "hand"):
            opp_ids.update(_zone_ids(opp, zone))

    bench_kos = [k for k in our_kos if k["area"] == "bench"]
    return {
        "episode_id": episode_id,
        "won": won,
        "turns": last_state.get("turn") if last_state else None,
        "our_prizes_taken": max_our_prizes_taken,
        "opp_prizes_taken": max_opp_prizes_taken,
        "attacks": attacks,
        "attack_count": len(attacks),
        "alakazam_attacks": sum(1 for a in attacks if a["attacker_id"] == ALAKAZAM),
        "avg_hand_on_attack": (
            round(sum(a["hand"] for a in attacks) / len(attacks), 2) if attacks else None
        ),
        "first_attack_turn": attacks[0]["turn"] if attacks else None,
        "our_kos": our_kos,
        "bench_kos": bench_kos,
        "bench_ko_count": len(bench_kos),
        "bench_damage_total": sum(e["damage"] for e in bench_damage_events),
        "bench_damage_events": bench_damage_events,
        "shaymin_played": shaymin_ever_played,
        "shaymin_zone_at_bench_ko": shaymin_zone_when_bench_ko,
        "opp_card_ids": dict(opp_ids.most_common(40)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seats = _seat_from_manifest(args.source, args.submission_id)
    results = []
    for episode_id, replay in _iter_replays(args.source):
        eid = episode_id.replace("episode_", "")
        seat = seats.get(eid)
        if seat is None:
            seat = _our_seat(replay, args.submission_id)
        if seat is None:
            continue
        try:
            results.append(analyse_episode(eid, replay, seat))
        except Exception as exc:  # pragma: no cover - diagnostic tool
            results.append({"episode_id": eid, "error": repr(exc)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"submission_id": args.submission_id, "episodes": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    ok = [r for r in results if "error" not in r]
    print(f"analysed {len(ok)}/{len(results)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
