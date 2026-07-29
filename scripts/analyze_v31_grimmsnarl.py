"""Deep-dive the v31 ladder run's Marnie's Grimmsnarl ex matchups.

Reconstructs a readable turn-by-turn log from our seat's observation logs and
aggregates the numbers that decide this matchup: who went first, when each
side's main attacker came online, hand size at every Powerful Hand, how much
damage each attack actually put on the board, and the prize trade.

Usage:
  python scripts/analyze_v31_grimmsnarl.py <run_dir> --submission-id <ID>
      [--opponent "Marnie's Grimmsnarl ex"] [--log EPISODE_ID]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

CARDS: dict[int, dict[str, Any]] = {
    c["cardId"]: c
    for c in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}
ATTACKS: dict[int, dict[str, Any]] = {
    a["attackId"]: a
    for a in json.loads(
        (ROOT / "vendor" / "cg" / "attacks.json").read_text(encoding="utf-8")
    )
}

ALAKAZAM, KADABRA, ABRA = 743, 742, 741
DUDUNSPARCE, DUNSPARCE = 66, 305
SHAYMIN, FEZANDIPITI = 343, 140
RARE_CANDY = 1079
GRIMMSNARL_EX, MUNKIDORI, FROSLASS = 648, 112, 104

# LogType values that matter here (see vendor/cg/api.py::LogType)
TURN_START = 2
MOVE_CARD = 6
SWITCH = 8
PLAY = 10
ATTACH = 11
EVOLVE = 12
ATTACK = 15
HP_CHANGE = 16
RESULT = 23

AREA_ACTIVE, AREA_BENCH, AREA_DISCARD = 4, 5, 3


def name(card_id: int | None) -> str:
    if card_id is None:
        return "?"
    card = CARDS.get(card_id)
    return card["name"] if card else f"#{card_id}"


def attack_name(attack_id: int | None) -> str:
    if attack_id is None:
        return "?"
    atk = ATTACKS.get(attack_id)
    return atk["name"] if atk else f"attack#{attack_id}"


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


def seat_map(run: Path, submission_id: int) -> dict[str, int]:
    path = run / "episodes.csv"
    rows = list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))
    mapping: dict[str, int] = {}
    for row in rows:
        for seat in (0, 1):
            if str(row.get(f"agent_{seat}_submission_id")) == str(submission_id):
                mapping[str(row["episode_id"])] = seat
    return mapping


def _stream(run: Path, episode_id: str, agent: int) -> list[dict[str, Any]]:
    """Concatenate one agent's per-step observation logs, dropping repeats.

    The engine re-emits the same log block on every step where the agent is
    only observing, so consecutive identical blocks are deduplicated.
    """
    path = (
        run / "episodes" / episode_id / f"agent_{agent}"
        / f"agent_{agent}_observation_logs.json"
    )
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    out: list[dict[str, Any]] = []
    previous: str | None = None
    for entry in entries:
        blob = json.dumps(entry["logs"], sort_keys=True)
        if blob == previous:
            continue
        previous = blob
        out.extend(entry["logs"])
    return out


def merged_logs(run: Path, episode_id: str, seat: int) -> list[dict[str, Any]]:
    """Return the log stream that covers the most turns.

    A losing agent stops being polled once it can no longer act, so its own
    stream is missing the final turn. The other seat's stream sees the same
    public events and runs to the end, so prefer whichever is longer.
    """
    ours = _stream(run, episode_id, seat)
    theirs = _stream(run, episode_id, 1 - seat)
    turns_ours = sum(1 for log in ours if log.get("type") == TURN_START)
    turns_theirs = sum(1 for log in theirs if log.get("type") == TURN_START)
    return theirs if turns_theirs > turns_ours else ours


def decks_from_replay(replay: dict[str, Any]) -> list[list[int] | None]:
    decks: list[list[int] | None] = [None, None]
    steps = replay["steps"]
    if len(steps) > 1:
        for seat in (0, 1):
            action = steps[1][seat].get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[seat] = action
    return decks


def state_timeline(replay: dict[str, Any], seat: int) -> list[dict[str, Any]]:
    out = []
    for step in replay["steps"]:
        if seat >= len(step):
            continue
        obs = step[seat].get("observation") or {}
        state = obs.get("current")
        if isinstance(state, dict) and (state.get("players") or []):
            out.append(state)
    return out


def analyse(run: Path, episode_id: str, seat: int) -> dict[str, Any]:
    replay = json.loads(
        (run / "episodes" / episode_id / "replay"
         / f"episode_{episode_id}.json").read_text(encoding="utf-8")
    )
    decks = decks_from_replay(replay)
    logs = merged_logs(run, episode_id, seat)
    opp = 1 - seat
    # Both seats observe the same public board; the loser stops being polled
    # first, so take whichever timeline reaches the later turn.
    states = max(
        (state_timeline(replay, seat), state_timeline(replay, opp)),
        key=lambda ts: ts[-1].get("turn") or 0 if ts else -1,
    )

    reward = replay["steps"][-1][seat].get("reward")
    won = bool(reward and reward > 0)

    turn = 0
    went_first: int | None = None
    events: list[str] = []
    my_attacks: list[dict[str, Any]] = []
    opp_attacks: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    first_evolve: dict[int, int] = {}
    damage_taken: Counter = Counter()
    my_evolves: Counter = Counter()
    my_plays: Counter = Counter()
    shaymin_turn: int | None = None
    our_kos: list[dict[str, Any]] = []
    their_kos: list[dict[str, Any]] = []

    for log in logs:
        kind = log.get("type")
        player = log.get("playerIndex")
        side = "US " if player == seat else "OPP"

        if kind == TURN_START:
            turn += 1
            if went_first is None:
                went_first = player
            pending = None
            events.append(f"T{turn:>2} --- {side.strip()} turn ---")
        elif kind == EVOLVE:
            cid = log.get("cardId")
            if cid is not None and cid not in first_evolve:
                first_evolve[cid] = turn
            if player == seat:
                my_evolves[cid] += 1
            events.append(f"T{turn:>2} {side} evolve -> {name(cid)}")
        elif kind == PLAY:
            cid = log.get("cardId")
            if player == seat:
                my_plays[cid] += 1
                if cid == SHAYMIN and shaymin_turn is None:
                    shaymin_turn = turn
            events.append(f"T{turn:>2} {side} play {name(cid)}")
        elif kind == SWITCH:
            events.append(
                f"T{turn:>2} {side} switch active -> "
                f"{name(log.get('cardIdBench'))}"
            )
        elif kind == ATTACK:
            attacker = log.get("cardId")
            pending = {
                "turn": turn,
                "player": player,
                "attacker": attacker,
                "attack_id": log.get("attackId"),
                "damage": 0,
                "targets": [],
            }
            (my_attacks if player == seat else opp_attacks).append(pending)
            events.append(
                f"T{turn:>2} {side} ATTACK {name(attacker)} / "
                f"{attack_name(log.get('attackId'))}"
            )
        elif kind == HP_CHANGE:
            # `value` is the HP delta on the card owned by `playerIndex`:
            # negative = damage taken, positive = healed / counter removed.
            value = log.get("value")
            cid = log.get("cardId")
            if not isinstance(value, int) or value >= 0:
                if isinstance(value, int) and value > 0:
                    events.append(
                        f"T{turn:>2}      heal +{value} on {side}{name(cid)}"
                    )
                continue
            dealt = -value
            victim_side = "US " if player == seat else "OPP"
            damage_taken[player] += dealt
            if pending is not None and player != pending["player"]:
                pending["damage"] += dealt
                pending["targets"].append((cid, dealt))
            events.append(
                f"T{turn:>2}      dmg {dealt:>3} -> {victim_side}{name(cid)}"
                f"{' (counters)' if log.get('putDamageCounter') else ''}"
            )
        elif kind == MOVE_CARD:
            # A Pokemon body moving from Active/Bench into the discard pile
            # is a Knock Out (evolutions move via PRE_EVOLUTION instead).
            if (
                log.get("fromArea") in (AREA_ACTIVE, AREA_BENCH)
                and log.get("toArea") == AREA_DISCARD
                and CARDS.get(log.get("cardId"), {}).get("cardType") == 0
            ):
                where = "active" if log["fromArea"] == AREA_ACTIVE else "bench"
                record = {
                    "turn": turn,
                    "card": log.get("cardId"),
                    "area": where,
                }
                (our_kos if player == seat else their_kos).append(record)
                events.append(
                    f"T{turn:>2}      KO {side}{name(log.get('cardId'))} "
                    f"({where})"
                )
        elif kind == RESULT:
            events.append(f"T{turn:>2} RESULT {json.dumps(log)}")

    # Board / prize snapshots. Prize lists are empty until setup finishes,
    # so only start tracking once a full 6-prize layout has been seen.
    hand_at_turn: dict[int, int] = {}
    bench_at_turn: dict[int, int] = {}
    my_prizes = opp_prizes = 0
    started = False
    for state in states:
        state_turn = state.get("turn") or 0
        players = state["players"]
        mine, theirs = players[seat], players[opp]
        mine_left = len(mine.get("prize") or [])
        theirs_left = len(theirs.get("prize") or [])
        if not started:
            if mine_left == 6 and theirs_left == 6:
                started = True
            else:
                continue
        # You take prizes from your OWN pile when you KO something.
        my_prizes = max(my_prizes, 6 - mine_left)
        opp_prizes = max(opp_prizes, 6 - theirs_left)
        hand_at_turn[state_turn] = mine.get("handCount") or 0
        bench_at_turn[state_turn] = min(
            bench_at_turn.get(state_turn, 9), len(mine.get("bench") or [])
        )

    final = states[-1] if states else None
    loss_mode = ""
    if final and not won:
        mine = final["players"][seat]
        bodies = len(mine.get("active") or []) + len(mine.get("bench") or [])
        if (mine.get("deckCount") or 0) == 0:
            loss_mode = "deck-out"
        elif opp_prizes >= 6:
            loss_mode = "prized-out"
        elif bodies <= 1:
            loss_mode = "board-out"
        else:
            loss_mode = f"other(bodies={bodies})"

    powerful_hand = [a for a in my_attacks if a["attacker"] == ALAKAZAM]
    shadow_bullet = [a for a in opp_attacks if a["attacker"] == GRIMMSNARL_EX]

    return {
        "episode_id": episode_id,
        "won": won,
        "seat": seat,
        "we_went_first": went_first == seat,
        "turns": turn,
        "opp_archetype": archetype(decks[opp]) if decks[opp] else "unknown",
        "opp_deck": decks[opp],
        "my_prizes_taken": my_prizes,
        "opp_prizes_taken": opp_prizes,
        "my_alakazam_turn": first_evolve.get(ALAKAZAM),
        "opp_grimmsnarl_turn": first_evolve.get(GRIMMSNARL_EX),
        "my_attacks": my_attacks,
        "opp_attacks": opp_attacks,
        "powerful_hand_count": len(powerful_hand),
        "powerful_hand_damage": [a["damage"] for a in powerful_hand],
        "shadow_bullet_count": len(shadow_bullet),
        "opp_attack_damage": [a["damage"] for a in opp_attacks],
        "my_total_damage": sum(a["damage"] for a in my_attacks),
        "opp_total_damage": sum(a["damage"] for a in opp_attacks),
        "first_attack_turn": my_attacks[0]["turn"] if my_attacks else None,
        "opp_first_attack_turn": (
            opp_attacks[0]["turn"] if opp_attacks else None
        ),
        "my_deck_left": (
            final["players"][seat].get("deckCount") if final else None
        ),
        "loss_mode": loss_mode,
        "our_kos": our_kos,
        "their_kos": their_kos,
        "our_ko_count": len(our_kos),
        "our_bench_ko_count": sum(1 for k in our_kos if k["area"] == "bench"),
        "their_ko_count": len(their_kos),
        "our_kos_by_card": dict(Counter(name(k["card"]) for k in our_kos)),
        "their_kos_by_card": dict(Counter(name(k["card"]) for k in their_kos)),
        "alakazam_evolves": my_evolves.get(ALAKAZAM, 0),
        "kadabra_evolves": my_evolves.get(KADABRA, 0),
        "rare_candy_played": my_plays.get(RARE_CANDY, 0),
        "shaymin_turn": shaymin_turn,
        "abra_played": my_plays.get(ABRA, 0),
        "bench_at_turn": bench_at_turn,
        "hand_sizes_on_powerful_hand": [
            a["damage"] // 20 for a in powerful_hand
        ],
        "my_bench_left": (
            len(final["players"][seat].get("bench") or []) if final else None
        ),
        "damage_we_took": damage_taken.get(seat, 0),
        "damage_we_dealt": damage_taken.get(opp, 0),
        "events": events,
        "hand_at_turn": hand_at_turn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--opponent", default="Marnie's Grimmsnarl ex")
    parser.add_argument("--log", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    seats = seat_map(args.run, args.submission_id)
    results = []
    for ep_dir in sorted((args.run / "episodes").iterdir()):
        seat = seats.get(ep_dir.name)
        if seat is None:
            continue
        res = analyse(args.run, ep_dir.name, seat)
        if args.opponent and res["opp_archetype"] != args.opponent:
            continue
        results.append(res)

    print(f"=== {args.opponent}: {len(results)} games, "
          f"{sum(r['won'] for r in results)} wins ===\n")
    header = (
        f"{'episode':>9} {'R':>4} {'1st':>4} {'trn':>3} {'zam':>4} "
        f"{'grm':>4} {'atk1':>4} {'oat1':>4} {'prz':>5} {'PH':>2} "
        f"{'dealt':>6} {'taken':>6} {'bch':>3} {'dck':>3}  {'loss mode':<12} "
        f"{'PH dmg'}"
    )
    print(header)
    for r in sorted(results, key=lambda x: (x["won"], x["episode_id"])):
        print(
            f"{r['episode_id']:>9} {'WIN' if r['won'] else 'LOSS':>4} "
            f"{'US' if r['we_went_first'] else 'OPP':>4} {r['turns']:>3} "
            f"{str(r['my_alakazam_turn']):>4} "
            f"{str(r['opp_grimmsnarl_turn']):>4} "
            f"{str(r['first_attack_turn']):>4} "
            f"{str(r['opp_first_attack_turn']):>4} "
            f"{r['my_prizes_taken']}-{r['opp_prizes_taken']:<3} "
            f"{r['powerful_hand_count']:>2} "
            f"{r['damage_we_dealt']:>6} {r['damage_we_took']:>6} "
            f"{str(r['my_bench_left']):>3} {str(r['my_deck_left']):>3}  "
            f"{r['loss_mode']:<12} "
            f"{r['powerful_hand_damage']}"
        )

    wins = [r for r in results if r["won"]]
    losses = [r for r in results if not r["won"]]

    def avg(rows: list[dict[str, Any]], key: str) -> str:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return f"{sum(vals) / len(vals):.2f}" if vals else "-"

    print("\n--- WIN vs LOSS averages ---")
    for key in (
        "turns", "my_alakazam_turn", "opp_grimmsnarl_turn",
        "first_attack_turn", "opp_first_attack_turn",
        "my_prizes_taken", "opp_prizes_taken",
        "powerful_hand_count", "damage_we_dealt", "damage_we_took",
        "my_deck_left", "alakazam_evolves", "kadabra_evolves",
        "rare_candy_played", "abra_played", "shaymin_turn",
        "our_ko_count", "our_bench_ko_count", "their_ko_count",
    ):
        print(f"  {key:24s} win={avg(wins, key):>7}  loss={avg(losses, key):>7}")
    print(f"  {'went first':24s} "
          f"win={sum(r['we_went_first'] for r in wins)}/{len(wins)}  "
          f"loss={sum(r['we_went_first'] for r in losses)}/{len(losses)}")

    print("\n--- per-game resources ---")
    print(f"{'episode':>9} {'R':>4} {'zamEvo':>6} {'kadEvo':>6} "
          f"{'candy':>5} {'abra':>4} {'shaymin@':>8}  hand sizes on "
          f"Powerful Hand / bench by turn")
    for r in sorted(results, key=lambda x: (x["won"], x["episode_id"])):
        bench = " ".join(
            f"{t}:{n}" for t, n in sorted(r["bench_at_turn"].items())
        )
        print(
            f"{r['episode_id']:>9} {'WIN' if r['won'] else 'LOSS':>4} "
            f"{r['alakazam_evolves']:>6} {r['kadabra_evolves']:>6} "
            f"{r['rare_candy_played']:>5} {r['abra_played']:>4} "
            f"{str(r['shaymin_turn']):>8}  "
            f"{r['hand_sizes_on_powerful_hand']}\n"
            f"{'':>46}bench {bench}\n"
            f"{'':>46}we lost {r['our_ko_count']} mons "
            f"({r['our_bench_ko_count']} on bench) {r['our_kos_by_card']}; "
            f"we KO'd {r['their_ko_count']} {r['their_kos_by_card']}"
        )

    print("\n--- opponent deck (union across games) ---")
    union: Counter = Counter()
    per_game: defaultdict[int, list[int]] = defaultdict(list)
    for r in results:
        counts = Counter(r["opp_deck"] or [])
        for cid, n in counts.items():
            per_game[cid].append(n)
        union.update(counts.keys())
    for cid, games in sorted(
        per_game.items(), key=lambda kv: (-len(kv[1]), -max(kv[1]))
    ):
        card = CARDS.get(cid, {})
        print(f"  {len(games):>2}/{len(results)} games  "
              f"x{min(games)}-{max(games)}  {cid:>5} {card.get('name', '?'):32s}"
              f" hp={card.get('hp')}")

    for episode_id in args.log:
        match = next(
            (r for r in results if r["episode_id"] == str(episode_id)), None
        )
        if match is None:
            print(f"\n[log] episode {episode_id} not in filtered set")
            continue
        print(f"\n===== FULL LOG {episode_id} "
              f"({'WIN' if match['won'] else 'LOSS'}) =====")
        for line in match["events"]:
            print(line)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
