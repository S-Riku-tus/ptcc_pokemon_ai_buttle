"""Analyze HOW the six prizes get taken, across many replays.

For each episode we reconstruct, per player:
  * the public prize-count timeline (len(players[i].prize)),
  * that player's OWN board (serial -> cardId), which is fully visible in that
    player's own observation view,
  * the winner (top-level rewards).

A prize event = the opponent's prize count drops. We attribute it to the
Pokemon that vanished from the losing player's own board at that moment and
record its prize value (megaEx=3, ex=2, else 1). The ordered list of prize
values a player collected is that player's "prize plan" (a multiset summing to
6 for the winner).

We then aggregate, from several angles, which prize plans win, how many KOs /
turns they take, and how it differs between our submission and the top-ladder
field.

Usage:
  python scripts/analyze_prize_taking.py \
      --dataset ours:data/runs/alakazam/20260724_v19_sub54938555 \
      --dataset top40:data/runs/alakazam/20260724_top40_current \
      --our-submission 54938555 \
      --out data/analysis/prize_taking_v19.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor"))
from cg.api import all_card_data  # noqa: E402

CARD = {c.cardId: c for c in all_card_data()}


def prize_value(card_id: int) -> int:
    c = CARD.get(card_id)
    if c is None:
        return 1
    return 3 if getattr(c, "megaEx", False) else 2 if getattr(c, "ex", False) else 1


def card_name(card_id: int) -> str:
    c = CARD.get(card_id)
    return getattr(c, "name", None) or f"#{card_id}" if c else f"#{card_id}"


def is_pokemon(card_id: int) -> bool:
    c = CARD.get(card_id)
    if c is None:
        return False
    # Pokemon cards carry hp; energy/trainer do not.
    return bool(getattr(c, "hp", 0))


def board_bodies(player_state: dict) -> dict:
    """base_serial -> top cardId for each Pokemon *body* on the board.

    A body keeps a stable identity across evolutions: the basic Pokemon's
    serial persists in the top card's preEvolution list, so we key each body by
    the minimum serial in its stack. This prevents an evolution (which mints a
    new top serial) from looking like a KO of the lower stage.
    """
    out = {}
    for zone in ("active", "bench"):
        for card in player_state.get(zone) or []:
            if not isinstance(card, dict) or card.get("serial") is None:
                continue
            serials = [card["serial"]]
            for pe in card.get("preEvolution") or []:
                if isinstance(pe, dict) and pe.get("serial") is not None:
                    serials.append(pe["serial"])
            out[min(serials)] = card.get("id")
    return out


def iter_episode_replays(dataset_dir: Path):
    """Yield (episode_id, replay_path) for a dataset directory.

    Supports both layouts we have on disk:
      * fetch_submission_logs: <root>/episodes/<id>/replay/episode_<id>.json
      * top40 collector:       <root>/replays/episode_<id>.json
    """
    epdir = dataset_dir / "episodes"
    if epdir.is_dir():
        for sub in sorted(epdir.iterdir()):
            rp = sub / "replay" / f"episode_{sub.name}.json"
            if rp.exists():
                yield sub.name, rp
        return
    rdir = dataset_dir / "replays"
    if rdir.is_dir():
        for rp in sorted(rdir.glob("episode_*.json")):
            yield rp.stem.replace("episode_", ""), rp


def analyze_episode(replay_path: Path):
    """Return a per-episode record, or None if unparsable."""
    try:
        data = json.loads(replay_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return None

    rewards = data.get("rewards") or []
    winner = None
    if (len(rewards) == 2 and isinstance(rewards[0], (int, float))
            and isinstance(rewards[1], (int, float))
            and rewards[0] != rewards[1]):
        winner = 0 if rewards[0] > rewards[1] else 1

    info = data.get("info") or {}
    team_names = info.get("TeamNames") or ["", ""]

    # Whole-game reconstruction. Each player's OWN board is fully visible only
    # in that player's own observation view, and a player's view refreshes only
    # on their own turns -- so per-step KO timing is unreliable. Instead we diff
    # the whole game: a body that appears and is gone from the final board was
    # KO'd (retreat/evolve keep the serial+presence, so they are not confused
    # for a KO). We then reconcile to the known prize totals.
    final_board = {0: {}, 1: {}}         # serial -> cardId (last own view)
    last_seen_step = {0: {}, 1: {}}      # serial -> step index last on board
    seen_pokemon = {0: set(), 1: set()}
    ever_serial_card = {0: {}, 1: {}}    # serial -> cardId (last known)
    min_prize = {0: 6, 1: 6}             # lowest prize count observed (public)
    max_turn = 0

    for si, step in enumerate(steps):
        for agent_pos, agent in enumerate(step):
            if not isinstance(agent, dict) or agent_pos > 1:
                continue
            cur = (agent.get("observation") or {}).get("current")
            if not isinstance(cur, dict):
                continue
            max_turn = max(max_turn, int(cur.get("turn") or 0))
            players = cur.get("players")
            if not isinstance(players, list) or len(players) < 2:
                continue
            for pi in (0, 1):
                pc = len(players[pi].get("prize") or [])
                if pc:
                    min_prize[pi] = min(min_prize[pi], pc)
            # agent_pos is the absolute index; players[agent_pos] is own board.
            bs = board_bodies(players[agent_pos])
            if bs:
                final_board[agent_pos] = bs
                for serial, cid in bs.items():
                    ever_serial_card[agent_pos][serial] = cid
                    last_seen_step[agent_pos][serial] = si
                    if is_pokemon(cid):
                        seen_pokemon[agent_pos].add(cid)

    def prize_plan_against(loser: int) -> list[int]:
        """Prize values of loser's Pokemon that the opponent KO'd, in KO order.

        Bodies gone from the loser's final own-board are confirmed KOs. The
        winning KO is often untracked (the game ends before the loser's view
        refreshes), so we reconcile the remainder up to 6 by pulling the
        highest-value bodies still stale-present in the final board.
        """
        gone = [
            (last_seen_step[loser].get(s, 0), ever_serial_card[loser][s])
            for s in ever_serial_card[loser]
            if s not in final_board[loser]
            and is_pokemon(ever_serial_card[loser][s])
        ]
        gone.sort(key=lambda t: t[0])
        observed = [prize_value(c) for _, c in gone]
        remainder = 6 - sum(observed)
        if remainder > 0:
            # Bodies still on the loser's stale final board, richest first --
            # the untracked finishing KO(s).
            standing = sorted(
                (prize_value(c) for c in final_board[loser].values()
                 if is_pokemon(c)),
                reverse=True,
            )
            for v in standing:
                if remainder <= 0:
                    break
                observed.append(v)
                remainder -= v
        return observed

    # For a decisive game the winner takes exactly 6; build both plans but the
    # winner's is the reconciled-to-6 one.
    prize_plans = {0: [], 1: []}
    if winner is not None:
        prize_plans[winner] = prize_plan_against(1 - winner)
        # loser's partial plan: what they managed to KO off the winner (no
        # reconciliation; they took < 6).
        loser = 1 - winner
        gone = [
            (last_seen_step[winner].get(s, 0), ever_serial_card[winner][s])
            for s in ever_serial_card[winner]
            if s not in final_board[winner]
            and is_pokemon(ever_serial_card[winner][s])
        ]
        gone.sort(key=lambda t: t[0])
        prize_plans[loser] = [prize_value(c) for _, c in gone]

    return {
        "episode_id": replay_path.stem.replace("episode_", ""),
        "winner": winner,
        "team_names": team_names,
        "max_turn": max_turn,
        "min_prize": min_prize,
        "prize_plans": prize_plans,        # {0: [...], 1: [...]} (winner sums 6)
        "seen_pokemon": {k: sorted(v) for k, v in seen_pokemon.items()},
    }


def deck_archetype(pokemon_ids: list[int]) -> dict:
    megas = sorted({card_name(c) for c in pokemon_ids if getattr(CARD.get(c), "megaEx", False)})
    exes = sorted({card_name(c) for c in pokemon_ids if getattr(CARD.get(c), "ex", False)
                   and not getattr(CARD.get(c), "megaEx", False)})
    if megas:
        cls = "mega"
    elif exes:
        cls = "ex"
    else:
        cls = "single"
    return {"class": cls, "megas": megas, "exes": exes}


def plan_signature(plan: list) -> str:
    """Multiset of KO prize values as a compact string, e.g. '3+2+1'."""
    vals = []
    lumps = 0
    for x in plan:
        if isinstance(x, tuple):
            lumps += x[1]
        else:
            vals.append(x)
    vals.sort(reverse=True)
    sig = "+".join(str(v) for v in vals)
    if lumps:
        sig = (sig + "+" if sig else "") + f"?{lumps}"
    return sig or "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True,
                    help="label:path (repeatable)")
    ap.add_argument("--our-submission", type=str, default="")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    datasets = []
    for spec in args.dataset:
        label, _, path = spec.partition(":")
        datasets.append((label, Path(path)))

    # Map episode -> our absolute index, from episodes.csv when present.
    def our_index_map(dataset_dir: Path) -> dict:
        m = {}
        epcsv = dataset_dir / "episodes.csv"
        if not args.our_submission or not epcsv.exists():
            return m
        with epcsv.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                eid = row.get("episode_id")
                if row.get("agent_0_submission_id") == args.our_submission:
                    m[eid] = 0
                elif row.get("agent_1_submission_id") == args.our_submission:
                    m[eid] = 1
        return m

    records = []
    for label, path in datasets:
        omap = our_index_map(path)
        n = 0
        for eid, rp in iter_episode_replays(path):
            rec = analyze_episode(rp)
            if rec is None:
                continue
            rec["dataset"] = label
            rec["our_index"] = omap.get(eid)
            records.append(rec)
            n += 1
        print(f"[{label}] parsed {n} episodes from {path}")

    # ---- Aggregations -------------------------------------------------------
    report = {"datasets": {}, "notes": [
        "prize value: mega=3, ex=2, single=1.",
        "A winning plan is the multiset of prize values of opponent bodies the "
        "winner KO'd; it should sum to ~6 (final KO may overshoot).",
        "plan_sum_distribution gauges reconstruction quality; composition "
        "stats below use only cleanly-reconstructed games (sum in 6..8).",
    ]}

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 2) if xs else None

    for label, _ in datasets:
        recs = [r for r in records
                if r["dataset"] == label and r["winner"] is not None]
        plan_sum = Counter()
        body_values = Counter()        # value -> count of KO'd bodies (winners)
        field_arch = Counter()
        # clean games: reconstructed winner plan sums to 6..8
        clean = []
        for r in recs:
            w = r["winner"]
            plan = [x for x in r["prize_plans"][w] if isinstance(x, int)]
            s = sum(plan)
            plan_sum[s] += 1
            for pi in (0, 1):
                field_arch[deck_archetype(r["seen_pokemon"][pi])["class"]] += 1
            if 6 <= s <= 8:
                clean.append((r, plan))
                for v in plan:
                    body_values[v] += 1

        # composition signature distribution (clean only)
        comp = Counter(plan_signature(p) for _, p in clean)
        # KOs-to-win and turns, split by opponent (loser) archetype
        by_opp = defaultdict(lambda: {"kos": [], "turns": [],
                                      "n3": [], "n2": [], "n1": [], "games": 0})
        for r, plan in clean:
            w = r["winner"]
            opp = deck_archetype(r["seen_pokemon"][1 - w])["class"]
            b = by_opp[opp]
            b["games"] += 1
            b["kos"].append(len(plan))
            b["turns"].append(r["max_turn"])
            b["n3"].append(plan.count(3))
            b["n2"].append(plan.count(2))
            b["n1"].append(plan.count(1))
        opp_summary = {}
        for opp, b in by_opp.items():
            opp_summary[opp] = {
                "games": b["games"],
                "avg_kos_to_win": avg(b["kos"]),
                "avg_turns": avg(b["turns"]),
                "avg_mega_kos(3)": avg(b["n3"]),
                "avg_ex_kos(2)": avg(b["n2"]),
                "avg_single_kos(1)": avg(b["n1"]),
            }

        # Efficiency angle: vs mega/ex opponents, does taking the big body
        # (a 3- or 2-prize KO) shorten the game?
        eff = {}
        for opp in ("mega", "ex"):
            big = 3 if opp == "mega" else 2
            took = [(r, p) for r, p in clean
                    if deck_archetype(r["seen_pokemon"][1 - r["winner"]])["class"] == opp
                    and p.count(big) >= 1]
            skip = [(r, p) for r, p in clean
                    if deck_archetype(r["seen_pokemon"][1 - r["winner"]])["class"] == opp
                    and p.count(big) == 0]
            eff[f"vs_{opp}"] = {
                "took_big_body": {
                    "games": len(took),
                    "avg_kos": avg([len(p) for _, p in took]),
                    "avg_turns": avg([r["max_turn"] for r, _ in took]),
                },
                "avoided_big_body": {
                    "games": len(skip),
                    "avg_kos": avg([len(p) for _, p in skip]),
                    "avg_turns": avg([r["max_turn"] for r, _ in skip]),
                },
            }

        report["datasets"][label] = {
            "n_decisive": len(recs),
            "n_clean": len(clean),
            "plan_sum_distribution": dict(sorted(plan_sum.items())),
            "field_archetype": dict(field_arch),
            "winner_ko_body_value_totals": dict(body_values),
            "winner_composition_distribution(clean)": comp.most_common(20),
            "by_opponent_archetype(clean)": opp_summary,
            "efficiency_take_big_body(clean)": eff,
        }

    # (2) Our-submission-specific view.
    ours = [r for r in records
            if r["our_index"] is not None and r["winner"] is not None]
    if ours:
        our_win = [r for r in ours if r["winner"] == r["our_index"]]
        our_loss = [r for r in ours if r["winner"] != r["our_index"]]

        def clean_plans(recs, idx_fn):
            out = []
            for r in recs:
                idx = idx_fn(r)
                plan = [x for x in r["prize_plans"][idx] if isinstance(x, int)]
                if 6 <= sum(plan) <= 8:
                    out.append((r, plan))
            return out

        win_clean = clean_plans(our_win, lambda r: r["our_index"])
        loss_clean = clean_plans(our_loss, lambda r: 1 - r["our_index"])

        our_comp = Counter(plan_signature(p) for _, p in win_clean)
        opp_comp = Counter(plan_signature(p) for _, p in loss_clean)
        beat_arch = Counter(
            deck_archetype(r["seen_pokemon"][1 - r["our_index"]])["class"]
            for r in our_win)
        lost_arch = Counter(
            deck_archetype(r["seen_pokemon"][1 - r["our_index"]])["class"]
            for r in our_loss)
        # our own deck archetype (sanity: should be single-prize)
        our_arch = Counter(
            deck_archetype(r["seen_pokemon"][r["our_index"]])["class"]
            for r in ours)
        report["our_submission"] = {
            "submission_id": args.our_submission,
            "games": len(ours),
            "wins": len(our_win),
            "losses": len(our_loss),
            "our_deck_archetype": dict(our_arch),
            "our_winning_composition(clean)": our_comp.most_common(),
            "our_avg_kos_to_win(clean)": avg([len(p) for _, p in win_clean]),
            "our_avg_turns_to_win(clean)": avg([r["max_turn"] for r, _ in win_clean]),
            "opp_winning_composition_when_we_lose(clean)":
                opp_comp.most_common(),
            "opp_avg_kos_when_we_lose(clean)":
                avg([len(p) for _, p in loss_clean]),
            "opp_archetype_when_we_win": dict(beat_arch),
            "opp_archetype_when_we_lose": dict(lost_arch),
            "winrate_by_opp_archetype": {
                a: {
                    "wins": beat_arch.get(a, 0),
                    "losses": lost_arch.get(a, 0),
                    "winrate": round(beat_arch.get(a, 0) /
                                     max(1, beat_arch.get(a, 0) + lost_arch.get(a, 0)), 2),
                }
                for a in set(beat_arch) | set(lost_arch)
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
