"""Does Punk Up feed the Ogerpon counter its own damage?

Teal Mask Ogerpon ex attacks with Myriad Leaf Shower: 30 damage, "30 more damage
for each Energy attached to both Active Pokemon", for GGG. Marnie's Grimmsnarl ex
has Grass weakness, so the multiplier lands on a total that *we* control half of:
Punk Up attaches up to 5 Basic {D} Energy to Marnie's Pokemon "in any way you
like", and Shadow Bullet only costs two.

For every turn we hand back to an Ogerpon deck, this reconstructs the damage
their Active can deal to our Active, and asks the counterfactual that matters:
would that attack still have been lethal if our own Active had been holding at
most --cap energy, with the surplus on the bench instead?

`hp` on an in-play card is remaining HP, so this compares against the real
number, not the printed one. Teal Dance attaches one {G} per turn, so the
lethal-check is also reported with one more Energy on their side.

Usage:
    python experiments/v6_meta_gap_analysis/measure_ogerpon_exposure.py \
        --out ogerpon_exposure.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v5"))

import ml_features as mf  # noqa: E402

from ml.core.replay_io import extract_fast_header_from_file  # noqa: E402

DECK_HASH = "9714ab5c3996f6cc"
CORPUS = ROOT / "data" / "kaggle_grimmsnarl_top50"
OGERPON_DECKS = ("0dede7cb8026e473", "97df7a2a423da1d8", "c377bbb15d6cbbb0")
OGERPON_EX = 96
GRASS = 1  # weakness enum: Basic {G} Energy
MAIN = 0
BASE_DAMAGE = 30
PER_ENERGY = 30


def energy_count(card: dict) -> int:
    return len(card.get("energyCards") or card.get("energies") or [])


def active(player: dict) -> dict:
    cards = player.get("active") or []
    return cards[0] if cards and isinstance(cards[0], dict) else {}


def scan(job: tuple[str, int, int]) -> dict | None:
    path, seat, cap = job
    try:
        head = extract_fast_header_from_file(path)
    except Exception:
        return None
    hashes = head.get("deck_hashes") or ["", ""]
    rewards = head.get("rewards") or [None, None]
    if len(hashes) < 2 or hashes[seat] != DECK_HASH:
        return None
    opponent_deck = hashes[1 - seat]
    if opponent_deck not in OGERPON_DECKS:
        return None
    try:
        won = int(int(rewards[seat]) > int(rewards[1 - seat]))
    except (TypeError, ValueError):
        return None
    try:
        replay = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None

    cards = {int(card["cardId"]): card for card in mf.CARDS} \
        if hasattr(mf, "CARDS") else {}
    per_turn: dict[int, dict] = {}
    for step in replay.get("steps") or []:
        if seat >= len(step):
            continue
        record = step[seat] or {}
        if record.get("status") != "ACTIVE":
            continue
        observation = record.get("observation") or {}
        select = observation.get("select") or {}
        current = observation.get("current") or {}
        if not select or int(select.get("context", -1)) != MAIN:
            continue
        players = current.get("players") or []
        your = int(current.get("yourIndex", seat))
        if len(players) < 2 or your >= len(players):
            continue
        me, opponent = players[your], players[1 - your]
        mine, theirs = active(me), active(opponent)
        if not mine or not theirs:
            continue
        if int(theirs.get("id", -1)) != OGERPON_EX:
            continue
        turn = int(current.get("turn", -1))
        # The last MAIN state of our turn is the board they attack into.
        per_turn[turn] = {
            "our_energy": energy_count(mine),
            "their_energy": energy_count(theirs),
            "our_hp_left": int(mine.get("hp", 0)),
            "our_active_id": int(mine.get("id", -1)),
            "cards": cards,
        }

    if not per_turn:
        return None
    exposed = exposed_capped = exposed_teal = exposed_teal_capped = 0
    our_energy: list[int] = []
    for state in per_turn.values():
        weak = weakness_multiplier(state["our_active_id"])
        for extra, keys in ((0, ("exposed", "capped")),
                            (1, ("teal", "teal_capped"))):
            total = state["our_energy"] + state["their_energy"] + extra
            damage = (BASE_DAMAGE + PER_ENERGY * total) * weak
            capped_total = (
                min(state["our_energy"], cap) + state["their_energy"] + extra
            )
            capped = (BASE_DAMAGE + PER_ENERGY * capped_total) * weak
            lethal = damage >= state["our_hp_left"]
            lethal_capped = capped >= state["our_hp_left"]
            if keys[0] == "exposed":
                exposed += int(lethal)
                exposed_capped += int(lethal_capped)
            else:
                exposed_teal += int(lethal)
                exposed_teal_capped += int(lethal_capped)
        our_energy.append(state["our_energy"])

    turns = len(per_turn)
    return {
        "opponent_deck": opponent_deck,
        "won": won,
        "turns_facing_ogerpon": turns,
        "our_active_energy_mean": round(statistics.fmean(our_energy), 3),
        "our_active_energy_max": max(our_energy),
        "exposed_share": round(exposed / turns, 4),
        "exposed_share_capped": round(exposed_capped / turns, 4),
        "exposed_share_with_teal": round(exposed_teal / turns, 4),
        "exposed_share_with_teal_capped": round(exposed_teal_capped / turns, 4),
        "self_inflicted_turns": exposed_teal - exposed_teal_capped,
    }


_WEAK: dict[int, int] = {}


def weakness_multiplier(card_id: int) -> int:
    if not _WEAK:
        for card in json.loads(
            (ROOT / "vendor" / "cg" / "cards.json").read_text("utf-8")
        ):
            _WEAK[int(card["cardId"])] = int(card.get("weakness") or 0)
    return 2 if _WEAK.get(card_id) == GRASS else 1


def jobs(cap: int) -> list[tuple[str, int, int]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, int, int]] = []
    for row in csv.DictReader(
        open(CORPUS / "indexes" / "replay_index.csv", encoding="utf-8-sig")
    ):
        if row["deck_hash"] != DECK_HASH:
            continue
        if row["agent_0_submission_id"] == row["agent_1_submission_id"]:
            continue
        key = (row["episode_id"], row["seat_index"])
        if key in seen:
            continue
        seen.add(key)
        path = CORPUS / Path(row["replay_path"].replace(chr(92), "/"))
        if path.exists():
            out.append((str(path), int(row["seat_index"]), cap))
    return out


KEYS = (
    "turns_facing_ogerpon", "our_active_energy_mean", "our_active_energy_max",
    "exposed_share", "exposed_share_capped", "exposed_share_with_teal",
    "exposed_share_with_teal_capped", "self_inflicted_turns",
)


def aggregate(rows: list[dict]) -> dict:
    out: dict = {"games": len(rows)}
    for key in KEYS:
        values = [row[key] for row in rows if row.get(key) is not None]
        out[key] = round(statistics.fmean(values), 4) if values else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    work = jobs(args.cap)
    print(f"replays={len(work)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for row in pool.map(scan, work, chunksize=16) if row]
    print(f"games_vs_ogerpon={len(rows)}", flush=True)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["opponent_deck"]].append(row)
    report = {
        "cap": args.cap,
        "attack": "Myriad Leaf Shower = 30 + 30 per Energy on both Actives",
        "note": (
            "Grimmsnarl ex, Morgrem and Impidimp all have Grass weakness, so "
            "the doubled total includes the Dark Energy we attached ourselves."
        ),
        "pooled": {
            "all": aggregate(rows),
            "wins": aggregate([row for row in rows if row["won"]]),
            "losses": aggregate([row for row in rows if not row["won"]]),
        },
    }
    for deck, group in grouped.items():
        report[deck] = {
            "all": aggregate(group),
            "wins": aggregate([row for row in group if row["won"]]),
            "losses": aggregate([row for row in group if not row["won"]]),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{'group':<20}{'split':<8}" + "".join(f"{k[:13]:>15}" for k in KEYS))
    for name, block in report.items():
        if not isinstance(block, dict) or "all" not in block:
            continue
        for split in ("all", "wins", "losses"):
            cells = "".join(
                f"{block[split][key]:>15.4f}"
                if block[split].get(key) is not None else " " * 15
                for key in KEYS
            )
            print(f"{name:<20}{split:<8}{cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
