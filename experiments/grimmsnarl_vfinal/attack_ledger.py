"""Every attack aimed at us, with the board quantities that set its damage.

The current top meta's three losing cells all attack with damage formulas that
read *our* board, not theirs:

* Teal Mask Ogerpon ex - Myriad Leaf Shower is 30 + 30 per Energy on **both**
  Active Pokemon, then doubled by Grimmsnarl ex's Grass Weakness.  Every Energy
  Punk Up puts on our own Active is worth 60 damage to them.
* Mega Froslass ex - Resentful Refrain is 50 per card in **our hand**, for one
  Energy.
* Hydrapple ex - Syrup Storm is 30 + 30 per {G} on all of *their* Pokemon, so
  it is not ours to control, and that is worth knowing too.

This walks every stored replay and writes one row per attack that hit us, so
those formulas can be checked against what actually happened rather than
assumed.  Output: ``attack_ledger.csv``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts", ROOT / "agents" / "grimmsnarl" / "grimmsnarl_ml_v22"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ml.core.replay_io import deck_hash  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import family  # noqa: E402
from build_grimmsnarl_version_games import RUNS, RUN_ROOT, OUR_DECK_HASH, _decks  # noqa: E402

OPT_ATTACK = 13
GRASS = 1
OUT = Path(__file__).resolve().parent / "attack_ledger.csv"

CARDS = {
    int(card["cardId"]): card
    for card in json.loads(
        (ROOT / "vendor" / "cg" / "cards.json").read_text(encoding="utf-8")
    )
}
ATTACKS = {
    int(entry["attackId"]): entry
    for entry in json.loads(
        (ROOT / "vendor" / "cg" / "attacks.json").read_text(encoding="utf-8")
    )
}


def bodies(player: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key in ("active", "bench"):
        for card in player.get(key) or []:
            if isinstance(card, dict):
                out.append(card)
    return out


def energy_count(card: dict[str, Any] | None) -> int:
    if not isinstance(card, dict):
        return 0
    return len(card.get("energies") or [])


def typed_energy(player: dict[str, Any], energy_type: int) -> int:
    return sum(
        sum(1 for e in (card.get("energies") or []) if int(e) == energy_type)
        for card in bodies(player)
    )


def active_of(player: dict[str, Any]) -> dict[str, Any] | None:
    cards = player.get("active") or []
    return cards[0] if cards and isinstance(cards[0], dict) else None


def main() -> int:
    rows: list[dict[str, Any]] = []
    for label, submission, run_dir in RUNS:
        episodes = ROOT / RUN_ROOT / run_dir / "episodes"
        if not episodes.is_dir():
            continue
        for episode_dir in sorted(episodes.iterdir()):
            replay_path = episode_dir / "replay" / "replay.json"
            if not replay_path.exists():
                candidates = list((episode_dir / "replay").glob("*.json"))
                if not candidates:
                    continue
                replay_path = candidates[0]
            try:
                replay = json.loads(replay_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            steps = replay.get("steps") or []
            decks = _decks(steps)
            seat = next(
                (
                    s for s in (0, 1)
                    if decks[s] and deck_hash(decks[s]) == OUR_DECK_HASH
                ),
                None,
            )
            if seat is None:
                continue
            opponent = 1 - seat
            opp_family = family(decks[opponent])
            rewards = replay.get("rewards") or [None, None]
            if rewards[seat] is None:
                continue
            won = int(rewards[seat] > (rewards[opponent] or 0))

            for index, step in enumerate(steps[:-1]):
                if opponent >= len(step):
                    continue
                entry = step[opponent] or {}
                observation = entry.get("observation") or {}
                action = entry.get("action")
                select = observation.get("select") or {}
                options = select.get("option") or []
                if not isinstance(action, list) or len(action) != 1:
                    continue
                choice = action[0]
                if not isinstance(choice, int) or not 0 <= choice < len(options):
                    continue
                option = options[choice]
                if int(option.get("type", -1)) != OPT_ATTACK:
                    continue

                current = observation.get("current") or {}
                players = current.get("players") or []
                if len(players) < 2:
                    continue
                us, them = players[seat], players[opponent]
                our_active = active_of(us)
                their_active = active_of(them)
                if our_active is None or their_active is None:
                    continue

                # HP after: the next observation either side sees.
                after_hp = None
                for later in steps[index + 1: index + 4]:
                    for actor in (0, 1):
                        if actor >= len(later):
                            continue
                        later_current = (
                            (later[actor] or {}).get("observation") or {}
                        ).get("current") or {}
                        later_players = later_current.get("players") or []
                        if len(later_players) < 2:
                            continue
                        later_active = active_of(later_players[seat])
                        if (
                            later_active is not None
                            and later_active.get("serial") == our_active.get("serial")
                        ):
                            after_hp = int(later_active.get("hp", -1))
                            break
                    if after_hp is not None:
                        break

                attack_id = int(option.get("attackId", -1))
                our_id = int(our_active.get("id", -1))
                rows.append({
                    "version": label,
                    "submission": submission,
                    "episode": episode_dir.name,
                    "won": won,
                    "opp_family": opp_family,
                    "opp_deck_hash": deck_hash(decks[opponent]),
                    "turn": int(current.get("turn", -1)),
                    "attack_id": attack_id,
                    "attack_name": (ATTACKS.get(attack_id) or {}).get("name", "?"),
                    "attacker_id": int(their_active.get("id", -1)),
                    "attacker_name": (
                        CARDS.get(int(their_active.get("id", -1))) or {}
                    ).get("name", "?"),
                    "our_active_id": our_id,
                    "our_active_name": (CARDS.get(our_id) or {}).get("name", "?"),
                    "our_active_hp": int(our_active.get("hp", -1)),
                    "our_active_maxhp": int(our_active.get("maxHp", -1)),
                    "our_active_energy": energy_count(our_active),
                    "our_hand": int(us.get("handCount") or 0),
                    "our_bench": len(us.get("bench") or []),
                    "their_active_energy": energy_count(their_active),
                    "their_grass_energy": typed_energy(them, GRASS),
                    "hp_after": after_hp if after_hp is not None else "",
                    "damage": (
                        int(our_active.get("hp", 0)) - after_hp
                        if after_hp is not None else ""
                    ),
                    "ko": (
                        1 if after_hp is None or after_hp <= 0 else 0
                    ),
                })

    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} attack rows to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
