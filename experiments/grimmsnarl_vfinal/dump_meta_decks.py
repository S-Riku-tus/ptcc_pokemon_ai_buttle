"""Dump every distinct 60-card list observed among the current top teams.

The endgame diagnosis counted families; a matchup fix needs the actual lists so
they can be loaded into the local arena and their cards read.  Writes
``meta_decks.json`` keyed by deck hash with the family, the observed count, and
the card list.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ml.core.replay_io import deck_hash, extract_fast_header_from_file  # noqa: E402
from analyze_grimmsnarl_matchup_ceiling import CARDS, archetype, family  # noqa: E402

CACHE = ROOT / "experiments" / "grimmsnarl_endgame_20260816" / "deck_probe_cache"
OUT = pathlib.Path(__file__).resolve().parent / "meta_decks.json"


def main() -> int:
    seen: dict[str, dict] = {}
    for path in sorted(CACHE.glob("episode_*.json")):
        header = extract_fast_header_from_file(str(path))
        for seat in (0, 1):
            deck = list((header.get("decks") or [[], []])[seat] or [])
            if len(deck) != 60:
                continue
            key = deck_hash(deck)
            entry = seen.setdefault(key, {
                "deck_hash": key,
                "family": family(deck),
                "archetype": archetype(deck),
                "count": 0,
                "deck": sorted(deck),
                "episodes": [],
            })
            entry["count"] += 1
            entry["episodes"].append(path.stem)

    ordered = sorted(seen.values(), key=lambda e: (-e["count"], e["family"]))
    OUT.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for entry in ordered:
        by_family[entry["family"]].append(entry)

    for fam, entries in sorted(
        by_family.items(), key=lambda kv: -sum(e["count"] for e in kv[1])
    ):
        total = sum(e["count"] for e in entries)
        print(f"\n=== {fam}  ({total} observations, {len(entries)} lists) ===")
        for entry in entries:
            print(f"  {entry['deck_hash']}  x{entry['count']}  {entry['archetype']}")
        top = entries[0]
        counts = collections.Counter(top["deck"])
        print(f"  -- most common list {top['deck_hash']} --")
        pokemon, other = [], []
        for card_id, n in counts.most_common():
            card = CARDS.get(card_id, {})
            line = f"     {n}x {card_id:>5} {card.get('name', '?')}"
            (pokemon if card.get("cardType") == 0 else other).append(line)
        for line in pokemon:
            print(line)
        for line in other:
            print(line)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
