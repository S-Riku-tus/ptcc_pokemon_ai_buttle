"""Read the cached top-team replays and report the deck meta they show.

Each cached replay carries both seats' 60-card decks, so one file yields two
observations of what the current field plays. Cheap: no network.
"""
import collections
import json
import pathlib
import sys

sys.path.insert(0, ".")
from ml.core.replay_io import deck_hash, extract_fast_header_from_file  # noqa

OURS = "9714ab5c3996f6cc"
GRIMMSNARL_EX = 648

try:
    from scripts.analyze_deck_archetypes import classify  # noqa
except Exception:
    classify = None

files = sorted(pathlib.Path("experiments/grimmsnarl_endgame_20260816/deck_probe_cache").glob("episode_*.json"))
print(f"cached replays: {len(files)}")

hashes = collections.Counter()
grim = 0
total = 0
examples = {}
for f in files:
    header = extract_fast_header_from_file(str(f))
    for seat in (0, 1):
        deck = list((header.get("decks") or [[], []])[seat] or [])
        if len(deck) != 60:
            continue
        total += 1
        h = deck_hash(deck)
        hashes[h] += 1
        examples.setdefault(h, sorted(collections.Counter(deck).items()))
        if GRIMMSNARL_EX in deck:
            grim += 1

print(f"60-card decks observed: {total}")
print(f"contains Grimmsnarl ex (648): {grim} ({grim/max(total,1):.1%})")
print(f"exactly our list {OURS}: {hashes.get(OURS, 0)}")
print()
print("top deck hashes:")
for h, c in hashes.most_common(15):
    tag = "  <== OURS" if h == OURS else ""
    print(f"  {h}  n={c}{tag}")

# Which observed decks contain Grimmsnarl ex at all, and how close to ours.
ours_cards = None
for h, cards in examples.items():
    if h == OURS:
        ours_cards = dict(cards)
print()
if ours_cards is None:
    print("our exact 60 was not observed in this sample")
grim_lists = [(h, dict(cards)) for h, cards in examples.items()
              if GRIMMSNARL_EX in dict(cards)]
print(f"distinct Grimmsnarl-ex lists observed: {len(grim_lists)}")
