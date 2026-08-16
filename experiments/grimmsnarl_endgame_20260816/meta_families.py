"""Current top-of-board deck meta, by family, from the cached replays."""
import collections
import pathlib
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from ml.core.replay_io import deck_hash, extract_fast_header_from_file  # noqa
from analyze_grimmsnarl_matchup_ceiling import family  # noqa

files = sorted(pathlib.Path("experiments/grimmsnarl_endgame_20260816/deck_probe_cache").glob("episode_*.json"))
fams = collections.Counter()
hash_fam = {}
for f in files:
    header = extract_fast_header_from_file(str(f))
    for seat in (0, 1):
        deck = list((header.get("decks") or [[], []])[seat] or [])
        if len(deck) != 60:
            continue
        fm = family(deck)
        fams[fm] += 1
        hash_fam[deck_hash(deck)] = fm

total = sum(fams.values())
print(f"decks observed among top-35 teams' latest games: {total}")
print(f"distinct 60-card lists: {len(hash_fam)}")
print()
for fm, c in fams.most_common():
    print(f"  {fm:36s} {c:3d}  {c/total:6.1%}")
