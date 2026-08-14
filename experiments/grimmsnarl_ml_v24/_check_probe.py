import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open("experiments/grimmsnarl_ml_v24/top60_decks.json", encoding="utf-8"))
rows = d["rows"]
print("probed:", d["probed"], "with deck hash:", sum(1 for r in rows if r.get("deck_hash")))
print("errors:", Counter(
    (r.get("error") or "")[:40] for r in rows if r.get("error")).most_common())
print()
print("deck hash share in top 60:")
for h, n in Counter(r["deck_hash"] for r in rows if r.get("deck_hash")).most_common():
    best = max(r["score"] for r in rows if r.get("deck_hash") == h)
    ranks = [r["rank"] for r in rows if r.get("deck_hash") == h]
    print(f"  {h}  slots={n:>2}  best={best:>7.1f}  ranks={ranks}")
