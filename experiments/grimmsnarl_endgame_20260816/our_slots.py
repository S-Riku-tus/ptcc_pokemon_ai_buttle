"""One cheap call: what does the board show for our team right now, and what
are our two live slots? Used to decide whether a reroll is safe.

A new submission truncates the OLDEST live slot, and the board shows the max of
the two live ones. So a reroll only risks nothing when the newer live slot is
already at least as high as the older one.
"""
import json
import sys

sys.path.insert(0, "scripts")
from fetch_kaggle_top100_snapshot import (  # noqa: E402
    COMPETITION_ID, fetch_leaderboard, fetch_public_session,
    fetch_team_public_submissions,
)

OUR_TEAM = 16487165

opener, headers, _html = fetch_public_session()
rows = fetch_leaderboard(opener, headers).get("publicLeaderboard") or []
print(f"teams on board: {len(rows)}  (competition {COMPETITION_ID})")
for k in (1, 10, 25, 50, 100, 150, 175, 200, 250, 300, 400, 500, 600, 800):
    if k <= len(rows):
        print(f"  rank {k:4d} = {rows[k - 1]['displayScore']}")

us = [r for r in rows if r.get("teamId") == OUR_TEAM]
if us:
    print(f"\nUS: rank {us[0]['rank']} / {len(rows)}  "
          f"score {us[0]['displayScore']}  sub {us[0]['submissionId']}")

subs = fetch_team_public_submissions(opener, headers, OUR_TEAM)
items = subs.get("submissions") or subs.get("publicSubmissions") or []
print(f"\nlive public submissions: {len(items)}")
for s in items:
    print("  " + json.dumps(
        {k: s.get(k) for k in
         ("id", "publicScoreFormatted", "date", "submittedAt", "status",
          "fileName", "description")},
        ensure_ascii=False))
