# grimmsnarl_ml_v17 — finish the route v16 already chose

## Decision

v16 is not an unfinished training run.  It deliberately kept the v15 model
and deck byte-identical because all proposed learned/planner levers were
offer-side on the 110 rated v15 games.  Its two changes are rule-level and were
complete enough to submit, but the stored-board audit exposed one conservative
gap inside its wall route.

v17 keeps v16's model, deck, mirror escalation gate and active BREAK route.  It
changes only `wall_break.py` at runtime:

1. **Complete PRESERVE.** v15 evolved the last Impidimp/Morgrem into another
   Grimmsnarl ex 8 times under a dead wall despite already having a fuelled
   Grimmsnarl ex.  v16 refused 2 of the 8.  The other 6 failed BREAK's
   immediate eight-turn test because Impidimp's current 10 damage was too slow,
   even though keeping it was the only route to a later 60-damage Morgrem.  All
   6 offer a dead Shadow Bullet and END, so v17 closes with the free Shadow and
   preserves the future breaker.
2. **Finish one Punk Up allocation.** There are 83 Punk Up target decisions in
   the 15 wall games.  Only 3 occur after the triggering Grimmsnarl is ready
   while an underfuelled route-viable Morgrem is offered.  v16 selects Morgrem
   on 2.  On the third it splits the last two Energy between Morgrem and
   Impidimp, leaving the Morgrem one short; v17 puts the second Energy on the
   Morgrem.  The Punk Up search count is unchanged.

No retraining was performed.  The 2,000-tree ranker SHA-256 remains
`dabc15894cae4ebf49ab6fa6d91e7af0ad81b2c88751da5ad2cb05a326b93f79`, and
the 60-card deck hash remains `9714ab5c3996f6cc`.

## Candidates measured and rejected

The unresolved v16 notes were checked before adding policy:

| candidate | stored evidence | verdict |
| --- | ---: | --- |
| general Punk Up retarget | 2 correct / 3 actionable wall allocations | one narrow miss only; fixed, no general rewrite |
| mirror Unfair Stamp refusal | 12 live turns offered / 12 played | saturated; opponent's 25 uses are availability-side |
| two-turn Boss route | 1 added prize / 446 Shadow Bullets | rejected in v16 |
| lethal Bench-30 target | 2 misses / 397 shots | rejected in v16 |
| Adrena-Brain uptake | 98.6% of offered turns | rejected in v16 |

The mirror's post-first-Shadow outcome gap therefore remains unexplained by a
decision refusal.  v17 does not invent a mirror override from that correlation.

## Stored-board footprint

`scripts/probe_grimmsnarl_v17_footprint.py` walks all 110 v15 games and applies
the v16 and v17 wall guards to the same teacher-forced decisions.

| scope | decisions | v16 → v17 differences |
| --- | ---: | ---: |
| wall | 1,742 | **7** |
| mirror | 3,367 | **0** |
| Alakazam | 1,496 | **0** |
| Festival | 177 | **0** |
| other | 2,971 | **0** |
| **total** | **9,753** | **7** |

The seven differences are six `Grimmsnarl ex evolve → Shadow Bullet` choices
and one Punk Up target redirection, spread over five wall episodes.  PRESERVE
is now 8/8 rather than v16's 2/8.  Both guards reported zero errors.

This is a scope and invariant proof, not an outcome estimate: after a changed
action resolves, the real future board diverges from the stored replay.

## Verification

- v17 agent suite: **262 passed**.
- Targeted wall tests: **33 passed**, including trigger-first Punk Up ordering,
  strict last-breaker preservation, and stand-down on damageable Actives.
- Python bytecode compilation: passed.
- `scripts/validate_agent.py`: passed; 60 cards, 19 unique card IDs, no warnings.
- Submission archive: 23 entries, extracted deck handshake 60 cards, and all
  six runtime load-error fields null.
- Local arena vs v16, 20 games: 8-12; 0 crashes, 0 illegal selections,
  43.68 ms/v17 move.
- Local arena vs the first-policy Crustle deck, 20 games: 18-2; 0 crashes,
  0 illegal selections, 50.35 ms/v17 move.

The arena results are legality/runtime smoke tests.  The native shuffle cannot
be paired and the first-policy Crustle agent does not construct a competitive
wall, so neither record is a promotion claim.

The repository-wide test command still has 12 pre-existing collection/smoke
failures caused by absent unrelated agents (`alakazam_ml_v2_expanded` and
`kashiwashira_spidops_reconstruction_v1`).  The v17-specific suite is clean.

Submission: `artifacts/submission.tar.gz`, SHA-256
`FC2CA99AA62415868AEF50E9A15468264BCBFCBF657073C31D5FB23E08470A92`.

## Promotion gate and remaining work

Keep v15 as the proven development champion until a closed-loop ladder run.
Promote v17 only with zero runtime faults and positive rating drift over at
least 50 games started from rating 1,000+, while retaining first Shadow ≤ 3.1
and T2 Shadow ≥ 35%.

The next evidence targets are unchanged where the logs genuinely remain thin:

- mirror play after the first Shadow Bullet, using new v17 outcomes rather
  than another offer-side heuristic;
- Festival/Dipplin, currently 0-4 with only four games;
- catastrophic losses to sub-800 opponents as a separate tail KPI.
