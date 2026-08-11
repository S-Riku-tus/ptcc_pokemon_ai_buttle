# grimmsnarl_ml_v18 — post-Shadow mirror audit and safe state reset

## Decision

v18 is built and ready to submit.  The additional mirror evidence does **not**
support a broad late-game target override.  It supports one narrow arithmetic
invariant, and the implementation audit found one independent state-leak bug:

1. `mirror_prize.py` guarantees that, after our first Shadow Bullet in a
   publicly identified mirror, damage placed now never passes an immediate
   Benched knockout worth more Prize cards.  It does not rank non-lethal
   targets.
2. `policy_router.py` resets its sticky matchup route when the turn counter
   moves backwards into a new episode.  v16/v17 did not do this, so a process
   reused after a mirror could incorrectly keep the mirror-only Froslass gate
   active in a later non-mirror game.

The deck, 2,000-tree ranker, feature code and every v17 wall decision are byte
identical.  No retraining was performed.

## Additional-log corpus

The refreshed evidence contains 152 exact-60 mirrors:

| source | public rating | exact mirrors | record |
| --- | ---: | ---: | ---: |
| deployed v15, submissions 55404196 + 55409394 | mixed | 35 | 20-15 |
| Raihan, submission 55177269 | 1151.0 | 31 | 18-13 |
| kd, submission 55187358 | 1116.3 | 49 | 30-19 |
| Sixth Sense, submission 55138264 | 1113.7 | 37 | 26-11 |
| **high-rated references** | **1114-1151** | **117** | **74-43** |

The two v15 submissions now contain 117 rated games in total, 70-47.  The
exact mirror cut is intentionally stricter than the older label-based 40-game
cut: both seats must reveal the identical 60 cards.

## What the post-first-Shadow audit found

### Rejected broad changes

- Adrena-Brain uptake is saturated.  In deployed exact-mirror losses it was
  used whenever offered; lower use is availability/state-side.
- A general "hit Benched Grimmsnarl" rule is wrong.  Stored examples show a
  40-HP Munkidori taking the Bench 30 and then being knocked out by Freezing
  Shroud at checkup.  Moving that 30 to a non-lethal Grimmsnarl delays a Prize.
- Adrena-Brain source choice already conditions on survival: in deployed
  losses it normally removes counters from a Benched Munkidori when the Active
  Grimmsnarl cannot be saved, and from the Active when the heal crosses the
  opponent's next-hit threshold.
- The 181-210 HP Grimmsnarl "set up the next 180" pattern is not deterministic:
  high-rated pilots took 5 of 15 clean offers, not enough to encode as an
  invariant.

### Accepted invariant

- Immediate Shadow Bench knockout: deployed 25/25; high-rated pilots 103/103.
- Immediate two-Prize Adrena-Brain finish while a one-Prize target also exists:
  high-rated pilots 5/5.
- Replaying those five exact observations through v17: v17 also chose the
  two-Prize Grimmsnarl 5/5.

The new layer is therefore defensive: it seals a result that is already
saturated in observed data and cannot overwrite the non-lethal plans where
the correct choice depends on Freezing Shroud, future Adrena-Brain count and
remaining attacks.

## Stored-action footprint

`scripts/probe_grimmsnarl_v18_footprint.py` replays actual actions through the
guard.

| scope | games | single-pick decisions | post-Shadow target prompts | immediate-KO prompts | changes | errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deployed | 35 | 3,278 | 342 | 25 | 0 | 0 |
| high-rated references | 117 | 11,841 | 1,171 | 102 | 0 | 0 |
| **total** | **152** | **15,119** | **1,513** | **127** | **0** | **0** |

Some of the 103 reference Shadow knockouts are single-option prompts or occur
outside v18's post-first-Shadow gate, hence the guard footprint is 102 total
immediate-KO prompts after combining Shadow and Adrena contexts.  This is a
scope difference, not a miss.

## Verification

- v18 agent suite: **271/271 passed**.
- Analysis and runtime modules: Python bytecode compilation passed.
- Static agent validation: passed; 60 cards, 19 unique card IDs, no warnings.
- Model SHA-256: `dabc15894cae4ebf49ab6fa6d91e7af0ad81b2c88751da5ad2cb05a326b93f79`
  (byte-identical to v17).
- Local arena v18 vs v17, 20 games: 9-11; **0 crashes, 0 illegal selections**,
  45.76 ms/v18 move.  This is a process-reuse/runtime smoke test, not a rating
  estimate.
- Submission archive: 24 entries; SHA-256
  `FA43A3922039A191B1494139C881045E25B91DD30993973768B3F3AEB9E53804`.

## Live ladder gate

The requested live ladder run has **not started**.  This workspace has no
Kaggle CLI, `KAGGLE_API_TOKEN`, `~/.kaggle/kaggle.json`, or Kaggle access
token.  Public EpisodeService access can download and analyse a submission
after it exists, but cannot create an authenticated submission.

Upload `artifacts/submission.tar.gz` as a new simulation submission beginning
near rating 1000, then provide its submission ID.  The closed-loop gate is:

- at least 50 completed public games;
- zero runtime/illegal-selection failures;
- rating drift measured from the first game's initial rating to the last
  game's updated rating, not peak rating;
- first Shadow Bullet own turn <= 3.1 and Shadow by own turn 2 >= 35%;
- mirror rating drift >= 0 and Froslass uptake when offered >= 80%;
- separately report results against opponents rated 1000+.

Until this gate is observed, v15 remains the proven champion and v18 remains a
submission-ready challenger.
