# Grimmsnarl ML v23

## Decision

V23 is a phase-routed successor to v22, not a new ranker.  The deck, 2,000
trees, 823 features, fallback policy, feature extractor, and inherited
one-ply arithmetic planner are byte-identical to v22.

The deployed policy has three layers:

1. Own turns 1-2 use same-deck teacher 16494330 (stored code 16), whose v8
   ladder behaviour reached Grimmsnarl and an attack on turn 2 more often.
2. Own turn 3 onward uses v22's 1220.2-rated teacher 16371703 (stored code 0),
   preserving v22's strong turn-3 conversion.
3. A lexicographic goal layer takes a legal route-completing search/evolution
   for the first attacker, and builds a second route through own turn 5 when a
   visible Abra/Kadabra/Alakazam line makes continuity the measured bottleneck.
   The phase ranker's score breaks ties inside the goal-satisfying set.

No generic Boss-rate increase was added: the 11 v22 losses did not expose a
missed game-winning attack, while indiscriminate Boss use would spend the
supporter for the turn.  An energy-allocation continuity proposal was also
removed because it fired zero times over all 44 public v22 games.

## Ladder evidence used

V22 played 44 public games at 33-11 and finished at 1006.9262.  All 11 losses
were concentrated in Alakazam (6) and the mirror (5).  All five mirror losses
lacked a Grimmsnarl ex on board by own turn 2.  The Alakazam losses included
two first attacks delayed to own turn 4/5 and one game that attacked on turns
2-4 before going four consecutive own turns without another attack.

Concrete declined legal routes:

- Episode 92626895: on own turn 2, Grimmsnarl ex was in hand and the deck search
  offered Rare Candy, but v22 selected another card and did not attack until
  own turn 5.
- Episode 92623138: an Impidimp was already in play and deck search offered
  Morgrem, but both the code-0 and code-16 ranker conditions selected another
  Impidimp.  V23 selects the bridge.
- Episode 92641973: the only attacker had 20 HP and an Impidimp backup lacked
  its bridge; v22 searched an optional Froslass piece.  V23 selects Morgrem.

## Teacher-forced footprint

The shared footprint walker was fixed before final measurement: ranker history
now advances once with the stored action, not once with the proposal and again
with the stored action.

On the 44 public v22 games (4,079 single-pick decisions):

| Candidate | Changed | Share | Per game | Games touched |
|---|---:|---:|---:|---:|
| Phase router only | 120 | 2.94% | 2.73 | 42/44 |
| Full v23 | 259 | 6.35% | 5.89 | 43/44 |

The full footprint was concentrated in MAIN (181) and deck search (70).  The
opening goal was exposed on 118 stored decisions in 34 games; Alakazam
continuity was exposed on 46 stored decisions in 10 of the 14 Alakazam games.
These counters are an upper bound: teacher forcing keeps following v22's stored
action, so the same evolution can remain offered repeatedly after v23 would
have taken it once and changed the live board.

Reports:

- `footprint_v22_public.json`
- `footprint_v22_phase_only.json`

This establishes legality, scope, and exact policy exposure.  It does not
establish causal win-rate improvement or a stable 1150 rating; those require a
new ladder run with matchup-stratified sample targets.

## Validation and package

- All tests under `agents/grimmsnarl/grimmsnarl_ml_v23/tests` pass.
- The teacher-forcing regression tests and agent-loader isolation tests pass.
- One pre-existing build-submission test is not runnable because it references
  the absent directory `agents/grimmsnarl/marnies_grimmsnarl_ex_v1`; the actual
  v23 build completed successfully.
- `artifacts/grimmsnarl_ml_v23_submission.tar.gz` contains 19 entries and has
  SHA-256 `939594e39ae289b8e17fde24ca2210c9c7fbfdc5193fbce5629b43df37fc2530`.
- All 12 packaged Python files compile.  The packaged model parses as 2,000
  trees / 823 features, and equals the source model after the submission
  builder's documented CRLF-to-LF normalization.
- A local import from an extracted archive containing the official `cg` bundle
  timed out after 180 seconds on Windows.  The source package was fully loaded
  by four replay workers with zero errors; the timeout is recorded rather than
  relabelled as a pass.
