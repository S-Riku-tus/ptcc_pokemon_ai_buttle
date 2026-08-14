# Grimmsnarl ML v24

V24 is a deliberately narrow challenger: v22 to the byte except for a
public-information veto of a selected Froslass evolution in the mirror. It
does not inherit v23's phase/continuity strategy, does not use `shroud_net`,
and does not change Snorunt setup or Froslass searches.

## Why v22 is the base

The pooled v22 corpus contains 194 games and 17,376 decisions. Teacher-forced
replay reproduces all 17,376 stored v22 actions. Going second, Alakazam,
first-attack timing, attack-access, Munkidori setup, Stamp, Boss, Rare Candy
and Grimmsnarl evolution no longer provide a controlled decision lever. V23
changes 6.30% of all decisions and intervenes most heavily in Alakazam, now
v22's best matchup, so none of that strategy layer is carried into v24.

## Correction to the supplied Froslass result

The source CSV column called `froslass_evolves` is broader than its name. Its
extractor increments for every selected Froslass card action other than an
attack, Ability, end or retreat. Searches and evolution-resolution prompts are
therefore included. The supplied 53-event result is a valid broad
Froslass-line proxy, but it is not 53 literal evolutions.

Re-parsing the same 56 mirrors with `action_type == "evolve"` gives:

| true Froslass evolution | games | record | win rate |
|---|---:|---:|---:|
| none | 39 | 25-14 | 0.641 |
| one or more | 17 | 5-12 | 0.294 |

The corrected controlled effect is -297.9 Elo (p=0.018); Fisher p=0.0218.
This remains a useful challenger direction, but it does not survive a fresh
20-lever Bonferroni correction and does not replicate in all four runs (the
two v22_d evolution games both won). V24 is therefore an experiment, not an
automatic replacement for v22.

The historical 12/12 opponent uptake does not reverse the direction. That was
a 36-game v15 offered-turn imitation measurement. In the current 50 exact-list
mirrors, winners make 0.18 true evolutions per game and losers 0.46; in our
losses we make 0.478 and the winning opponent 0.13.

## Offline binding

`footprint_v22_v24.json` replays all 194 games:

| check | result |
|---|---:|
| v22 action infidelity | 0 / 17,376 |
| v24 changed decisions | 18 |
| games touched | 17 |
| changes in 56 mirrors | 18 |
| changes outside the mirror | 0 |
| changes on v22 wins / losses | 6 / 12 |

All 18 true evolution events are publicly identifiable and all 18 bind. Every
replaced action is a Froslass evolution. Replacements include five Shadow
Bullets and one Filch, and no END action, so the veto is not merely turning
tempo into a pass. V22's ranker score remains the tie-break among alternatives.

## Ladder decision

Run v22 control and v24 challenger simultaneously in the two submission slots
and submit nothing else until both finish. The same-code noise floor is about
77 Elo, so a single sequential 50-game rating is not a promotion test.

Submission artifact:

- `artifacts/grimmsnarl_ml_v24_submission.tar.gz`
- 10,928,225 bytes, 19 entries
- SHA-256 `1b768103f04536b76df6b91cb5c21bb345fdd1de8782081755bd483f3fbf29a3`
- extracted compile and import smoke: PASS
