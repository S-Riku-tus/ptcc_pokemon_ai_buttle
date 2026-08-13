# Grimmsnarl ML v12

## Decision

v12 is v11.1 with the search layer's **coverage** raised from 16.6% of our
decisions to ~99%, paid for out of a compute budget that was 96.6% idle, with
the acceptance gate tightened in step and two leaf defects fixed that the
field data identifies directly.

The deck, the v9 ranker, the fallback policy, the features and the arithmetic
planner are byte-identical to v11.1. Only `main.py` and `arithmetic_search.py`
change.

## 1. Where the rating actually went

The handover analysis of submission 55353978 is right on its central point and
should be read first: the ~100-point drop is mostly the opponent draw, and the
rating-controlled estimate (-49 Elo, p = 0.246) does not separate from an
identical-code noise floor of ±76. Nothing below argues with that.

Two of its downstream conclusions do not survive being recomputed against the
field, and one of them would have sent v12 in the wrong direction.

### 1a. The mirror is not a weakness

The analysis found v11.1 at 9-10 in the mirror and proposed making search more
conservative there. Pooled over every version we have run on this deck, against
the 3,642 archived field games on the same 60 cards:

The 9-10 was one run of 19 games. Tuning the mirror would have been tuning
noise, and the Boss's Orders shift the analysis flagged (21.2% -> 58.3%,
p = 0.00158) does not correspond to any outcome difference.

### 1b. One matchup is the entire going-second deficit

| | field | ours |
|---|---|---|
| going first | 1166-712 (0.621) | 105-51 (0.673) |
| going second | 980-784 (0.556) | 59-60 (0.496) |
| going second, excluding Alakazam | 866-736 (0.541) | 49-42 (0.538) |

Take Alakazam out and our going-second rate is the field's to within a
thousandth. Alakazam going second is 28 games at 0.357 against 0.704, worth
about **7 wins in 119 going-second games**. It is the only measurable deficit
left on this deck. This confirms
`grimmsnarl-alakazam-is-turn-order` on a larger pool (v8 + v9 + v11a + v11b +
v11.1).

### 1c. But the *mechanism* recorded for it does not survive the larger pool

The recorded mechanism is "the attacker never arrives": 3.6 bodies at turn 4
against the field's 5.21, first attack on turn 8 against 4. Those came from
`alakazam_second_tempo.json`, which is v8's **five** going-second games against
Alakazam. Re-measured over all 28 such games we have (v8, v9, v11a, v11b,
v11.1), the arrival gap is small. In own-turn ordinals, which is what the
following tables use throughout - the source artifact keys by the engine's
shared turn counter, so its "turn 4" is our own turn 2 when going second:

| going second, all opponents | own turn 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| field bodies | 4.41 | 5.13 | 5.33 | 5.38 |
| our bodies | 4.52 | 5.23 | 5.51 | 5.52 |

Against Alakazam specifically the bodies gap is 4.96 against 5.21 at own turn 2
(not 3.6 against 5.21), and our first attack lands on own turn 2.80 against the
field's 2.48 (not turn 8 against turn 4). Across all opponents we are *ahead* of
the field on both. There is no general early-setup tempo defect, so nothing in
v12 is built for one.

What *is* behind, and only against Alakazam, is Grimmsnarl ex uptime:

| Alakazam, going second | own turn 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| field, all | 0.45 | 0.78 | 0.79 | 0.77 | 0.79 | 0.75 |
| field, games it won | 0.49 | 0.83 | 0.88 | 0.85 | 0.89 | 0.91 |
| field, games it lost | 0.35 | 0.65 | 0.58 | 0.57 | 0.36 | 0.40 |
| **ours** | **0.39** | **0.70** | **0.63** | **0.73** | **0.56** | **0.80** |

We track the field's *losing* profile in the one matchup we lose. In the mirror
we are at 0.89 / 0.93 / 0.98 / 0.97 - i.e. exactly the field - so this is not a
general failure to hold an attacker. Alakazam takes one prize per knockout and
our Grimmsnarl ex gives two, so each ex we hand over is a third of the game, and
we hand them over without replacing the body. Our attack rate on own turn 3 is
0.33 against the field's 0.58, and we finish 10.7% of these games having never
attacked at all against the field's 3.1%.

## 2. Why v12 is a coverage change

### The search saw one sixth of the game

Recomputed from v11.1's own 59 ladder games (`.tmp` diagnostic, method in
section 5):

* 394 of our own turns, 41.6 MAIN selects per game;
* **2,372** MAIN selects offered a real choice - **6.02 per own turn**;
* the once-per-turn rule searched **393** of them: **16.6%**.

The one it always took was the first of the turn, which is the widest and least
decidable: 177 of the 393 had ten or more options. Every ordering decision that
follows - the class a full-turn leaf is actually good at, and the class both of
v11's genuine counterfactual wins came from (Petrel before Munkidori, Rare Candy
before Munkidori) - was never searched.

### It was not a cost decision

The competition configuration is `actTimeout: 0`, so every second we spend comes
out of the 600 s per-episode overage bank and nothing else. Across v11.1's 59
games the bank at game end was:

```
min 579.5   p10 583.4   median 588.2   max 594.1
```

v11.1's worst game spent **20.5 s of 600**, or 3.4%.

## 3. What changed

### 3a. Whole-turn coverage under an explicit budget

`SearchBudget` reads `remainingOverageTime` (authoritative on Kaggle) *and*
keeps its own `perf_counter` total (the only meter that exists under the local
`vendor/cg` shim, which does not supply the bank). The tighter of the two
governs, with a documented degradation ladder rather than a cliff:

| headroom | behaviour |
|---|---|
| bank > 150 s reserve and < 240 s spent, headroom >= 60 s | search every MAIN decision, cap 14/turn |
| headroom < 60 s | once per turn - v11.1 behaviour |
| headroom < 1.5x the measured mean search cost | off |

A refilled bank is read as a new episode, so per-game accounting survives the
turn-rewind detector missing an edge.

### 3b. Three determinizations instead of two

Six times the coverage is six times the exposure to a hand-written leaf, so the
gate tightens with it: a candidate must beat v9 in **all three** hidden-state
samples. The third sample is nearly free because samples are now re-screened
after *every* determinization, not only the first - a candidate that has already
failed one can never satisfy the consensus, so it is not simulated again.

### 3c. `exposed_prizes`: what the opponent collects on their reply

v11's leaf priced end-of-turn survival as `active_survival_margin` at x10, so
the difference between ending on a body that lives and one that dies was worth
less than one point of setup progress. v12 adds the prize itself:

```
exposed_prizes = prize_value(our final active) if it dies to the opponent's
                 best available attack next turn, else 0
```

with two rules in `_grade_upgrade`:

* a candidate may **not** increase it unless it attacks or deals more damage -
  a hard refusal, not a price;
* decreasing it counts as a **major**, alongside landing an attack or readying
  a Grimmsnarl ex.

This is the term the Alakazam-going-second table asks for, and it is
matchup-agnostic: no deck hash, no opponent identity, just prize arithmetic.
It also makes a Munkidori Adrena-Brain that lifts our active out of knockout
range visible to the search for the first time.

### 3d. Board width past three bodies

v11's leaf capped body value at `min(bodies, 3)`, making a 3-body and a 6-body
end of turn identical. The field's own win rate by bodies at the end of own
turn 3, on this exact 60:

| bodies | going second | going first |
|---|---|---|
| 3 | 0.206 (n=68) | 0.403 (n=62) |
| 4 | 0.413 (n=225) | 0.524 (n=246) |
| 5 | 0.538 (n=403) | 0.607 (n=519) |
| 6 | 0.631 (n=1024) | 0.665 (n=1018) |

The gradient runs to a full bench, so the cap is raised to 5 and a body gain
becomes a *medium* signal. Deliberately a medium and not a major: this evidence
is a correlation across whole games, so it earns the right to break a tie, not
to force one. An extra body still has to pay for itself out of the hand.

`grimmsnarl-roadmap-to-1100` records the opposite-signed fact - across pilots,
`bench>=3 at turn end` correlates **-0.564** with rating. The two are not in
conflict and the tension is deliberate: that one compares different pilots over
whole games (better pilots trade and close faster), while this compares two
lines from the *same* root at the *same* point in the same turn, which is the
only comparison the leaf ever makes.

### 3e. Removed: the Alakazam-going-second special case

v11.1 gave that matchup a second search per turn. Under full coverage every
matchup gets every search, so a matchup-keyed branch could only ever be a
constraint. Removed rather than weakened.

## 4. Verification

### Static

* 188 v12 tests pass, covering every inherited v3-v11 invariant plus new tests
  for coverage, the per-turn cap, budget exhaustion, the degradation ladder,
  bank-refill episode detection, missing-bank fallback, the re-screening rule,
  `exposed_prizes` in both directions, and the body-width terms.
* `validate_agent.py --agent grimmsnarl_ml_v12` passed: 60 cards, 19 unique, no
  warnings.
* SHA-256 equality with v11.1 verified for `deck.csv`, `fallback_policy.py`,
  `ml_features.py`, `ml_planner.py`, `ml_runtime.py`, `policy_base.py` and the
  33 MB ranker.
* `tests/` shows the same 2 failures and 10 setup errors as the v11 run, all
  pre-existing missing-fixture problems for `alakazam_ml_v2_expanded` and the
  Spidops package. None imports v12.
  Ranker SHA-256 is `b0397a4a0270e2c8d3bb5088c759a173307f4446c8577c68895214493b91836c`,
  the same file v9 and v11.1 shipped.
* Submission archive: 19 entries, 8,543,227 bytes, SHA-256
  `03e08ecd93ee72ed7e51eb75bf7b213ddbf1f63dd998f902a2cfae5334607771`.

### Counterfactual over v11.1's own 59 ladder games

`ladder_counterfactual_v11_vs_v12.json`, via
`scripts/probe_grimmsnarl_v12_coverage.py`. Both agents are teacher-forced on
the identical stored action at every one of the 5,653 decisions, multi-picks
included, so the only difference between them is the search layer.

**Coverage held on real ladder states:**

| | v11.1 | v12 |
|---|---|---|
| MAIN decisions offered a choice | 2,372 | 2,372 |
| searched | 393 (16.6%) | **2,368 (99.83%)** |
| own turns reached | 393 | 393 |
| skipped: no second candidate | - | 4 |
| skipped: budget, per-turn cap, planner guard | - | **0 / 0 / 0** |
| branch errors, incomplete branches | - | **0 / 0** |
| search seconds only | not instrumented in v11 | **2,978** (50.5 s/game, 2.20 s/search) |
| search overrides | - | **344** (14.5% of searches) |

(The probe records only the candidate's snapshot, so the v11.1 column is its
coverage arithmetic - one search per own turn, 393 own turns - rather than a
counter read back from this run. The 50.5 s/game here is the search layer alone.)

**Fidelity.** v11.1 - the deployed artifact - reproduces **95.52%** of its own
logged actions here, matching the 95.63% in the handover analysis; the remaining
4.5% is this harness's ceiling for every agent, not something v12 introduced.
More importantly, on the 314 decisions where the two policies disagree, the
action actually played matches **v11.1 at least 241 times and v12 at most 73**.
That is the correct sign and it settles the worry recorded in
`grimmsnarl-offline-probe-skips-multipick`: on this run the deployed policy does
dominate its own counterfactual. The earlier inversion (v8 49/70 against v11
13/70) is most likely an artifact mismatch - the v11a/v11b submissions shipped
v8's ranker while the repository's `grimmsnarl_ml_v11` now carries v9's - rather
than a harness defect, though that was not re-run to confirm.

**What changed.** 314 divergences, 5.32 a game, 5.55% of all decisions; 312 of
them in MAIN. The behavioural direction is one-sided and is the whole story of
this diff:

| card the layer moves toward | v11.1 would play | v12 plays |
|---|---|---|
| Marnie's Grimmsnarl ex | 16 | **83** |
| Munkidori | **40** | 10 |
| Marnie's Morgrem | 17 | 2 |
| Basic {D} Energy | 23 | 22 |

v12 systematically buys Grimmsnarl ex board presence and sells Munkidori
tempo. That is the exact statistic that separates the field's wins from its
losses in the one matchup we lose (section 1c), and it comes out of the leaf's
`ready_grimms` / `active_ready` majors now being applied to the whole turn
instead of its first decision. It is also the *opposite* of the direction the
handover analysis suspected v11 of taking. Divergences peak on own turns 5-8,
i.e. the mid-game rebuild, not the opening.

**Two caveats, stated rather than buried:**

* This probe **cannot estimate strength.** The 59 games were played by v11.1;
  whether v12 would have won them is unknown. The "games with an override went
  37-18" split in the JSON has a 4-game control arm and is reported only for
  completeness - it is not evidence and is not part of the gate. (The v11
  release read a split of this shape as reassurance; that was a mistake.)
* This probe **cannot test the budget governor.** It replays v11.1's stored
  `remainingOverageTime` values, which never fell below 588 s, so the throttle
  and the degradation ladder were never engaged (`budget_stops = 0`,
  `budget_degraded = 0`). Those paths are covered by unit tests and by the
  internal wall-clock meter, not here.

**One unexplained residual.** 2 of the 5,653 decisions diverge in a *non-MAIN*
context (16), which the search layer never touches - both are a Grimmsnarl
ex / Munkidori target choice. 0.035% of decisions, and the architecture is
v11's, so it is not introduced by v12, but it means some module-level state
survives a search that should not. It is the first thing to chase in the next
iteration's measurement work.

## 5. Method

Every number above is recomputed from stored replays, not taken from a summary.
Each is reproducible from a script in `scripts/`:

`probe_grimmsnarl_v12_coverage.py` supersedes
`probe_grimmsnarl_v11_ladder_overrides.py`: `--base` / `--candidate` are
arbitrary agent directories and every output key is named after the role, so the
next iteration re-points it instead of copying and half-renaming it.

* coverage, budget and option-count distributions: walk our own seat through
  `data/runs/grimmsnarl/20260809_grimmsnarl_ml_v11_sub55353978`, count MAIN
  (`select.context == 0`) selects with more than one option per own turn, and
  read `remainingOverageTime` from the observation;
* matchup and tempo tables: 3,642 field games from
  `data/kaggle_grimmsnarl_top50` filtered to our deck hash `9714ab5c3996f6cc`,
  plus 275 of our own across v8, v9, v11a, v11b and v11.1, with board state
  keyed by **own-turn ordinal** rather than the engine's shared turn counter;
* turn order is read from a late step because `current.firstPlayer` is -1 until
  the flip (`replay-firstplayer-sentinel`);
* non-public validation episodes are excluded from every count.

## 6. Deliberately not done

* **No mirror tuning.** Section 1a: it is not a weakness.
* **No Alakazam-keyed rule.** The deficit is real and large, but a deck-hash
  override is the wrong shape; `exposed_prizes` addresses the same prize
  arithmetic without naming a matchup.
* **No Ogerpon work.** 0.195 for the field over 128 going-second games. It is a
  structural counter (`grimmsnarl-ogerpon-structural-counter`).
* **No ranker retrain.** `grimmsnarl-imitation-saturated`; the model is
  byte-identical to v9's on purpose so that this run measures the search layer
  and nothing else.
* **No claim about rating.** See the gate below.

## 7. Promotion gate

The v11 release shipped on a gate its run could not evaluate - it asked for
60-80 games against 1000+ opponents and got one. This gate is written to be
answerable from whatever the ladder actually deals.

Read in this order:

1. **Legality and containment.** Zero illegal selections, zero branch errors,
   zero timeouts, and `overage_remaining_min` never below 150 s. Any failure
   here is an immediate rollback regardless of rating.
2. **Coverage held.** `searched / considered` above 0.9 on the ladder. If the
   budget governor throttled, the run measures a different agent than the one
   tested here and the rest of the gate is void.
3. **Alakazam going second.** The pre-registered target: it is 5-16 across v8,
   v9 and v11.1 with the field at 0.704. Any run with fewer than 8 such games
   cannot move this number and should not be read as if it had.
4. **Override-joined outcome.** Games with at least one search override against
   games without, opponent-rating controlled.

Do **not** promote or roll back on the final rating. Four runs of this deck have
now peaked early and decayed, the same-code noise floor is ±76 Elo, and rating
does not track win rate on this deck at all (`grimmsnarl-roadmap-to-1100`:
spearman -0.138, p = 0.55). Champion stays v9 until points 1-3 are satisfied.
