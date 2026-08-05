# Grimmsnarl ML v4 — the per-turn denominator, and two of the analysis's five priorities refuted

Date: 2026-08-04
Parent: `grimmsnarl_ml_v3` (ladder 950.8 / 905.2 over 129 games, 61.2% win rate)
Sibling: `grimmsnarl_ml_v2` (ladder 967.4 over 59 games, 61.0% win rate)

**Headline, stated before the detail — and it is not the one this report was
drafted with.** The single most valuable finding of v4 is that **every version in
this line has been shipping an under-trained model.** v3 early-stopped at 678
trees on patience 200; the same feature set with patience 700 runs to 3,395 trees
and scores 0.8503 on the held-out block against 0.8455. That 0.48 points is free
and it is available to v2 and v3 as well.

Which means the honest reading of v4's own numbers is a negative one. v4's 28 new
columns reach test Top-1 **0.8503 — identical to v3's features trained properly,
to four decimal places.** The features buy no accuracy. What they buy is
*compression*: v4 reaches that ceiling in 1,238 trees where v3's set needs 3,395.

So v4 delivers: two genuine correctness fixes in the Freezing Shroud ledger, one
provable planner rule whose measured firing rate is zero, the same accuracy
ceiling at a third of the trees, and parity in paired self-play (51.7% against v3,
55.0% against v2, both intervals straddling 50%) — plus three measurement findings
that are worth more than any of it.

**Recommendation: promote v4 over v3, keep v2 as champion until a ladder run says
otherwise.** v4 dominates v3 on every axis measured (same fidelity ceiling but
reproducibly, two bug fixes, 51.7% head to head) and is not distinguishable from
v2. The 0.48 points v3 left on the table by under-training is the one change here
that is certain to be real.

Two of the analysis's five priorities are **refuted by the teacher data** and are
deliberately not implemented; one more is already at teacher parity. Each would
have been a change in the wrong direction.

## 0. The measurement error that produced three of the five priorities

`scripts/analyze_grimmsnarl_v4_gaps.py` re-measures every MAIN behaviour **per own
turn** instead of per MAIN decision. MAIN is re-asked after every intermediate
action, and attacking or passing closes the turn, so a per-decision denominator
counts how many actions a turn contains rather than whether the player took the
action. The two readings are not close:

| shape | per MAIN decision | per own turn |
|---|---:|---:|
| teachers attack into a damage-immune Active | 18.9% | **88.6%** |
| teachers make the dark-energy attachment | 23.6% | **81.9%** |

The per-decision figure is just 1 / (actions per turn). Both tables look
plausible; only one is a rate. Every number below is per own turn, and the
instrument asserts its own turn count against the replay (6.27 own turns a game
against ~11.7 total) after a bare `except` was found silently eating a
`TypeError` on 13,514 of ~23,000 turns.

## 1. What the analysis got right, wrong, and already-solved

| # | analysis priority | per-turn teacher rate | ours | verdict |
|---|---|---:|---:|---|
| 1 | hard-ban the 0-effect Shadow Bullet | 88.6% field, **91.1% elite** | 94.1% | **refuted** |
| 5 | Adrena-Brain target selection | 97.1% take when offered | **97.6%** | already parity |
| 4 | Froslass by multi-turn ledger | 73.6% mirror field, 80.8% pin | **100%** | real, ledger fixed |
| 3 | Boss: whether, not just where | 63.0% on the wall shape | **20%** | real, rule added |
| 2 | 900-mirror / going second | — | — | see §4 |

**Priority 1 is the important one to get wrong.** The claim was that Shadow Bullet
into Crustle for literally zero damage is waste. Per turn, on the 666 teacher
turns where the last available swing was worthless, the field takes it 88.6% of
the time and the elite pair 91.1% — *more* than average. Attacking is free once
the turn is otherwise spent: it costs no energy and exposes nothing. A veto here
would have moved us away from every teacher in the corpus on a 666-turn
denominator. Not shipped.

**Priority 5 was already solved.** The Adrena-Brain deficit (4.77 uses a game
against 6.07) is not a targeting error — we take the ability on 97.6% of the turns
it is offered against the field's 97.1%. It decomposes entirely as availability:
1.46 Munkidori in play a turn against 1.55, of which 0.90 are fuelled against
1.07. And Punk Up attaches only to "your Marnie's Pokemon", so **a Munkidori is
fuelled by the once-per-turn hand attachment or never**. The mechanism the
analysis intuited is right; the decision to fix is upstream of the one it named.

## 2. The defect the measurement found instead

The single largest decision gap is not against the field, it is against **the
pilot we are already pinned to**, and v3's pooled 84.6% Top-1 hides it because it
is a small share of decisions.

| per-turn take rate | field | elite pair | pinned 16494330 | v3 |
|---|---:|---:|---:|---:|
| dark energy attached | 81.9% | 86.3% | **83.5%** | **75.1%** |
| Grimmsnarl ex evolve | 70.3% | 72.5% | 67.4% | 60.4% |
| Boss's Orders played | 38.0% | 33.2% | 33.3% | 29.1% |
| Froslass evolve, mirror only | 73.6% | 71.8% | 80.8% | **100%** |
| Adrena-Brain uses / game | 6.07 | 5.83 | 5.80 | 4.77 |

75.1% is **below all 21 teachers** — the lowest pilot is 74.5%. With comparable
Dark Energy in hand per turn (0.72 against 0.79), so it is a preference, not a
draw. All three of the first rows point the same way: v4's diagnosis is that the
agent systematically under-takes proactive board-building and over-takes
turn-closing actions.

One shape inside this is not a preference but arithmetic, and it is where the
`enabling` row of §5 comes from: an attachment onto a dry Munkidori switches
Adrena-Brain on for the rest of the game. Measured separately, that shape is
**already at parity** — 97.4% for v3 against 98.0% for the field — so it has no
headroom and a planner rule on it would fire on 0.10 turns a game and buy
nothing. That is why v4 attacks the general rate with features rather than a
dominance rule.

## 3. What v4 changes

**28 feature columns (794 → 822).**

* *The cost of closing the turn.* v3 had a column for every reason to attach
  (`energy_enables_munkidori` is the 7th-highest-gain feature in the whole
  model) and none for the cost of not attaching. The argmax is a comparison
  between candidates, so for the attachment to win more often, END and attack
  have to score less: `ends_turn_wasting_attachment`,
  `ends_turn_leaving_munkidori_dry`, `ends_turn_wasted_enabling_count`.
* *Battle Cage in the shroud ledger — a real bug.* Battle Cage prevents damage
  counters on Benched Pokemon "by effects of attacks and Abilities from **the
  opponent's** Pokemon". Freezing Shroud is an Ability, so under Battle Cage our
  Froslass is stopped on *their* Bench but not on our own, and not on either
  Active. Battle Cage therefore makes Freezing Shroud strictly *worse* for us,
  and v3's stadium-blind count said the opposite. `shroud_side` is now
  side-aware; a test asserts the same board flips from `shroud_net = +1` to `-1`.
* *The shroud priced more than one checkup deep.* v3 asked only "does something
  die at the very next checkup", which is why its Froslass guard never fired: at
  10 HP the answer is yes, at 20 it is no, and the whole exchange lives in
  between. `shroud_checkups_to_kill` plus prize-weighted and 3-checkup net
  columns replace it.

**One new planner rule, on the same dominance-only terms.** `_wall_unlock`
refuses to *close* the turn — by the swing or by passing — while the Active is
damage-immune, the swing takes no prize anywhere, a gust is legal, and a body
Shadow Bullet damages sits on their Bench. Both closing moves leave the board
unchanged; Boss makes the same swing worth ≥180, and the supporter slot was
about to go unused. It is the correct form of priority 1: it converts the dead
turn instead of banning the swing.

Probing v3 over its own 65 games is what set the trigger. Keyed on the attack
alone the rule fires *never* — on all 5 turns with this shape v3 never picked the
swing at a decision where the gust was also offered, which is exactly how v3's
own Boss rule ended at 0 firings. Including END is what makes it reachable.

## 4. Offline result: an early-stopping artifact, not a feature gain

Same corpus, same hyperparameters, same never-touched per-team chronological
block as v3.

| block | n | v2.1 | v3 | **v4** |
|---|---:|---:|---:|---:|
| per-team chronological test, Top-1 | 34,611 | 0.8484 | 0.8455 | **0.8503** |
| validation Top-1 | 35,061 | — | 0.8471 | **0.8504** |
| trees at early stop | | — | 678 | 1,238 |
| MAIN (ctx 0) | 17,883 | — | 0.7955 | **0.8015** |
| counter placement (ctx 13) | 2,538 | — | 0.8440 | **0.8534** |
| genuine divergence | | — | 2.44% | **2.34%** |
| same-turn ordering | | — | 9.30% | **9.03%** |

0.8503 is above v3's test Wilson upper bound (0.8493) and 19 of 21 pilots
improve, including the elite 16371703 by +1.31. Read alone, that table says the
28 columns worked.

**The control says otherwise, and it is the most important number in this
report.** Retraining *v3's own feature set* with the early-stopping patience
raised from 200 to 700:

| run | trees | validation Top-1 | test Top-1 | test Wilson 95% |
|---|---:|---:|---:|---|
| v3 features, patience 200 — **shipped v3** | 678 | 0.8471 | 0.8455 | 0.8417–0.8493 |
| v3 features, patience 700 — **control** | 3,395 | 0.8504 | **0.8503** | 0.8465–0.8540 |
| v4 features, patience 200 — **shipped v4** | 1,238 | 0.8504 | **0.8503** | 0.8465–0.8541 |

Identical to four decimal places. **The 28 new columns buy no accuracy at all**;
the entire +0.48 is v3 having stopped 2,717 trees early. The validation curve on
this corpus is flat and noisy above ~600 trees — v3 saw 0.8471 at 678, 0.8459 at
800, and would have needed to survive to 3,395 to find 0.8504 — so patience 200
is simply too small a window, and this has been true of v2 and v1 as well.

What the new columns *do* buy is compression: the same ceiling in 1,238 trees
instead of 3,395, a 2.7× smaller booster for equal accuracy. They take 1.94% of
total model gain, led by the Munkidori-fuel cluster (`munkidori_dry_count` rank
70 of 823, `grimmsnarl_one_energy_short` 94, `energy_in_hand` 106).

This is the **third** consecutive attempt to raise imitation fidelity on this
archetype by adding informative columns (v3's 63, v4's 28) and the third to
return nothing. Combined with v3's finding that conditioning on a stronger pilot
lowers fidelity without fixing the habit, the case that this objective is
saturated is now as strong as offline evidence gets.

One thing the new columns *are* good for, and it is the only defensible reason to
prefer v4's feature set: **v4 finds the ceiling at the default patience.**
Retrained with patience 700, v4 stops at the same 1,238 trees and the same 0.8503
— it was already converged. v3's set needs patience 700 to get there at all.

| run | patience | trees | validation | test | test Wilson 95% |
|---|---:|---:|---:|---:|---|
| v3 features | 200 | 678 | 0.8471 | 0.8455 | 0.8417–0.8493 |
| v3 features | 700 | 3,395 | 0.8504 | 0.8503 | 0.8465–0.8540 |
| v4 features | 200 | 1,238 | 0.8504 | 0.8503 | 0.8465–0.8541 |
| v4 features | 700 | 1,238 | 0.8504 | 0.8503 | 0.8465–0.8541 |

So the ceiling on this corpus and objective is 0.8503 and both feature sets reach
it; v4 reaches it reproducibly and in a third of the trees, and v3 only reaches it
if you already know about the patience defect.

## 5. And the honest negative: the targeted behaviour did not move

Both agents teacher-forced through the pinned pilot's 18 held-out games — the one
block where the *target* is known, because `PILOT` is what that pilot actually did
on these exact boards.

| per-turn take rate | offers | PILOT | v3 | v4 | closer |
|---|---:|---:|---:|---:|:--|
| dark energy attached | 75 | 0.8533 | 0.7067 | 0.7200 | v4 |
| ...enabling Adrena-Brain / an attack | 54 | 0.9815 | 0.8889 | 0.8704 | v3 |
| Grimmsnarl ex evolve | 33 | 0.7273 | 0.6970 | **0.7273** | v4 |
| Froslass evolve | 19 | 1.0000 | 0.8947 | 0.8947 | tie |
| Boss's Orders played | 33 | 0.2424 | 0.3030 | **0.2727** | v4 |
| all-context agreement | 1,706 | — | 0.9156 | 0.9144 | tie |

v4 is closer on three rows, v3 on one, tied on one — and **every one of those
differences is one or two decisions.** The attachment gap against the pilot is
14.7 points for v3 and 13.3 for v4: unchanged in any sense that matters. The
END-side cost columns are used by the trees (ranks 252–459) and do not move the
argmax.

The same table computed on our own 129 ladder boards shows the rates moving
slightly *down* — but that table has no known target, because nobody knows the
right attachment rate for a board v3 created, and it should not be read as
evidence either way. Its one honest use is as a self-consistency check: the
`played_*` column reproduces `analyze_grimmsnarl_v4_gaps.py` exactly on all six
shapes, which is what validates both instruments.

`_wall_unlock` has fired **0 times** in 147 games of stored evidence (129 ladder,
18 held-out): the 5 candidate ladder turns need the END trigger *and* a fuelled
Grimmsnarl Active, and on those 5 turns v3 was not attacking. It is a guardrail
whose firing rate is measured at zero, not a demonstrated fix.

## 5b. Outcome evidence: parity

Paired local self-play on the cg engine, alternating seats, the mirror.

| pairing | games | v4 wins | v4 rate | Wilson 95% | first-seat | crashes | illegal |
|---|---:|---:|---:|---|---|---:|---:|
| v4 vs v3 | 120 | 62 | 51.7% | 42.8–60.4% | 32–30 | 0 | 0 |
| v4 vs v2 | 120 | 66 | 55.0% | 46.1–63.4% | 36–30 | 0 | 0 |

Both intervals straddle 50%: **parity**, which is the same answer v3 got against
v2 over 299 games (49.5%). 0 draws, 0 crashes and 0 illegal selects in 47,787
moves at 24–39 ms/move.

These are one 120-game run each and must be read as such. The v3 report's lesson
applies directly: its first 60-game run read 56.7% and the pooled 299 read 49.5%,
because `local_arena --seed` seeds Python's RNG and not the cg engine's shuffles,
so repeated runs are independent samples of one number rather than seeds. 55.0%
against v2 is therefore not a claim that v4 beats v2; it is enough to rule out a
large regression against the highest-rated agent in the line.

## 6. Reading this against v2's higher rating

v2 sits at 967.4 and v3 at 950.8 / 905.2, and the two v3 runs are the same code.
Win rates are 61.0% (n=59) and 61.2% (n=129) — Wilson intervals 48.3–72.4% and
52.6–69.2%. The rating gap is inside the known noise of this ladder (an identical
agent has scored 842.8 and 804), and the per-turn behaviour table shows v2 and v3
within a point of each other on every shape except Boss.

Offline the shipped ordering was **v2.1 0.8484 > v3 0.8455**, which now reads as
v3 having been trained less carefully rather than featured worse: at equal patience
both v3 and v4 sit at 0.8503. The user's instinct that v2 looked stronger than v3
was pointing at something real, and this is the mechanism — not the 63 v3 columns,
but 2,717 missing trees.

## 7. What v4 does not settle

0. **Whether v4 is stronger.** 240 paired self-play games say parity. A bucketed
   ladder run is the only remaining evidence, and the promotion gate from v3 §6
   still applies: 1000+ opponents at least even, mirror ≥ 60%, Mega Kangaskhan ex
   ≥ 60%, first/second gap under ~5 points. `analyze_grimmsnarl_v4_gaps.py`
   makes each of those a per-turn number rather than a rating.
1. **Whether v2 should be retrained at patience 700.** This is the cheapest
   untested win in the whole line: v2.1 shipped at 0.8484 under the same
   patience-200 defect, so it is very likely leaving points on the table too. If
   v2 is the champion on rating, retraining *it* is a better next move than
   promoting anything.
2. **The attachment rate.** Three independent attempts have now failed to move a
   MAIN preference by adding columns (v3's 63, v4's 28). The gap is 13 points
   against our own pinned pilot and is the direct cause of the Adrena-Brain
   deficit that separates our won mirrors from our lost ones. The next lever is
   not a feature: either a dominance rule with a wider proof than "this
   attachment enables something" (the enabling shape is already at 97.4%), or the
   outcome learning v3 §6.3 named.
2. **Froslass in the mirror.** We take 100% of mirror offers; the pin takes 80.8%
   and the pilot that wins 84% of its mirrors takes 53.8%. The corrected ledger
   did not change it (0.8947 both). This is the strongest remaining single-shape
   divergence from our own teacher.
3. **Going second.** 50.9% against 69.4% first. No decision-level defect has been
   isolated for it yet; `first_player_is_self` has been a column since v2, so it
   is not feature blindness.

## Artifacts

- `gaps.json` — the per-turn table for 3,655 teacher games, 21 pilots, both v3
  ladder runs and v2, split by seat and mirror
- `corpus_v4_report.json` — 287,828 decisions, 822 features
- `train_v4_base.json` — training run, per-context, per-pilot, feature gains
- `control_v3_longpatience.json` — v3's features with patience 700
- `behaviour_v{3,4}_pinned_holdout.json` — §5, the table with a known target
- `behaviour_v{3,4}_on_v3{a,b}.json` — the same table on our own ladder boards
- `selfplay_v4_vs_v3.log`, `selfplay_v4_vs_v2.log`

Reproduce with:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_grimmsnarl_damage_tables.py --variant v4
.\.venv\Scripts\python.exe .\scripts\analyze_grimmsnarl_v4_gaps.py --workers 12 `
  --ladder "v3_a:55216787:data/runs/grimmsnarl/20260804_grimmsnarl_ml_v3_sub55216787" `
  --ladder "v3_b:55217233:data/runs/grimmsnarl/20260804_grimmsnarl_ml_v3_sub55217233" `
  --ladder "v2:55205556:data/runs/grimmsnarl/20260803_grimmsnarl_ml_v2_sub55205556" `
  --out experiments/grimmsnarl_ml_v4/gaps.json
.\.venv\Scripts\python.exe .\scripts\build_grimmsnarl_v2_corpus.py `
  --agent-dir agents/grimmsnarl/grimmsnarl_ml_v4 `
  --output data/ml/grimmsnarl/processed/corpus_v4.npz `
  --report experiments/grimmsnarl_ml_v4/corpus_v4_report.json --workers 10
.\.venv\Scripts\python.exe .\scripts\train_grimmsnarl_v2_teacher.py `
  --corpus data/ml/grimmsnarl/processed/corpus_v4.npz `
  --output-model data/ml/grimmsnarl/models/ranker_v4.txt `
  --report experiments/grimmsnarl_ml_v4/train_v4_base.json `
  --team-feature --split-mode per-team --threads 16
.\.venv\Scripts\python.exe .\scripts\export_grimmsnarl_v1_model.py `
  --model data/ml/grimmsnarl/models/ranker_v4.txt `
  --corpus data/ml/grimmsnarl/processed/corpus_v4.npz `
  --teacher-team 16494330 --min-context-support 90 --min-context-top1 0.5 `
  --output agents/grimmsnarl/grimmsnarl_ml_v4/ranker_model.json `
  --report experiments/grimmsnarl_ml_v4/train_v4_base.json
.\.venv\Scripts\python.exe .\scripts\analyze_grimmsnarl_v3_behaviour.py `
  --agent-dir agents/grimmsnarl/grimmsnarl_ml_v4 `
  --data-root data/kaggle_grimmsnarl_top50 --teams 16494330 `
  --min-episode 89418031 --corpus data/ml/grimmsnarl/processed/corpus_v4.npz `
  --report experiments/grimmsnarl_ml_v4/behaviour_v4_pinned_holdout.json
```
