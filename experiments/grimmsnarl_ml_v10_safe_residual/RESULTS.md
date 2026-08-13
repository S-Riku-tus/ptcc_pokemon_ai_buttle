# grimmsnarl_ml_v10 — a safe residual over v8

**Verdict.** The matchup this version was commissioned to fix does not survive
its own test, and neither does the mechanism it was commissioned to use. One
class does, it is the class v6 measured and pre-registered, and it is what
ships. v10 changes **6 decisions in 4,480** on v8's own ladder run, all in one
context, and leaves every measured attachment, evolve, attack, knockout and
prize rate **bit-identical** to v8.

Everything below is recomputed from `data/submissions/submission_55317804` and
`data/kaggle_grimmsnarl_top50`; none of the numbers in the brief were taken on
trust, and two of them turned out to mean something different from what they
were used for.

---

## 1. The brief's facts, recomputed

`scripts/analyze_grimmsnarl_v10_ladder.py` → `ladder_v8_55317804.json`.

51 episodes, one of them `EPISODE_TYPE_VALIDATION` (90643182) and
dropped, leaves **50 rated games**. Our deck hashes to `9714ab5c3996f6cc` in all
50. Turn order is read from `current.firstPlayer` on a late step, never from the
seat index.

So the arithmetic is right. What it is used for is not.

---

## 2. The Alakazam premise does not survive a test

`cc38cb450b86770a` is **one of five** deck hashes in the run whose heaviest line
is Alakazam. Taken as an archetype the record is **4-7**, and the split that
matters is turn order:

| | going first | going second |
| --- | --- | --- |
| Alakazam family (11 games) | **4-2** | **0-5** |
| everything else (39 games) | 16-6 | 8-9 |

* Alakazam family against the rest, unconditional: 4-7 vs 24-15, Fisher
  **p = 0.178**.
* Going first, Alakazam against the rest: 4-2 vs 16-6, **p = 1.00**. The
  matchup is *indistinguishable* from the field when we are on the play.
* Matched on both confounders — going second **and** opponent ≥ 1000 — Alakazam
  is 0-5 against 4-7 for everything else, **p = 0.245**.
* The single hash's 1-6 does reach p = 0.035 on its own, but that hash was
  selected out of ~30 for looking bad. At that multiplicity it is noise.

The structural read agrees with the statistics. That list is Abra / Kadabra /
Alakazam — a 140 HP, **one-prize** attacker — plus Shaymin, plus 4 Enhanced
Hammer that are dead against Basic Darkness. They need three knockouts on
Grimmsnarl ex; we need six on theirs. Shadow Bullet one-shots an Alakazam for a
single prize. That is a prize-trade the deck is on the wrong side of, and it is
not a decision the ranker is getting wrong.

**Consequence for this version:** the brief's acceptance criterion "the override
on the Alakazam losing boards repeats across several games and is supported by
several independent lines of evidence" is **not met, and cannot be met by any
candidate** — see §5 for the direct measurement. A deck-hash-keyed Alakazam
override would be fitting five games of noise, which the brief separately
forbids.

## 3. What *is* significant: turn order — and it is ours, not the deck's

Within the run: **20-8 going first against 8-14 going second, Fisher
p = 0.021.** That is the only cut in the 50 games that reaches significance.

The control that decides what it means is the field's record with the **same 60
cards**, over 3,642 archived games
(`scripts/analyze_grimmsnarl_v10_turn_order.py`):

| | first | second | odds ratio |
| --- | --- | --- | --- |
| v8 | 20-8 (0.714) | 8-14 (0.364) | **4.38** |
| field, 1100+ band | 162-99 (0.621) | 126-106 (0.543) | 1.29 |
| field, all bands | 1166-712 (0.621) | 980-784 (0.555) | 1.31 |

The deck has a ~7-point turn-order penalty. v8 has a 35-point one. v8 is
*above* the field on the play and below it on the draw — though at n = 22 the
going-second cell alone only reaches p = 0.085 against the field, so this is the
strongest signal in the run rather than an established regression.

### It is not visible in any behaviour we can measure

Per **own turn** (never per decision — the same data reads 18.9% or 88.6%
depending on the denominator):

| take rate, per own turn | v8 all | v8 1st | v8 2nd | field 1100+ 1st | field 1100+ 2nd |
| --- | --- | --- | --- | --- | --- |
| Dark attachment | 0.808 | 0.805 | 0.813 | 0.797 | 0.847 |
| attachment that turns an attack on | 0.994 | 1.000 | 0.985 | 0.932 | 0.891 |
| Froslass evolve | 0.791 | 0.808 | 0.765 | 0.785 | 0.862 |
| Grimmsnarl ex evolve | 0.617 | 0.660 | 0.559 | 0.666 | 0.680 |
| Boss's Orders | 0.317 | 0.233 | 0.410 | 0.353 | 0.365 |
| attack | 0.968 | 0.956 | 0.987 | 0.921 | 0.921 |
| bench a Basic | 0.870 | 0.833 | 0.929 | 0.803 | 0.850 |

v8 is at or above the elite band on five of seven, and its going-second numbers
are its *better* ones. Whatever costs the games on the draw, it is not one of
these.

### And no take rate has a rating gradient to chase

Spearman of each per-turn rate against pilot rating, over the 21 same-deck
pilots (`pilot_gradient.json`):

| rate | rho | p |
| --- | --- | --- |
| Dark attachment | +0.171 | 0.457 |
| enabling attachment | −0.396 | 0.075 |
| Froslass evolve | −0.355 | 0.115 |
| Grimmsnarl ex evolve | +0.275 | 0.227 |
| Boss's Orders | +0.216 | 0.348 |
| attack | −0.530 | **0.013** |
| bench | −0.397 | 0.074 |

Seven tests; only `attack` is nominally significant, it is *negative*, and it is
not monotone at the top (the 1220-rated pilot is at 0.972, higher than v8). This
reproduces the standing finding that Grimmsnarl imitation is saturated: there is
no resource behaviour left to imitate our way into.

## 4. Two hypotheses tested and rejected outright

**Board-out.** Episode 90672919 was lost on turn 5 with Grimmsnarl ex alone and
an empty Bench — both sides still on five prizes, so the knockout ended the game
rather than the prize race. It looked like a gate worth writing.
`scripts/analyze_grimmsnarl_v10_lone_body.py`: v8 closes a turn on a lone body
**2 times in 296** (0.68%), the field **8 in 1,545** (0.52%), and in **zero** of
v8's cases was a Basic benchable — that game had no Basic in hand from turn 4.
It was a draw, not a decision. Rejected.

```
v7 v3 v7 v3 v7 v7 v7 v3 v3 v3
v7 v3 v3 v3 v3 v7 v7 v3 v7 v7
v3 v3 v7 v7 v7 v7 v7 v7 v7 v3
```

## 5. The mechanism the brief asked for, measured and rejected

`probe_grimmsnarl_v10_advisors.py` re-walked all 50 rated games decision by
decision with **five independent advisors** scoring the identical boards — the
separately-trained current-top-four model (v9) and the shipped trees re-pinned
as 16371703, 16422241, 16452116 and 16561259. All teacher-forced, so every
advisor sees the same board at every step and none drifts onto its own
distribution. 4,480 decisions.

A k-of-n consensus disagreeing with v8 fires at a sane rate. It is simply
**uncorrelated with winning**:

| consensus | in games v8 lost | in games v8 won | Fisher p |
| --- | --- | --- | --- |
| 3 of 5 | 115/1729 = 6.65% | 214/2751 = 7.78% | 0.176 |
| 4 of 5 | 59/1729 = 3.41% | 101/2751 = 3.67% | 0.680 |
| 5 of 5 | 23/1729 = 1.33% | 32/2751 = 1.16% | 0.676 |

At every threshold the panel disagrees with v8 *slightly more often in the games
v8 won*. A general consensus residual is a policy perturbation with no relation
to the outcome — which is the same lesson the Alakazam line paid 5.16 points of
measured agreement for.

Restricted to the seven Alakazam-family **losses** (429 decisions), a 4-of-5
consensus produces 18 overrides spread over **12 distinct shapes**, the most
repeated of which occurs **twice**. There is no repeated, multiply-supported
Alakazam shape to gate on. That is the direct measurement behind §2.

## 6. What ships: the Petrel search, one class

v6 measured this class, found the only rating gradient in the line that survives
a significance test, and deliberately held it back so that one ladder run
measured one change. v7 shipped a value-search layer instead and v8 removed it,
so the class has still never been deployed. It is re-verified here from scratch
(`scripts/analyze_grimmsnarl_v10_stamp.py`, 3,642 games, 5,206 Petrel searches).

Unfair Stamp is playable **only if the opponent took a Prize card during their
last turn**. Taking it out of a Petrel search when they did not is a card that
does nothing this turn.

| dead Unfair Stamp, taken | rate | n |
| --- | --- | --- |
| **v8 on its own ladder run** | **0.743** | 35 |
| v8's own pin, 16494330 (1077.6) | 0.708 | 89 |
| field, all pilots | 0.570 | 2,477 |
| 16561259 (1126.3) | 0.340 | 100 |
| 16463316 (1141.3) | 0.426 | 211 |
| 16531269 (1121.6) | 0.443 | 194 |
| 16452116 (1101.8) | 0.471 | 193 |
| 16371703 (1220.2) | 0.512 | 203 |
| 16422241 (1172.6) | 0.521 | 71 |

**Spearman against pilot rating: rho = −0.607, p = 0.0036 over 21 pilots.** The
**live** case does not run with rating (rho = −0.607, p = 0.148, n = 7) and the
field takes a live Stamp *more* often than a dead one (0.767 against 0.570), so
the refusal is specific to the dead board — a policy, not a dislike of the card.
Ten gradient tests were run in this report in total; at Bonferroni 0.05/10 =
0.005 this one still stands.

On **v8's own 35 dead-Stamp offers**, every advisor takes it less often than v8:

| | v8 | 16371703 | 16422241 | 16452116 | 16561259 | v9 |
| --- | --- | --- | --- | --- | --- | --- |
| dead Stamp taken | **0.743** | 0.600 | 0.657 | 0.571 | 0.514 | 0.514 |

Unanimous in direction, measured on v8's boards rather than on the advisors'
own. A wider eight-pilot panel (`stamp_panel_wide.json`) is unanimous too,
0.514-0.686.

### The rule

`agents/grimmsnarl/grimmsnarl_ml_v10/ml_residual.py`. v8's index is returned
unless **all** of:

1. context 7 and the select's `effect` resolves to Team Rocket's Petrel;
2. an Unfair Stamp is on offer and none is already in hand;
3. v8's own pick **is** that Stamp;
4. the Stamp is **dead** — the opponent's prize count has not fallen since the
   last turn we observed;
5. at least **3 of 4** re-pinned pilots pick the same other option;
6. that option is not a second copy of the Stamp.

Any exception, any missing feature, fewer than three advisors scoring, a board
shape it was never measured on — all return v8's index and are counted.

The panel costs **no extra model file**: `teacher_team_id` is a categorical
input, so a second opinion is one more pass over the shipped trees. It is
computed only on boards that already passed gates 1-4, roughly once every two
games.

### Deployed effect on v8's own 50 rated games

`scripts/probe_grimmsnarl_v10_residual.py` walks v8 and v10 side by side on the
identical stored decisions (`residual_ledger.json`):

```
decisions               4480
differences                6      (0.134%)
contexts changed        {7: 6}
distinct episodes          6
of which v8 lost           1

petrel_searches           69
stamp_offered             37
dead_stamp_chosen         26   <- the override ceiling, 0.58% of decisions
panel_agreed_with_v8      18
panel_short                2
overrides                  6
errors                     0
stamp_live_kept            0
stamp_already_in_hand      0
new_game_detected          0
```

All six replace the Stamp with a card the panel agreed on — Lillie's
Determination five times, Poké Pad once — at turns 3-6 of six different games.

## 7. Acceptance

| criterion | result |
| --- | --- |
| deck hash identical to v8 | `9714ab5c3996f6cc`, byte-identical `deck.csv` |
| no regression in v8's tests | 163 v8 tests pass unchanged; **190** total in v10 |
| unit tests for the new gate | 27, including one per protected context |
| attachment / attack-enabling attachment / Grimmsnarl / Froslass not degraded | **bit-identical**, see below |
| Alakazam override repeated and multiply supported | **not met — see §2 and §5** |
| mirror and sub-1000 strengths not broken | no decision in those games changed except one Petrel search |

Replaying all 51 stored games through both agents
(`behaviour_v10_on_v8_ladder.json` against `baseline_v8_on_v8_ladder.json`),
**every** one of these is identical to the digit:

```
attachment 0.7902   attack-enabling attachment 0.9750
Froslass 0.7442     Grimmsnarl ex 0.5854      Boss 0.3133
best-prize knockout 0.9778    max counters moved 1.000
planner: heal 12/205, punk 0/215, boss-route 0/0, wall-unlock 0/0
ranker: 3871 used, 0 feature errors, 0 score errors, escalation moved 45
```

The only moved number in the whole report is context 7's agreement with the
replay, 0.9908 → 0.9817 — the six overrides. This is not a coincidence to be
checked per version: context 7 is "put a card in hand", so it cannot be an
attachment, an evolve, an attack, a knockout, a prize pick or a gust target.
The invariants hold **by construction**, and
`tests/test_v10_residual.py::test_no_other_context_is_ever_reached` pins that
for contexts 0, 3, 4, 15, 16, 21, 40 and 43.

### 7.2 Submission artifact — built, **not** submitted

```
artifacts/submission_grimmsnarl_ml_v10.tar.gz
10,934,419 bytes, 19 entries, no tests / metadata / __pycache__
sha256 c5d3811d81ec0abf053423374be50c7e062f45f9967de74d1677b78f87058b6b
```

`scripts/validate_agent.py` passes with no warnings. Extracted to a clean
directory the archive imports with `load_error`, `planner_load_error` and
`residual_load_error` all `None` and returns a 60-card deck hashing to
`9714ab5c3996f6cc`. Nothing was uploaded to Kaggle and no existing submission
was touched.

## 8. Honest limits

* **The effect is small.** Six decisions in fifty games. The class ceiling — the
  most any policy could change here — is 26 decisions, 0.58%. Given that an
  identical agent has scored 842.8 and 804 on this ladder, no realistic run
  length will resolve a change this size. What the evidence supports is that the
  change is in the right direction and cannot hurt the rest of the agent; it
  does not support a rating prediction.
* **The going-second deficit is unexplained and unaddressed.** It is the only
  significant cut in the run and the field does not share it, but no measured
  behaviour differs there, so there is nothing to gate on yet. §9 says what
  would be needed.
* **One threshold is not validated out of sample.** There is one ladder run. The
  3-of-4 rule was chosen as a strict supermajority, not fitted: 2, 3 and 4 of 4
  fire on 6, 6 and 3 decisions respectively, so the result is insensitive to it.
* **Live Stamps were never tested in deployment.** v8 picked a live Stamp on 0
  of 2 opportunities, so gate 4 never had to fire on the stored run. It is
  covered by a unit test, not by a replay. The same applies to the guard that
  clears the prize history when a turn number goes backwards — nothing calls
  `diag_reset` between Kaggle episodes, so a reused process would otherwise
  read the previous game's prize count on the first Petrel search of the next
  one. Both are unit-tested; neither is exercised by the stored replays.

### Repository suite

`pytest tests -m "not smoke"` reports 1 failure and 10 errors both with and
without v10 present (`test_ml_paths` wants an `alakazam_ml_v2_expanded` that
does not exist in this checkout; the Spidops package test fails to import
`main`). Neither is touched by this work. Agent test directories also cannot be
collected two at a time — v8 and v9 collide the same way — so each is run on its
own.

## 9. Not implemented, and what it would take

**Petrel → Boss's Orders.** The same pass found a second significant gradient on
the same select, in the opposite direction: taking Boss out of a Petrel search
runs *with* pilot rating, **rho = +0.581, p = 0.0058**, and v8 takes it on **0
of 64** offers against 0.129-0.161 for the top pilots. It is not acted on
because every panel pilot is also near zero on v8's boards (0.016-0.109), so
there is no consensus to gate on, and a preference rule would be exactly the
unmeasured shell this version exists to avoid. At Bonferroni over the ten
gradients tested here (0.005) it also does not clear. To act on it: a panel that
actually plays it on our boards, which means either pilots selected for that
behaviour or a model trained with the Petrel search as its target.

**The going-second gap.** To turn it into something gateable, the next run needs
per-turn measurements the current probes do not take: hand size and prize-race
position at the end of each own turn, the turn the first knockout lands, and
what the *opponent* did on the turn before each of our losing sequences. The
resource take rates are exhausted — they are all at or above the elite band.

**Alakazam.** If the archetype is to be attacked at all it needs games, not
analysis: 11 of them cannot separate a matchup from a turn-order split. The
prize-trade arithmetic (their one-prize 140 HP attacker against our two-prize
320) suggests the answer is a deck question, which this line has fixed.
