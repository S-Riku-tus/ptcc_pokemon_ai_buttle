# Grimmsnarl ML v5 — the attachment gap was starvation, and it was upstream in a rule

Date: 2026-08-05
Parent: `grimmsnarl_ml_v4` (ladder 1031.2 over 52 games, 61.5% win rate, peak 1137.5)
Corpus: `data/kaggle_grimmsnarl_top50`, 3,710 same-60 replays (deck hash `9714ab5c3996f6cc`), 22 pilots
Run under analysis: `data/runs/grimmsnarl/20260805_grimmsnarl_ml_v4_sub55253296` (52 games)

**Headline.** v3 and v4 each set out to raise the once-per-turn Dark Energy
attachment rate and each failed — v4's own report concludes that "three
attempts across v3 and v4 have now failed to move a MAIN preference by adding
columns, which is the case for the next version being outcome-based rather than
imitation-based." That conclusion was wrong about the cause. The attachment
rate is not a MAIN preference at all. It is downstream of a **multi-pick select
the ranker structurally cannot see**, where the rule policy has been taking
every card the game offers since v1.

Punk Up searches the deck for *up to* five Basic Darkness. `maxCount` is 5, so
`Ranker.is_scorable` (which requires `maxCount == 1`) never routes it; it falls
to `normalize_selection`, which keeps every option with a positive score, and
`score_card` gives a Basic Darkness a flat 100. **v4 therefore took the maximum
on 100% of 90 stored activations.** The deck holds ten Darkness in total.

## 1. The behaviour, against the field, in rating order

Punk Up activations, per own activation, over the same-60 corpus:

| band | n | mean searched | took every card offered |
|---|---:|---:|---:|
| pilots ≥ 1120 | 1,997 | 2.65 | **36.9%** |
| 1070–1119 | 3,002 | 2.74 | 45.2% |
| < 1070 | 1,875 | 3.03 | 63.7% |
| **v4 (ladder)** | **90** | **3.67** | **100.0%** |

At the `maxCount == 5` node specifically the elite band takes 2 on 30%, 3 on
35%, 4 on 9% and 5 on 23% of activations. v4 takes 5 on 47 of 47.

Nothing in this line has ever scored this decision: it is not in MAIN, it is
not a Top-1 candidate, and the per-turn instrument v4 built measures *whether*
Punk Up fired, not how much it took.

## 2. Why it costs games: the deck is the fuel line

Punk Up attaches only to "your Marnie's Pokémon", so a Munkidori — the deck's
whole damage-movement engine — can only ever be fuelled by the once-a-turn hand
attachment. Mining five Darkness at a time out of a ten-card supply starves
exactly that. Measured **per own turn**:

| | ≥1120 | 1070–1119 | <1070 | **v4** |
|---|---:|---:|---:|---:|
| Darkness left in deck | 4.23 | 4.13 | 4.02 | **3.69** |
| a Darkness in hand | 71.0% | 72.0% | 72.8% | **66.9%** |
| attachment made when offered | 83.9% | 82.4% | 79.3% | **76.0%** |
| attachment made | 59.6% | 59.3% | 57.7% | **50.8%** |
| own turns | 6,652 | 9,881 | 6,408 | 305 |

From turn 5 on, where the deck is thin and it matters: 2.71 / 2.59 / 2.44
Darkness left against our **1.98**, and the attachment made on 53.2% / 52.8% /
50.4% of turns against our **40.3%**.

Every link is monotone in pilot rating and v4 is off the end of all four. This
is the chain the v3 and v4 feature work was trying to enter halfway down.

Two corroborating facts from the same run say the shortfall is misallocation
rather than scarcity: v4 has an attack-ready Grimmsnarl ex on board on 73.2% of
mirror turns against the field's 62.6% and the rank-3 pilot's 54.8%, and its
Punk Up *allocation order* is better than every teacher band — over 330 stored
attaches it fed a body already at two Darkness while a hungrier body was on the
menu **zero times** (elite 1.8%, mid 1.4%). v4 does not put the energy in the
wrong place. It takes energy it has no place for.

## 3. What v5 does

### P0 — `fallback_policy.punk_search_budget`

    searched = min(offered,
                   max(2, deficit_of_the_triggering_body_to_two
                        + one_per_other_Marnie's_body_below_two))

Fitted against the corpus, not invented. Reproducing each band's own count:

| rule | ≥1120 exact | within 1 | mean predicted vs actual |
|---|---:|---:|---|
| **v5 budget** | **51.9%** | **90.1%** | 2.61 vs 2.65 |
| always take max (v1–v4) | 36.9% | 53.4% | 3.95 vs 2.65 |
| always take 2 | 50.8% | 80.5% | 1.91 vs 2.65 |
| two per hungry body | 45.7% | 80.0% | 3.13 vs 2.65 |
| deficits + 1 slack | 40.2% | 81.1% | 3.27 vs 2.65 |

The v5 budget's fit degrades in rating order — 51.9% for ≥1120, 45.4% for
1070–1119, 36.9% below, 34.4% for v4 — while "always take max" improves in
rating order (36.9% / 45.2% / 63.7% / 100%). That ordering is what makes the
budget *the elite policy* rather than a curve that happens to fit.

### P0b — the same defect on Buddy-Buddy Poffin

`normalize_selection` maximises every multi-pick, and Poffin is the other one
that matters. v4 takes both Basics on 100% of 49 activations; the ≥1100 pilots
on 74.3% of 1,298, mean 1.64. Their discriminator is Bench space (1.07 bodies
with two slots open, 1.55 with three, 1.90 with four, 1.97 with five) and how
deep the Marnie's line already is (1.96 with none down, 1.19 with two, 0.58
with three). v5 takes one when fewer than three slots are open or the line is
already two bodies deep, both otherwise.

### P0c — one planner guard, so a tighter budget cannot be spent wrong

The budget is computed against one promise: the body that just evolved ends the
activation able to Shadow Bullet. `ml_planner._punk_allocation` makes that hold
when the budget *is* the deficit, and refuses a fifth energy onto a body already
holding four while a lighter one is offered.

Both clauses cost nothing today — the teachers fuel a still-hungry trigger on
96.1% / 98.8% / 99.5% of such offers and **v4 on 115 of 115** — and the stack
cap fires on 0.16% of elite attaches against v4's 0.91%. They are insurance for
the new budget, not a fix for the ranker.

## 4. Measured effect on v4's own boards

`scripts/probe_grimmsnarl_v5_search_budget.py` replays the 52 stored ladder
games decision by decision, asks the candidate at every Punk Up and Poffin
search, and advances the game with the action actually taken, so the two agents
are compared on identical states. The instrument reproduces v4's stored counts
exactly, which is the control.

| | v4 (control) | **v5** | ≥1120 band |
|---|---:|---:|---:|
| Punk Up, mean searched (n=90) | 3.67 | **2.62** | 2.65 |
| Punk Up, took every card offered | 100.0% | **34.4%** | 36.9% |
| Punk Up, five-card searches | 47 | **2** | — |
| Poffin, mean taken (n=55) | 1.89 | **1.60** | 1.64 |
| Poffin, took both | 100.0% | **70.9%** | 74.3% |
| illegal selects | 0 | **0** | — |

Both land on the elite band's own means.

## 5. Three analysis priorities measured and **not** implemented

Same discipline as v4, which refuted two of its five inherited priorities.

**A second Darkness on Munkidori — already correct, no change needed.** Across
v4's 52 games all 126 hand attachments to a Munkidori were the first energy on
a dry body; there were zero voluntary second attachments. (The one Munkidori
seen holding two got the second from the opponent's Handheld Fan.) The
corpus-wide rate is 0.78%. The proposed cap would have changed nothing.

**Attaching before a non-lethal attack — refuted.** The shape the priority
names is a turn closing while a Dark Energy attachment onto a dry Munkidori is
still legal. Across 764 v4 MAIN decisions with an attachment on the menu, that
happened **0 times** (elite 0.1%, n=15,555), and when a dry-Munkidori
attachment *is* offered v4 takes it 41.2% of the time against the elite band's
42.6% — parity. The attachment deficit is upstream fuel, which is §2.

**Petrel → Boss's Orders — not expressible as a dominance rule.** Petrel is
itself a Supporter, so at the moment of its search the supporter slot for the
turn is already spent: a Boss fetched off Petrel can never be played that turn.
There is nothing to prove. Conditioning does not rescue it either — on the exact
conjunction (Stamp and Boss both offered, no Boss in hand, a fuelled Grimmsnarl
on board) even the ≥1100 pilots take Boss on **4.1%** of 363 offers. v4 takes it
on 0 of 17. A planner rule that forced Boss there would be a 96-point error in
the opposite direction.

What *is* real in that select, and is left for a future version because it is a
ranker preference on a routed context rather than anything a shell can prove:
when a fresh Unfair Stamp is offered off Petrel, v4 takes it on **81.0%** of 42
offers against 46.3% for ≥1100, 65.6% for 1060–1099 and 67.9% below. Monotone
in rating again, and v4 again off the end.

**The mirror Froslass — real, but no board condition separates the refusals.**
v4 evolves on 10 of 10 mirror turns where it is offered; the field is 70.8%
(≥1100), 72.5% and 82.4%. But no measured condition discriminates: the
three-checkup prize-weighted shroud ledger being net negative for us moves the
≥1100 rate from 70.8% to 71.0%, having no fuelled Munkidori moves it *up* to
93.5%, and the strongest signal available — turn ≥ 7 — only reaches 61.0%. A
hard veto at the tightest conjunction the corpus offers would be a 61-point
error where the current behaviour is a 39-point one. v4's existing narrow
guard (refuse only when the next checkup knocks out one of ours and none of
theirs) is kept unchanged, and this stays open.

## 6. Validation

- 144 agent tests pass: 116 inherited from v4, plus 28 new — 17 pinning the
  budget against the counts the elite band plays, 11 pinning the planner
  guard's firing *and* its stand-down conditions.
- v4's 116 tests still pass unchanged in v4's own directory.
- 0 illegal selects, 0 crashes in the counterfactual probe over 52 games and in
  paired self-play.

**Ladder rating is not measured.** Local self-play and behavioural probes are
what this report contains; see [[kaggle-ladder-rating-noise]] for why a 52-game
ladder sample cannot settle a difference this size on its own.

## 7. What to check on the next ladder run

The behavioural targets, in the order the causal chain runs:

1. Punk Up takes every card offered on **30–45%** of activations (v4: 100%).
2. Five-card searches ≈ **0%** (v4: 52% of `maxCount == 5` activations).
3. Darkness left in deck per own turn ≥ **4.0** (v4: 3.69), and ≥ **2.4** from
   turn 5 (v4: 1.98).
4. A Darkness in hand on ≥ **70%** of own turns (v4: 66.9%).
5. The attachment made on ≥ **57%** of own turns (v4: 50.8%).
6. Adrena-Brain uses per game ≥ **6.0** in the mirror (v4: 5.50).
7. Unchanged, and to be watched for regression: lethal-attack take rate,
   Bench-KO target choice (100%), Unfair Stamp use rate, the Alakazam matchup,
   and the first/second-seat balance.

If 1 and 2 move but 3–6 do not, the fuel model in §2 is wrong and the change
should be reverted rather than tuned.
