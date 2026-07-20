# Validation report — Marnie's Grimmsnarl ex v3

## Summary

v3 is a **logic-only** change over v2 (the 60-card list and `policy_base.py` are
identical; the deck SHA is unchanged).  It adds one validated change (Unfair Stamp
hand-denial) and one archetype-faithful, measured-neutral change (second
Munkidori), and it **rejects** a data-suggested change that turned out to hurt.

## Measurement methodology

The bundled `cg` engine is **not seed-reproducible** — the native shuffle RNG is
not controllable from Python, so the same command produces different results
(observed: 35.3% then 33.7% on an identical 300-game call).  Single 1000-game runs
therefore carry ~±3pt of noise.  All headline numbers below are large-sample and
reported with 95% confidence intervals.  Self-play was run with
`scripts/local_arena.py`; failure-mode diagnosis with `scripts/diag_grimmsnarl.py`.

## Head-to-head results (win rate, excl. draws)

Against `alakazam_ml_v11` (4000 games each):

| Agent | Win rate | 95% CI | vs v2 |
|---|---:|---:|---:|
| v2 | 31.8% | [30.3, 33.2] | — |
| **v3** | **34.7%** | [33.2, 36.2] | **+2.9pt (z=2.8)** |

Against `alakazam_ml_v10` (2000 games each) — a *different* tuned Alakazam, used to
check the change is not overfit to one opponent:

| Agent | Win rate | 95% CI | vs v2 |
|---|---:|---:|---:|
| v2 | 38.6% | [36.5, 40.8] | — |
| **v3** | **43.0%** | [40.9, 45.2] | **+4.4pt (z=2.8)** |

The improvement is statistically significant and **generalises across both
opponents**.

## What was kept

1. **Unfair Stamp as pre-attack hand denial.**  Unfair Stamp is a free Item that
   resets the opponent to two cards.  v2 scored it below Shadow Bullet, so on a
   turn it could attack it attacked-and-ended and never stamped.  v3 scores it just
   above Shadow Bullet (and below board development), scaled to the opponent's hand
   size, so it strips a large hand and then still attacks.  This directly starves
   hand-size damage/draw engines — Alakazam's *Powerful Hand* deals 2 damage
   counters per card in hand.  **This was the measured driver of the win-rate gain.**

2. **Second Munkidori vs a racing opponent.**  A second Munkidori doubles the
   Adrena-Brain relocation that both heals our attacker and spreads damage back.
   v2 only opened the second copy on a fully developed board; v3 also opens it
   against a racing opponent.  This measured **neutral** vs the tested Alakazam
   agents (34.7% with vs 34.7% without; 43.0% vs 42.8%) but is retained for its
   spread value against the wider field, per the archetype's design.

## What was rejected (measured to hurt)

- **Gating Froslass development + an explicit "heal the threatened attacker"
  bias.**  A diagnostic correlation (Froslass online in 58% of losses vs 39% of
  wins) suggested Froslass was self-sabotaging by chipping our own attacker.
  Acting on it, however, *lowered* the win rate: the Unfair-Stamp-only build scored
  34.7% vs `ml_v11` while adding these two changes dropped it to 33.5%.  The
  correlation was confounded (longer, losing games simply have more time to evolve
  Froslass), so both changes were dropped.

## Diagnostic finding (why v2 lost)

Instrumented self-play (`diag_grimmsnarl.py`) showed the loss mechanism cleanly.
The single biggest win/loss discriminator was **opponent maximum hand size**
(~17.6 in our wins vs ~23 in our losses).  Because Grimmsnarl ex costs the
opponent 2 prizes but Alakazam only 1, trading one-for-one loses the prize race;
Alakazam reliably stacked a 17+ card hand and one-shot our 320 HP attacker with
Powerful Hand.  The winning line is to **race with Shadow Bullet and deny the
hand** — which is exactly what the Unfair Stamp change enforces.

## Completed static checks

- `deck.csv`: exactly 60 cards; identical to v1/v2 (logic-only change).
- `main.py` / `policy_base.py`: compile and import under the fake-`cg` harness.
- Golden-state tests: **14 passed** (`tests/test_v2_static.py`).
- Self-contained runtime (`main.py`, `policy_base.py`, `deck.csv`).

## Caveats

- Only Alakazam-family opponents (`ml_v11`, `ml_v10`) and the mirror were available
  for live testing.  The second-Munkidori change is retained on archetype grounds
  and measured neutral here; it is not independently validated against the wider
  field.
- These are self-play results against specific agents, not Kaggle-ladder ratings.
