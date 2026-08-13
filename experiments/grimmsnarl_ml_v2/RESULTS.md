# Grimmsnarl ML v2 — re-targeted pilot, all-context imitation

Date: 2026-08-03
Result: **91.74% end-to-end all-context agreement** with a pilot 29 rating
points stronger than v1's, up from **81.41%**.

## The two v1 defects this fixes

### 1. The pilot was selected by maximising agreement

v1 compared pilots by offline Top-1 and pinned the winner: team 16421840,
rank 48, rating 1048.5 — the weakest teacher in the corpus.

That was not bad luck, it is what the criterion selects for. Agreement with a
pilot is inversely related to that pilot's rating, so ranking by fidelity
walks *down* the leaderboard:

| pilot | rating | v1 MAIN Top-1 |
|---|---:|---:|
| 16371703 | 1220.2 | 0.7029 |
| 16422241 | 1172.6 | 0.7450 |
| 16556346 | 1086.7 | 0.8682 |
| 16494330 | 1077.6 | 0.8839 |
| 16421840 | **1048.5** | **0.9155** ← v1 pinned this |

### 2. Half the decisions were never measured

v1 imitated MAIN (~41 decisions/game) and left every other context to the
inherited v7 rule policy (~47 decisions/game), which had never been compared
against a single top-50 pilot. Measured against v1's own teacher:

| context | decides | v1 agreement |
|---|---|---:|
| MAIN | play sequencing | 90.5% |
| **TO_HAND / deck** | **which card to search out** | **39.5%** |
| REMOVE_DAMAGE_COUNTER | Adrena-Brain counter removal | 50.0% |
| DAMAGE_COUNTER | Adrena-Brain counter placement | 64.5% |
| TO_HAND / discard | Night Stretcher recovery | 57.9% |
| DAMAGE | Shadow Bullet bench-snipe target | 76.5% |
| ATTACH_FROM | Punk Up energy source | 93.2% |
| **all contexts** | | **81.41%** |

v1 copied the teacher's play sequencing faithfully while searching for the
wrong cards and putting damage in the wrong places. That is the most likely
explanation for 91% MAIN agreement with a 1048-rated teacher producing a
rating of 871.

## Was the strongest pilot reachable?

Not at v1's fidelity, and the reason is not stochasticity.

`analyze_grimmsnarl_pilot_compute.py` reads `remainingOverageTime` deltas as
per-decision compute. The rank-4 pilot (1220.2) spends 19.1 s/game with a p95
of 69 ms — a fast deterministic program, not a search agent. Two pilots that
*are* slow (rank 11 at 107 s/game, rank 49 with a 4.7 s p95) rate no better.

Per-pilot menu-level self-consistency explains the difficulty instead:

| pilot | rating | self-consistency | v1 model | model − self |
|---|---:|---:|---:|---:|
| 16421840 | 1048.5 | 0.861 | 0.916 | **+5.4** |
| 16371703 | 1220.2 | 0.773 | 0.703 | **−7.0** |

For the weak pilot, board features add information over a menu lookup. For
the strong pilot the model does *worse* than a menu lookup: the strong pilot
conditions on state our features do not express. Closing that is open work,
not something this version claims to have solved.

So v2 targets the Pareto point rather than either extreme: **16494330, rank
30, rating 1077.6** — the strongest pilot whose model already exceeded its own
self-consistency.

## What v2 changes

* Corpus extended from MAIN to every single-pick select context: **287,828
  decisions**, up from 148,028 (+94%), 698 features.
* Optional selects (`minCount == 0`) are included as plain single-picks: across
  3,655 games the teachers declined one **zero** times, so there is no decline
  branch to model.
* `ctx_*` feature block for the new contexts: candidate card identity resolved
  across deck/discard/board/opponent zones, copies already held, and whether
  the pending damage swing kills the target.
* Semantic key extended with `ctx_number` so `{number: 1|2|3}` options stay
  distinct instead of collapsing into one candidate.
* Runtime routes every scorable context through the ranker. Multi-pick selects
  and face-down prize picks stay rule-driven — the prize zone exposes no card
  ids, so nothing is learnable there.

## Results

Offline, pilot-conditioned, per-pilot chronological split, strict Top-1 on
decisions never used for fitting or selection.

| block | n | v1 (rules) | v2 (ranker) |
|---|---:|---:|---:|
| MAIN | 758 | 0.905 | 0.885 |
| TO_HAND deck search | 120 | **0.395** | **0.925** |
| DAMAGE_COUNTER | 86 | 0.645 | 0.930 |
| REMOVE_DAMAGE_COUNTER | 45 | 0.500 | 0.911 |
| DAMAGE snipe target | 50 | 0.765 | 0.960 |
| ATTACH_FROM | 81 | 0.932 | 0.988 |
| **all contexts, end-to-end** | 1331 | **0.8141** | **0.9174** |

Offline all-context Top-1 for the pinned pilot is 0.9223 (n=1480, Wilson 95%
0.9068–0.9342); the 0.49-point runtime gap is the multi-pick contexts the
runtime still leaves to rules. Pooled across all 21 pilots the model scores
0.8471.

MAIN drops 2 points against v1's number, but those are different teachers —
v1's figure is against a 1048.5-rated pilot and v2's against a 1077.6-rated
one, and the harder pilot was chosen deliberately.

## Where the remaining 8% is, and why 95% did not land

Target was ≥95%. Achieved 92.23% offline / 91.74% end-to-end. The residual on
the pinned pilot:

- 92.16% correct
- 4.32% same-turn ordering (the model played something the teacher also
  played that turn, in a different order)
- 1.76% same-action-type divergence
- 1.76% genuine divergence

Order-insensitive turn-set agreement is 96.49%. Reaching 95% strict therefore
means recovering about 2.8 of the 4.3 ordering points without breaking a
correct choice. That is the same wall the Alakazam v36 report hit, and the
model families that failed there (action hierarchy, empirical pair
precedence, conditional pairwise, Top-K cascade) have not been retried here.

Upweighting the target pilot 6x during fitting was tried and did nothing
(0.9216 vs 0.9223 unweighted), consistent with v1's finding that single-pilot
corpora underperform the pooled conditioned model.

## Open items

1. No ladder rating for v2. v1 rated 871 over 61 games (39-22); per the
   rating-noise finding, compare by opponent-bucketed win rate, not rating.
2. v1's ladder logs show the losing match-ups are Fezandipiti ex (3-6),
   Mega Kangaskhan ex (2-4) and Mega Lucario ex (2-3). Imitation does not
   target these directly.
3. Multi-pick selects are still rule-driven, including **Punk Up's five-energy
   attachment**, which is arguably the highest-leverage decision in the deck.
   Sequential/greedy decoding over the same ranker is the obvious next step.
4. The strong-pilot feature gap (16371703 at 78.4% all-context) is unexplained
   and is the ceiling on targeting the 1220-rated pilot.

## Artifacts

- `pilot_compute.json` — per-pilot compute and determinism probe
- `teacher_selfagree.json` — per-pilot self vs field agreement
- `context_agreement_16421840.json`, `context_agreement_areas.json` — v1
  per-context baseline
- `corpus_v2_report.json` — 287,828-decision extraction
- `train_v2_allctx.json`, `train_v2_focus6.json` — training runs
- `runtime_v2_16494330.json` — end-to-end all-context agreement
