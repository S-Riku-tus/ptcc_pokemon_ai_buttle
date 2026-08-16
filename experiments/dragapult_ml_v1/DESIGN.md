# Dragapult ML v1 design decision

## 1. Decision

Build an exact-list Dragapult ex imitation baseline first.  Do not add search,
value arbitration, or deck variants to v1.

This choice isolates one question: can a deck with both current top-field
support and abundant strong-pilot logs be reproduced safely?  Adding search at
the same time would make a good or bad result impossible to attribute to the
deck, teacher data, imitation model, or planner.

## 2. Why this deck

The supplied 2026-08-16 analysis gives Dragapult the best evidence combination:

- 24.2% of the observed upper metagame, so it is not a single-pilot anomaly.
- Our historical result was 29-17 (0.630), unlike the weak evidence for the
  newly popular Conkeldurr (0-4) and Hydrapple (4-10).
- Nine independent high-ranked submissions used one identical 60-card list,
  allowing deck identity and pilot policy to be separated from archetype labels.

This is a data-quality decision, not a claim that Dragapult is the strongest
deck in isolation.

## 3. Experimental contract

The information flow is deliberately one-way:

`verified exact-list replay -> public-observation features -> chronological split -> ranker -> guarded runtime -> untouched outcome test`

The following rules prevent optimistic leakage:

1. Re-hash the full 60-card deck at the target seat in every replay.
2. Deduplicate by `(episode_id, seat_index)`.
3. Never use the opponent's hidden hand/deck, future observations, final reward,
   or a post-decision state as an input feature.
4. Split chronologically inside each teacher, so every teacher's test games are
   later than that teacher's train games.
5. Give each episode equal base mass, then each teacher equal total mass, so one
   long game or prolific pilot cannot dominate training.
6. Pin one real teacher identity at runtime.  An averaged or invented teacher
   code was not observed during training and would be out of distribution.

## 4. Runtime authority

The model receives authority only for mandatory, single-card selections with at
least two distinct semantic candidates, a supported context, and known card and
attack identities.  The deterministic Dragapult policy handles:

- optional decline decisions;
- multi-pick decisions;
- unseen candidate identities;
- thin or excluded contexts;
- feature, model-load, or scoring failures.

This is important because accepted-action logs contain no explicit negative
example for the choice to decline an optional action.  Ranking only the visible
accepted candidates must not silently remove that legal alternative.

## 5. Predeclared gates

| Gate | Required | v1 result | Status |
|---|---:|---:|---|
| Verified exact-list trajectories | at least 1,000 | 854 | fail |
| Independent teachers | at least 5 | 9 | pass |
| Deck/seat integrity errors | 0 | 0 | pass |
| Held-out top-3 imitation | at least 0.9700 | 0.9634 | fail |
| Submitted-shell legal actions | 1.0000 | 1.0000 | pass |
| Submitted-shell exceptions | 0 | 0 | pass |

The runtime test used 102 chronological test episodes and 10,612 decisions.  Its
semantic agreement was 0.7327 overall and 0.7573 for mandatory single-pick
decisions.  ML was used on 8,357 decisions; the rest took the safe fallback.

These are fidelity and safety measurements, not win-rate measurements.  A
policy can copy frequent easy decisions accurately and still lose games on a
small number of strategically critical choices.

## 6. Consequence

`dragapult_ml_v1` is a reproducible offline candidate and a clean control arm,
not a submission recommendation.  Do not spend a live submission slot on it
until both failed gates are addressed and a later frozen batch supplies an
untouched outcome test.

The next experiment should change one axis only:

1. Extend the exact-list corpus beyond 1,000 verified trajectories and freeze a
   new future test batch.
2. Improve MAIN/context-0 decisions, where runtime agreement is 0.6528, while
   leaving already reliable mechanical contexts alone.
3. Compare the revised imitation agent with v1 using the same frozen episodes.
4. Only then create v2 search arbitration: allow the value model to override
   imitation in late turns where its calibration is demonstrated, logging every
   override and retaining imitation when the value margin is uncertain.

This order makes the eventual search comparison causal: v1 supplies the
no-search baseline that the earlier versions never cleanly established for this
new deck.
