# Dragapult ML v2 design decision

## 1. What the v1 ladder run actually said

v1 was submitted as 55545828. As of the 2026-08-16 refetch it has produced 22
public games plus one validation game: 11-11, displayed rating 507.8 after
starting at 600. 507.8 is not a recovery — the opponents it was paired against
averaged 493.8 initial rating, so at a 0.500 win rate the Elo fixed point is
493.8 and the displayed number is noise around it. Bucketed by *opponent*
rating: below 400 it is 4-1, 400–500 it is 2-2, 500–600 it is 5-6, above 600 it
is 0-2.

The obvious readings — bad deck, bad matchups, too little data — are all wrong,
and the logs say so.

The deck is fine. On 2026-08-16 Dragapult is 13 of the top 50, the largest
archetype, and 8 of those 13 play this exact list. The same-deck teachers win
0.651 over 1,392 games.

The matchups are fine. The teachers' worst common matchups are Conkeldurr and
Mega Kangaskhan at 0.500 and Hydrapple at 0.548; against the current field mix
that is about 0.60 expected.

The action rates are fine. Per own turn, v1 attached energy on 0.894 of the
turns it was offered against the teachers' 0.892, evolved Drakloak 0.958 vs
0.954, used Adrena-Brain 0.926 vs 0.903, used Recon Directive 0.838 vs 0.823.

What was wrong was the *argument* of the action:

| | v1.0 live | teachers |
|---|---:|---:|
| duplicate-colour attachments / game | 0.870 | 0.037 |
| completes a Fire+Psychic pair / game | 1.043 | 1.701 |
| Dragapult evolved onto a 2-colour body / game | 0.174 | 0.809 |
| first Phantom Dive, own-turn mean | 6.5 | 4.0 |
| games that ever used Phantom Dive | 63.6% | 94.0% |
| own turns per game | 8.05 | 6.86 |

Replaying the run through the bundle that produced it attributes every one of
its 23 duplicate-colour attachments to the ranker, not to the fallback: the
submitted v1.0 has no guard, and it reproduces 2,031 of the run's 2,053
decisions (0.9893). In 12 of those 23 the same decision offered an attachment
that completed the Fire+Psychic pair.

## 2. Why that is a representation failure, not a preference failure

v1's row for an attachment carried the target's card id and its *total* energy
count. "Fire onto a Dragapult holding Psychic" and "Fire onto a Dragapult
holding Fire" produced identical rows. No amount of training data or model
capacity can separate two identical rows.

This distinction matters because the correct response is opposite in the two
cases. v1.1 chose the other response — a deterministic override of energy,
evolution, retreat, Boss and search — and on the frozen test split that
override seized 2,322 decisions and matched the teachers on 40.2% of them,
against the model's own 72.7%. That is the same shape as the Grimmsnarl v24
Froslass guard and the Alakazam v31-v34 safety shell: a hand-written policy
inserted in front of imitation loses more than the case it was written for.

## 3. Scope

One axis at a time, in this order:

1. Give the model the columns it needs to express the distinction, and the
   resolved card knowledge it needs to price a knock-out.
2. Refresh the corpus, which was below its own 1,000-trajectory gate.
3. Delete the broad override; keep only what is mechanically dominated.
4. Fix the deterministic policy where it owns decisions the ranker never sees.

No search, no value model, no deck change. The five pilots above 1180 have all
cut the two Jamming Tower, but team 16380946 scored 1229.3 on this list and
1224.0 after making that change, so the list is not the difference.

## 4. Leakage and distribution controls

The v1 controls are kept: full 60-card re-hash per seat, dedup by
`(episode_id, seat_index)`, no opponent hidden state, no future state, no
reward, per-teacher chronological splits, equal episode and teacher mass, one
real pinned teacher identity.

One is added. `remainingOverageTime` is removed from the feature set. It was
v1's 15th highest-gain column and it describes the pilot's compute, not the
game: teacher logs run 572.3–592.3 s and ours run 591.2–598.9 s, so nearly
every split learned on it sends our rows down a branch the training data did
not cover.

## 5. Predeclared gates

| Gate | Required | v1 | v2 | Status |
|---|---:|---:|---:|---|
| Verified exact-list trajectories | >= 1,000 | 854 | 1,392 | pass |
| Independent teachers | >= 5 | 9 | 15 | pass |
| Deck/seat integrity errors | 0 | 0 | 0 | pass |
| Held-out top-3 imitation | >= 0.9700 | 0.9634 | 0.9634 | fail |
| Submitted-shell legal actions | 1.0000 | 1.0000 | 1.0000 | pass |
| Submitted-shell exceptions | 0 | 0 | 0 | pass |
| Duplicate route attachments at or below teacher rate | yes | no | 2/681 vs 8/614 | pass |
| Shell agreement not below v1 on the same episodes | yes | 0.6862 | 0.7295 | pass |

Top-3 is still short of the predeclared 0.9700 and is reported as a miss. It is
also the gate least connected to play: the v2 test split spans 15 pilots rather
than 9, and pooled Top-k falls as pilot disagreement rises.

## 6. Consequence

v2 is a submission candidate rather than an offline control. Its behavioural
curve now sits on the teachers': 0.900 of local games reach both Phantom Dive
colours at own-turn 3.72, against the teachers' 0.943 at 3.69 and the submitted
v1.0's 0.617 at 4.24, over 60 games in which v2 wins 44.

The stronger evidence is the counterfactual, because it is measured on the
decisions that actually lost the run rather than on held-out teacher games.
Forced onto the real trajectory, v2 chooses 0 duplicate-colour route
attachments where v1.0 chose 21, and 109 pair-completing ones where v1.0 chose
24, while still reproducing 82.7% of v1.0's live actions. At the 12 decisions
where v1.0 took a duplicate with a completing attachment on offer, v2 takes the
completing one 11 times and the duplicate never.

What is still unmeasured, in priority order for v3:

1. **The pinned teacher.** 16380946 is pinned because it is the highest rated,
   but it is also the pilot this model reproduces least accurately (Top-1 0.714
   against 0.803 for the best-imitated pilot). Whether a more imitable pin
   plays better has never been tested for this deck.
2. **Multi-pick beyond Ultra Ball.** Buddy-Buddy Poffin's "up to 2 Basics"
   still runs on a hand-written score at 0.728 agreement, and no imitation
   metric covers it because the ranker is single-pick by construction.
3. **The Hydrapple and Conkeldurr cells**, where the teachers themselves are at
   0.548 and 0.500 and imitation therefore has no headroom to give.
