# Dragapult ML v2 design decision

## 1. What the v1 ladder run actually said

v1 was submitted as 55545828 and produced 21 public games: 10-11, rating 493
after starting at 600. The obvious readings — bad deck, bad matchups, too
little data — are all wrong, and the logs say so.

The deck is fine. On 2026-08-16 Dragapult is 13 of the top 50, the largest
archetype, and 8 of those 13 play this exact list. The same-deck teachers win
0.651 over 1,392 games.

The matchups are fine. The teachers' worst common matchups are Conkeldurr and
Mega Kangaskhan at 0.500 and Hydrapple at 0.548; against the current field mix
that is about 0.60 expected.

The action rates are fine. Per own turn, v1 attached energy on 0.889 of the
turns it was offered against the teachers' 0.877, evolved Drakloak 0.951 vs
0.963, used Adrena-Brain 0.917 vs 0.932, used Recon Directive 0.821 vs 0.895.

What was wrong was the *argument* of the action:

| | v1.0 live | teachers |
|---|---:|---:|
| duplicate-colour attachments / game | 0.95 | 0.03 |
| completes a Fire+Psychic pair / game | 1.05 | 1.62 |
| Dragapult evolved onto a 2-colour body / game | 0.19 | 0.70 |
| first Phantom Dive, own-turn mean | 6.2 | 3.8 |
| games that ever used Phantom Dive | 57% | 96% |

In 9 separate decisions v1 attached a colour a body already held while the
same decision offered the colour that would have armed Phantom Dive.

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
curve now sits on the teachers': 0.950 of local games reach both Phantom Dive
colours at own-turn 3.97, against the teachers' 0.957 at 3.78 and the submitted
v1.0's 0.550 at 5.03.

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
