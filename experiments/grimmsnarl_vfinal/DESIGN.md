# grimmsnarl_ml_vfinal — design and evidence

Written 2026-08-17 JST, on top of
`experiments/grimmsnarl_endgame_20260816/DIAGNOSIS.md`.

## 1. What state the line was actually in

Two things had to be separated before choosing what to build.

**The displayed score had collapsed for a reason that has nothing to do with
the Grimmsnarl policy.** At 2026-08-17 00:0x JST our team stood at rank
2077 / 6859 with a public score of **694.1**, and both live slots were
Dragapult submissions (`55550682` = 694.1, `55545828` = 507.8). The converged
Grimmsnarl slots — v28's 968.7 and the v22 rerun's 886.2 — had been truncated
away by later submissions. Nothing in this document is needed to recover most
of that: v22 alone is an implied-strength ~1010 agent and putting it back in a
slot is worth roughly +300 displayed points with no code and no risk.

**The policy itself has been flat since v22.** Opponent-mean-adjusted implied
strength over 552 stored games:

| run | n | record | opp mean | implied |
|---|---:|---|---:|---:|
| v22 (a-e pooled) | 215 | 135-80 | 913 | ~1010 |
| v25 | 85 | 48-37 | 825 | ~890 |
| v27 | 34 | 21-13 | 789 | 872 |
| v28 | 35 | 24-11 | 861 | 996 |
| v29 | 47 | 30-17 | 758 | 858 |

Seven versions, no separation from v22 beyond the ~130 Elo single-run noise
floor.

## 2. Where the remaining win rate actually is

`where_we_lose.py` splits all 552 games. Against opponents rated 950+ — the
only band whose games move the rating — we are 64-73 (0.467):

| cell | n | share | win rate | implied |
|---|---:|---:|---:|---:|
| Ogerpon | 13 | 9.5% | **0.000** | — |
| Hydrapple ex | 9 | 6.6% | 0.333 | 892 |
| Mega Lopunny / Froslass | 22 | 16.1% | 0.409 | 947 |
| Grimmsnarl mirror | 34 | 24.8% | 0.500 | 1000 |
| Kangaskhan / Crustle | 15 | 10.9% | 0.600 | 1077 |
| Dragapult | 18 | 13.1% | 0.556 | 1081 |
| Alakazam | 15 | 10.9% | 0.733 | 1182 |

Two things follow, and they set the honest ceiling for this deck.

**The Grass cells are structural, not a policy defect.** Teal Mask Ogerpon ex's
Myriad Leaf Shower is 30 + 30 per Energy on both Active Pokemon and Marnie's
Grimmsnarl ex is Grass-Weak, so from the turn Ogerpon holds three Energy the
attack is `(30 + 30x5) x 2 = 360` against a 320 HP body: a one-shot on our
two-prize attacker, every turn. Shadow Bullet's 180 into a 210 HP Ogerpon ex is
a two-shot back. `attack_ledger.py` also refutes the obvious policy fix — the
suspicion that Punk Up over-feeds the formula by loading our own Active. Over
86 Myriad Leaf Showers aimed at our Active Grimmsnarl, our Energy count on it
was **2.00 on average and 2 at the median**, i.e. exactly the Shadow Bullet
cost and never more. There is nothing to take away.

**Turn order is no longer the gap.** Pooled across all opponents we are 0.680
going first and 0.579 second, but against 950+ opponents the split is 0.456 /
0.478. The going-second deficit that v3-v15 chased lives entirely in the weak
band.

Repairing every one of Ogerpon, Hydrapple and Lopunny to a 0.5 would be worth
about **+47 Elo** in that band. That is the whole identified matchup budget,
and it is the reason the target in the user's brief (1200, about rank 10 of
6859) is not reachable on this 60-card list. What is reachable is roughly
1050-1120.

## 3. The one lever that had never been tested

The endgame diagnosis' section 6 is the starting point: three versions shipped
a search layer and **none of them ever changed a played action**.

| version | machinery | overrides |
|---|---|---|
| v7 | real-engine branching + 381-column value model | 1 in 1,706 decisions |
| v11 | belief search, 16.6% of decisions searched | 0 |
| v27 | adaptive belief search, 301 considered / 23 searched | 0 |

Each of them searched *past* the end of our own turn. That forces two things
we cannot do well: a belief over the opponent's hidden cards, and a learned
value head to score the resulting leaf. v27's own gate list — mirror-only, turn
>= 5, top-3 candidates, rank margin 3.0, belief confidence 0.55, mean value
gain 0.04, and "no candidate may be worse than v22 in any sampled world" — is
what a designer writes when they do not trust the estimate, and it is why the
layer was inert.

`turn_search.py` inverts the design.

* **It stops at the end of our own turn.** Inside our own turn the opponent
  does not draw, play or attack, so their hidden cards cannot change the line;
  the only hidden information that matters is the order of *our* deck, and our
  60-card list is known exactly. The determinization is honest rather than
  believed, and the opponent's hidden zones are filled with inert Basic Energy.
* **The leaf is scored on prizes, not on a model.** No value head is involved.
* **The authority is narrow and hard.** The layer may overrule the v22 ranker
  only when some other first action leads to a complete line that wins the game
  or takes strictly more prizes this turn than any line starting from the
  ranker's own action, and only when the same first action wins on every
  determinization sampled. Everything else, every exception, and any budget
  pressure returns v22's answer unchanged.

The engine work is done through the official `cg.api.search_begin` /
`search_step`, which is the same facility v11 and v27 used, so nothing new is
required of the Kaggle runtime.

### Two implementation faults that had to be found first

Both of these produce a silently inert layer, which is exactly the failure mode
of the three previous attempts, so they are recorded here.

1. **Depth-first search with a node cap is not a search.** The first build
   explored one arbitrary corner of the tree: at 800 nodes it found a better
   prize line on 2.6% of turns, and raising the budget 15x to 12,000 nodes
   changed that number to 2.6%. Replacing it with a breadth-limited beam that
   always keeps every attack and END option found more with *fewer* nodes.
2. **A plain top-k beam prunes its own baseline.** Judging "is another opening
   better than the ranker's opening" requires the ranker's opening to survive
   to a complete line, and a value-ordered beam dropped it in **24%** of
   searches, which the layer then correctly but uselessly declined. Reserving a
   beam slot per distinct first action cut that to 0.5%.

A third fidelity fault is fixed in the shipped build: inside the search,
multi-select decisions were taking the offered maximum, while v22 actually
trims Punk Up to a budget and Poffin to bench space. The search was modelling
an agent we do not ship. It now calls the rule policy for multi-picks, with the
policy's module-level per-turn caches snapshotted and restored so the search
leaves no trace on the live game.

## 4. Offline evidence that there is room to find

`probe_turn_search.py`, 45 stored ladder games, 233 of our turns. Paired
design: determinize once, open one search tree, walk it once following the v22
agent action by action, then enumerate the same tree. Both walks descend from
the same root, so the difference is not a shuffle difference.

```
our turns searched: 233   (1.98s per turn, 0 positions unopenable)
turns where a better line exists:            104 (44.6%)
  ... strictly more prizes this turn:         11 (4.7%), 13 extra prizes
  ... same prizes, more damage:               41 (17.6%)
  node-budget truncations:                     0
```

4.7% of our turns had a line taking a prize v22 did not take, on the board v22
actually had. At ~5.2 own turns per game that is about **0.29 extra prizes per
game**, against a six-prize game and an average 3.22 prizes still on our side
in a loss.

That is a small number and it is stated as a small number. It is also the first
non-zero search lever in this line: v18's two guards bound 0 times in 33 games,
and v7/v11/v27's search bound 0-1 times.

## 5. First ladder-proxy verdict: the layer bound, and it was slightly harmful

320-game local mirror arena, vfinal versus v22, seats alternating
(`arena_mirror_vfinal_v22.json`):

```
vfinal 153-167   win rate 0.478   Wilson 95% [0.424, 0.533]
  as first seat  86-74 (0.538)
  as second seat 67-93 (0.419)
search: 11,214 decisions considered, 480 overrides (4.3%), 0 errors,
        18,091 s over 320 games = 57 s per game against a 600 s bank
```

So the layer is no longer inert - it is the first search build in this line
with a real footprint - but the point estimate is 2.2 points *below* v22 and
the interval covers zero. On this evidence it must not ship.

## 6. Why, measured rather than guessed

Two explanations fit: the extra prize is an artifact of the determinized deck
order, or the extra prize is real but arrives at the *end* of a line while the
agent only plays the line's first action and then hands the turn back to the
greedy ranker. `probe_commitment.py` separates them on 90 stored games and 465
of our turns, using the same paired tree as the offline probe:

```
turns where the layer would have overridden:  23
  the hand-back turn collects the whole prize   9  (39.1%)
  the hand-back turn collects nothing extra    14  (60.9%)
  the hand-back turn is worse than greedy       0  ( 0.0%)

prizes over those 23 turns: greedy 10, hand-back 19, line-if-committed 36
mean damage delta hand-back - greedy: -73.5
best-line depth: mean 10.7, median 10
```

The prize is real: committing the line is worth 36 prizes where the ranker
takes 10. But an override that is not committed collects barely half of that
**and costs 73 damage a turn**, because the opening it plays is an opening the
ranker would not have chosen and the ranker then walks somewhere else. A line's
first action without its continuation is worse than not overriding at all.
That is the mechanism behind the 0.478, and it is a design fault, not a
refutation of the search.

The fix is commitment: once the layer overrules, it plays the whole line to the
end of the turn. Replay is by option *signature*, not index - our deck was
shuffled differently in reality, so the same card sits elsewhere - and the plan
is abandoned the moment a step does not resolve, or the turn changes. Steps the
opponent takes inside our turn are dropped from the plan, because the live
agent is never shown them.

### Three more faults that only a committed plan exposes

Committing surfaced a class of bug the uncommitted layer could hide, because an
uncommitted override only has to name *one* option correctly.

3. **`Option.cardId` is never populated.** Over one whole stored episode, 1,993
   options and not one carried it - an option is named purely by
   `(area, index, playerIndex)`. The first commitment build therefore dropped
   `index` from the signature to survive a reshuffle and, with no `cardId` to
   fall back on, collapsed all 1,181 CARD options in that episode to a handful
   of identical signatures. It replayed whatever happened to sit first.
   Measured: **91-229, 0.284** over 320 games. That run measures the bug, not
   the design. The fix resolves the position back to a card through the state -
   by `serial` for anything already in play, by card id for anything coming out
   of the deck.
4. **Prize cards are face down.** Taking a prize is a TO_HAND selection over
   six unknown cards; the determinization invents identities for them and the
   real ones never match, so *every* plan that took a prize - which is every
   plan the layer exists to play - was abandoned on its last step. Prizes carry
   no identity in the signature at all; they are interchangeable.
5. **The opponent is asked things inside our turn.** Those selections are in
   the searched line but the live agent is never shown them, so they are
   dropped from the plan.

After all five: 13 plans over 12 games, 57 replayed steps, **2 abandoned**,
both on a genuine board divergence (the live turn was asked a MAIN where the
plan expected a switch), which returns the turn to v22.

## 6. Reproduce

```powershell
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_vfinal\dump_meta_decks.py
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_vfinal\attack_ledger.py
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_vfinal\read_ledger.py
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_vfinal\where_we_lose.py
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_vfinal\probe_turn_search.py `
    --games 45 --nodes 20000 --branch 30 --beam 48 --seconds 40
.\.venv\Scripts\python.exe .\scripts\arena_grimmsnarl.py `
    --a agents\grimmsnarl\grimmsnarl_ml_vfinal `
    --b agents\grimmsnarl\grimmsnarl_ml_v22 --games 320 --workers 8 `
    --report experiments\grimmsnarl_vfinal\arena_mirror_vfinal_v22.json
```
