# Why the rating keeps falling, and what a "current-meta retrain" can and cannot buy

Diagnosis written 2026-08-16 JST, against the 08-16 14:15 leaderboard snapshot
(6,841 teams) and 552 stored ladder games across v22 … v29 plus the fresh
v22 rerun `55542305`.

Reproduce (all scripts in this directory, run from the repository root):

```powershell
.\.venv\Scripts\python.exe .\scripts\fetch_kaggle_top100_snapshot.py --top-n 60 --output-root .\.tmp\lb_20260816
.\.venv\Scripts\python.exe .\scripts\fetch_submission_logs.py --submission 55542305 `
    --output data\runs\grimmsnarl\20260816_grimmsnarl_sub55542305 `
    --deck-dir agents\grimmsnarl\grimmsnarl_ml_v22
.\.venv\Scripts\python.exe .\scripts\build_grimmsnarl_version_games.py `
    --output experiments\grimmsnarl_endgame_20260816\version_games.csv
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_ml_v6\measure_refresh_opportunity.py `
    --submissions .\.tmp\lb_20260816\latest\public_submissions_top60.csv `
    --scratch .\.tmp\deck_probe_0816 --top 60 --out .\.tmp\refresh_opportunity_0816.json
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_endgame_20260816\v29_strength.py
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_endgame_20260816\settled.py
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_endgame_20260816\drift2.py
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_endgame_20260816\pace.py
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_endgame_20260816\vs_current_meta.py
.\.venv\Scripts\python.exe .\experiments\grimmsnarl_endgame_20260816\meta_families.py
```

`logs_*.txt` beside each script holds the output quoted below. The
`measure_refresh_opportunity` sweep is rate-limited by Kaggle after roughly 35
submissions (HTTP 429); the 31 replays it did fetch are cached in
`deck_probe_cache/` so the meta tables replay offline.

---

## 0. Summary

Four separate things are being read as one "the rating is going down", and
they need different responses.

| # | Claim | Verdict |
|---|---|---|
| 1 | Our team's displayed score fell 968.7 → 874.7 | **True, and self-inflicted.** We truncated the converged v28 slot with two fresh 600-start submissions. |
| 2 | The same v22 code now scores ~130 lower than on 08-13 | **Mostly a rating-scale artifact.** Controlled on a common-date opponent rating the calendar effect drops from −55.6 to −12.6 Elo/day and stops being significant. Also n=25, unconverged. |
| 3 | The deck has stopped being competitive | **True and now total.** 0 of 62 decks observed among the current top-35 teams' games contain Grimmsnarl ex at all. |
| 4 | "Retrain V22-style on current top data" | **Not executable for this deck.** Same-deck teacher episodes currently on offer: **0**. |

The single largest *identified* strength gap is unchanged and now worse:
Conkeldurr + Hydrapple + Mega Lopunny/Froslass are **51.6% of the current top
meta**, and our all-version record in those three cells is 20-34 (0.370), with
Conkeldurr at **0-4** and never modelled.

---

## 1. The displayed drop is mostly slot management, not policy

Every team has exactly 2 live submission slots (verified on 30 of 30 top-30
teams, `public_submissions_top60_20260816.csv`), the board shows the **max** of
them, and a new submission truncates the **oldest** live slot.

Our team `16487165`:

| date | board rank | board score | which slot |
|---|---|---|---|
| 2026-08-15 23:16 | 175 / 6827 | **968.7** | v28 `55526859` |
| 2026-08-16 14:15 | 487 / 6841 | **874.7** | v22 rerun `55542305` |

v28's 968.7 no longer exists. Submitting v29 (`55530747`) and then the v22
rerun (`55542305`) pushed it out. The two live slots are now 853.4 and 874.7.

This is a structural cost of the iteration loop itself: with only two slots and
truncate-oldest, **every submission destroys the older of the two converged
scores**, and nothing since v22 has been better than v22 by more than the
~130 Elo single-run noise floor (`experiments/grimmsnarl_ml_v27/WHY_THE_RATING.md`).
13 of the current top-30 teams carry their best score on their *older* slot,
which is what "stop resubmitting once the pair is good" looks like from
outside.

## 2. Per-version strength, opponent-mean adjusted

`logs_v29_strength.txt`. Rating converges to
`mean(opponent rating) + 400·log10(w/(1−w))`, so the raw final rating is not
comparable across runs with different draws.

| run | n | record | win rate | opp mean | implied strength | final |
|---|---:|---|---:|---:|---:|---:|
| v22_a | 45 | 33-12 | 0.733 | 836.4 | 1012.2 | 1000.6 |
| v22_b | 38 | 27-11 | 0.711 | 884.8 | 1040.8 | 1018.6 |
| v22_c | 57 | 31-26 | 0.544 | 1000.1 | 1030.7 | 1020.4 |
| v22_d | 50 | 29-21 | 0.580 | 912.4 | 968.5 | 952.8 |
| v25_a | 51 | 27-24 | 0.529 | 896.2 | 916.7 | 910.7 |
| v27 | 34 | 21-13 | 0.618 | 789.0 | 872.3 | 853.3 |
| v28 | 35 | 24-11 | 0.686 | 860.5 | 996.0 | 975.7 |
| v29 | 47 | 30-17 | 0.638 | 758.3 | 857.0 | 847.5 |
| **v22_e (08-16 rerun)** | **25** | **15-10** | **0.600** | **809.6** | **880.0** | **874.8** |

Two cautions on reading v22_e as "v22 is now only an 880 agent":

* **It is 25 games.** Across the 13 stored runs, the rating after 20 games
  ranged 753 → 1099 and moved by −79 … +103 afterwards (`logs_pace.txt`).
* Its draw is 190 points weaker than v22_c's.

## 3. Field drift versus rating deflation

`logs_drift2.txt`. Logistic on outcome with opponent rating and turn order held
fixed, plus a calendar-day term. Coefficients converted to Elo with each fit's
own rating slope.

| opponent measured by | subset | n | day effect | p |
|---|---|---:|---:|---:|
| rating **at pairing time** | all versions | 552 | **−55.6 Elo/day** | 0.0056 |
| rating **at pairing time** | v22 code only | 215 | −43.7 Elo/day | 0.114 |
| **settled** score on the 08-16 board | all versions | 174 | **−12.6 Elo/day** | 0.59 |
| **settled** score on the 08-16 board | v22 code only | 51 | −21.0 Elo/day | 0.39 |

Switching the opponent onto a single common rating scale removes ~77% of the
apparent decline and all of its significance. The mechanism is that the pool
keeps absorbing fresh 600-start submissions from strong teams, so a *displayed*
800 today is a stronger agent than a displayed 800 on 08-13.

Honest limits: the matched subsample is 174 of 552 games, the settled-rating
95% CI on the day term is roughly [−59, +34] Elo/day and therefore does not
*exclude* −55.6, and matching only succeeds for opponents who have not
resubmitted since. The claim supported is "the evidence for genuine
meta-driven decay is weak", not "there is provably none".

## 4. The deck is gone from the top of the meta

`logs_meta_families.txt`, from 31 cached replays of the current top-35 teams'
latest submissions — 62 sixty-card decks, 27 distinct lists, mostly from the
1050+ band because pairing is rating-proximate.

| family | share of the current top meta | distinct lists | our all-version record |
|---|---:|---:|---|
| Dragapult | 24.2% | 7 | 29-17 (0.630) |
| **Conkeldurr** | **21.0%** | 7 | **0-4** |
| Hydrapple ex | 17.7% | 4 | 4-10 (0.286) |
| Mega Lopunny / Froslass | 12.9% | 1 | 16-20 (0.444) |
| Alakazam | 8.1% | 3 | 78-35 (0.690) |
| Mega Lucario | 8.1% | 2 | 35-9 (0.795) |
| Kangaskhan / Crustle | 4.8% | 1 | 27-11 (0.711) |
| Team Rocket's Mewtwo ex | 1.6% | 1 | 3-2 |
| Arboliva ex | 1.6% | 1 | 0-1 |
| **Grimmsnarl ex (card 648) in any list** | **0.0% (0 / 62)** | 0 | — |

Trajectory of the archetype: 51% of the top 50 on 08-02, 6 of the top 40 on
08-06, 2 of the top 60 on our exact list on 08-14, **0 of 62 today**. Even the
1220.2-rated pilot our v22 ranker is pinned to (`16371703`) has resubmitted and
now sits at rank 75 / 1024.6.

Conkeldurr is a genuinely new archetype: 7 distinct lists across 13
observations, so it is a field-wide adoption and not one team's pet deck. We
have met it 4 times and lost 4 times, and the 08-16 loss was a blowout — 6 of
our prizes still on the board against their 1.

**What we are actually paired against is not this meta.** Over 552 stored games
our opponents were 25.7% Grimmsnarl mirror and 20.5% Alakazam — 46% of our
games are archetypes that are 0–8% of the top. We win a lot of games that do
not buy rating, and lose the cells that do.

## 5. Consequence for the proposed "current-meta V22-style retrain"

The v22 method is: take a strong *same-deck* pilot, imitate it, condition the
ranker on that pilot. `measure_refresh_opportunity.py` reports the input that
method needs:

```
"deck_hash": "9714ab5c3996f6cc",
"same_deck_new_episodes_available": 0
```

There is no fresh same-deck teacher to imitate. Retraining the existing
pipeline on "current top data" is therefore not a thing that can be done while
keeping the deck. The three executable variants are, in increasing cost:

1. **Retrain on the archive we already have.** Cheap, but the archive is the
   same 08-05/08-06 corpus that produced v22 through v29; the frozen selection
   is exactly what has been mined for 29 versions and
   [[grimmsnarl-imitation-saturated]] recorded that 63 new features and every
   teacher pin bought nothing.
2. **Change the deck and imitate a current top pilot.** This is what the user's
   plan reduces to once the same-deck constraint is dropped. Precedent: we did
   exactly this on 2026-08-02 for Mega Lopunny — 386 verified teacher
   trajectories, 32,497 decisions from the rank-1 pilot — and the resulting
   agent laddered at **767.1**. Our Alakazam line's best is 916.9. Grimmsnarl
   v22's ~1020 is still our best result on any deck.
3. **Stop imitating and learn from outcomes.** See §6.

## 6. Multi-turn lookahead has never actually been given authority

The user's reading that the agent "only predicts one move" is correct in
effect, but the reason is not that lookahead is missing — it is that every
version that shipped it clamped it to inert.

| version | search machinery | how often it changed the played action |
|---|---|---|
| v7 | real-engine branching of the top-3 MAIN candidates from turn 5, plus a 381-column public-state value model (test AUC 0.785 pooled, 0.858 on turn 9+) | **1 override in 1,706 decisions** |
| v11 | belief search, 16.6% of decisions searched on 3.4% of the 600 s bank | **0 overrides** in the offline probe |
| v27 | adaptive belief search, 301 considered / 23 searched / 336 branches | **0 overrides** |
| v29 | explicitly "one-ply arithmetic planner, no belief search" | n/a |

So the components exist — engine rollout, a trained value head, a per-episode
600-second budget that has never been more than ~3% used. What has never been
tested is a version where the value model is allowed to *disagree* with the
imitation ranker. That is a real untested lever, and it is the only lever left
that does not need teacher data we cannot obtain.

Two caveats before betting on it. The value model's turn-band AUC is 0.63 for
turns 1-4, so early-game authority is not supported by the fit we have. And the
single-run noise floor is ~130 Elo, so a ladder run cannot verify the change;
it has to be validated offline, on held-out games, against the played action.

## 7. The endgame is a stopping rule, and it is worth more than any model change

Goal for the remaining half-day-to-two-days: maximise the final displayed
score. Under that goal slot management dominates modelling, because a single
run cannot resolve anything under ~130 Elo but the slot rule is worth tens of
points with no variance.

### 7.1 Live state (`our_slots.py`, 2026-08-16 ~15:05 JST)

```
US: rank 438 / 6841  score 886.2  sub 55542305
live public submissions: 2
  55542305  886.2   <- v22 rerun, newest, still playing (874.7 -> 886.2)
  55530747  847.5   <- v29, oldest, frozen at its stored final
```

Each submission plays a bounded number of rated games (34-57 across our 14
stored runs) and then goes idle at a frozen score. v29 has stopped; v22_e is
still moving.

### 7.2 The rule

With slots `(old, new)` the board shows `max(old, new)`, and submitting
replaces the pair with `(new, fresh)`. So the post-submission floor is `new`:

> **Submit only while the newer live slot is at least as high as the older
> one.** Then the displayed score can never fall. Submitting when
> `new < old` throws away `old − new` points.

Right now `new = 886.2 > old = 847.5`, so a submission is free of downside: it
truncates the dead 847.5 slot, freezes v22_e at its final, and buys a fresh
draw.

### 7.3 What it is worth (`logs_reroll_value.txt`)

200,000 trials per cell, draws ~ N(fixed point, 63), ranks read off the live
board.

| policy | fixed point 890 | fixed point 920 | fixed point 950 | P(worse than now) |
|---|---:|---:|---:|---:|
| do nothing | 886.2 (rank 446) | 886.2 | 886.2 | 0% |
| burn both slots at once | 925.4 (rank 298) | 955.6 (rank 212) | 985.5 (rank 141) | 8.8-22.8% |
| **reroll, 1 cycle** | 913.3 | 931.9 | 955.2 | **0%** |
| **reroll, 2 cycles** | 919.3 | 943.9 | 975.3 | **0%** |
| **reroll, 3 cycles** | 920.1 (rank 319) | 946.2 (rank 241) | 979.9 (rank 154) | **0%** |
| reroll, 4+ cycles | 920.1 | 946.6 | 980.8 | 0% |

Three cycles capture essentially all of it, and each cycle costs one
convergence window (2-12 h). Burning both slots at once has a slightly higher
mean when the fixed point is well above 886 but carries real downside; the
stopping rule gets ~95% of the upside with none.

For scale: the entire Ogerpon + Lopunny + Hydrapple matchup repair was priced
at +19.9 Elo pooled in `experiments/grimmsnarl_ml_v28/LADDER_ANALYSIS.md`. The
reroll is worth +34 to +94 and needs no code.

### 7.4 Plan

1. Let v22_e (`55542305`) finish its game allocation and freeze.
2. Resubmit **v22** — `artifacts/grimmsnarl_ml_v22_submission.tar.gz`,
   sha256 `c2b1097e…fb33b`, verified against
   `agents/grimmsnarl/grimmsnarl_ml_v22/metadata.json`. v22 is the
   best-evidenced agent we have: 190 stored games, implied strength 1008.8,
   against v28's 996.0 over 35.
3. When the fresh slot freezes, apply the rule: submit again only if it landed
   at or above the slot it would truncate. Otherwise stop permanently.
4. Stop submitting once less than ~12 hours remain: a 600-start that cannot
   converge would replace a converged slot.
5. Do not use a slot on a challenger. Nothing since v22 has separated from it,
   and a challenger that lands low both wastes a cycle and lowers the floor.

## 8. Open items this diagnosis did not settle

* The competition deadline is not in any artifact in this repository and the
  Kaggle page is JS-rendered, so the time budget in §7 is unparameterised.
* `measure_refresh_opportunity` was rate-limited after ~35 submissions; ranks
  36-60 of the current board are unprobed, so "0 same-deck teachers" is
  established for the top 35, not the top 60.
* Whether teacher `16371703` switched decks or merely resubmitted the same list
  is unprobed for the same reason (HTTP 429).
* Conkeldurr is 21% of the top meta and we have four games against it. Its
  mechanism against us is unanalysed.
