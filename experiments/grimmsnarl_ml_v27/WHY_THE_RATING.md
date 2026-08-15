# What produces a rating on this ladder, and what we would have to change

Follow-up to `LADDER_ANALYSIS.md`, which established that v22, v26 and v27 are
one policy (16 differing decisions in 264 games). That left the real question
open: if the code is the same, what *does* separate 853.3 from 1020.4, why is
the top of the leaderboard stable if the number is this noisy, and what would
actually have to improve.

Reproduce:

```bash
python scripts/analyze_grimmsnarl_rating_mechanics.py
python scripts/analyze_kaggle_leaderboard_persistence.py
python scripts/analyze_grimmsnarl_true_strength.py
python scripts/analyze_grimmsnarl_pairing.py
python scripts/analyze_grimmsnarl_equilibrium.py
```

Outputs: `rating_mechanics.json`, `leaderboard_persistence.json`,
`true_strength.json`, `pairing.json`, `equilibrium.json`, and a `logs_*.txt`
for each.

---

## 0. The update rule, recovered

The per-game update is plain Elo with a decaying K:

```
delta = K(n) * (result - 1 / (1 + 10 ** ((opponent - ours) / 400)))
```

`K(n)` fitted on the 445 stored games is identical across all eleven runs and
replays every final rating to within 16 points (usually under 5):

| game | 1 | 2 | 3 | 5 | 10 | 20 | 34 |
|---|---|---|---|---|---|---|---|
| K | 215.9 | 199.4 | 173.6 | 122.6 | 61.8 | 30.8 | 17.8 |

**Games 1-10 carry 63.4% of all the K a 34-game run will ever have.**

## 1. The opening was not the problem

v27's first ten games: **7-3 against an opponent mean of 770, rating 868.0 at
game 10** - the fourth-best opening of eleven runs, and against the
second-strongest opening draw. v22_a, the eventual 1000.6, was at 873.7 after
ten games against a *weaker* draw (690).

Nor did v27 lose to weak opponents. Its 13 losses averaged an opponent rating
of 818; its 21 wins averaged 771. Only 2 of 13 losses were to an opponent under
750. The hypothesis "we lost early to weak opponents" is false on both halves.

What actually happened is games 10-18, which went 2-7 and took 868.0 down to
772.9. Ten of the thirteen losses came in the second seat: **v27 alone was
14-3 (0.824) going first and 7-10 (0.412) going second.**

## 2. Rating is capped by the draw, and the draw follows the rating

Pairing is rating-proximate with a slope under one:

```
opponent = 20.3 + 0.931 * our rating        (n = 445, pearson 0.692)
```

| our rating | 800 | 900 | 1000 | 1100 | 1200 |
|---|---|---|---|---|---|
| pool supplied | 765 | 858 | 952 | 1045 | 1138 |
| offset | -35 | -42 | -48 | -55 | -62 |

The pool always sits ~40-60 points below us, so a converged run needs about
0.56 just to stand still. Combined with the K decay this is a trap: the
opening decides which pool the rest of the run is sampled from, and by game 15
there is no longer enough K to leave it.

Stated as correlations over the eleven runs:

* corr(rating at game 10, opponent mean afterwards) = **0.905** (p = 0.0001)
* corr(rating at game 10, final rating) = **0.683** (p = 0.021)
* corr(win rate after game 10, final rating) = **-0.391** (p = 0.23)

The win rate after the opening does not predict the finish. It is slightly
*negative*, because a good opening buys a harder pool.

This is also the whole 08-13 versus 08-15 story, and it corrects the reading in
`LADDER_ANALYSIS.md` section 3. Pooling the v22-equivalent policy by day:

| ladder | record | win rate | opponent mean | implied 34-game rating |
|---|---|---|---|---|
| 08-13/14 | 120-70 | 0.632 | 915 | 995 +/- 63 |
| 08-15 | 49-25 | **0.662** | **744** | 860 +/- 60 |

The win rate on 08-15 was *higher* (Fisher p = 0.67). The entire difference is
the opponent mean, and on 08-15 33.6% of opponents were rated under 700 against
11.1% on 08-13. The field did not get better at playing; the active pool got
diluted with fresh submissions. "-100 Elo/day of field difficulty" was the
wrong mechanism for the right observation.

v27's own draw (mean 789) caps a 90%-winning agent at 971.

## 3. The top of the leaderboard is skill, not luck

Three full snapshots, 6.3k-6.8k teams, 2026-08-05 / 08-07 / 08-14.

* Spearman between 08-05 and 08-14: **0.881** over 6196 teams in both.
* Teams ranked 1-10 on 08-05 kept **84%** of their edge over the field nine
  days later; 70% were still in the top 50.
* Variance decomposition: field sd 185.4, sd of a team's own nine-day change
  89.0, so sd(noise) = 62.9 and sd(skill) = 174.4. **Reliability 0.885** -
  which matches the observed Pearson of 0.885 exactly.

So ~88% of leaderboard position is real and ~12% is noise. Noise of sd 63 does
not carry anyone from rank 600 to rank 50; that gap is 190 points, three sd.

Two caveats that inflate the top and are worth exploiting rather than envying:

* **The displayed score is the maximum of a team's two live slots** - verified
  on 60 of 60 top teams. With sd 63, the better of two independent runs of the
  *same* agent is worth **+35.5 points** in expectation; three runs, +68.
  The median top-60 team's two slots differ by 92 points.
* The #1 team is a different team in each of the three snapshots (1300.3,
  1158.9, 1254.5), so the very top is genuinely churny even though the top
  *band* is not.

## 4. Our actual strength, stripped of the draw

40% of our 445 opponents can be matched to a team on the newest leaderboard,
giving a settled strength estimate instead of the number they carried at the
time. Our record against it:

| settled opponent | n | record | win rate | Wilson |
|---|---|---|---|---|
| 0-700 | 35 | 33-2 | 0.943 | [0.81, 0.98] |
| 700-850 | 56 | 40-16 | 0.714 | [0.59, 0.82] |
| 850-1000 | 59 | 33-26 | 0.559 | [0.43, 0.68] |
| 1000-1100 | 21 | 7-14 | **0.333** | [0.17, 0.55] |
| 1100+ | 5 | 3-2 | 0.600 | [0.23, 0.88] |

Logistic fit: `logit P(win) = 5.674 - 0.00586 * settled_score`
(se 0.00127, z = -4.60, n = 176). A pure Elo agent has slope -0.00576, so
**we are 1.02x as steep - rating-consistent, not abnormally fragile.**

Solving the fixed point `r* = pool(r*) + 400 log10(w/(1-w))`:

> **r\* = 970** (rank ~175 of 6812), against a pool of 923 at a 0.566 win rate.
> Sensitivity over +/-1 sd of the slope: 953 to 995.

That single number reconciles everything. v22's 1000-1020 was **above** our
true level, v27's 853 **below** it, and both sit inside the +/-63 run noise.
There is no v22-to-v27 regression to explain.

## 5. What each rank actually demands

Applying a uniform logit shift until r\* lands on the score a rank required on
08-14:

| target rank | rating | shift needed | win rate vs a 950 pool |
|---|---|---|---|
| 200 | 956.0 | -14 | 0.508 (now 0.528) |
| 150 | 978.1 | +8 | 0.540 |
| 100 | 1005.4 | **+36** | 0.579 |
| 50 | 1046.5 | **+78** | 0.636 |
| 25 | 1089.8 | +122 | 0.693 |
| 10 | 1158.5 | +192 | 0.771 |
| 1 | 1254.5 | +289 | 0.855 |

And what the known holes are worth, at 0.55 in each cell:

| matchup | n | share | win rate | delta Elo |
|---|---|---|---|---|
| Ogerpon | 23 | 5.2% | 0.130 | +16.4 |
| Hydrapple ex | 12 | 2.7% | 0.333 | +4.4 |
| Mega Lopunny / Froslass | 30 | 6.7% | 0.467 | +4.2 |
| **all three, through the fixed point** | 65 | 14.6% | | **r\* 970 -> 995, rank 123** |

So the full matchup repair is worth about a third of what rank 50 needs, and
it is still the largest single identified item. Rank 50 requires a general
+78, not a patch.

## 6. Why the ladder cannot see our work

* A 34-game run of the pooled policy has a **90% spread of 215 points**
  (bootstrap, sd 65). Nothing under ~130 points is measurable in one run.
* Of 176 strength-graded games, only 26 (14.8%) were against a settled-1000+
  opponent. **v27 met zero in its 18 graded games.** The matchups that set the
  ceiling are essentially unsampled during the runs used to judge a version.
* Our exposure to strong opponents only rises if our rating rises, so the
  measurement improves exactly when we no longer need it.

## 7. What follows

1. **Put the same agent in both slots when the goal is a score** (+35.5
   expected, free), and both slots on challenger-plus-control when the goal is
   a measurement. Never both at once, and never compare across days.
2. **Stop treating a ladder run as the fitness function.** Nothing below ~130
   points is visible in 34 games. Offline evaluation against strong opponents
   is the only instrument with the resolution we need.
3. **Ogerpon first** among matchups: 3-20 across all versions, 5.2% of games,
   +16 Elo alone and the only double-digit item on the board.
4. **The second seat on the current field**: v27 was 14-3 first, 7-10 second.
5. Everything already ruled out stays ruled out - the belief search, the
   restored Froslass veto, runtime, and the K-schedule opening are all not the
   explanation.

The honest summary: our agent is a ~970 agent. v22 was a lucky draw from that
distribution and v27 an unlucky one. Reaching rank 50 needs +78 Elo of real
strength, of which the known matchup holes supply at most +25.
