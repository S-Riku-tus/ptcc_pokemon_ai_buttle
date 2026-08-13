# Grimmsnarl: what is actually between v8 and 1100

Written 2026-08-08 from the same artifacts as `RESULTS.md`, plus three new
passes: `matchup_ceiling.json`, `rating_levers.json`, `alakazam_gap.json` and
`alakazam_second_tempo.json`.

---

## 0. A correction to RESULTS.md §2

`RESULTS.md` concluded the Alakazam matchup was "not supported" as a defect. That
conclusion used the wrong baseline. It asked *is Alakazam worse than our other
matchups, inside our own 50 games* — a test with almost no power, which returned
p = 0.18. The sharp test is against the field playing the identical 60 cards,
where the baseline has 326 games instead of 39.

| | games | record | win rate |
| --- | ---: | ---: | ---: |
| field vs Alakazam | 326 | 242-84 | **0.742** |
| v8 vs Alakazam | 11 | 4-7 | 0.364 |
| field vs Alakazam, **going second** | 162 | 114-48 | **0.704** |
| v8 vs Alakazam, **going second** | 5 | **0-5** | **0.000** |

Ours against the field, going second: **Fisher p = 0.0028**. Going first,
p = 0.62 — normal. And the field's *own* first/second split inside this matchup
is p = 0.129, i.e. **the field does not collapse on the draw here**, so our 0-5
is not the general turn-order effect.

The mirror is the control that rules out "we are simply weaker": there we are
11-5 overall and 3-3 going second against the field's 0.564, p = 1.00.

The Alakazam matchup is a real, matchup-specific, seat-specific defect and it is
the largest single number in this whole analysis.

### Mechanism: the attacker never arrives

Alakazam, going second (`alakazam_second_tempo.json`):

| | field (n=162) | v8 (n=5) |
| --- | ---: | ---: |
| first attack, turn (median/mean) | 4.0 / 4.36 | **8.0 / 7.20** |
| Grimmsnarl ex first in play | 6.0 / 5.55 | **10.0 / 8.80** |
| bodies in play at turn 4 | 6.0 / 5.21 | **4.0 / 3.60** |
| own turns in the game | 5.0 / 5.37 | 5.0 / 5.00 |
| prizes taken | 4.13 | **0.80** |

The game lasts about five of our turns. The field is attacking on turn 4 and has
5.2 bodies down by then. We attack on turn 8 — after the game is over — with 3.6
bodies. We take 0.8 prizes a game where they take 4.1.

And it is specific, not our general slowness on the draw. v8's own first attack
going second is turn 5.0 against everything else; the field's is 4.0 against
everything and **4.0 against Alakazam**. The field speeds up into this matchup.
We lose three more turns.

---

## 1. Is 1100 reachable on this deck? Yes — but rating is not win rate

**6 of the 21 archived pilots playing deck hash `9714ab5c3996f6cc` are rated
above 1100**, topping out at 1220.2. The deck is not the ceiling.

But the thing that would normally follow — "so raise the win rate" — does not
hold here. Across those same 21 pilots:

```
spearman(archived win rate, leaderboard rating) = -0.138,  p = 0.552
spearman(games in archive,   leaderboard rating) = +0.405,  p = 0.069
```

The 1220.2 pilot wins 60.0%. The 1073.9 pilot wins **70.6%**. Win rates run
0.541-0.706 with no relation to rating at all. Volume and opponent pool carry
more of the rating than the win rate does.

**Consequence for goal-setting:** "+100 rating" is not a well-posed policy
target. The well-posed one is **win rate against opponents rated ≥1000**, where
v8 is 10-18 (0.357). That is the number a policy change can move and the number
the ladder converts into rating.

---

## 2. Where skill buys anything at all on this deck

| opponent family | share of meta | elite | rest | elite − rest | v8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grimmsnarl (mirror) | 40.7% | 0.605 | 0.596 | **+0.009** | 0.688 (16) |
| Mega Lopunny / Froslass | 10.8% | 0.509 | 0.500 | **+0.009** | 0.400 (5) |
| Kangaskhan / Crustle | 9.9% | 0.707 | 0.613 | +0.095 | 0.667 (3) |
| Alakazam | 8.9% | 0.722 | 0.750 | **−0.028** | 0.364 (11) |
| Ogerpon | 7.6% | 0.185 | 0.295 | **−0.111** | 0.000 (2) |
| Team Rocket's Mewtwo ex | 4.0% | 0.800 | 0.581 | **+0.219** | — (0) |
| Dragapult | 3.7% | 0.769 | 0.604 | **+0.165** | 0.667 (3) |
| Cynthia's Garchomp ex | 3.4% | 0.667 | 0.558 | +0.108 | — (0) |
| Fezandipiti ex | 2.7% | 0.571 | 0.435 | +0.136 | — (0) |
| Mega Lucario | 2.5% | 0.806 | 0.797 | +0.010 | 0.750 (4) |

Three readings, all actionable:

1. **70.6% of the meta is skill-inert.** The mirror is 40.7% of all games and a
   1100+ pilot is 0.9 points better in it than a 1050 pilot. Work spent on the
   mirror cannot pay. The same is true of Lopunny/Froslass and Lucario.
2. **Ogerpon is negative-skill.** Elite 0.185 against the rest's 0.295 — the
   better you play the worse you do, over 278 games. 7.6% of the meta is a
   near-auto-loss and trying harder makes it worse. Concede it in the model.
3. **The elite band's entire surplus lives in 23.6% of games** — Kangaskhan,
   Mewtwo, Dragapult, Garchomp, Fezandipiti. **We have 6 games total in that
   slice.** We have essentially no measurement of ourselves where skill is the
   thing that decides.

Alakazam sits oddly in this table: it is *skill-inert but high* — everyone wins
it about three games in four, and we do not. That is what makes §0 a bug rather
than a matchup.

---

## 3. Why imitation stalled at ~1080

v8's ranker is pinned to team 16494330, **rated 1077.6**. Copying a 1077 pilot
perfectly lands near 1077, and the measured fidelity says we are close to that:
Top-1 is 0.928 for the pin against 0.797-0.839 for the 1130-1220 pilots. The
better the pilot, the less imitable — so "pin a stronger teacher" does not work
either, and the pooled consensus is the ~1060 policy.

To find where a *different* policy actually exists, 36 statistics were computed
per pilot across the archive and correlated with rating, Benjamini-Hochberg
controlled (`rating_levers.json`). This inverts the method every version so far
has used: the gradient picks the target instead of us picking it first.

**The take rates five versions have optimised have no gradient at all:**
Dark attachment p=0.46, Grimmsnarl evolve p=0.23, Boss's Orders p=0.35, bench
p=0.07, supporter choice p=0.17-0.64, stadium p=0.12, attack-this-turn p=0.87.

**What does have one is a single style axis**, and v8 is at the wrong end of
every part of it:

| statistic | rho | p | elite | rest | **v8** | v8 rank of 21 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Petrel searches Boss's Orders | +0.581 | 0.0058 | 0.093 | 0.046 | **0.000** | **last** |
| dead Unfair Stamp taken | −0.588 | 0.0064 | 0.453 | 0.636 | **0.743** | near top |
| bench ≥3 at turn end | −0.564 | 0.0078 | 0.874 | 0.891 | 0.897 | 76th pct |
| retreat taken | +0.534 | 0.0127 | 0.213 | 0.171 | **0.147** | 14th pct |
| Night Stretcher taken | −0.534 | 0.0127 | 0.655 | 0.838 | 0.798 | 57th pct |
| attack taken when offered | −0.530 | 0.0135 | 0.941 | 0.957 | **0.968** | 76th pct |

Read together: **the elite reposition and keep options; v8 commits.** It never
searches a gust, retreats least of 21 pilots, attacks most, takes the dead ACE
SPEC most, and keeps the widest bench. Six statistics, one policy difference.

*A caveat worth stating.* Three more statistics clear BH — prizes per own turn
(rho −0.765), own turns per game (+0.752), turns with a prize (−0.682) — but
they are one mechanically-linked family (six prizes divided by game length) and
almost certainly an opponent-strength proxy: stronger pilots draw stronger
opponents and those games run longer. Multiplied out, total prizes per game are
3.40 / 3.58 / 3.30 for elite / rest / v8 and do not order by rating. **Do not
chase them.**

---

## 4. The plan

### Stage 0 — submit v10 as the challenger (built, unsubmitted)
It closes one of the six style-axis statistics (dead Stamp) and is worth 6
decisions in 50 games. Treat it as a **control**, not a lever: its value is that
it proves the class-escalation machinery is safe in production.

### Stage 1 — Alakazam going second (highest expected value)
8.9% of the meta, 38 points below the field, mechanism identified, and the field
gives a 162-game target profile to aim at. Not an imitation target — an outcome
target with a measurable intermediate: **first attack by turn 4-5, ≥5 bodies by
turn 4.**

Work: replay our 5 losses and a matched sample of the field's 114 wins side by
side over turns 1-4 and find the divergence. Candidates from the tempo table are
bench width (3.6 vs 5.2 bodies) and whatever is eating our early turns. Expected
value if closed to the field's rate: **+3.4 points of overall win rate**, and
more against the ≥1000 band where those opponents live.

### Stage 2 — extend the class-escalation ladder along the style axis
Same machinery as v10, one class per version, each with a pre-registered
gradient and a replay-measured override count:
1. **Petrel → Boss's Orders** (rho +0.581, v8 last of 21, 64 offers in 50 games).
   Blocked today because no panel pilot plays it on *our* boards either
   (0.016-0.109) — so this one needs a panel selected *for that behaviour*
   (16388654 at 0.161, 16371703 at 0.153, 16422241 at 0.129), not the current
   top-rated four.
2. **Retreat** (rho +0.534, v8 14th percentile, 218 offers).
3. **Attack commitment** (rho −0.530) — the riskiest; a wrong veto here costs
   prizes, so it needs the arithmetic gate of Stage 4 behind it, not a pin.

### Stage 3 — get data in the 23.6% slice that decides rating
We have 6 games against Kangaskhan / Mewtwo / Dragapult / Garchomp /
Fezandipiti. The archive has 862. Build the opponent panel from those and run
the counterfactual probe there before optimising anything, or Stage 2 will be
tuned on a meta slice that does not decide the rating.

### Stage 4 — the one lever that is not bounded by the pin's rating
Every stage above is still bounded by "what some 1100-1220 pilot did". The only
route past that is search, and it has been tried once and misdiagnosed. v7 wired
the engine's real `search_begin` / `search_step` API — 216 searches, 0 branch
errors — and got **zero overrides because the learned value head returned equal
leaf values on every candidate**. The search worked; the evaluation was flat.

Replace the learned value head with turn-level arithmetic the planner already
computes for single candidates — prizes taken this turn, whether each of our
bodies survives the opponent's best reply, whether the attacker ends fuelled —
and evaluate whole *action sequences* rather than single candidates. That is
what would catch the orderings the current single-decision planner rules cannot
see (they fired 0 times in 51 games: boss-route 0, wall-unlock 0, punk-alloc 0).

Trigger search only on the turns that matter — a lethal is
available, or the opponent's next attack kills our Active — not on low-margin
MAIN decisions from turn 5, which is the trigger that produced v7's 8 searches
a game.

### What not to do
* Do not tune the mirror (40.7% of the meta, +0.009 skill premium).
* Do not tune Ogerpon (7.6%, elite are 11 points *worse* than the rest).
* Do not add features for the per-turn take rates — six versions and 91 feature
  columns of evidence say they are saturated and none has a rating gradient.
