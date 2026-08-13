# Grimmsnarl ML v3 — route arithmetic as features, and a 0.11% dominance shell

Date: 2026-08-04
Parent: `grimmsnarl_ml_v2` (v2.1 model, ladder rating 967.4 over 59 games)

The 59-game v2 ladder analysis named three defects. This version treats them as
three different *kinds* of problem, because they are:

| defect | v2 | elite pilots | what kind of problem it is | v3 |
|---|---:|---:|---|---|
| Boss spent on a body the Bench-30 already kills | 4/7 (57.1%) | 3/35 (8.6%) | one-turn arithmetic across two selects | **2/7 (28.6%)**, from features |
| Adrena-Brain passes over a damaged Grimmsnarl ex | 77/181 (42.5%) | 11.9% | a threshold over a whole turn | **71/181 (39.2%)**, from the planner |
| Froslass evolved while Freezing Shroud is net negative | 18/22 (81.8%) | 21.4% | a judgement | **19/22, not fixed** |

**Headline, stated before the detail: two of the three behaviours moved and
imitation fidelity held.** v3 is therefore a behaviour-corrected, instrumented
challenger, not a demonstrated ladder improvement.

Everything below is measured on the same boards or the same held-out block, so
the numbers are comparable to v2.1's rather than to a fresh ladder run whose
noise (842.8 vs 804 for an identical agent) exceeds every effect here.

## 0. A measurement instrument first

`scripts/analyze_grimmsnarl_v3_behaviour.py` replays stored games decision by
decision, asks a candidate agent what it would do on each board, then advances
the game with the action that was actually taken. Teacher forcing keeps the
boards on-distribution and keeps the runtime's intra-turn history describing the
turn the replay described, so two runs differing only in a pin, a model or a
planner are compared on identical states.

It also fixes one thing the deep analysis got structurally right but measured
loosely: **Boss's Orders hands its target back as a SWITCH select (context 3),
not TO_ACTIVE (4)**, so the 4/7 figure mixed a numerator taken at the target
select with a denominator taken at the Boss play. Counted consistently over 24
gusts, v2 spends 4 of its 7 chances on a free-kill body and v3 spends 2.

## 1. The feature work, and an honest negative result

63 columns added (794 total), all from the observation alone so the corpus
builder and the runtime cannot drift:

* **Route table.** `turn_routes` enumerates the one swing available this turn:
  prizes with no gust (180 to their Active + 30 to the best Bench body) against
  prizes for every gust candidate, counting the *displaced* Active as a snipe
  target because gusting benches it. On the 89678716 board this scores v2's
  actual play — gust the 20 HP Mega Lucario ex — at 3 prizes and three other
  gusts at 4. The analysis proposed leaving Boss unplayed there; the route table
  finds a strictly better line than either.
* **Freezing Shroud ledger.** `ABILITY_POKEMON_IDS` is generated from the card
  database: 217 Pokemon with an Ability, minus the Froslass line, which is
  exactly what the card text exempts. v2 counted three card ids, so outside the
  mirror the opponent's side of the ledger read zero however many Abilities they
  had in play.
* **Return damage.** A generated `(energy cost, printed damage)` frontier per
  card plus type/Weakness/Resistance tables give
  `heals_needed(hp, threat)` — 140 HP against 180 needs two Adrena-Brain moves,
  not one — and `movable_counters`, which is also the damage the move deals, so
  a lightly damaged source is worse on both sides of the ledger.

**These features did not buy imitation fidelity.** Retrained with identical
hyperparameters, v3 stops 241 iterations earlier and scores:

| block | n | v2.1 | v3 |
|---|---:|---:|---:|
| all pilots, per-team chronological test | 34,611 | 0.8484 | 0.8455 |
| pinned pilot 16494330, offline | 1,480 | 0.9196 | 0.9142 |
| context 16 (counter source), offline test | 1,645 | 0.8705 | **0.8760** |
| context 43, offline test | 965 | 0.9772 | **0.9803** |
| elite pilot 16371703, offline test | 2,581 | 0.7854 | 0.7788 |

End-to-end, the real agent teacher-forced through the pinned pilot's later 18
games (n=1,420): **91.20% with the planner disabled** against 91.34% for v2.1 —
two decisions. Context 13 gains 2.2 points, context 15 gains 1.9, MAIN gains
0.1, deck search loses 1.5.

The reading is not that the features are wrong, it is that **the objective is
saturated against this teacher**. v2.1's residual was 4.3% same-turn ordering
plus 1.8% genuine divergence; a column describing good play cannot predict a
pilot who passes over a damaged attacker 48.8% of the time. Where features can
still change behaviour they did: the wasted-gust rate halved with the planner
rule never firing once.

## 2. The planner, and what it costs

`ml_planner` fires only where the observation *proves* the ranker's pick is
dominated on prizes taken this turn or on whether a body survives the next
attack, keeps the ranker's ordering as its tie-break inside the allowed set, and
counts every firing. Over v2's 59 games:

```
considered 4393 | heal_overrides 5 | boss_route_overrides 0
                | froslass_overrides 0 | errors 0
```

**0.11% of scored decisions.** The Alakazam line's unmeasured safety shell cost
5.16 points of agreement; this one costs 0.28 points end-to-end (91.20% → 90.92%),
and all of it is 4 context-16 decisions where it refuses the pinned pilot's own
habit. That is the shell disagreeing with the teacher on purpose, which is the
only reason to have one.

Three rules ship:

1. **Boss target.** Override only when the ranker's gust target dies to the free
   Bench-30 on its own *and* another gust takes strictly more prizes with the
   same swing. Denying an attacker or breaking a wall are reasons a prize count
   cannot see, so every other disagreement stays with the ranker. Fired 0 times:
   in both remaining wasted gusts our Active was not a fuelled Grimmsnarl, so
   the swing was not available and the rule correctly stood down.
2. **Adrena-Brain source.** Override only when an offered body of ours is knocked
   out by the opponent's best printed hit but survives the heals still live this
   turn, is worth at least as many prizes as the ranker's pick, and carries at
   least as many movable counters — so the override never deals less damage than
   the answer it replaces. Only the Active is treated as threatened; spread and
   snipes onto the Bench are not modelled, and calling a benched body "savable"
   would move counters against a threat that was never coming.
3. **Froslass guard.** Refuse only the evolve whose very next checkup knocks out
   a body of ours and none of theirs. Fired 0 times, and it is a guardrail, not
   the fix for defect 3.

The heal rule needs state no feature can carry: Adrena-Brain is once per turn
*per Munkidori*, so the planner counts activations per turn from the actions
actually taken. `main.observe_external` exposes that to the evaluators, so
teacher-forced runs advance the heal budget along the teacher's line too.

## 3. Was re-pinning the teacher the answer instead?

Tested, twice, because it is cheap — the model is pilot-conditioned, so the pin
is an inference-time argument.

| variant | decisions changed | counter-source pass | net-negative Froslass evolve |
|---|---:|---:|---:|
| v3, pinned 1077.6 (shipped) | — | 39.2% | 86.4% |
| v3, context 16 pinned to 1220.2 | 0.1% | **40.9%** | 86.4% |
| v3, every context pinned to 1220.2 | 12.3% | 39.8% | **45.5%** |
| v2.1, context 16 pinned to 1220.2 | 0.2% | 44.8% (from 42.5%) | 81.8% |

Two conclusions, both load-bearing:

* **The elite pilot's Adrena-Brain policy is not expressible here.** Pinning it
  made that context slightly *worse* under both feature sets. Adding the threat
  and counter columns did not change that. This is the strongest evidence in the
  run that imitation has run out on this decision, and it is why the planner
  owns it instead.
* **The Froslass habit is a pin problem, not a feature problem.** Correcting the
  ability table did nothing; pinning the 1220.2 pilot everywhere moves 86.4% →
  45.5%, close to that pilot's own 21.4%. But it changes 12.3% of all decisions
  in favour of a policy this model reproduces at 77.9%, which is a large
  unmeasured bet. It is not shipped, and §6 says how to settle it.

## 5. The evaluation gap the analysis flagged first

`fetch_submission_logs.py` now stores `agent_<n>_initial_score` and
`agent_<n>_updated_score` from the same EpisodeService response that lists each
episode, and the behaviour probe reports win rate by opponent-rating bucket
(<900 / 900-999 / 1000-1099 / 1100+), by seat and by archetype. For the v2 run
every bucket reads `unknown`, which is the point: joining today's public
leaderboard matched 12 of 59 opponents and could not say what any of them was
rated at the time. From the next fetch on, "does it beat 900s" is answerable.

## 6. What v3 does not settle

## Artifacts

Reproduce with:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_grimmsnarl_damage_tables.py --variant v3
.\.venv\Scripts\python.exe .\scripts\build_grimmsnarl_v2_corpus.py `
  --agent-dir agents/grimmsnarl/grimmsnarl_ml_v3 `
  --output data/ml/grimmsnarl/processed/corpus_v3.npz `
  --report experiments/grimmsnarl_ml_v3/corpus_v3_report.json --workers 14
.\.venv\Scripts\python.exe .\scripts\train_grimmsnarl_v2_teacher.py `
  --corpus data/ml/grimmsnarl/processed/corpus_v3.npz `
  --output-model data/ml/grimmsnarl/models/ranker_v3.txt `
  --report experiments/grimmsnarl_ml_v3/train_v3_base.json `
  --team-feature --split-mode per-team --threads 18
.\.venv\Scripts\python.exe .\scripts\export_grimmsnarl_v1_model.py `
  --model data/ml/grimmsnarl/models/ranker_v3.txt `
  --corpus data/ml/grimmsnarl/processed/corpus_v3.npz `
  --teacher-team 16494330 --min-context-support 90 --min-context-top1 0.5 `
  --output agents/grimmsnarl/grimmsnarl_ml_v3/ranker_model.json `
  --report experiments/grimmsnarl_ml_v3/train_v3_base.json
.\.venv\Scripts\python.exe .\scripts\analyze_grimmsnarl_v3_behaviour.py `
  --agent-dir agents/grimmsnarl/grimmsnarl_ml_v3 `
  --run-dir data/runs/grimmsnarl/20260803_grimmsnarl_ml_v2_sub55205556 `
  --corpus data/ml/grimmsnarl/processed/corpus_v3.npz `
  --report experiments/grimmsnarl_ml_v3/behaviour_v3_planner_on.json
```

`--min-context-support 90` keeps context 8 routed, which is the threshold v2.1
chose deliberately after a wall-heavy replay set showed its inherited rule was
worse than the ranker despite a validation sample of 11 decisions. The default
gate (400) drops it; keeping it means v3 differs from v2.1 only in the features,
the model and the planner.
