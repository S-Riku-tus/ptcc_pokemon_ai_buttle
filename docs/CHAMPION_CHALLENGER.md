# Champion–Challenger Evaluation

One command compares the current **Champion** agent against a new **Challenger**,
aggregates result / safety / tactical / ML metrics, computes a statistical
confidence interval, and writes a **promotion-recommendation report**.

> **It never promotes automatically.** The evaluation does not overwrite the
> Champion, swap models, edit config, run git (commit/tag/push), delete agents,
> or submit to Kaggle. Formal promotion is a separate, human-invoked step
> (`scripts/promote_challenger.py`).

---

## What it is

- **Champion**: the current baseline agent. Today: `agents/alakazam_ml_v3`
  (the guarded ML runtime over the v12 fallback).
- **Challenger**: a newly trained/experimental agent, e.g.
  `agents/alakazam_ml_v4_candidate` or `agents/alakazam_ml_v3_challenger`.
- **Baseline**: a reference agent (usually the pure-rule fallback
  `alakazam741_v12_top_sync_full`) used to catch a Challenger that beats the
  Champion but is weaker than plain rules.

The pipeline: preflight → seat-swapped matches → metric aggregation →
confidence interval → promotion judgement → artifact bundle.

---

## Check the current Champion

The Champion is whatever `champion_agent` names in the config
(`configs/champion_challenger/alakazam.json`) or `--champion` on the CLI.

```bash
python scripts/validate_agent.py --agent alakazam_ml_v3
```

---

## Create a Challenger

Copy an existing agent and edit it:

```bash
python scripts/new_agent.py alakazam_ml_v3 alakazam_ml_v4_candidate
# edit agents/alakazam_ml_v4_candidate/ (model, runtime, ...)
python scripts/validate_agent.py --agent alakazam_ml_v4_candidate
```

Name it `*_candidate` or `*_challenger` so it is auto-detectable.

### Recommended metadata

Add these fields to `agents/<name>/metadata.json` so auto-detection and the
report can describe the Challenger:

```json
{
  "name": "alakazam_ml_v4_candidate",
  "role": "challenger",
  "archetype": "alakazam",
  "parent_agent": "alakazam_ml_v3",
  "model_version": "v4",
  "created_at": "2026-07-17T15:00:00+09:00",
  "training_dataset_version": "top50_plus_ladder_20260717"
}
```

An agent is treated as a candidate when its name matches `*_candidate` /
`*_challenger` **or** its metadata `role` is `challenger`.

---

## One-command evaluation

Full config-driven run (200 games by default):

```bash
python scripts/run_champion_challenger.py \
  --config configs/champion_challenger/alakazam.json \
  --challenger alakazam_ml_v4_candidate
```

Short form (no config file needed):

```bash
python scripts/run_champion_challenger.py \
  --champion alakazam_ml_v3 \
  --challenger alakazam_ml_v4_candidate \
  --games 200
```

Auto-detect the Challenger (omit `--challenger`):

```bash
# lists candidates if more than one; auto-picks if exactly one
python scripts/run_champion_challenger.py --config configs/champion_challenger/alakazam.json
python scripts/run_champion_challenger.py --list-challengers   # just list, don't run
```

The single command runs, in order: Champion/Challenger/Baseline existence
checks → static validation → 60-card deck check → import check → model-load
check → seat-swapped match generation → Champion vs Challenger games →
(optional) Baseline games → metric aggregation → promotion judgement →
JSON / CSV / Markdown reports.

If some games fail, the run continues and records the reasons in
`failures.csv`; only fatal errors (an agent that cannot be imported, no decided
games) stop it and produce an `INVALID_EVALUATION` report.

---

## Config

`configs/champion_challenger/alakazam.json`. Everything is overridable on the
CLI (CLI wins over config; config wins over defaults).

| Key | Meaning |
|---|---|
| `champion_agent` / `challenger_agent` / `baseline_agent` | agent directory names |
| `games` | total Champion-vs-Challenger games (even when `seat_swap`) |
| `seat_swap` | play each seed with seats swapped (fairness) |
| `seed` | base logical seed (see *Fairness* below) |
| `max_steps` | per-game step cap (draw if reached) |
| `parallel_workers` | reserved; the cg engine is single-instance so runs are sequential |
| `timeout_seconds_per_decision` | soft (wall-clock) per-decision timeout flag |
| `save_replays` / `save_trajectories` | write `replays/` / `trajectories/` |
| `run_baseline_comparison` / `baseline_games` | also play both agents vs the Baseline |
| `minimum_games` | games required for a PROMOTE verdict |
| `require_confidence_interval_above` | Wilson CI lower bound must clear this to PROMOTE |
| `alakazam_card_id` | key attacker card id (743) for tactical metrics |
| `output_root` | artifact root directory |
| `promotion_thresholds` | the promotion gates (below) |

---

## Fairness: same seed + seat swap

For `games = 200` with `seat_swap = true`, the schedule is **100 seed pairs ×
2 games**:

```
For each of 100 seeds:
  Game A: Champion=seat0, Challenger=seat1
  Game B: Challenger=seat0, Champion=seat1
```

Each agent occupies seat 0 and seat 1 the same number of times.

> **Engine limitation (stated honestly).** The bundled cg engine's
> `BattleStart(deck0, deck1)` accepts **no RNG seed** — the shuffle and coin
> flips are internal to the native library. So the two games of a seed pair are
> **not** bit-identical initial conditions. The per-pair `seed` is a *logical*
> identifier (it also seeds the Python-level `random`/`first` baselines).
> Fairness is therefore enforced by **seat swap**, not by shared shuffles. The
> generated report repeats this note.

---

## Metrics

Written to `agent_metrics.csv`, `matchup_metrics.csv`, `game_results.csv`,
`seed_pair_results.csv`, `ml_diagnostics.csv` and summarised in the report.

**Results**: total games, Champion/Challenger wins, draws, Challenger win rate
(ex-draws) with a **Wilson 95% CI**, seat-0/seat-1 win rate, first/second
player win rate, seed-pair breakdown (won both seats / one seat only).

**Safety**: crashes, illegal actions, (soft) timeouts, import failures, model
load failures, average and max decision time.

**Tactical**: average first-attack turn, attack-by-T2 rate, attack rate per own
turn, attacks per game, **Alakazam attacks per game**, idle turns after the
first attack in losses, average game turns, search uses per attack, average
hand size, average hand at Alakazam attack, average overkill, **deckout** and
**boardout** counts/rates, normal prize losses.

**ML diagnostics** (when the agent exposes `diag_snapshot()`): decisions,
model-selected count, adoption rate, fallback rate, low-confidence fallbacks,
per-scope fallback reasons (e.g. `lethal_guard`, `outside_training_scope`), etc.

### Metric caveats (heuristics)

- **Loss reason** is inferred from the terminal observation: if the winner has
  taken all prizes → `prizes`; else if the loser has no Pokemon in play →
  `boardout`; else if the loser's deck is empty → `deckout`. The engine does not
  expose a loss cause directly.
- **Overkill** and **Alakazam attack** damage use the archetype's
  hand-size-scaled attack (`20 × hand`); meaningful for the Alakazam attacker.
- **Timeouts** are soft wall-clock flags, not hard kills (in-process games
  cannot be interrupted). `max_steps` produces a draw, not a timeout.
- **Search uses** is approximated from nested (non-main) select contexts.

---

## Promotion judgement

The report ends with one of:

| Verdict | Meaning |
|---|---|
| `PROMOTE_RECOMMENDED` | all required gates pass, CI lower bound clears the floor, and games ≥ minimum. Stronger than the Champion. |
| `HOLD` | safe and not clearly worse, but evidence is insufficient (too few games, CI too wide, a tactical gate marginally missed). Needs more evaluation. |
| `REJECT` | a clear regression: a safety violation (any crash/illegal/timeout over threshold), a losing head-to-head, or an excessive deckout/boardout rate. |
| `INVALID_EVALUATION` | the comparison did not really happen (import failure, no decided games, pervasive engine start errors). |

Required gates (all configurable in `promotion_thresholds`): head-to-head win
rate, attack rate, Alakazam attacks per game, deckout rate, boardout rate,
post-first-attack idle turns in losses, crashes, illegal actions, timeouts.

**Any crash or illegal action blocks `PROMOTE_RECOMMENDED`** even with a good
win rate.

### Statistical uncertainty

The win rate is reported with a Wilson 95% CI, e.g.
`55.0% (95% CI 48.1%–61.7%)`. Clearing 53% once is not enough: if
`require_confidence_interval_above` (default 0.50) is not below the CI lower
bound, the verdict is `HOLD`, not `PROMOTE_RECOMMENDED`.

---

## Reading the report

`promotion_report.md` sections: identity (names, model/deck hashes, timestamp,
git commit, environment) → head-to-head → seat/turn-order breakdown → tactical
table → safety table → baseline comparison → ML diagnostics → per-condition
PASS/FAIL → final verdict + remaining concerns → formal promotion procedure.

`promotion_report.json` is the machine-readable equivalent (consumed by
`promote_challenger.py`).

---

## Formal promotion (human-only)

**Default is dry-run** — it changes nothing:

```bash
python scripts/promote_challenger.py \
  --report artifacts/champion_challenger/<run>/promotion_report.json \
  --new-agent-name alakazam_ml_v4 \
  --dry-run
```

Apply — copies the Challenger into a **new** `agents/<name>/` directory only:

```bash
python scripts/promote_challenger.py \
  --report artifacts/champion_challenger/<run>/promotion_report.json \
  --new-agent-name alakazam_ml_v4 \
  --apply
```

`--apply` refuses to overwrite an existing directory and refuses non-`PROMOTE`
verdicts unless `--allow-non-promote` is passed. It never edits the Champion,
runs git, tags, pushes, or submits to Kaggle — those remain manual human steps.

---

## Output layout

```text
artifacts/champion_challenger/
  20260717_153000_alakazam_ml_v3_vs_alakazam_ml_v4_candidate/
    config_resolved.json
    environment.json
    game_results.csv
    seed_pair_results.csv
    agent_metrics.csv
    matchup_metrics.csv
    ml_diagnostics.csv
    failures.csv
    promotion_report.json
    promotion_report.md
    run.log
    replays/          # only when save_replays
    trajectories/     # only when save_trajectories
```

---

## Baseline comparison

With `run_baseline_comparison` (or `--run-baseline`) and a `baseline_agent`,
both agents also play the Baseline for `baseline_games` games each:

```text
Challenger vs baseline
Champion   vs baseline
```

This flags a Challenger that beats the Champion but is weaker than pure rules.

---

## Large runs / parallelism

The cg engine is a **single global native instance**, so games run
**sequentially** within a process; `parallel_workers > 1` is accepted but logged
and ignored. For 200 games, expect the run to take a few minutes. To parallelise
you would run separate processes over disjoint seed ranges and merge the CSVs
manually (not automated here to keep results reproducible and the engine safe).

---

## Failure checklist

- `INVALID_EVALUATION` + import failure → run
  `python scripts/validate_agent.py --agent <name>` and check the model file.
- Many `failures.csv` rows with `battle_start` errors → deck legality / engine.
- `REJECT` from boardout/deckout → inspect `game_results.csv` loss reasons and
  saved `replays/`.
- Wide CI / `HOLD` → increase `games`.
