"""Marnie's Grimmsnarl ex v10 ML: v8, plus a residual on one decision class.

v10 changes nothing about v8 except one shape, and the reason it is only one
shape is a negative result rather than caution. Fifty rated ladder games of v8
(submission 55317804) were re-walked decision by decision with five independent
advisors scoring the identical boards: a k-of-n consensus among them disagrees
with v8 on 3.4% of decisions, and that rate is flat across the outcome - 3.41%
in the games v8 lost against 3.67% in the ones it won, Fisher p = 0.68. A
general consensus override is therefore a policy perturbation uncorrelated with
winning, which is exactly what the Alakazam line paid 5.16 points of measured
agreement to discover.

The one class that survives is the Petrel search, and specifically the Unfair
Stamp that cannot be played this turn. Over the 3,642 same-deck games in the
archive the rate at which a pilot takes a dead Stamp runs against pilot rating
at rho = -0.607, p = 0.0036 across 21 pilots, while the *live* case does not
(p = 0.148) - the refusal is board-specific, not a dislike of the card. v8 sits
at 0.743 on its own run against a field 0.570 and 0.34-0.52 for the five
highest-rated pilots, and on v8's own 35 dead-Stamp offers every advisor takes
it less often than v8 does. ``ml_residual`` overrides that pick, and only that
pick, when three of four re-pinned pilots agree on the same replacement.

Context 7 is "put a card in hand", so the class cannot be an attachment, an
evolve, an attack, a knockout or a prize pick: the four invariants this version
is held to are satisfied by construction, and replaying all 51 stored games
leaves every one of those rates bit-identical to v8's. Details, including the
matchup claim this version set out to fix and could not support, are in
``experiments/grimmsnarl_ml_v10_safe_residual/RESULTS.md``.

Inherited from v6, which changes one thing and it is the pin rather than a
rule: **the Froslass evolve is scored as a different pilot.**

The v5 ladder run closed every resource gap it set out to close - Punk Up 2.52
cards searched against the elite band's 2.65 and v4's 3.65, 4.07 Darkness left
in deck against 4.23 and 3.71, 6.08 Adrena-Brain a game against 6.07 - and left
one behaviour completely untouched: v5 takes the Froslass evolve on **59 of
59** turns it is offered, 9 of 9 in the mirror. The field is at 73.6% and the
1220.2-rated pilot at 53.8% in the mirror.

The reason it never moved is that it was never a defect in the model. The pin
is team 16494330, rated 1077.6, and measured over their own 161 games in the
corpus **that pilot evolves on 95.7% of the turns it is offered** - the highest
Froslass rate of any pilot the model was trained on, together with the highest
dead-Unfair-Stamp rate (71.3% against a field median near 52%). v5 is not
diverging from its teacher here. It is copying an outlier faithfully, and no
feature column can fix a policy the teacher does not have: v3 and v4 spent 91
columns between them failing to move a single MAIN preference.

So v6 keeps everything about v5 and asks a different pilot about this one
decision class. ``teacher_team_id`` is a categorical input, so a MAIN select
that offers a Froslass evolve is scored end to end as team 16371703 (1220.2,
Froslass 52.6% in the mirror), and every other decision in the game is scored
by the incumbent pin exactly as v5 scored it. One teacher code per argmax, so
the comparison inside a decision stays on one scale. The escalation's firing
rate, and what it changed against what v5 would have played, are counted in
``diag_snapshot`` - the Alakazam line lost 5.16 points of agreement to a shell
nobody was counting.

The escalation teacher was chosen by measurement, not by rating, because
fidelity runs inversely to rating on this deck: the model reproduces 16371703
at 0.797 Top-1 against 0.928 for the incumbent pin, so a stronger pilot does
not automatically deliver more behaviour change. Three candidates were probed
on 66 stored ladder games and 16371703 moved the Froslass rate furthest even
so - to 78.0% of offered turns against 83.1% for the more imitable 16561259 and
88.1% for 16422241 - which is why it ships. That probe, and the control where
the escalation pilot only gets a veto instead of the class, are in
``experiments/grimmsnarl_ml_v6/RESULTS.md``.

Inherited from v5, which changed one thing in the place three versions of
feature engineering could not reach: **how many cards a multi-pick search
takes.**

Punk Up's five-energy deck search is a multi-pick select, so it never reaches
the ranker at all - ``Ranker.is_scorable`` requires ``maxCount == 1``. It falls
to the rule policy, where ``normalize_selection`` keeps every option scoring
above zero and ``score_card`` scores a Basic Darkness at a flat 100. The result
is that v1 through v4 have taken **every energy the select offered, on 100% of
90 stored activations.** The same-60 top-50 corpus does that on 36.9% of 1,997
activations for the pilots rated >= 1120, 45.2% for 1070-1119 and 63.7% for the
rest: a rating-ordered gap with us off the end of it, on a decision no
agreement metric in this line has ever scored.

The cost is the deck. Ten Basic Darkness is the entire supply and Punk Up
reaches only "your Marnie's Pokemon", so a Munkidori is fuelled by the
once-a-turn hand attachment or not at all. Measured per own turn, mining five
at a time leaves us

                            ours   elite    from turn 5 on
    Darkness left in deck    3.69    4.23    1.98 / 2.71
    a Darkness in hand      66.9%   71.0%   62.2% / 67.0%
    the attachment made     50.8%   59.6%   40.3% / 53.2%

and every one of those three is monotone in pilot rating. The last line is the
statistic v3 and v4 each set out to move with feature columns and each failed
to move - v4's own report calls three such attempts "the case for the next
version being outcome-based rather than imitation-based". It was never a MAIN
preference. It was starvation, upstream, in a rule.

So v5 replaces the count with what the board wants: the triggering Grimmsnarl's
deficit to Shadow Bullet, plus one card for each other Marnie's body still
short of it, floored at two. That reproduces the elite band's own count 51.9%
exactly and 90.1% within one card, and the fit degrades in rating order (45.4%,
36.9%, 34.4% for us), which is what makes it their policy rather than a curve
that happens to fit. Buddy-Buddy Poffin has the identical defect and gets the
same treatment off its own discriminator, Bench space. Replayed through v4's
52 stored games, v5 searches 2.62 Darkness against v4's 3.67 (elite 2.65) and
takes the maximum on 34.4% of activations against 100% (elite 36.9%), with 0
illegal selects.

``ml_planner`` gains one guard, not a preference: the budget is computed
against the promise that the body which just evolved ends the activation able
to attack, so the allocation must honour it when the budget is the deficit. The
teachers put the energy on a still-hungry trigger on 96.1% / 98.8% / 99.5% of
such offers and v4 on 115 of 115, so it costs nothing today; it exists so a
tighter budget cannot be spent wrong. A second clause refuses a fifth energy
onto a body already holding four while a lighter one is offered (elite 0.16%,
v4 0.91%).

Three further priorities from the ladder analysis were measured and are
deliberately **not** implemented; see ``experiments/grimmsnarl_ml_v5``.

Inherited from v4, which was v3 re-aimed at three places the per-turn ladder
measurement found the agent diverging from the pilot it is *already pinned to*,
rather than from the field:

* the once-per-turn Dark Energy attachment is made on 75.1% of the turns it
  is legal, against 83.5% for the pinned pilot and 86.3% for the elite pair.
  Punk Up attaches only to "your Marnie's Pokemon", so a hand attachment is the
  *only* way a Munkidori is ever fuelled - which is why this is 0.90 fuelled
  Munkidori a turn against 1.07 and 4.77 Adrena-Brain uses a game against 6.07,
  the exact statistic that separates our won mirrors from our lost ones;
* the Froslass evolve is taken on 22 of 22 mirror offers, against 80.8% for the
  pinned pilot and 53.8% for the pilot that wins 84% of its mirrors;
* a Shadow Bullet into a damage-immune Active while Boss is in hand and a
  damageable body sits on their Bench: 1 of 5 gusts for v3, 63.0% over 216
  teacher turns, 83.3% for the pinned pilot.

Two things the v3 ladder analysis proposed are deliberately *not* done, because
the per-turn teacher rates contradict them; see `experiments/grimmsnarl_ml_v4`.

Decision split, unchanged from v2 except for the last line:

* Every single-pick select in the model's routed contexts is decided by the
  imitation ranker's argmax. v1 routed only MAIN and left the rest to the rule
  policy, which matched the pinned teacher 39.5% of the time on deck search.
* Multi-pick selects - Punk Up's five-energy attachment, skill ordering,
  two-card discards - and the face-down prize picks stay with the
  ``marnies_grimmsnarl_ex_v7`` rule policy. The prize zone exposes no card ids,
  so nothing can be learned there.
* v3 adds ``ml_planner``: the ranker's answer is overridden only where the
  observation proves it is dominated on prizes taken this turn or on whether a
  body of ours survives the next attack. The 59-game v2 ladder analysis found
  two such shapes - gusting a body the free Bench-30 already kills, and moving
  Adrena-Brain counters off something other than a body a heal would save - and
  both are arithmetic across a whole turn rather than a preference between
  candidates. Every override is counted in ``diag_snapshot`` so the deployed
  override rate stays a measured number: the Alakazam line lost 5.16 points of
  agreement to an unmeasured safety shell.
* v4 adds one more planner rule on the same terms - refusing to end the turn on
  a swing that deals nothing while a gust that exposes a damageable body is
  still in hand - and 28 feature columns, of which the load-bearing ones price
  what *ending* the turn throws away. v3 had a column for every reason to
  attach and none for the cost of not attaching, and the argmax is a comparison
  between candidates: for the attachment to win more often, ending the turn has
  to score less.

The rule policy is invoked on every observation regardless, so its prize
tracker and diagnostics stay consistent; where the ranker answers, the rule
answer is discarded.
"""

from __future__ import annotations

import os

import fallback_policy
from fallback_policy import agent as _fallback_agent
from ml_runtime import Ranker

_RANKER: Ranker | None = None
_LOAD_ERROR: str | None = None
if os.environ.get("GRIMMSNARL_ML_DISABLE") != "1":
    try:
        _RANKER = Ranker()
    except Exception as error:  # missing/corrupt model must not crash the game
        _LOAD_ERROR = f"{type(error).__name__}: {error}"

# The planner is an override layer, not a dependency: if it cannot be imported
# the agent must still play the ranker's answer rather than fail to load.
try:
    from ml_planner import Planner

    _PLANNER = Planner()
    _PLANNER_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    _PLANNER = None
    _PLANNER_ERROR = f"{type(error).__name__}: {error}"
_PLANNER_DISABLED = (
    _PLANNER is None or os.environ.get("GRIMMSNARL_PLANNER_DISABLE") == "1"
)

# Same contract as the planner: a residual that cannot be imported must leave
# v8 playing, not stop the agent from loading.
try:
    from ml_residual import Residual

    _RESIDUAL = Residual()
    _RESIDUAL_ERROR: str | None = None
except Exception as error:  # noqa: BLE001
    _RESIDUAL = None
    _RESIDUAL_ERROR = f"{type(error).__name__}: {error}"
_RESIDUAL_DISABLED = (
    _RESIDUAL is None or os.environ.get("GRIMMSNARL_RESIDUAL_DISABLE") == "1"
)


def _choose(observation):
    if not isinstance(observation, dict) or observation.get("select") is None:
        return _fallback_agent(observation)

    # Before any early return: whether an Unfair Stamp is playable is a fact
    # about the opponent's last turn, so the residual has to see every
    # observation, not only the ones it might act on. ``note`` is idempotent
    # per turn, so the duplicate call from ``observe_external`` is harmless.
    if _RESIDUAL is not None:
        _RESIDUAL.note(observation)

    rule_choice = _fallback_agent(observation)
    if _RANKER is None:
        return rule_choice

    select = observation.get("select") or {}
    if not _RANKER.is_scorable(select):
        chosen = (
            rule_choice[0]
            if isinstance(rule_choice, list) and len(rule_choice) == 1
            else None
        )
        if chosen is not None and not _RANKER.teacher_forced:
            _RANKER.observe_external(observation, chosen)
            if _PLANNER is not None:
                _PLANNER.note(observation, select, chosen)
        return rule_choice

    index = _RANKER.choose(observation)
    if index is None:
        # Feature or scoring failure: keep the rule answer, and keep the
        # intra-turn history aligned with what was actually played.
        chosen = (
            rule_choice[0]
            if isinstance(rule_choice, list) and rule_choice
            else 0
        )
        if not _RANKER.teacher_forced:
            _RANKER.observe_external(observation, chosen)
            if _PLANNER is not None:
                _PLANNER.note(observation, select, chosen)
        return rule_choice
    if not _PLANNER_DISABLED:
        index = _PLANNER.adjust(
            observation, select, index, _RANKER.last_scores
        )
    # Last, so the planner's arithmetic overrides are never second-guessed:
    # they and the residual own disjoint contexts, and this ordering keeps it
    # that way if a later version widens either one.
    if not _RESIDUAL_DISABLED:
        index = _RESIDUAL.adjust(
            observation, select, index, _RANKER, _RANKER.last_scores
        )
    _RANKER.commit(index)
    if _PLANNER is not None and not _RANKER.teacher_forced:
        _PLANNER.note(observation, select, index)
    return [index]


def observe_external(observation, chosen):
    """Advance both histories with an action we did not choose.

    Teacher-forced evaluation replays a stored game, so the ranker's intra-turn
    columns and the planner's per-turn heal budget both have to follow the
    teacher rather than our own suggestion. Evaluators that only called
    ``Ranker.observe_external`` would leave the planner counting nothing, and
    the residual reading an opponent prize history with gaps in it.
    """
    if _RANKER is not None:
        _RANKER.observe_external(observation, chosen)
    if _PLANNER is not None and isinstance(observation, dict):
        _PLANNER.note(observation, observation.get("select") or {}, chosen)
    if _RESIDUAL is not None and isinstance(observation, dict):
        _RESIDUAL.note(observation)


def diag_reset():
    fallback_policy.DIAG.clear()
    fallback_policy.DIAG.update(fallback_policy._fresh_diag())
    if _RANKER is not None:
        _RANKER.reset()
    if _PLANNER is not None:
        _PLANNER.reset()
    if _RESIDUAL is not None:
        _RESIDUAL.reset()


def diag_snapshot():
    return {
        "fallback": dict(fallback_policy.DIAG),
        "ml": _RANKER.snapshot() if _RANKER is not None else {},
        "planner": _PLANNER.snapshot() if _PLANNER is not None else {},
        "residual": _RESIDUAL.snapshot() if _RESIDUAL is not None else {},
        "load_error": _LOAD_ERROR,
        "planner_load_error": _PLANNER_ERROR,
        "residual_load_error": _RESIDUAL_ERROR,
    }


# IMPORTANT: Kaggle's loader selects the last callable defined in main.py.
def agent(observation):
    return _choose(observation)
