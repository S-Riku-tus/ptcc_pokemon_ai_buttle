"""Train and evaluate the v2 all-context Grimmsnarl imitation ranker.

Same machinery as v1 plus per-context reporting. v1 fitted MAIN only and
scored 90.5% there, but the rule policy that owned the remaining contexts
agreed with the same teacher 39.5% on deck search and 50-65% on damage
placement, dragging all-context agreement to 81.4%. Reporting one pooled
number would hide exactly that, so every block is broken out.

Original v1 notes follow.

Train and evaluate the Grimmsnarl imitation ranker.

Three defects from the Alakazam line are fixed by construction:

1. Early stopping is on strict Top-1, the metric the agent is actually judged
   on. v33 stopped on NDCG and shipped roughly half the trees it wanted.
2. Reported agreement is per-decision Top-1 on a chronological block that is
   never touched during fitting or configuration selection.
3. Teacher-cohort choice is an experiment, not an assumption. ``--teams`` and
   ``--min-agreement`` allow field-pooled, subset and single-pilot corpora to
   be compared on the same held-out block, and ``--leave-out-team`` measures
   whether the policy transfers to a pilot the model never saw.

The error taxonomy separates recoverable same-turn ordering errors (the model
picked something the teacher also played that turn, just in another order)
from genuine divergence. That distinction is what the Alakazam v36 report
identified as the real ceiling, so it is measured from the start.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _ranges(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(groups, dtype=np.int64)
    return np.r_[0, ends[:-1]], ends


class Corpus:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.features = data["features"]
        self.labels = data["labels"]
        self.groups = data["groups"]
        self.splits = data["splits"]
        self.episode_ids = data["episode_ids"]
        self.seats = (
            data["seats"]
            if "seats" in data.files
            else np.full(len(self.groups), -1, dtype=np.int8)
        )
        self.team_ids = data["team_ids"]
        self.submission_ids = (
            data["submission_ids"]
            if "submission_ids" in data.files
            else np.full(len(self.groups), -1, dtype=np.int64)
        )
        self.turns = data["turns"]
        self.contexts = (
            data["contexts"] if "contexts" in data.files
            else np.zeros(len(self.groups), dtype=np.int16)
        )
        self.won = data["won"]
        self.teacher_action_types = data["teacher_action_types"]
        self.names = [str(x) for x in data["feature_names"]]
        self.categorical = [str(x) for x in data["categorical"]]
        self.starts, self.ends = _ranges(self.groups)
        self.team_feature = False

    def rows_for(self, decisions: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.arange(self.starts[i], self.ends[i]) for i in decisions
        ]) if len(decisions) else np.zeros(0, dtype=np.int64)

    def add_team_feature(self) -> None:
        """Condition the ranker on which pilot is acting.

        Pooling 21 pilots buys data but forces one averaged policy. With the
        pilot exposed as a categorical the model can keep the shared mechanics
        and still express per-pilot habits; inference pins it to the pilot we
        want to copy. The column is materialised per split in ``matrix`` -
        widening the 2.3 GB base array in place costs a second copy and gets
        the process killed.
        """
        if not self.team_feature:
            self.team_feature = True
            # Dense 0..N-1 codes. Raw Kaggle team ids are ~1.6e7, and
            # LightGBM allocates categorical bins over the value range, so
            # feeding them raw makes construction pathologically slow.
            self.team_codes = {
                int(team): index
                for index, team in enumerate(sorted(set(
                    int(x) for x in self.team_ids
                )))
            }
            self.names.append("teacher_team_id")
            self.categorical.append("teacher_team_id")

    def resplit_per_team(self, validation: float, test: float) -> dict:
        """Hold out each pilot's own newest games instead of the field's.

        A single global chronological cut puts almost all of one pilot's games
        on one side of the boundary, so per-pilot test blocks come out tiny and
        wildly uneven. Cutting inside each pilot keeps the split honest - test
        games are still strictly later than that pilot's training games - and
        gives every pilot a test block worth quoting a confidence interval on.
        """
        splits = np.empty(len(self.groups), dtype=self.splits.dtype)
        boundaries: dict[str, list[int]] = {}
        for team in np.unique(self.team_ids):
            mask = self.team_ids == team
            episodes = np.sort(np.unique(self.episode_ids[mask]))
            total = len(episodes)
            test_size = max(1, int(round(total * test)))
            validation_size = max(1, int(round(total * validation)))
            train_end = max(1, total - test_size - validation_size)
            validation_min = int(episodes[train_end])
            test_min = int(
                episodes[min(total - 1, train_end + validation_size)]
            )
            block = self.episode_ids[mask]
            splits[mask] = np.where(
                block >= test_min, "test",
                np.where(block >= validation_min, "validation", "train"),
            )
            boundaries[str(int(team))] = [validation_min, test_min]
        self.splits = splits
        return boundaries

    def matrix(self, decisions: np.ndarray, pin_team: int | None = None):
        """Feature block for these decisions, built in bounded chunks."""
        rows = self.rows_for(decisions)
        width = self.features.shape[1] + int(self.team_feature)
        block = np.empty((len(rows), width), dtype=np.float32)
        step = 200_000
        for start in range(0, len(rows), step):
            window = rows[start:start + step]
            block[start:start + len(window), :self.features.shape[1]] = (
                self.features[window]
            )
        if self.team_feature:
            codes = (
                np.full(
                    len(decisions),
                    self.team_codes[int(pin_team)],
                    dtype=np.float32,
                )
                if pin_team is not None
                else np.asarray(
                    [self.team_codes[int(x)] for x in self.team_ids[decisions]],
                    dtype=np.float32,
                )
            )
            block[:, -1] = np.repeat(codes, self.groups[decisions])
        return block


def top1(scores: np.ndarray, corpus: Corpus, decisions: np.ndarray,
         row_offset: np.ndarray) -> np.ndarray:
    """Per-decision correctness of the argmax candidate."""
    correct = np.zeros(len(decisions), dtype=bool)
    for slot, decision in enumerate(decisions):
        start = row_offset[slot]
        size = int(corpus.groups[decision])
        window = scores[start:start + size]
        best = int(np.argmax(window))
        correct[slot] = bool(
            corpus.labels[corpus.starts[decision] + best] == 1
        )
    return correct


class Top1Metric:
    """LightGBM feval for strict Top-1 on the validation set."""

    def __init__(self, corpus: Corpus, decisions: np.ndarray):
        self.corpus = corpus
        self.decisions = decisions
        sizes = corpus.groups[decisions]
        self.offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)

    def __call__(self, preds: np.ndarray, dataset: lgb.Dataset):
        correct = top1(preds, self.corpus, self.decisions, self.offsets)
        return "top1", float(correct.mean()), True


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963985
    phat = successes / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * ((phat * (1 - phat) + z * z / (4 * total)) / total) ** 0.5
    return (
        round((centre - margin) / denominator, 4),
        round((centre + margin) / denominator, 4),
    )


def error_taxonomy(
    corpus: Corpus,
    decisions: np.ndarray,
    offsets: np.ndarray,
    scores: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    """Split misses into same-turn ordering errors and genuine divergence."""
    action_col = names.index("action_type_id")
    card_col = names.index("candidate_card_id")
    attack_col = names.index("candidate_attack_id")
    target_col = names.index("candidate_target_id")

    def identity(row: int) -> tuple:
        return (
            int(corpus.features[row, action_col]),
            int(corpus.features[row, card_col]),
            int(corpus.features[row, attack_col]),
            int(corpus.features[row, target_col]),
        )

    # What the teacher actually played across each whole turn.
    turn_plays: dict[tuple, set[tuple]] = defaultdict(set)
    for slot, decision in enumerate(decisions):
        start = int(corpus.starts[decision])
        chosen = start + int(np.flatnonzero(
            corpus.labels[start:int(corpus.ends[decision])] == 1
        )[0])
        key = (int(corpus.episode_ids[decision]), int(corpus.turns[decision]))
        turn_plays[key].add(identity(chosen))

    counts: Counter[str] = Counter()
    confusion: Counter[tuple] = Counter()
    for slot, decision in enumerate(decisions):
        start = int(corpus.starts[decision])
        size = int(corpus.groups[decision])
        window = scores[offsets[slot]:offsets[slot] + size]
        predicted = start + int(np.argmax(window))
        if corpus.labels[predicted] == 1:
            counts["correct"] += 1
            continue
        chosen = start + int(np.flatnonzero(
            corpus.labels[start:start + size] == 1
        )[0])
        key = (int(corpus.episode_ids[decision]), int(corpus.turns[decision]))
        predicted_identity = identity(predicted)
        if predicted_identity in turn_plays[key]:
            counts["same_turn_ordering"] += 1
        elif predicted_identity[0] == identity(chosen)[0]:
            counts["same_action_type_divergence"] += 1
        else:
            counts["divergence"] += 1
        confusion[(identity(chosen)[0], predicted_identity[0])] += 1

    total = sum(counts.values())
    return {
        "counts": dict(counts),
        "rates": {
            key: round(value / total, 4) for key, value in counts.items()
        },
        "order_insensitive_top1": round(
            (counts["correct"] + counts["same_turn_ordering"]) / max(1, total),
            4,
        ),
        "top_action_confusions": [
            {"teacher": int(a), "predicted": int(b), "count": int(n)}
            for (a, b), n in confusion.most_common(12)
        ],
    }


def topk(corpus: Corpus, decisions: np.ndarray, offsets: np.ndarray,
         scores: np.ndarray, k: int) -> float:
    hits = 0
    for slot, decision in enumerate(decisions):
        start = int(corpus.starts[decision])
        size = int(corpus.groups[decision])
        window = scores[offsets[slot]:offsets[slot] + size]
        order = np.argsort(-window)[:k]
        hits += int(any(
            corpus.labels[start + int(index)] == 1 for index in order
        ))
    return round(hits / max(1, len(decisions)), 4)


def select_decisions(corpus: Corpus, split: str,
                     teams: set[int] | None,
                     leave_out: int | None) -> np.ndarray:
    mask = corpus.splits == split
    if teams is not None:
        mask &= np.isin(corpus.team_ids, list(teams))
    if leave_out is not None:
        if split == "train":
            mask &= corpus.team_ids != leave_out
        else:
            mask &= corpus.team_ids == leave_out
    return np.flatnonzero(mask)


V20_HARD_STATES = (
    "delayed_attack_access", "attack_chain_gap", "punk_allocation",
    "ready_promotion", "wall_recovery", "mirror_endgame",
)


def hard_state_masks(
    corpus: Corpus,
    decisions: np.ndarray,
    mask_set: str = "v21",
) -> dict[str, np.ndarray]:
    """Observable decision slices that deserve more teacher signal.

    v19 weighted every decision from a won game four times.  That mostly
    repeats ordinary winning positions.  v20 instead concentrates capacity on
    states where the submitted policy demonstrably loses continuity: delayed
    access, no near-term backup attacker, Punk Up allocation, a ready-attacker
    promotion, a live wall route, or a one/two-Prize endgame.  All columns are
    public observation features and the eventual result is not consulted.
    """
    starts = corpus.starts[decisions]

    def state(name: str, default: float = 0.0) -> np.ndarray:
        if name not in corpus.names:
            return np.full(len(decisions), default, dtype=np.float32)
        return corpus.features[starts, corpus.names.index(name)]

    def offered(name: str) -> np.ndarray:
        if name not in corpus.names:
            return np.zeros(len(decisions), dtype=bool)
        column = corpus.names.index(name)
        return np.asarray([
            bool(np.max(corpus.features[
                corpus.starts[decision]:corpus.ends[decision], column
            ]) > 0)
            for decision in decisions
        ])

    if mask_set == "dragapult_v3":
        # Observable slices measured in the Dragapult v2 ladder autopsy.  The
        # failure tail is not an Energy-route failure any more; it is the lack
        # of a second completed attacker.  Weight whole ranking groups, never
        # individual candidates, so LambdaRank remains internally consistent.
        route_bodies = state("route_bodies")
        return {
            "board_width_gap": (
                (route_bodies <= 1)
                & (state("open_bench_slots") > 0)
            ),
            "backup_chain_gap": (
                (state("phantom_ready_active") > 0)
                & (state("backup_route_eta", 99.0) > 1)
            ),
            "board_search_offer": (
                (state("line_body_deficit_three") > 0)
                & offered("candidate_is_board_search")
            ),
            "needed_line_piece_offer": offered("candidate_line_piece_needed"),
            "backup_advance_offer": offered("candidate_advances_backup_route"),
        }
    if mask_set not in {"v20", "v21"}:
        raise ValueError(f"unknown hard-state set: {mask_set}")
    turn = state("turn")
    has_ready = state("has_ready_attacker")
    active_ready = state("active_attacker_ready")
    backup_eta = state("backup_grim_line_eta", 9.0)
    has_line = state("marnie_body_count") > 0
    masks = {
        "delayed_attack_access": (turn >= 5) & (has_ready <= 0) & has_line,
        "attack_chain_gap": (active_ready > 0) & (backup_eta > 1),
        "punk_allocation": offered("candidate_punk_targets_trigger"),
        "ready_promotion": offered("candidate_ready_promotion_offered"),
        "wall_recovery": (
            (state("opp_active_is_damage_immune") > 0)
            & (state("shadow_bullet_prizes_now") <= 0)
        ),
        "mirror_endgame": (
            (state("mirror_match_signal") > 0)
            & (state("self_prize_count", 6.0) <= 2)
        ),
        # v21 adds the two slices the v19/v20 ladder autopsy sized.  Both
        # degrade to empty on a pre-v21 corpus because ``state`` defaults a
        # missing column to zero, so v20's reports stay reproducible.
        #
        # ``retreat_lock`` is the own turn where the lock is still preventable:
        # a non-Marnie Active holding no Darkness with a Marnie line benched.
        # ``bench_locked`` is the turn it has already cost us the attack.  Over
        # the 529 stored ladder games the lock occurs on 128 own turns in 82
        # games, and the single-decision fix binds only 4 times in 1,881 manual
        # attachments - which is why it is taught here rather than guarded.
        "retreat_lock": state("retreat_lock_risk") > 0,
        "bench_locked": state("bench_locked_ready_attacker") > 0,
        # Prize conversion.  v20 dropped v19's win weighting and did not
        # replace it with anything that saw a Prize route, and its Boss rate
        # fell from 0.303 to 0.180 of offers against an elite band at 0.385.
        # A same-turn Boss gain is rare (8 of 457 stored Shadow Bullets), so
        # this weights the decision rather than forcing the play.
        "boss_prize_route": state("route_boss_prize_gain") >= 1,
        "dead_shadow_with_route": (
            (state("shadow_bullet_prizes_now") <= 0)
            & (state("route_boss_prize_gain") >= 1)
        ),
    }
    if mask_set == "v20":
        # The control: exactly the six slices v20 shipped, so a v21 candidate
        # can be compared against a reproduction rather than a report.
        masks = {name: masks[name] for name in V20_HARD_STATES}
    return masks


def make_dataset(corpus: Corpus, decisions: np.ndarray,
                 reference: lgb.Dataset | None = None,
                 focus_team: int | None = None,
                 focus_weight: float = 1.0,
                 win_weight: float = 1.0,
                 hard_state_weight: float = 1.0,
                 hard_state_set: str = "v21",
                 rating_weight: float = 0.0,
                 ratings: dict[int, float] | None = None,
                 episode_weights: dict[tuple[int, int], float] | None = None,
                 episode_equal_weight: bool = False,
                 teacher_equal_weight: bool = False) -> lgb.Dataset:
    """Optionally tilt the fit toward the pilot we intend to pin.

    Conditioning on pilot id already lets the model express per-pilot habits,
    but the loss is still dominated by the other twenty. Upweighting the
    target trades a little shared-mechanic signal for fidelity to the one
    policy that ships. v1 showed the opposite extreme - a single-pilot corpus -
    is worse than pooling, so this is the middle of that range, and the weight
    is chosen on validation like any other hyperparameter.

    Seven weightings compose here, all default off:

    * ``episode_equal_weight`` prevents long, highly interactive games from
      contributing more total loss merely because they contain more choices.
    * ``teacher_equal_weight`` prevents the most prolific submission from
      becoming the implicit target policy.  It equalises each teacher's base
      mass after episode equalisation; explicit focus/rating tilts are applied
      afterwards and therefore remain intentional.

    * ``focus_weight`` on the pinned pilot's decisions.
    * ``win_weight`` on decisions from games the teacher won. The corpus has
      carried a ``won`` flag since v1 and nothing ever read it, so every
      decision in a lost game counted the same as one in a won game. Kept
      small: a loss is often a bad opening, not a bad decision, so a large
      weight would just learn "teachers who drew well".
    * ``hard_state_weight`` on observation-defined attack access, continuity,
      allocation, promotion, wall and mirror-endgame states.  Unlike result
      weighting it remains available at inference and directly targets the
      sparse decisions v20 is intended to improve.
    * ``rating_weight`` interpolates each pilot's weight by leaderboard rating,
      so the shared mechanics are still fitted on all 21 pilots but the
      stronger ones pull harder. Capped by construction at 1 + rating_weight.
    * ``episode_weights`` is an explicit, auditable sidecar keyed by
      ``episode_id:seat``.  Dragapult uses this for a small tilt toward
      teacher games that sustained four Phantom Dives; it is kept generic so
      the trainer never infers a future outcome from candidate features.
    """
    rows = corpus.rows_for(decisions)
    categorical = [
        name for name in corpus.categorical if name in corpus.names
    ]
    weights = np.ones(len(decisions), dtype=np.float32)
    if episode_equal_weight:
        episode_ids = corpus.episode_ids[decisions]
        _, inverse, counts = np.unique(
            episode_ids, return_inverse=True, return_counts=True
        )
        weights *= 1.0 / counts[inverse].astype(np.float32)
    if teacher_equal_weight:
        teacher_ids = corpus.team_ids[decisions]
        teachers, inverse = np.unique(teacher_ids, return_inverse=True)
        totals = np.bincount(inverse, weights=weights, minlength=len(teachers))
        positive = totals[totals > 0]
        target = float(positive.mean()) if len(positive) else 1.0
        scales = np.where(totals > 0, target / totals, 1.0)
        weights *= scales[inverse].astype(np.float32)
    if focus_team is not None and focus_weight != 1.0:
        weights *= np.where(
            corpus.team_ids[decisions] == focus_team, focus_weight, 1.0
        )
    if win_weight != 1.0:
        weights *= np.where(corpus.won[decisions] == 1, win_weight, 1.0)
    if episode_weights:
        weights *= np.asarray([
            episode_weights.get(
                (int(corpus.episode_ids[decision]), int(corpus.seats[decision])),
                1.0,
            )
            for decision in decisions
        ], dtype=np.float32)
    if hard_state_weight != 1.0:
        masks = hard_state_masks(corpus, decisions, hard_state_set)
        difficult = np.logical_or.reduce(list(masks.values()))
        weights *= np.where(difficult, hard_state_weight, 1.0)
    if rating_weight and ratings:
        values = np.array(
            [ratings.get(int(t), 0.0) for t in corpus.team_ids[decisions]],
            dtype=np.float32,
        )
        low, high = float(values.min()), float(values.max())
        span = max(high - low, 1e-6)
        weights *= 1.0 + rating_weight * (values - low) / span
    # Keep LightGBM's effective regularisation scale comparable across modes.
    mean_weight = float(weights.mean()) if len(weights) else 1.0
    if mean_weight > 0:
        weights /= mean_weight
    use_weights = not np.allclose(weights, 1.0)
    # free_raw_data lets LightGBM drop the dense copy once it is binned;
    # evaluation rebuilds the block it needs one split at a time.
    return lgb.Dataset(
        corpus.matrix(decisions),
        label=corpus.labels[rows],
        group=corpus.groups[decisions],
        weight=(
            np.repeat(weights, corpus.groups[decisions])
            if use_weights else None
        ),
        feature_name=corpus.names,
        categorical_feature=categorical,
        reference=reference,
        free_raw_data=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-model", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--teams", default="",
                        help="Comma separated team ids; empty uses all.")
    parser.add_argument("--focus-team", type=int,
                        help="Upweight this pilot's decisions during fitting.")
    parser.add_argument("--focus-weight", type=float, default=1.0)
    parser.add_argument(
        "--win-weight", type=float, default=1.0,
        help="Weight decisions from games the teacher won.",
    )
    parser.add_argument(
        "--hard-state-set", default="v21",
        choices=["v20", "v21", "dragapult_v3"],
        help=(
            "Which observable slices get the extra weight. 'v20' is the six "
            "shipped slices, kept so v20 can be reproduced as a control."
        ),
    )
    parser.add_argument(
        "--hard-state-weight", type=float, default=1.0,
        help=(
            "Weight observable recovery/continuity/wall/endgame decisions; "
            "unlike --win-weight this does not use the eventual result."
        ),
    )
    parser.add_argument(
        "--rating-weight", type=float, default=0.0,
        help="Scale each pilot's weight by leaderboard rating (0 = off).",
    )
    parser.add_argument(
        "--episode-equal-weight", action="store_true",
        help="Give every training episode equal total base loss mass.",
    )
    parser.add_argument(
        "--teacher-equal-weight", action="store_true",
        help="Give every training teacher equal total base loss mass.",
    )
    parser.add_argument(
        "--episode-weight-map", type=Path,
        help=(
            "JSON sidecar with a 'weights' object keyed by episode_id:seat. "
            "Only the training loss is weighted; validation/test stay clean."
        ),
    )
    parser.add_argument(
        "--ratings", default="",
        help="team_id:rating pairs, comma separated, for --rating-weight.",
    )
    parser.add_argument("--eval-team", type=int,
                        help="Restrict validation and test to this pilot.")
    parser.add_argument("--leave-out-team", type=int,
                        help="Train without this team, evaluate only on it.")
    parser.add_argument("--objective", default="lambdarank",
                        choices=["lambdarank", "binary"])
    parser.add_argument("--num-leaves", type=int, default=255)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-data-in-leaf", type=int, default=40)
    parser.add_argument("--feature-fraction", type=float, default=0.5)
    parser.add_argument("--bagging-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--lambda-l2", type=float, default=1.0)
    parser.add_argument("--num-boost-round", type=int, default=4000)
    parser.add_argument("--early-stopping", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--team-feature", action="store_true",
        help="Expose the acting pilot as a categorical feature.",
    )
    parser.add_argument(
        "--split-mode", default="global", choices=["global", "per-team"],
    )
    parser.add_argument("--validation-fraction", type=float, default=0.12)
    parser.add_argument("--test-fraction", type=float, default=0.12)
    args = parser.parse_args()

    if args.focus_weight <= 0:
        parser.error("--focus-weight must be positive")
    if args.win_weight <= 0:
        parser.error("--win-weight must be positive")
    if args.hard_state_weight <= 0:
        parser.error("--hard-state-weight must be positive")
    if args.rating_weight < 0:
        parser.error("--rating-weight cannot be negative")

    corpus = Corpus(args.corpus)
    split_boundaries = None
    if args.split_mode == "per-team":
        split_boundaries = corpus.resplit_per_team(
            args.validation_fraction, args.test_fraction
        )
    if args.team_feature:
        corpus.add_team_feature()
    teams = {
        int(value) for value in args.teams.split(",") if value.strip()
    } or None

    train = select_decisions(corpus, "train", teams, args.leave_out_team)
    validation = select_decisions(corpus, "validation", teams,
                                  args.leave_out_team)
    test = select_decisions(corpus, "test", teams, args.leave_out_team)
    if not len(train) or not len(validation) or not len(test):
        raise SystemExit(
            f"empty split: train={len(train)} validation={len(validation)} "
            f"test={len(test)}"
        )
    print(
        f"decisions train={len(train)} validation={len(validation)} "
        f"test={len(test)} features={len(corpus.names)}",
        flush=True,
    )

    if args.eval_team is not None:
        validation = validation[
            corpus.team_ids[validation] == args.eval_team
        ]
        test = test[corpus.team_ids[test] == args.eval_team]
        if not len(validation) or not len(test):
            raise SystemExit("eval-team has no held-out decisions")
        print(
            f"eval restricted to {args.eval_team}: "
            f"validation={len(validation)} test={len(test)}",
            flush=True,
        )

    ratings = {
        int(pair.split(":")[0]): float(pair.split(":")[1])
        for pair in args.ratings.split(",") if ":" in pair
    }
    episode_weights: dict[tuple[int, int], float] = {}
    episode_weight_metadata: dict[str, Any] | None = None
    if args.episode_weight_map:
        episode_weight_metadata = json.loads(
            args.episode_weight_map.read_text(encoding="utf-8")
        )
        raw_weights = episode_weight_metadata.get("weights") or {}
        for raw_key, raw_weight in raw_weights.items():
            parts = str(raw_key).split(":")
            if len(parts) != 2:
                parser.error(
                    f"invalid episode weight key {raw_key!r}; expected episode:seat"
                )
            weight = float(raw_weight)
            if weight <= 0:
                parser.error(f"episode weight must be positive: {raw_key}={weight}")
            episode_weights[(int(parts[0]), int(parts[1]))] = weight
    if args.rating_weight:
        missing_ratings = sorted(
            set(map(int, corpus.team_ids[train])) - set(ratings)
        )
        if missing_ratings:
            raise SystemExit(
                "--rating-weight requires a rating for every training team; "
                f"missing={missing_ratings}"
            )
    train_set = make_dataset(
        corpus, train,
        focus_team=args.focus_team, focus_weight=args.focus_weight,
        win_weight=args.win_weight,
        hard_state_weight=args.hard_state_weight,
        hard_state_set=args.hard_state_set,
        rating_weight=args.rating_weight, ratings=ratings,
        episode_weights=episode_weights,
        episode_equal_weight=args.episode_equal_weight,
        teacher_equal_weight=args.teacher_equal_weight,
    )
    validation_set = make_dataset(corpus, validation, reference=train_set)

    params: dict[str, Any] = {
        "objective": args.objective,
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.lambda_l2,
        "seed": args.seed,
        "verbosity": -1,
        "num_threads": args.threads,
        # No built-in metric. lgb.early_stopping stops on whichever tracked
        # metric stalls first, so leaving ndcg@k enabled would let NDCG pick
        # the iteration count for a model deployed on strict Top-1. That is
        # the v33 defect; Top1Metric below is the only metric.
        "metric": "None",
    }
    if args.objective == "lambdarank":
        params["lambdarank_truncation_level"] = 12
        params["label_gain"] = [0, 1]

    metric = Top1Metric(corpus, validation)
    evals: dict[str, dict[str, list[float]]] = {}
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[validation_set],
        valid_names=["validation"],
        feval=metric,
        callbacks=[
            lgb.early_stopping(args.early_stopping, first_metric_only=False),
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals),
        ],
    )

    results: dict[str, Any] = {
        "corpus": str(args.corpus.resolve()),
        "teams": sorted(teams) if teams else "all",
        "leave_out_team": args.leave_out_team,
        "focus_team": args.focus_team,
        "focus_weight": args.focus_weight,
        "win_weight": args.win_weight,
        "hard_state_weight": args.hard_state_weight,
        "hard_state_set": args.hard_state_set,
        "hard_state_support": {
            split: {
                name: int(mask.sum())
                for name, mask in hard_state_masks(
                    corpus, block, args.hard_state_set
                ).items()
            }
            for split, block in (
                ("train", train),
                ("validation", validation),
                ("test", test),
            )
        },
        "rating_weight": args.rating_weight,
        "episode_equal_weight": bool(args.episode_equal_weight),
        "teacher_equal_weight": bool(args.teacher_equal_weight),
        "ratings": {
            str(team): rating for team, rating in sorted(ratings.items())
        },
        "episode_weight_map": (
            str(args.episode_weight_map.resolve()) if args.episode_weight_map else None
        ),
        "episode_weight_entries": len(episode_weights),
        "episode_weight_metadata": (
            {key: value for key, value in episode_weight_metadata.items()
             if key != "weights"}
            if episode_weight_metadata else None
        ),
        "eval_team": args.eval_team,
        # Support per context decides which contexts the runtime is allowed to
        # route through the ranker: context 8 had 9 held-out decisions and
        # scored 22%, which is noise, not a policy.
        "train_context_support": {
            str(int(context)): int(count)
            for context, count in zip(
                *np.unique(corpus.contexts[train], return_counts=True)
            )
        },
        "params": params,
        "best_iteration": int(booster.best_iteration),
        "num_boost_round": args.num_boost_round,
        "team_feature": bool(args.team_feature),
        "split_mode": args.split_mode,
        "split_boundaries": split_boundaries,
        "split_decisions": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "split_episodes": {
            name: int(len(np.unique(corpus.episode_ids[block])))
            for name, block in
            (("train", train), ("validation", validation), ("test", test))
        },
        "split_submissions": {
            name: int(len(np.unique(corpus.submission_ids[block])))
            for name, block in
            (("train", train), ("validation", validation), ("test", test))
        },
    }

    for name, block in (("validation", validation), ("test", test)):
        matrix = corpus.matrix(block)
        scores = booster.predict(
            matrix, num_iteration=booster.best_iteration
        )
        del matrix
        sizes = corpus.groups[block]
        offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
        correct = top1(scores, corpus, block, offsets)
        hits = int(correct.sum())
        low, high = wilson(hits, len(block))
        results[name] = {
            "decisions": int(len(block)),
            "top1": round(float(correct.mean()), 4),
            "top1_wilson95": [low, high],
            "top2": topk(corpus, block, offsets, scores, 2),
            "top3": topk(corpus, block, offsets, scores, 3),
            "taxonomy": error_taxonomy(
                corpus, block, offsets, scores, corpus.names
            ),
            "top1_by_teacher_action": {},
        }
        by_action: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for slot, decision in enumerate(block):
            action = int(corpus.teacher_action_types[decision])
            by_action[action][0] += 1
            by_action[action][1] += int(correct[slot])
        results[name]["top1_by_teacher_action"] = {
            str(action): {
                "decisions": total,
                "top1": round(agree / total, 4),
            }
            for action, (total, agree) in sorted(by_action.items())
        }
        by_context: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for slot, decision in enumerate(block):
            context = int(corpus.contexts[decision])
            by_context[context][0] += 1
            by_context[context][1] += int(correct[slot])
        results[name]["top1_by_context"] = {
            str(context): {
                "decisions": total,
                "top1": round(agree / total, 4),
            }
            for context, (total, agree) in sorted(
                by_context.items(), key=lambda kv: -kv[1][0]
            )
        }

        by_team: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for slot, decision in enumerate(block):
            team = int(corpus.team_ids[decision])
            by_team[team][0] += 1
            by_team[team][1] += int(correct[slot])
        results[name]["top1_by_team"] = {
            str(team): {
                "decisions": total,
                "top1": round(agree / total, 4),
            }
            for team, (total, agree) in sorted(
                by_team.items(), key=lambda kv: -kv[1][1] / max(1, kv[1][0])
            )
        }
        results[name]["top1_by_hard_state"] = {}
        for slice_name, mask in hard_state_masks(
            corpus, block, args.hard_state_set
        ).items():
            total = int(mask.sum())
            results[name]["top1_by_hard_state"][slice_name] = {
                "decisions": total,
                "top1": (
                    round(float(correct[mask].mean()), 4) if total else None
                ),
            }

    importance = sorted(
        zip(corpus.names, booster.feature_importance("gain")),
        key=lambda item: -item[1],
    )
    results["top_features"] = [
        {"name": name, "gain": round(float(gain), 1)}
        for name, gain in importance[:30]
    ]
    results["validation_curve_tail"] = {
        key: [round(float(v), 4) for v in values[-5:]]
        for key, values in evals.get("validation", {}).items()
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.output_model:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(
            str(args.output_model), num_iteration=booster.best_iteration
        )

    print(json.dumps({
        "best_iteration": results["best_iteration"],
        "validation_top1": results["validation"]["top1"],
        "test_top1": results["test"]["top1"],
        "test_top1_wilson95": results["test"]["top1_wilson95"],
        "test_top3": results["test"]["top3"],
        "test_order_insensitive_top1":
            results["test"]["taxonomy"]["order_insensitive_top1"],
        "test_taxonomy": results["test"]["taxonomy"]["rates"],
        "test_top1_by_context": results["test"]["top1_by_context"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
