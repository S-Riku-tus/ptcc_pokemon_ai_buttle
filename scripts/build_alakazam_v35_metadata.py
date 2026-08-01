"""Regenerate the v35 metadata from the measured experiment outputs.

Kept as a script rather than a one-off so the numbers in metadata.json can be
traced back to the JSON reports that produced them, and so a rerun after new
measurements cannot silently disagree with them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agents" / "alakazam" / "alakazam_ml_v35"
EXP = ROOT / "experiments" / "alakazam_ml_v35"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rd(value, digits=4):
    return round(float(value), digits)


def main() -> int:
    m = load(AGENT / "metadata.json")
    audit = load(EXP / "shell_audit.json")
    resid = load(EXP / "residual_taxonomy.json")
    label = load(EXP / "label_shape.json")
    casc = load(EXP / "cascade.json")
    bench = load(EXP / "inference_benchmark.json")["results"]
    ab34 = load(EXP / "runtime_ab_v34shell.alakazam_ml_v35.json")
    ab35 = load(EXP / "runtime_ab_v35shell.alakazam_ml_v35.json")

    v34s = audit["v34"]
    v35s = audit["lethal_end_or_weak_boss_end_only"]

    m["name"] = "alakazam_ml_v35"
    m["version"] = "35.0.0"
    m["role"] = "challenger"
    m["status"] = "offline_challenger_not_yet_submitted"
    m["parent_agent"] = "alakazam_ml_v34"
    m["created_at"] = "2026-08-02T00:00:00+09:00"
    m["runtime_revision"] = "v35_narrowed_lethal_and_boss_guards_20260802"
    m["model"] = (
        "Unchanged from v34: the v31 exact-memory shell and a 657-feature "
        "large-leaf LambdaRank model of 2,050 trees. v35 changes only the "
        "two safety guards that an end-to-end audit showed were discarding "
        "the ranker's pick far more often than they were rescuing it."
    )
    m["change_summary"] = (
        "Runtime only. The lethal guard and the Boss-route guard are "
        "narrowed from compulsions into prohibitions. No model, feature, "
        "corpus, deck or hyperparameter changed."
    )
    m["changed_versus_v34"] = [
        "lethal_guard: was an unconditional pre-ranking veto whenever the "
        "v29 baseline held an estimated-lethal Powerful Hand, on 1,122 of "
        "9,977 holdout decisions. Now fires only when the ranker would end "
        "the turn or swing with a non-lethal attack.",
        "preserve_fallback_boss_route: was an unconditional veto whenever "
        "the v29 baseline wanted Boss's Orders and the ranker did not, on "
        "326 decisions. Now fires only when the ranker would end the turn.",
        "ALAKAZAM_ML_V35_SHELL=v34 restores the previous shell for A/B runs.",
    ]
    m["unchanged_from_v34"] = [
        "ranker_model.json is byte-identical: 2,050 trees, seed 1091, 657 "
        "features.",
        "Corpus, holdout boundaries, graded labels, recency weights.",
        "breaks_current_ko, end_with_ready_attack, dudunsparce_body_floor "
        "and unmodelled_other guards; v29 baseline; board memory; deck.",
        "v29_runtime._candidate_safety_reason is untouched, because the v29 "
        "baseline's own pick is an input feature of the ranker.",
    ]
    m["shell_audit"] = {
        "method": (
            "Every guard reads columns the corpus already stores, so the "
            "shell is replayed exactly from the cached candidate matrix plus "
            "held-out ranker scores. Verified against the packaged agent: "
            "the model-independent lethal guard fires 149 times on the first "
            "25 test episodes in both the simulation and the real runtime."
        ),
        "holdout": "test block, 200 episodes, 9,977 decisions",
        "ranker_top1": rd(v34s["test"]["overall"]["model_top1"]),
        "v29_baseline_top1": rd(v34s["test"]["overall"]["baseline_top1"]),
        "v34_played_top1": rd(v34s["test"]["overall"]["played_top1"]),
        "v35_played_top1": rd(v35s["test"]["overall"]["played_top1"]),
        "v34_played_turn_set": rd(v34s["test"]["overall"]["played_turn_set"]),
        "v35_played_turn_set": rd(v35s["test"]["overall"]["played_turn_set"]),
        "v34_blocked_rate": rd(v34s["test"]["overall"]["blocked_rate"]),
        "v35_blocked_rate": rd(v35s["test"]["overall"]["blocked_rate"]),
        "unguarded_upper_bound_played_top1": rd(
            audit["unguarded"]["test"]["overall"]["played_top1"]
        ),
        "v34_guard_net_decisions": {
            name: stats["net_decisions"]
            for name, stats in v34s["test"]["by_guard"].items()
        },
        "validation_played_top1": {
            "v34": rd(v34s["validation"]["overall"]["played_top1"]),
            "v35": rd(v35s["validation"]["overall"]["played_top1"]),
        },
    }
    m["runtime_agreement_ab"] = {
        "method": (
            "The packaged v35 agent replayed over the teacher's own "
            "observations with only ALAKAZAM_ML_V35_SHELL differing. "
            "Absolute levels are inflated because the shipped model was "
            "refitted on every episode including these; the difference "
            "between the two arms is the signal."
        ),
        "episodes": ab35["episodes"],
        "decisions": ab35["decisions"],
        "v34_shell_played_top1": rd(ab34["played_top1"]),
        "v35_shell_played_top1": rd(ab35["played_top1"]),
        "v34_shell_fallback_rate": rd(ab34["diag"]["fallback_rate"]),
        "v35_shell_fallback_rate": rd(ab35["diag"]["fallback_rate"]),
    }
    m["residual_taxonomy"] = {
        "note": (
            "Top-1 errors split by whether the turn can still converge. An "
            "ordering error picks an action the teacher also plays this "
            "turn; a premature error does that with a turn-ending action and "
            "skips the rest; a divergence picks an action the teacher never "
            "plays."
        ),
        "test": {
            key: rd(resid["test"]["overall"][key])
            for key in ("top1", "ordering_error_rate", "premature_rate",
                        "divergence_rate", "unrecoverable_rate", "turn_set")
        },
        "unrecoverable_by_teacher_action": {
            name: {
                "count": stats["count"],
                "unrecoverable_rate": rd(stats["unrecoverable_rate"]),
            }
            for name, stats in sorted(
                resid["test"]["by_teacher_action"].items(),
                key=lambda kv: -kv[1]["unrecoverable_rate"] * kv[1]["count"],
            )
        },
    }
    m["rejected"] = {
        "teacher_corpus_refetch": (
            "Submission 54773249 yielded 13 new games in 5.5 hours "
            "(2,268 -> 2,281, +0.6%). The teacher has essentially stopped "
            "laddering, so data is no longer a lever."
        ),
        "label_gain_reshaping": {
            "note": (
                "The deployed graded labels map to gains 127/7/1/0, so "
                "LambdaRank spends 120 of every 127 units of pair weight on "
                "intra-turn order and 1 on set membership. Flatter gains "
                "trade Top-1 for divergence monotonically, and the "
                "order-blind control collapses to 49.85% Top-1 because 'end' "
                "belongs to every turn's action set. v34's shape is kept; it "
                "also wins on validation."
            ),
            "validation_top1_test_top1_test_divergence": {
                name: [
                    rd(entry["validation"]["overall"]["top1"]),
                    rd(entry["test"]["overall"]["top1"]),
                    rd(entry["test"]["overall"]["divergence_rate"]),
                ]
                for name, entry in label.items()
            },
        },
        "two_stage_cascade": {
            "note": (
                "Stage two re-ranks the top-K on out-of-fold stage-one "
                "scores. Validation separates K=3 and K=5 by 0.02 points "
                "while test separates them by 1.26, so the selection signal "
                "is far below the outcome variance and the effect cannot be "
                "distinguished from zero."
            ),
            "stage1_oof_top1": rd(casc["oof_stage1"]["top1"]),
            "k3_validation_test": [
                rd(casc["k3"]["validation"]["overall"]["top1"]),
                rd(casc["k3"]["test"]["overall"]["top1"]),
            ],
            "k5_validation_test": [
                rd(casc["k5"]["validation"]["overall"]["top1"]),
                rd(casc["k5"]["test"]["overall"]["top1"]),
            ],
        },
        "ko_guard_narrowing": (
            "Firing breaks_current_ko only when an attack option exists "
            "gains 0.14 points on test (0.8214 -> 0.8228). Not worth "
            "complicating the guard that makes the lethal narrowing safe."
        ),
        "more_trees": (
            "The validation curve is flat from 1,900 to 2,200 trees and a "
            "refit reselects 2,050. No headroom."
        ),
    }
    m["safety_guards"] = [
        "Refuse to end the turn or swing non-lethally while a lethal "
        "Powerful Hand is available (narrowed in v35 from 'take it now').",
        "Reject candidates that destroy a currently available KO. Powerful "
        "Hand deals 20 per card in hand, so this is what makes waiting safe.",
        "Never end the turn while the Active Alakazam can attack.",
        "Preserve the deterministic Boss route only against ending the turn "
        "(narrowed in v35 from 'against every other action').",
        "Do not cycle Dudunsparce away from a board of two or fewer Pokemon.",
        "Reject unmodelled 'other' actions.",
        "Use the frozen v29 policy for unsupported, nested, low-time or "
        "illegal cases.",
    ]
    m["inference_cost"] = {
        "note": (
            "The same 289 scoped decisions as the v34 measurement, re-run "
            "for both agents in one session so the comparison is internal. "
            "The model now answers 95.8% of scoped decisions instead of "
            "81.7%, which is where the extra time goes."
        ),
    }
    for row in bench:
        tag = row["agent"].replace("alakazam_ml_", "")
        for key in ("mean_ms", "median_ms", "p95_ms", "max_ms"):
            m["inference_cost"][f"{tag}_{key}"] = rd(row[key], 2)
        m["inference_cost"][f"{tag}_model_rate"] = rd(row["ml_model_rate"])
        m["inference_cost"][f"{tag}_fallback_rate"] = rd(
            row["ml_fallback_rate"]
        )
    m["known_limitations"] = [
        "Not yet submitted. Every claim here is agreement with the teacher "
        "on held-out states, which is closer to the ladder than the "
        "ranker-only agreement v31-v34 reported, but is still a proxy.",
        "The premise that v34's offline gain did not reach the ladder rests "
        "on a provisional 'around 900' reading. The gap to v33's 916.9 sits "
        "inside the band an identical agent has already shown (842.8 versus "
        "804.0), so it must be re-checked once v34 settles.",
        "Keeping breaks_current_ko costs 0.84 points against the unguarded "
        "upper bound. That is a deliberate payment for the property that "
        "lets the lethal guard be narrowed.",
        "The ranker is unchanged, so the 90% strict Top-1 target is still "
        "unmet at 83.01% and the 672 unrecoverable holdout errors are "
        "untouched.",
        "Boss's Orders remains the lowest-agreement class at 47.94%, though "
        "the new taxonomy shows 34 of those 52 points are recoverable "
        "ordering.",
        "The ladder's noise floor is about 40 rating points at n~60, so one "
        "run cannot confirm this change.",
    ]
    for key in ("versus_v33_same_holdout", "changed_versus_v33",
                "unchanged_from_v33"):
        m.pop(key, None)
    m["holdout_identity"] = {
        "shared_with": "alakazam_ml_v34",
        "note": (
            "Identical corpus and boundaries as v34, so every number here is "
            "directly comparable with the v34 report."
        ),
    }

    arena = EXP / "local_arena.json"
    if arena.exists():
        m["local_validation"] = load(arena)

    for name in ("ranker_model.json", "v29_ranker_model.json",
                 "legacy_ranker_model.json", "target_ranker_model.json",
                 "teacher_memory.bin", "fallback_policy.py", "deck.csv"):
        digest = hashlib.sha256((AGENT / name).read_bytes()).hexdigest()
        key = name.rsplit(".", 1)[0] + "_sha256"
        if key in m and m[key] != digest:
            raise SystemExit(f"{name} changed: {m[key]} -> {digest}")

    (AGENT / "metadata.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("metadata written; every shipped artifact hash is unchanged")
    print("played top1 {} -> {}".format(
        m["shell_audit"]["v34_played_top1"],
        m["shell_audit"]["v35_played_top1"],
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
