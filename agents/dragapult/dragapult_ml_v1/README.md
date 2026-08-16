# Dragapult ML v1

This is the first conservative implementation of the new-deck plan.  It is a
teacher-conditioned imitation baseline, not a search agent.

The runtime intentionally falls back on optional choices, multi-pick choices,
unseen candidate identities, low-support contexts, or any model error.  The
checked-in `ranker_model.json` is pinned to teacher 16380946.  Without it the
directory is a legal deterministic Dragapult agent and is useful for smoke
tests only.

This version is an offline baseline, not a submission recommendation.  Its
102-game chronological runtime test had 100% legal actions and no exceptions,
but the corpus has only 854 verified trajectories and held-out top-3 imitation
accuracy (96.34%) missed the predeclared 97% gate.

Reproduction commands and the frozen teacher cohort live in
`experiments/dragapult_ml_v1/README.md`.
