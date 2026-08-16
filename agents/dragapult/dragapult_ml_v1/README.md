# Dragapult ML v1.1 (guarded hybrid)

This directory now contains a guarded version of the original
teacher-conditioned imitation baseline.  It is not a search agent.

The checked-in `ranker_model.json` is pinned to teacher 16380946.  The model is
still used for supported choices, but v1 did not encode attached Energy types.
The deterministic policy therefore owns direct attachments, evolution,
critical switches, Boss, and immediate evolution-line searches.  It completes
Fire + Psychic on a viable Dragapult route before spending a manual attachment
on Munkidori.  Optional choices, multi-pick choices, unseen candidates,
low-support contexts, and model errors also fall back as before.

The submitted v1 pilot went 3-4 in its first seven games.  It used Phantom Dive
in only four games and made four duplicate Fire/Psychic route attachments.  On
the same 529 live decisions, v1.1 produced no duplicate route attachment, with
100% legal actions and no exceptions.  Its frozen 102-game chronological test
also remained 100% legal with no exceptions; semantic agreement is 70.31%
because guarded choices deliberately depart from the untyped model.

This remains a guarded candidate, not proof of a target ladder rating.  The
corpus has only 854 verified trajectories, held-out top-3 imitation accuracy
(96.34%) missed the predeclared 97% gate, and the revised policy has not yet had
a sufficiently large untouched prospective outcome test.

Reproduction commands and the frozen teacher cohort live in
`experiments/dragapult_ml_v1/README.md`.
