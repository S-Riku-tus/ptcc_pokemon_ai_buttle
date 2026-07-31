"""Recompute the v33 intra-turn columns in place.

Only the eight appended columns are re-derived; the extracted base features,
labels, groups and splits are reused, so this avoids a full replay pass when
the intra-turn definition changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_alakazam_v33_corpus import (  # noqa: E402
    TURN_FEATURES,
    build_turn_state,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as cached:
        payload = {key: cached[key] for key in cached.files}
    names = payload["feature_names"].astype(str).tolist()
    if names[-len(TURN_FEATURES):] != list(TURN_FEATURES):
        raise RuntimeError("cache does not end with the v33 turn columns")

    base_names = names[:-len(TURN_FEATURES)]
    base = payload["features"][:, :len(base_names)]
    extra = build_turn_state(
        base, payload["labels"], payload["groups"],
        payload["episode_ids"], base_names,
    )
    changed = int(
        np.count_nonzero(
            payload["features"][:, len(base_names):] != extra
        )
    )
    payload["features"] = np.concatenate([base, extra], axis=1)
    np.savez_compressed(args.cache, **payload)
    print(f"rewrote {args.cache} ({changed} turn-column cells changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
