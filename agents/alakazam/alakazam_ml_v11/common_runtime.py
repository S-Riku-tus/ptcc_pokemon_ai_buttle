from __future__ import annotations

import hashlib
from typing import Any


MAJKEL_KEY_CARDS = {
    13, 19, 66, 140, 305, 343, 741, 742, 743, 1079, 1081, 1086,
    1097, 1129, 1152, 1182, 1184, 1197, 1225, 1231, 1266,
}
HIGH_IMPORTANCE_CARDS = {13, 19, 66, 1079, 1081, 1182, 1197, 1225, 1231}


def stable_code(value: Any, buckets: int = 4093) -> int:
    raw = str(value if value is not None else "").encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=4).digest(), "little") % buckets + 1

