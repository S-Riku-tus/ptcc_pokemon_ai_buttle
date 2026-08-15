"""A last-mile deck-out veto for optional Lillie's Determination draws.

v25 lost two games while ahead on prizes after several optional deck-thinning
actions shortened an already stalled clock.  This guard does not globally
suppress search or Punk Up.  It only replaces Lillie when shuffling the rest
of the hand back and drawing six (eight while all six prizes remain) can empty
the deck, and only when we are ahead or already have a ready attacker.
The best v22-scored non-Lillie action is used, so all normal sequencing still
runs before the turn eventually closes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import ml_features as mf


class DeckClockGuard:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stats: Counter[str] = Counter()
        self.records: list[dict[str, int]] = []

    def adjust(self, observation: dict[str, Any], select: dict[str, Any],
               index: int, scores: dict[int, float]) -> int:
        try:
            options = list(select.get("option") or [])
            if (
                int(select.get("context", -1)) != mf.MAIN_CONTEXT
                or not 0 <= index < len(options)
            ):
                return index
            current = observation.get("current") or {}
            players = current.get("players") or []
            your = int(current.get("yourIndex", 0) or 0)
            if len(players) < 2 or your not in (0, 1):
                return index
            me, opponent = players[your], players[1 - your]
            chosen = mf.candidate_card(current, options[index], select)
            if int((chosen or {}).get("id", -1)) != mf.LILLIE_ID:
                return index
            self.stats["lillie_considered"] += 1
            turn = int(current.get("turn", 0) or 0)
            deck = int(me.get("deckCount", 0) or 0)
            hand_count = len(mf._cards(me, "hand"))
            if not hand_count:
                hand_count = int(me.get("handCount", 0) or 0)
            draw_target = 8 if len(me.get("prize") or []) == 6 else 6
            # Lillie itself has been played; every other hand card is first
            # returned to the deck.  This is the net number removed afterward.
            refill = max(0, draw_target - max(0, hand_count - 1))
            ready = any(
                int(body.get("id", -1)) == mf.GRIMMSNARL_EX_ID
                and mf._dark_energy_count(body) >= 2
                for body in mf._in_play(me)
            )
            prize_lead = len(opponent.get("prize") or []) - len(me.get("prize") or [])
            if turn < 10 or deck > refill or not (ready or prize_lead > 0):
                return index
            safe = []
            for candidate, score in scores.items():
                if not 0 <= candidate < len(options) or candidate == index:
                    continue
                card = mf.candidate_card(current, options[candidate], select)
                if int((card or {}).get("id", -1)) == mf.LILLIE_ID:
                    continue
                safe.append((float(score), candidate))
            if not safe:
                self.stats["no_alternative"] += 1
                return index
            replacement = max(safe)[1]
            self.stats["lillie_deckout_vetoed"] += 1
            if len(self.records) < 24:
                self.records.append({
                    "turn": turn,
                    "deck": deck,
                    "hand": hand_count,
                    "refill": refill,
                    "from_index": index,
                    "to_index": replacement,
                })
            return replacement
        except Exception:
            self.stats["errors"] += 1
            return index

    def snapshot(self) -> dict[str, Any]:
        return {**dict(self.stats), "records": list(self.records)}
