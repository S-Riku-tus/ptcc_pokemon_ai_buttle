from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
import pandas as pd

from .features import option_features


class SafeHybridRanker:
    """Rank only supplied legal actions and fall back when confidence is low.

    The surrounding submission agent remains responsible for enumerating legal options and
    implementing the proven rule-based fallback. This class never emits an action outside
    the caller-provided legal option list.
    """

    def __init__(self, artifact_dir: str | Path):
        artifact_dir = Path(artifact_dir)
        self.model = joblib.load(artifact_dir / "ranker.joblib")
        self.schema = json.loads((artifact_dir / "model_schema.json").read_text(encoding="utf-8"))

    def _matrix(self, current: dict[str, Any], select: dict[str, Any], legal_options: Sequence[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame([option_features(current, select, option) for option in legal_options])
        columns = self.schema["feature_columns"]
        maps = self.schema.get("category_maps", {})
        for column in columns:
            if column not in frame:
                frame[column] = -1
            if column == "action_type":
                action_map = self.schema.get("action_type_map", maps.get(column, {}))
                frame[column] = frame[column].astype(str).map(action_map).fillna(-1).astype(int)
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(-1)
        return frame[columns]

    def choose_index(
        self,
        current: dict[str, Any],
        select: dict[str, Any],
        legal_options: Sequence[dict[str, Any]],
        fallback: Callable[[], int],
    ) -> tuple[int, dict[str, Any]]:
        if not legal_options:
            raise ValueError("legal_options must not be empty")
        if len(legal_options) == 1:
            return 0, {"source": "forced", "confidence": 1.0, "margin": 1.0}
        matrix = self._matrix(current, select, legal_options)
        scores = np.asarray(self.model.predict(matrix), dtype=float)
        temperature = max(float(self.schema.get("temperature", 1.0)), 1e-6)
        shifted = scores / temperature
        shifted = shifted - shifted.max()
        probs = np.exp(np.clip(shifted, -50, 50)); probs /= probs.sum()
        order = np.argsort(-scores)
        best = int(order[0])
        confidence = float(probs[best])
        margin = float(probs[order[0]] - probs[order[1]])
        predicted_action_type = str(pd.DataFrame([option_features(current, select, legal_options[best])]).iloc[0]["action_type"])
        probability_threshold = float(self.schema.get("action_type_thresholds", {}).get(predicted_action_type, self.schema["fallback_probability"]))
        if confidence < probability_threshold or margin < float(self.schema["fallback_margin"]):
            fallback_index = int(fallback())
            if not (0 <= fallback_index < len(legal_options)):
                fallback_index = best
            return fallback_index, {"source": "fallback", "confidence": confidence, "margin": margin, "predicted_action_type": predicted_action_type}
        return best, {"source": "ml", "confidence": confidence, "margin": margin, "predicted_action_type": predicted_action_type}
