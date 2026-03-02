from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd

from .config import FEATURE_COLUMNS, METRICS_PATH, MODEL_PATH


def load_artifacts() -> Tuple[Any, Dict]:
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    return model, metrics


def predict_one(model: Any, payload: Dict, threshold: Optional[float] = None) -> Dict:
    row = {feature: payload[feature] for feature in FEATURE_COLUMNS}
    frame = pd.DataFrame([row])

    probability = float(model.predict_proba(frame)[0, 1])
    decision_threshold = 0.5 if threshold is None else float(threshold)
    prediction = int(probability >= decision_threshold)

    if probability >= 0.75:
        band = "High"
    elif probability >= 0.45:
        band = "Medium"
    else:
        band = "Low"

    return {
        "prediction": prediction,
        "conversion_probability": round(probability, 4),
        "lead_priority": band,
        "decision_threshold": round(decision_threshold, 4),
    }
