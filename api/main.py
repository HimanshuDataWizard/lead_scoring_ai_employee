from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from lead_scoring.config import FEATURE_COLUMNS
from lead_scoring.inference import load_artifacts, predict_one

app = FastAPI(
    title="Lead Scoring API",
    version="2.0.0",
    description="Predicts conversion probability and lead priority using the trained pipeline.",
)


class LeadRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    company_size: int = Field(..., ge=1)
    industry: str = Field(..., min_length=1)
    region: str = Field(..., min_length=1)
    prev_purchases: int = Field(..., ge=0)
    response_time: int = Field(..., ge=0)
    last_contact: int = Field(..., ge=0)
    source: str = Field(..., min_length=1)
    deal_value: float = Field(..., ge=0)


model = None
metrics = {}


@app.on_event("startup")
def startup() -> None:
    global model, metrics
    try:
        model, metrics = load_artifacts()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Model artifacts were not found. Run `python3 scripts/train.py` first."
        ) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "features": FEATURE_COLUMNS}


@app.get("/model-info")
def model_info() -> dict:
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics metadata not found.")
    return metrics


@app.post("/predict")
def predict(body: LeadRequest) -> dict:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    threshold = metrics.get("decision_threshold", 0.5) if metrics else 0.5
    output = predict_one(model, body.model_dump(), threshold=threshold)
    return {
        "input": body.model_dump(),
        **output,
    }
