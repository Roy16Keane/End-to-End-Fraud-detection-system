from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict

from fraud.inference.predictor import FraudPredictor


ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

app = FastAPI(title="Fraud Detection API", version="0.1.0")

predictor = FraudPredictor(artifacts_dir=ARTIFACTS_DIR, model_dir=MODEL_DIR)


class PredictRequest(BaseModel):
    # Keep it flexible: accept any transaction fields as dict
    transaction: Dict[str, Any] = Field(..., description="Single transaction fields as key/value pairs")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    fraud_proba: float
    fraud_label: int
    threshold: float


@app.on_event("startup")
def _startup():
    predictor.load()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        out = predictor.predict(req.transaction, threshold=req.threshold)
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
