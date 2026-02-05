import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud.inference.predictor import FraudPredictor

app = FastAPI(title="Fraud Detection API", version="0.1.0")

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
predictor = FraudPredictor(artifacts_dir=ARTIFACTS_DIR)


class PredictRequest(BaseModel):
    transaction: Dict[str, Any] = Field(...)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


@app.on_event("startup")
def startup():
    predictor.load()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return predictor.predict(req.transaction, threshold=req.threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
