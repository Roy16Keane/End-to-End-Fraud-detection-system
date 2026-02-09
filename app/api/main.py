import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud.inference.predictor import FraudPredictor

app = FastAPI(title="Fraud Detection API", version="0.1.0")

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
TEST_MODE = os.getenv('TEST_MODE','0') == '1'
predictor = FraudPredictor(artifacts_dir=ARTIFACTS_DIR)


class PredictRequest(BaseModel):
    transaction: Dict[str, Any] = Field(...)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


@app.on_event("startup")
def load_predictor():
     # In CI/unit tests we don't want to load real artifacts
    if TEST_MODE:
        return
    predictor.load()


@app.get("/health")
def health():
    return {"status": "ok"}

print("TEST_MODE =", TEST_MODE, "ARTIFACTS_DIR =", ARTIFACTS_DIR)


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # If TEST_MODE, return a deterministic dummy response
        if TEST_MODE:
            return {"fraud_proba": 0.123, "fraud_label": 0, "threshold": float(req.threshold)}
        return predictor.predict(req.transaction, threshold=req.threshold)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
   
