import os
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from fraud.inference.predictor import FraudPredictor


app = FastAPI(
    title="Fraud Detection API",
    version="0.2.0",
)

# Standard HTTP metrics


Instrumentator().instrument(app).expose(app)

# Custom business metrics


PREDICTIONS_TOTAL = Counter(
    "fraud_model_predictions_total",
    "Total number of successful fraud predictions",
    ["prediction"],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "fraud_model_prediction_errors_total",
    "Total number of prediction requests that failed",
    ["error_type"],
)

PREDICTION_DURATION_SECONDS = Histogram(
    "fraud_model_prediction_duration_seconds",
    "Time spent processing fraud predictions",
    buckets=(
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
)

LAST_PREDICTION_TIMESTAMP = Gauge(
    "fraud_model_last_prediction_timestamp_seconds",
    "Unix timestamp of the most recent successful prediction",
)

LAST_FRAUD_PROBABILITY = Gauge(
    "fraud_model_last_prediction_probability",
    "Fraud probability returned by the most recent successful prediction",
)


# Initialise labelled counters so that they appear before first use.
PREDICTIONS_TOTAL.labels(prediction="fraud")
PREDICTIONS_TOTAL.labels(prediction="normal")
PREDICTION_ERRORS_TOTAL.labels(error_type="validation")
PREDICTION_ERRORS_TOTAL.labels(error_type="internal")

# Application configuration


ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

predictor = FraudPredictor(artifacts_dir=ARTIFACTS_DIR)


class PredictRequest(BaseModel):
    transaction: Dict[str, Any] = Field(...)
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )


@app.on_event("startup")
def startup() -> None:
    """
    Load model artifacts when the API starts.

    CI and unit tests can skip loading real artifacts by setting TEST_MODE=1.
    """
    if TEST_MODE:
        return

    predictor.load()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    """
    Generate a fraud prediction and update business metrics.
    """
    start_time = time.perf_counter()

    try:
        if TEST_MODE:
            result: Dict[str, Any] = {
                "fraud_proba": 0.123,
                "fraud_label": 0,
                "threshold": float(req.threshold),
            }
        else:
            result = predictor.predict(
                req.transaction,
                threshold=req.threshold,
            )

        fraud_label = int(result["fraud_label"])
        fraud_probability = float(result["fraud_proba"])

        prediction_type = "fraud" if fraud_label == 1 else "normal"

        PREDICTIONS_TOTAL.labels(
            prediction=prediction_type
        ).inc()

        LAST_PREDICTION_TIMESTAMP.set_to_current_time()
        LAST_FRAUD_PROBABILITY.set(fraud_probability)

        return result

    except ValueError as exc:
        PREDICTION_ERRORS_TOTAL.labels(
            error_type="validation"
        ).inc()

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        PREDICTION_ERRORS_TOTAL.labels(
            error_type="internal"
        ).inc()

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error.",
        ) from exc

    finally:
        duration = time.perf_counter() - start_time
        PREDICTION_DURATION_SECONDS.observe(duration)


print(
    "TEST_MODE =",
    TEST_MODE,
    "ARTIFACTS_DIR =",
    ARTIFACTS_DIR,
)