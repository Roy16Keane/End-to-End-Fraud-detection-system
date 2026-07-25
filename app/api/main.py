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
    version="0.3.0",
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

EXPLANATIONS_TOTAL = Counter(
    "fraud_model_explanations_total",
    "Total number of successful local SHAP explanations",
)

EXPLANATION_DURATION_SECONDS = Histogram(
    "fraud_model_explanation_duration_seconds",
    "Time spent generating local SHAP explanations",
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


# Initialise labelled counters so they appear before first use.
PREDICTIONS_TOTAL.labels(prediction="fraud")
PREDICTIONS_TOTAL.labels(prediction="normal")

PREDICTION_ERRORS_TOTAL.labels(error_type="validation")
PREDICTION_ERRORS_TOTAL.labels(error_type="internal")



# Application configuration


ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

predictor = FraudPredictor(
    artifacts_dir=ARTIFACTS_DIR,
)



# Request models


class PredictRequest(BaseModel):
    transaction: Dict[str, Any] = Field(
        ...,
        description="Raw transaction fields required by the fraud model.",
    )

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold used to assign the fraud label.",
    )

    explain: bool = Field(
        default=False,
        description="Whether to include a local SHAP explanation.",
    )

    max_explanation_features: int = Field(
        default=6,
        ge=1,
        le=20,
        description=(
            "Maximum number of risk-increasing and "
            "risk-reducing features returned."
        ),
    )



# Application lifecycle


@app.on_event("startup")
def startup() -> None:
    """
    Load the model, feature engineering artifacts and SHAP explainer.

    CI and unit tests can skip loading real artifacts by setting TEST_MODE=1.
    """
    if TEST_MODE:
        return

    predictor.load()


# API endpoints


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "test_mode": TEST_MODE,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    """
    Generate a fraud prediction.

    A local SHAP explanation is generated only when explain=True.
    """
    request_start_time = time.perf_counter()

    try:
        if TEST_MODE:
            result: Dict[str, Any] = {
                "fraud_proba": 0.123,
                "fraud_label": 0,
                "threshold": float(req.threshold),
            }

            if req.explain:
                result["explanation"] = {
                    "available": False,
                    "reason": "SHAP explanations are disabled in test mode.",
                }

        else:
            if req.explain:
                explanation_start_time = time.perf_counter()

                try:
                    result = predictor.predict(
                        payload=req.transaction,
                        threshold=req.threshold,
                        explain=True,
                        max_explanation_features=(
                            req.max_explanation_features
                        ),
                    )
                finally:
                    explanation_duration = (
                        time.perf_counter() - explanation_start_time
                    )

                    EXPLANATION_DURATION_SECONDS.observe(
                        explanation_duration
                    )

            else:
                result = predictor.predict(
                    payload=req.transaction,
                    threshold=req.threshold,
                    explain=False,
                )

        fraud_label = int(result["fraud_label"])
        fraud_probability = float(result["fraud_proba"])

        prediction_type = (
            "fraud"
            if fraud_label == 1
            else "normal"
        )

        PREDICTIONS_TOTAL.labels(
            prediction=prediction_type
        ).inc()

        if (
            req.explain
            and not TEST_MODE
            and "explanation" in result
        ):
            EXPLANATIONS_TOTAL.inc()

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

    except RuntimeError as exc:
        PREDICTION_ERRORS_TOTAL.labels(
            error_type="internal"
        ).inc()

        raise HTTPException(
            status_code=503,
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
        request_duration = (
            time.perf_counter() - request_start_time
        )

        PREDICTION_DURATION_SECONDS.observe(
            request_duration
        )


print(
    "TEST_MODE =",
    TEST_MODE,
    "ARTIFACTS_DIR =",
    ARTIFACTS_DIR,
)