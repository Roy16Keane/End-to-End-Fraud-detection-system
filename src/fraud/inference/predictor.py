from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from fraud.utils.io import load_joblib
from fraud.features.preprocessing import make_numeric_matrix


@dataclass
class InferenceArtifacts:
    featurizer: Any
    feature_cols: list[str]
    dropped_cols: list[str]
    meta: Dict[str, Any]


class FraudPredictor:
    """
    Loads artifacts + model once and serves predictions.

    Expected files (from your new training script):
      artifacts/featurizer.joblib
      artifacts/train_meta.joblib
      models/xgb_model.joblib
      models/meta.json (optional)
    """

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        model_dir: str = "models",
        model_filename: str = "xgb_model.joblib",
        meta_filename: str = "train_meta.joblib",
        featurizer_filename: str = "featurizer.joblib",
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.model_dir = Path(model_dir)

        self.model_path = self.model_dir / model_filename
        self.meta_path = self.artifacts_dir / meta_filename
        self.featurizer_path = self.artifacts_dir / featurizer_filename

        self._artifacts: Optional[InferenceArtifacts] = None
        self._model = None

    def load(self) -> "FraudPredictor":
        # 1) featurizer (fit/transform object)
        featurizer = load_joblib(self.featurizer_path)

        # 2) training metadata (feature cols, dropped cols, params, etc.)
        meta = load_joblib(self.meta_path)

        feature_cols = meta.get("feature_cols", [])
        dropped_cols = meta.get("dropped_cols", [])

        # 3) model
        model = load_joblib(self.model_path)

        self._artifacts = InferenceArtifacts(
            featurizer=featurizer,
            feature_cols=list(feature_cols) if feature_cols else [],
            dropped_cols=list(dropped_cols) if dropped_cols else [],
            meta=meta if isinstance(meta, dict) else {},
        )
        self._model = model
        return self

    def _ensure_loaded(self) -> None:
        if self._artifacts is None or self._model is None:
            raise RuntimeError("Predictor not loaded. Call load() at startup.")

    def _prepare_features(self, payload: Dict[str, Any]) -> pd.DataFrame:
        self._ensure_loaded()
        assert self._artifacts is not None

        df = pd.DataFrame([payload])

        # same feature engineering as training
        df_feat = self._artifacts.featurizer.transform(df)

        # drop leakage cols if present
        df_feat = df_feat.drop(
            columns=[c for c in ["TransactionID", "UID", "isFraud"] if c in df_feat.columns],
            errors="ignore",
        )

        # numeric matrix + fill
        X = make_numeric_matrix(df_feat)

        # NOTE: dropped_cols were already removed during training by drop_allnan_and_constant_cols(X)
        # Keeping this drop is harmless and makes inference robust across versions:
        X = X.drop(columns=self._artifacts.dropped_cols, errors="ignore")

        # align to training columns (order + missing)
        train_cols = self._artifacts.feature_cols
        if train_cols:
            for c in train_cols:
                if c not in X.columns:
                    X[c] = -1
            X = X[train_cols]  # drop extras + order

        return X

    def predict_proba(self, payload: Dict[str, Any]) -> float:
        self._ensure_loaded()
        X = self._prepare_features(payload)
        return float(self._model.predict_proba(X)[0, 1])

    def predict(self, payload: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
        proba = self.predict_proba(payload)
        label = int(proba >= threshold)
        return {"fraud_proba": proba, "fraud_label": label, "threshold": float(threshold)}
