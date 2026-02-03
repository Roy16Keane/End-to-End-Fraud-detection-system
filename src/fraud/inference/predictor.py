from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from fraud.utils.io import load_joblib
from fraud.features.preprocessing import make_numeric_matrix


@dataclass
class InferenceArtifacts:
    featurizer: Any
    feature_cols: list[str]
    dropped_cols: list[str]


class FraudPredictor:
    """
    Loads artifacts + model once and serves predictions.
    """

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        model_dir: str = "models",
        model_filename: str = "xgb_model.json",
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / model_filename

        self._artifacts: Optional[InferenceArtifacts] = None
        self._model: Optional[xgb.XGBClassifier] = None

    def load(self) -> "FraudPredictor":
        # ---- Load featurizer + meta ----
        featurizer = load_joblib(str(self.artifacts_dir / "featurizer.joblib"))
        meta = load_joblib(str(self.artifacts_dir / "train_meta.joblib"))

        feature_cols = meta.get("feature_cols", [])
        dropped_cols = meta.get("dropped_cols", [])

        # ---- Load XGBoost model ----
        clf = xgb.XGBClassifier()
        clf.load_model(str(self.model_path))

        self._artifacts = InferenceArtifacts(
            featurizer=featurizer,
            feature_cols=list(feature_cols),
            dropped_cols=list(dropped_cols),
        )
        self._model = clf
        return self

    def _ensure_loaded(self) -> None:
        if self._artifacts is None or self._model is None:
            raise RuntimeError("Predictor not loaded. Call load() at startup.")

    def _prepare_features(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """
        payload -> single-row DataFrame -> featurize -> align to training columns
        """
        self._ensure_loaded()
        assert self._artifacts is not None

        # 1) single row
        df = pd.DataFrame([payload])

        # 2) apply same feature engineering
        df_feat = self._artifacts.featurizer.transform(df)

        # 3) drop leakage cols if present
        df_feat = df_feat.drop(columns=[c for c in ["TransactionID", "UID", "isFraud"] if c in df_feat.columns], errors="ignore")

        # 4) numeric convert + fill
        X = make_numeric_matrix(df_feat)

        # 5) drop training-time allnan/constant cols
        X = X.drop(columns=self._artifacts.dropped_cols, errors="ignore")

        # 6) align to training feature columns (order + missing)
        #    - add missing columns with -1
        #    - drop extras
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

        # XGBClassifier predict_proba returns [p0, p1]
        proba = float(self._model.predict_proba(X)[0, 1])
        return proba

    def predict(self, payload: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
        proba = self.predict_proba(payload)
        label = int(proba >= threshold)
        return {"fraud_proba": proba, "fraud_label": label, "threshold": float(threshold)}
