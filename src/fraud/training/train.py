from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from fraud.features.preprocessing import (
    make_numeric_matrix,
    drop_allnan_and_constant_cols,
)
from fraud.features.featurizer import FraudFeaturizer
from fraud.utils.io import save_joblib


DEFAULT_PARAMS = dict(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    missing=-1,
    random_state=42,
)


def train_final_model(
    df_train: pd.DataFrame,
    target_col: str = "isFraud",
    params: dict | None = None,
    artifacts_dir: str = "artifacts",
    model_dir: str = "models",
) -> dict:
    params = params or DEFAULT_PARAMS

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # 1) Fit featurizer ON DATA
    # ---------------------------
    featurizer = FraudFeaturizer().fit(
        df_train.drop(columns=[target_col], errors="ignore")
    )

    # ---------------------------
    # 2) Transform full training data
    # ---------------------------
    X = featurizer.transform(
        df_train.drop(columns=[target_col], errors="ignore")
    )
    y = df_train[target_col].astype(int).values

    # ---------------------------
    # 3) Numeric cleanup + drop constants
    # ---------------------------
    X = make_numeric_matrix(X, drop_cols=["TransactionID", "UID"])
    X, dropped_cols = drop_allnan_and_constant_cols(X)

    feature_cols = X.columns.tolist()

    # ---------------------------
    # 4) Train final model
    # ---------------------------
    clf = xgb.XGBClassifier(**params)
    clf.fit(X, y, verbose=False)

    # ---------------------------
    # 5) Persist artifacts
    # ---------------------------
    save_joblib(featurizer, Path(artifacts_dir) / "featurizer.joblib")

    train_meta = {
        "feature_cols": feature_cols,
        "dropped_cols": dropped_cols,
        "params": params,
        "n_features": len(feature_cols),
    }
    save_joblib(train_meta, Path(artifacts_dir) / "train_meta.joblib")

    save_joblib(clf, Path(model_dir) / "xgb_model.joblib")

    with open(Path(model_dir) / "meta.json", "w") as f:
        json.dump(train_meta, f, indent=2)

    return train_meta


    