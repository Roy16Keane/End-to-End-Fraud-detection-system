from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from fraud.features.preprocessing import FraudFeaturizer, make_numeric_matrix, drop_allnan_and_constant_cols
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
    random_state=42
)


def train_final_model(
    df_train: pd.DataFrame,
    target_col: str = "isFraud",
    params: dict | None = None,
    artifacts_dir: str = "artifacts",
    model_dir: str = "models",
) -> dict:
    params = params or DEFAULT_PARAMS

    # 1) Fit featurizer on full training data (LOCKED feature logic)
    featurizer = FraudFeaturizer().fit(df_train.drop(columns=[target_col], errors="ignore"))

    # 2) Transform full train
    X = featurizer.transform(df_train.drop(columns=[target_col], errors="ignore"))
    y = df_train[target_col].astype(int).values

    # 3) Drop leakage columns if present
    X = X.drop(columns=[c for c in ["TransactionID", "UID"] if c in X.columns], errors="ignore")

    # 4) Numeric matrix + drop constants
    X = make_numeric_matrix(X)
    X, dropped = drop_allnan_and_constant_cols(X)

    # 5) Train model on all data (no validation here; validation already done in notebook/CV)
    clf = xgb.XGBClassifier(**params)
    clf.fit(X, y, verbose=False)

    # 6) Persist artifacts + model
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    save_joblib(featurizer, f"{artifacts_dir}/featurizer.joblib")
    save_joblib({"dropped_cols": dropped, "feature_cols": X.columns.tolist()}, f"{artifacts_dir}/train_meta.joblib")

    # xgb native save
    clf.save_model(f"{model_dir}/xgb_model.json")

    meta = {
        "n_features": int(X.shape[1]),
        "params": params,
        "dropped_cols": dropped,
    }
    with open(f"{model_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta
