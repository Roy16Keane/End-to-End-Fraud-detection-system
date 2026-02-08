from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import xgboost as xgb
import mlflow

from sklearn.metrics import roc_auc_score

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


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def train_final_model(
    df_train: pd.DataFrame,
    target_col: str = "isFraud",
    params: dict | None = None,
    artifacts_dir: str = "artifacts",
    model_dir: str = "models",
    mlflow_experiment: str = "ieee-fraud-xgb",
) -> dict:
    params = params or DEFAULT_PARAMS

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # MLflow setup
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name="xgb_final_train") as run:
        run_id = run.info.run_id

        # ---------------------------
        # 0) Log run metadata
        # ---------------------------
        mlflow.log_params(params)
        mlflow.log_param("target_col", target_col)
        mlflow.log_param("git_commit", _git_commit())
        mlflow.log_param("data_file", "dataset/merged_data_pruned.parquet")

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

        mlflow.log_metric("n_features", float(len(feature_cols)))
        mlflow.log_metric("n_dropped_cols", float(len(dropped_cols)))

        # ---------------------------
        # 4) Train final model
        # ---------------------------
        clf = xgb.XGBClassifier(**params)
        clf.fit(X, y, verbose=False)

        # Minimal sanity metric (full train AUC)
        train_auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))
        mlflow.log_metric("train_auc", train_auc)
        # ---------------------------
        # 4b) Feature importance report
        # ---------------------------
        Path("reports").mkdir(parents=True, exist_ok=True)

        importances = clf.feature_importances_

        fi_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)

        fi_path = Path("reports") / "feature_importance.json"
        fi_df.to_json(fi_path, orient="records", indent=2)


        # ---------------------------
        # 5) Persist artifacts (DVC will version these)
        # ---------------------------
        save_joblib(featurizer, Path(artifacts_dir) / "featurizer.joblib")

        train_meta = {
            "feature_cols": feature_cols,
            "dropped_cols": dropped_cols,
            "params": params,
            "n_features": len(feature_cols),
            "train_auc": train_auc,
            "mlflow_run_id": run_id,         # 🔑 artifact ↔ MLflow linkage
            "git_commit": _git_commit(),
        }
        save_joblib(train_meta, Path(artifacts_dir) / "train_meta.joblib")

        save_joblib(clf, Path(model_dir) / "xgb_model.joblib")

        # Also write lightweight meta.json and log it to MLflow
        meta_path = Path(model_dir) / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(train_meta, f, indent=2)

        mlflow.log_artifact(str(meta_path))

        return train_meta
