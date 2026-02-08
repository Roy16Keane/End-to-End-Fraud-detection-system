from __future__ import annotations

import json
import subprocess
import os 
from dataclasses import asdict
from pathlib import Path
from typing import Any
import gc

import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
from sklearn.metrics import roc_auc_score

from fraud.features.preprocessing import SECONDS_IN_DAY, make_numeric_matrix, drop_allnan_and_constant_cols
from fraud.features.featurizer import FraudFeaturizer


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def add_month_group(df: pd.DataFrame, days_per_month: int = 30) -> pd.Series:
    return (df["TransactionDT"] // (days_per_month * SECONDS_IN_DAY)).astype(int)


def forward_month_cv(
    df_raw: pd.DataFrame,
    y: np.ndarray,
    featurizer: FraudFeaturizer,
    params: dict,
    min_train_months: int = 2,
    days_per_month: int = 30,
    max_folds: int = 4,
) -> dict[str, Any]:
    """
    Fold-safe forward month CV:
      - for each val month:
          fit featurizer on TRAIN months only
          transform TRAIN and VAL with that featurizer
          train model on TRAIN
          evaluate on VAL
    """
    df = df_raw.copy()
    df["month"] = add_month_group(df, days_per_month=days_per_month)

    months = np.sort(df["month"].unique())
    fold_rows: list[dict[str, Any]] = []
    aucs: list[float] = []
    folds_done = 0

    for i in range(min_train_months, len(months)):
        if folds_done >= max_folds:
            break
        val_month = int(months[i])

        tr_mask = df["month"].isin(months[:i])
        va_mask = df["month"].eq(val_month)

        df_tr = df.loc[tr_mask].drop(columns=["month"])
        df_va = df.loc[va_mask].drop(columns=["month"])

        y_tr = y[tr_mask.to_numpy()]
        y_va = y[va_mask.to_numpy()]

        # ---------- fit featurizer on TRAIN ONLY ----------
        fz = FraudFeaturizer().fit(df_tr.drop(columns=["isFraud"], errors="ignore"))

        X_tr = fz.transform(df_tr.drop(columns=["isFraud"], errors="ignore"))
        X_va = fz.transform(df_va.drop(columns=["isFraud"], errors="ignore"))

        # numeric + drop fold-specific constants (train-derived)
        X_tr = make_numeric_matrix(X_tr, drop_cols=["TransactionID", "UID"])
        X_va = make_numeric_matrix(X_va, drop_cols=["TransactionID", "UID"])

        X_tr, dropped = drop_allnan_and_constant_cols(X_tr)
        X_va = X_va.drop(columns=dropped, errors="ignore")

        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            
        )

        folds_done +=1

        preds = clf.predict_proba(X_va)[:, 1]
        auc = float(roc_auc_score(y_va, preds))
        aucs.append(auc)

        row = {
            "val_month": val_month,
            "train_rows": int(tr_mask.sum()),
            "val_rows": int(va_mask.sum()),
            "auc": auc,
            "n_features": int(X_tr.shape[1]),
        }
        fold_rows.append(row)

        print(
            f"Val month={val_month:>3} | train={row['train_rows']:>7} "
            f"| val={row['val_rows']:>6} | AUC={auc:.5f}"
        )
        # --- hard memory cleanup per fold ---
        del df_tr, df_va, X_tr, X_va, y_tr, y_va, clf, preds, fz
        gc.collect()


    aucs_np = np.array(aucs, dtype=float)
    summary = {
        "folds": int(len(aucs_np)),
        "mean_auc": float(aucs_np.mean()) if len(aucs_np) else float("nan"),
        "std_auc": float(aucs_np.std()) if len(aucs_np) else float("nan"),
        "min_auc": float(aucs_np.min()) if len(aucs_np) else float("nan"),
        "max_auc": float(aucs_np.max()) if len(aucs_np) else float("nan"),
        "folds_detail": fold_rows,
    }

    print("\nSummary:", {k: summary[k] for k in ["folds", "mean_auc", "std_auc", "min_auc", "max_auc"]})
    return summary


def evaluate_and_log(
    df_all: pd.DataFrame,
    params: dict,
    experiment_name: str = "ieee-fraud-xgb",
    run_name: str = "forward_month_cv",
    reports_dir: str = "reports",
) -> dict[str, Any]:
    """
    Creates an MLflow run and logs:
      - fold aucs
      - summary mean/std/min/max
      - JSON report artifact
    """
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    y = df_all["isFraud"].astype(int).values

    # We create a dummy featurizer instance; per-fold CV uses a fresh fit anyway.
    dummy_fz = FraudFeaturizer()

    
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.log_params(params)
        mlflow.log_param("cv_scheme", "forward_month")
        mlflow.log_param("min_train_months", 2)
        mlflow.log_param("days_per_month", 30)
        mlflow.log_param("git_commit", _git_commit())
        mlflow.log_param("data_file", "dataset/merged_data_pruned.parquet")

        summary = forward_month_cv(
            df_raw=df_all,
            y=y,
            featurizer=dummy_fz,
            params=params,
            min_train_months=2,
            days_per_month=30,
        )

        # Log summary metrics
        mlflow.log_metric("cv_mean_auc", summary["mean_auc"])
        mlflow.log_metric("cv_std_auc", summary["std_auc"])
        mlflow.log_metric("cv_min_auc", summary["min_auc"])
        mlflow.log_metric("cv_max_auc", summary["max_auc"])
        mlflow.log_metric("cv_folds", summary["folds"])

        # Log fold metrics
        for row in summary["folds_detail"]:
            mlflow.log_metric(f"fold_auc_month_{row['val_month']}", row["auc"])

        # Write + log report artifact
        report = {
            "mlflow_run_id": run_id,
            "git_commit": _git_commit(),
            "params": params,
            "summary": summary,
        }
        report_path = Path(reports_dir) / "cv_summary.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        mlflow.log_artifact(str(report_path))

        return report
