from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from fraud.features.preprocessing import SECONDS_IN_DAY, make_numeric_matrix, drop_allnan_and_constant_cols


def add_month_group(df: pd.DataFrame, days_per_month: int = 30) -> pd.Series:
    return (df["TransactionDT"] // (days_per_month * SECONDS_IN_DAY)).astype(int)


def forward_month_cv(
    df_all: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    params: dict,
    min_train_months: int = 2,
    days_per_month: int = 30,
) -> dict:
    df = df_all.copy()
    df["month"] = add_month_group(df, days_per_month=days_per_month)

    months = np.sort(df["month"].unique())
    aucs = []

    for i in range(min_train_months, len(months)):
        val_month = months[i]
        tr_mask = df["month"].isin(months[:i])
        va_mask = df["month"].eq(val_month)

        X_tr = df.loc[tr_mask, feature_cols]
        y_tr = y[tr_mask.to_numpy()]

        X_va = df.loc[va_mask, feature_cols]
        y_va = y[va_mask.to_numpy()]

        # numeric + drop fold-specific constants
        X_tr = make_numeric_matrix(X_tr)
        X_va = make_numeric_matrix(X_va)

        X_tr, dropped = drop_allnan_and_constant_cols(X_tr)
        X_va = X_va.drop(columns=dropped, errors="ignore")

        clf = xgb.XGBClassifier(**params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=50
        )

        preds = clf.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, preds)
        aucs.append(auc)

        print(f"Val month={val_month:>3} | train={tr_mask.sum():>7} | val={va_mask.sum():>6} | AUC={auc:.5f}")

    aucs = np.array(aucs)
    summary = {
        "folds": int(len(aucs)),
        "mean_auc": float(aucs.mean()),
        "std_auc": float(aucs.std()),
        "min_auc": float(aucs.min()),
        "max_auc": float(aucs.max()),
    }
    print("\nSummary:", summary)
    return summary
