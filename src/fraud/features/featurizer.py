from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fraud.features.preprocessing import (
    ensure_cols_exist,
    safe_str_series,
    fit_frequency_maps,
    add_frequency_encoded,
    fit_factorize_maps,
    transform_with_maps,
    fit_agg_tables,
    add_agg_features,
    make_numeric_matrix,
    drop_allnan_and_constant_cols,
)


@dataclass
class PreprocessArtifacts:
    cat_cols: List[str]
    fe_cols: List[str]
    group_cols_list: List[List[str]]
    factor_maps: Dict[str, Dict[Any, int]]
    fe_maps: Dict[str, Dict[Any, int]]
    agg_tables: Dict[str, pd.DataFrame]
    dropped_cols: List[str]


class FraudFeaturizer:
    """
    Fit on train, transform any df consistently.
    """

    def __init__(self):
        self.artifacts: Optional[PreprocessArtifacts] = None

    def fit(self, train_df: pd.DataFrame) -> "FraudFeaturizer":
        Xtr = train_df.copy()

        cat_cols = [
            "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2",
            "P_emaildomain", "R_emaildomain",
            "DeviceType", "DeviceInfo",
        ]
        cat_cols = ensure_cols_exist(Xtr, cat_cols)

        fe_cols = ["card1", "addr1", "P_emaildomain"]
        fe_cols = ensure_cols_exist(Xtr, fe_cols)

        # combined key
        if "card1" in Xtr.columns and "addr1" in Xtr.columns:
            Xtr["card1_addr1"] = safe_str_series(Xtr["card1"]) + "_" + safe_str_series(Xtr["addr1"])
            fe_cols = fe_cols + ["card1_addr1"]

        fe_maps = fit_frequency_maps(Xtr, fe_cols)
        factor_maps = fit_factorize_maps(Xtr, cat_cols)

        group_cols_list: List[List[str]] = []
        if "card1" in Xtr.columns:
            group_cols_list.append(["card1"])
        if "card1" in Xtr.columns and "addr1" in Xtr.columns:
            group_cols_list.append(["card1", "addr1"])

        agg_tables = fit_agg_tables(Xtr, group_cols_list, target_col="TransactionAmt")

        # set artifacts early
        self.artifacts = PreprocessArtifacts(
            cat_cols=cat_cols,
            fe_cols=fe_cols,
            group_cols_list=group_cols_list,
            factor_maps=factor_maps,
            fe_maps=fe_maps,
            agg_tables=agg_tables,
            dropped_cols=[],
        )

        # compute dropped cols on TRAIN after transforms
        Xt = self.transform(train_df, _skip_drop=True)
        Xt_num = make_numeric_matrix(Xt, drop_cols=["isFraud", "TransactionID", "UID"])
        Xt_num, dropped_cols = drop_allnan_and_constant_cols(Xt_num)

        self.artifacts.dropped_cols = dropped_cols
        return self


    def transform(self, df: pd.DataFrame, _skip_drop: bool = False) -> pd.DataFrame:
        if self.artifacts is None:
            raise RuntimeError("Featurizer not fit. Call fit(train_df) first.")

        X = df.copy()

        # Ensure all expected raw cols exist (prevents KeyError like 'card2')
        for c in self.artifacts.cat_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.artifacts.fe_cols:
            if c not in X.columns:
                X[c] = np.nan

        #  combined key for FE (safe, because card1/addr1 now guaranteed to exist)
        if "card1" in X.columns and "addr1" in X.columns and "card1_addr1" not in X.columns:
            X["card1_addr1"] = safe_str_series(X["card1"]) + "_" + safe_str_series(X["addr1"])

        #  Frequency encoding
        X = add_frequency_encoded(X, self.artifacts.fe_maps, suffix="_FE")

        #  Aggregations FIRST (avoid dtype merge problems)
        X = add_agg_features(
            X, self.artifacts.agg_tables, self.artifacts.group_cols_list, target_col="TransactionAmt"
        )

        #  Factorize categoricals
        X = transform_with_maps(X, self.artifacts.factor_maps)

        # Drop training-time allnan/constant cols
        if not _skip_drop:
            X = X.drop(columns=self.artifacts.dropped_cols, errors="ignore")

        return X

