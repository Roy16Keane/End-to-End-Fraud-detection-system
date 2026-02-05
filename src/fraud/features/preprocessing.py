from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd


SECONDS_IN_DAY = 24 * 60 * 60


def ensure_cols_exist(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def safe_str_series(s: pd.Series) -> pd.Series:
    return s.astype("object").fillna("__NA__").astype(str)


def fit_frequency_maps(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[Any, int]]:
    maps: Dict[str, Dict[Any, int]] = {}
    for c in cols:
        maps[c] = df[c].value_counts(dropna=False).to_dict()
    return maps


def add_frequency_encoded(df: pd.DataFrame, fe_maps: Dict[str, Dict[Any, int]], suffix="_FE") -> pd.DataFrame:
    out = df.copy()
    for c, mp in fe_maps.items():
        out[f"{c}{suffix}"] = out[c].map(mp).fillna(0).astype("int32")
    return out


def fit_factorize_maps(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[Any, int]]:
    """
    Build mapping dicts from TRAIN only:
      value -> integer code
    Unknowns in transform become -1
    """
    maps: Dict[str, Dict[Any, int]] = {}
    for c in cols:
        # treat NaN as a category
        vals = df[c].astype("object").fillna("__NA__")
        uniques = pd.unique(vals)
        maps[c] = {v: i for i, v in enumerate(uniques)}
    return maps


def transform_with_maps(df: pd.DataFrame, factor_maps: Dict[str, Dict[Any, int]]) -> pd.DataFrame:
    out = df.copy()
    for c, mp in factor_maps.items():
        vals = out[c].astype("object").fillna("__NA__")
        out[c] = vals.map(mp).fillna(-1).astype("int32")
    return out
def fit_agg_tables(train_df, group_cols_list, target_col="TransactionAmt"):
    """
    Creates aggregation tables (mean,std) for each group spec.
    Returns dict key = "colA|colB".
    """
    aggs = {}
    df = train_df.copy()

    for group_cols in group_cols_list:
        # force stable join key dtype
        for c in group_cols:
            df[c] = safe_str_series(df[c])

        key = "|".join(group_cols)
        g = df.groupby(group_cols)[target_col].agg(["mean", "std"]).reset_index()
        g = g.rename(columns={"mean": f"{target_col}_{key}_mean", "std": f"{target_col}_{key}_std"})
        aggs[key] = g

    return aggs



def add_agg_features(df, agg_tables, group_cols_list, target_col="TransactionAmt"):
    out = df.copy()

    for group_cols in group_cols_list:
        # force stable join key dtype (matches fit_agg_tables)
        for c in group_cols:
            out[c] = safe_str_series(out[c])

        key = "|".join(group_cols)
        table = agg_tables[key]

        out = out.merge(table, on=group_cols, how="left")

        mean_col = f"{target_col}_{key}_mean"
        std_col = f"{target_col}_{key}_std"
        out[mean_col] = out[mean_col].fillna(out[target_col].mean()).astype("float32")
        out[std_col] = out[std_col].fillna(out[target_col].std()).astype("float32")

    return out


def drop_allnan_and_constant_cols(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    all_nan = X.columns[X.isna().all()].tolist()
    const = X.columns[X.nunique(dropna=True) <= 1].tolist()
    drop_cols = sorted(set(all_nan) | set(const))
    return X.drop(columns=drop_cols, errors="ignore"), drop_cols


def make_numeric_matrix(df: pd.DataFrame, drop_cols: Optional[List[str]] = None) -> pd.DataFrame:
    out = df.copy()
    if drop_cols:
        out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(-1)
    return out


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

        # FE maps on raw categories
        fe_maps = fit_frequency_maps(Xtr, fe_cols)

        # Factor maps
        factor_maps = fit_factorize_maps(Xtr, cat_cols)

        # Agg tables
        group_cols_list: List[List[str]] = []
        if "card1" in Xtr.columns:
            group_cols_list.append(["card1"])
        if "card1" in Xtr.columns and "addr1" in Xtr.columns:
            group_cols_list.append(["card1", "addr1"])

        agg_tables = fit_agg_tables(Xtr, group_cols_list, target_col="TransactionAmt")

        # IMPORTANT: set artifacts FIRST (temporary dropped_cols)
        self.artifacts = PreprocessArtifacts(
            cat_cols=cat_cols,
            fe_cols=fe_cols,
            group_cols_list=group_cols_list,
            factor_maps=factor_maps,
            fe_maps=fe_maps,
            agg_tables=agg_tables,
            dropped_cols=[],
        )

        # Now we can transform safely
        Xt = self.transform(train_df, _skip_drop=True)

        # Compute columns to drop based on TRAIN after transforms
        Xt_num = make_numeric_matrix(Xt, drop_cols=["isFraud", "TransactionID", "UID"])
        Xt_num, dropped_cols = drop_allnan_and_constant_cols(Xt_num)

        # Update artifacts with dropped cols
        self.artifacts = PreprocessArtifacts(
            cat_cols=cat_cols,
            fe_cols=fe_cols,
            group_cols_list=group_cols_list,
            factor_maps=factor_maps,
            fe_maps=fe_maps,
            agg_tables=agg_tables,
            dropped_cols=dropped_cols,
        )

        return self

    def transform(self, df: pd.DataFrame, _skip_drop: bool = False) -> pd.DataFrame:
        """
        Apply FE + factorize + agg features.
        Does not add UID features here (we keep UID separate).
        """
        if self.artifacts is None:
            # allow warm transform during fit by constructing minimal artifacts on the fly
            # (used only internally; fit() will overwrite self.artifacts)
            raise RuntimeError("Featurizer not fit. Call fit(train_df) first.")

        X = df.copy()

        # combined key for FE
        if "card1" in X.columns and "addr1" in X.columns and "card1_addr1" not in X.columns:
            X["card1_addr1"] = safe_str_series(X["card1"]) + "_" + safe_str_series(X["addr1"])

        # FE
        X = add_frequency_encoded(X, self.artifacts.fe_maps, suffix="_FE")

        # Agg features
        X = add_agg_features(X, self.artifacts.agg_tables, self.artifacts.group_cols_list, target_col="TransactionAmt")

        if not _skip_drop:
            # Drop cols that were identified as all-NaN/constant in training
            X = X.drop(columns=self.artifacts.dropped_cols, errors="ignore")
        # Factorize categoricals
        X = transform_with_maps(X, self.artifacts.factor_maps)


        return X
