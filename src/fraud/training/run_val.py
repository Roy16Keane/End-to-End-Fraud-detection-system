from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from fraud.data.load_data import load_train_data
from fraud.training.validate import evaluate_and_log
from fraud.training.train import DEFAULT_PARAMS


cv_params = dict(DEFAULT_PARAMS)
cv_params["n_jobs"] = 1
cv_params["n_estimators"] = min(cv_params.get("n_estimators", 500), 200)  # CV only
cv_params["max_depth"] = min(cv_params.get("max_depth", 8), 8)          # keep the same



def main():
    df = load_train_data("dataset/merged_data_pruned.parquet")
    report = evaluate_and_log(
        df_all=df,
        params=cv_params,
        experiment_name="ieee-fraud-xgb",
        run_name="forward_month_cv_3fold",
        reports_dir="reports",
    )
    print("Eval complete")
    print(report["summary"])
    


if __name__ == "__main__":
    main()


