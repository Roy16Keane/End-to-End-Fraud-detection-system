import pandas as pd
from fraud.training.train import train_final_model

def test_train_final_model_smoke(tmp_path):
    df = pd.DataFrame({
        "TransactionDT": [1, 2, 3, 4, 5, 6],
        "TransactionAmt": [10.0, 20.0, 15.0, 25.0, 12.0, 30.0],
        "card1": ["a", "a", "b", "b", "a", None],
        "addr1": ["x", "x", "y", "y", "x", "z"],
        "P_emaildomain": ["gmail.com", None, "yahoo.com", "yahoo.com", "gmail.com", "gmail.com"],
        "isFraud": [0, 0, 1, 0, 0, 1]
    })

    meta = train_final_model(
        df_train=df,
        artifacts_dir=str(tmp_path / "artifacts"),
        model_dir=str(tmp_path / "models"),
    )

    assert meta["n_features"] > 0
