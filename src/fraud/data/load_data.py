from pathlib import Path
import pandas as pd


DEFAULT_TRAIN_PATH = Path("dataset/merged_data_pruned.parquet")


def load_train_data(path: str | Path = DEFAULT_TRAIN_PATH) -> pd.DataFrame:
    """
    Load the final merged + pruned training dataset.

    This dataset:
    - has missingness-pruned columns applied
    - still contains raw features (NOT encoded)
    - includes the target column `isFraud`

    This is the canonical input for model training.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found at: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return df
