import joblib
from pathlib import Path
from typing import Union, Any

PathLike = Union[str, Path]

def save_joblib(obj: Any, path: PathLike) -> None:
    joblib.dump(obj, str(path))

def load_joblib(path: PathLike) -> Any:
    return joblib.load(str(path))



