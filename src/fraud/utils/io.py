import joblib

def save_joblib(obj, path: str) -> None:
    joblib.dump(obj, path)

def load_joblib(path: str):
    return joblib.load(path)
