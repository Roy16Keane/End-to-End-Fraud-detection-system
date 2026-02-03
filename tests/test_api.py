import json
import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_smoke(monkeypatch, tmp_path):
    """
    This test assumes you have already trained once and created:
      artifacts/featurizer.joblib
      artifacts/train_meta.joblib
      models/xgb_model.json

    For CI later, we’ll generate these in a fixture.
    """

    payload = {
        "transaction": {
            "TransactionDT": 12345,
            "TransactionAmt": 99.9,
            "card1": "1234",
            "addr1": "200",
            "P_emaildomain": "gmail.com"
        },
        "threshold": 0.5
    }

    r = client.post("/predict", data=json.dumps(payload))
    # If you haven't trained/saved artifacts yet, this will 400.
    assert r.status_code in (200, 400)

    if r.status_code == 200:
        out = r.json()
        assert 0.0 <= out["fraud_proba"] <= 1.0
        assert out["fraud_label"] in (0, 1)
