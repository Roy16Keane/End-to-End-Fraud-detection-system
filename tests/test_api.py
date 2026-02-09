import os 
os.environ.setdefault("TEST_MODE", "1")

from fastapi.testclient import TestClient
from app.api.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200


def test_predict_ok_contract():
    payload = {
        "transaction": {
            "TransactionDT": 100000,
            "TransactionAmt": 49.99,
            "ProductCD": "W",
            "card1": 1234,
            "addr1": 200,
            "P_emaildomain": "gmail.com",
        },
        "threshold": 0.5,
    }

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200, r.text

        data = r.json()
        assert set(["fraud_proba", "fraud_label", "threshold"]).issubset(data.keys())
        assert 0.0 <= float(data["fraud_proba"]) <= 1.0
        assert int(data["fraud_label"]) in (0, 1)
        assert float(data["threshold"]) == 0.5


def test_predict_missing_transaction_returns_422():
    with TestClient(app) as client:
        r = client.post("/predict", json={"threshold": 0.5})
        assert r.status_code == 422

