from fastapi.testclient import TestClient
from src.app import app
import src.app as app_module

class DummyVectorizer:
    def transform(self, texts):
        return [[0.1, 0.2]]

class DummyModel:
    def predict(self, X):
        return [1]

def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200

def test_predict(monkeypatch):
    monkeypatch.setattr(app_module, "model", DummyModel())
    monkeypatch.setattr(app_module, "vectorizer", DummyVectorizer())

    client = TestClient(app)
    r = client.post("/predict", json={"text": "free money"})
    assert r.status_code == 200
    assert r.json()["prediction"] == "spam"