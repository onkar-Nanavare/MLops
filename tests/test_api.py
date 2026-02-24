from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["service_version"] == "v1.0.0"
    assert "model_revision" in data
    assert "message" in data

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service_version"] == "v1.0.0"
    assert "model_revision" in data

def test_predict():
    test_text = "Congratulations! You have won a free lottery ticket."

    response = client.post("/v1/predict", json={"text": test_text})
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "class_id" in data