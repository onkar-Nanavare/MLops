from fastapi.testclient import TestClient
from src.app import app
import os

# Make sure Hugging Face token is set in environment
# Example: export HUGGINGFACE_TOKEN=<your-token>
# or pass it in CI/CD workflow
os.environ["HUGGINGFACE_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN", "")

client = TestClient(app)

# -------------------------
# Test root endpoint
# -------------------------
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# -------------------------
# Test health endpoint
# -------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# -------------------------
# Test predict endpoint
# -------------------------
def test_predict():
    test_text = "Congratulations! You have won a free lottery ticket."
    response = client.post("/predict", json={"text": test_text})
    
    assert response.status_code == 200
    # prediction should be either "spam" or "ham"
    assert response.json()["prediction"] in ["spam", "ham"]
    # class_id should be 0 or 1
    assert response.json()["class_id"] in [0, 1]