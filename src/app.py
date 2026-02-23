from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import os
from huggingface_hub import hf_hub_download

app = FastAPI()

# Hugging Face Hub settings
HF_REPO = os.getenv("HF_REPO", "your-username/spam-mlops-model")
HF_MODEL_FILE = os.getenv("HF_MODEL_FILE", "model.pkl")
HF_VECT_FILE = os.getenv("HF_VECT_FILE", "vectorizer.pkl")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Local paths to save artifacts
MODEL_PATH = Path("/models/model.pkl")
VECT_PATH = Path("/models/vectorizer.pkl")

model = None
vectorizer = None

def download_artifacts():
    """Download model and vectorizer from Hugging Face Hub if not present."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not MODEL_PATH.exists():
        MODEL_PATH.write_bytes(hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_MODEL_FILE,
            token=HF_TOKEN
        ))
    if not VECT_PATH.exists():
        VECT_PATH.write_bytes(hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_VECT_FILE,
            token=HF_TOKEN
        ))

def load_artifacts():
    """Load model and vectorizer into memory (lazy load)."""
    global model, vectorizer
    if model is None or vectorizer is None:
        download_artifacts()  # Ensure artifacts are available locally
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)

class TextIn(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    # Load artifacts on app startup
    load_artifacts()

@app.post("/predict")
def predict_spam(data: TextIn):
    load_artifacts()  # Lazy load if not already
    vec = vectorizer.transform([data.text])
    pred = int(model.predict(vec)[0])
    label = "spam" if pred == 1 else "ham"
    return {"prediction": label, "class_id": pred}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok"}