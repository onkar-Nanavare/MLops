from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import os
from huggingface_hub import hf_hub_download

app = FastAPI()

# Hugging Face Hub settings
HF_REPO = os.getenv("HF_REPO", "Onkar000007/spam-mlops-model")
HF_MODEL_FILE = os.getenv("HF_MODEL_FILE", "model.pkl")
HF_VECT_FILE = os.getenv("HF_VECT_FILE", "vectorizer.pkl")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Local paths (CI-safe & Docker-safe)
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "models"
MODEL_PATH = ARTIFACTS_DIR / HF_MODEL_FILE
VECT_PATH = ARTIFACTS_DIR / HF_VECT_FILE

model = None
vectorizer = None

def download_artifacts():
    """Download model and vectorizer from Hugging Face Hub if not present."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_MODEL_FILE,
            local_dir=ARTIFACTS_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )

    if not VECT_PATH.exists():
        hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_VECT_FILE,
            local_dir=ARTIFACTS_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )

def load_artifacts():
    """Load model and vectorizer into memory (lazy load)."""
    global model, vectorizer
    if model is None or vectorizer is None:
        download_artifacts()
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)

class TextIn(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    load_artifacts()

@app.post("/predict")
def predict_spam(data: TextIn):
    load_artifacts()
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