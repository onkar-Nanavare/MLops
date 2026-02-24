from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import os
from huggingface_hub import hf_hub_download

# =====================
# App & Version Info
# =====================
APP_VERSION = os.getenv("APP_VERSION", "v1.0.0")
HF_REVISION = os.getenv("HF_REVISION", "main")  # commit hash or tag

app = FastAPI(
    title="Spam Classifier API",
    version=APP_VERSION
)

# =====================
# Hugging Face Hub settings
# =====================
HF_REPO = os.getenv("HF_REPO", "Onkar000007/spam-mlops-model")
HF_MODEL_FILE = os.getenv("HF_MODEL_FILE", "model.pkl")
HF_VECT_FILE = os.getenv("HF_VECT_FILE", "vectorizer.pkl")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# =====================
# Local paths (CI-safe & Docker-safe)
# =====================
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
            revision=HF_REVISION,
            local_dir=ARTIFACTS_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )

    if not VECT_PATH.exists():
        hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_VECT_FILE,
            revision=HF_REVISION,
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

# =====================
# API Schemas
# =====================
class TextIn(BaseModel):
    text: str

# =====================
# App Lifecycle
# =====================
@app.on_event("startup")
def startup_event():
    load_artifacts()

# =====================
# Versioned API
# =====================
@app.post("/api/v1/predict")
def predict_spam(data: TextIn):
    load_artifacts()
    vec = vectorizer.transform([data.text])
    pred = int(model.predict(vec)[0])
    label = "spam" if pred == 1 else "ham"

    return {
        "prediction": label,
        "class_id": pred,
        "model_repo": HF_REPO,
        "model_revision": HF_REVISION,
        "service_version": APP_VERSION
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service_version": APP_VERSION,
        "model_revision": HF_REVISION
    }

@app.get("/")
def root():
    return {
        "message": "Spam Classifier API is running",
        "service_version": APP_VERSION,
        "model_revision": HF_REVISION
    }