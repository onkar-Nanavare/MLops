from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import os

app = FastAPI()

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/model.pkl"))
VECT_PATH = Path(os.getenv("VECT_PATH", "/models/vectorizer.pkl"))

model = None
vectorizer = None

def load_artifacts():
    global model, vectorizer
    if model is None or vectorizer is None:
        if not MODEL_PATH.exists() or not VECT_PATH.exists():
            raise RuntimeError("Model artifacts not found. Did you mount /models?")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)

class TextIn(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    # Load only when app actually starts (not on import)
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