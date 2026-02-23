from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI()

# Load model relative to container root (/app)
MODEL_PATH = Path("model.pkl")
VECT_PATH = Path("vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

class TextIn(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(data: TextIn):
    vec = vectorizer.transform([data.text])
    pred = model.predict(vec)[0]

    pred = int(pred)
    label = "spam" if pred == 1 else "ham"

    return {
        "prediction": label,
        "class_id": pred
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "v1.0.0"}

@app.get("/")
def root():
    return {"status": "ok"}