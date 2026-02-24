# src/upload_model.py
import os
from pathlib import Path
from huggingface_hub import HfApi

MODEL_FILE = Path("model.pkl")
VECT_FILE = Path("vectorizer.pkl")

for f in [MODEL_FILE, VECT_FILE]:
    if not f.exists():
        raise FileNotFoundError(f"{f} not found!")

hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not set!")

api = HfApi()

api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo="model.pkl",
    repo_id="Onkar000007/spam-mlops-model",
    repo_type="model",
    token=hf_token,
    commit_message="Upload model artifacts v1.0.0"
)

api.upload_file(
    path_or_fileobj=VECT_FILE,
    path_in_repo="vectorizer.pkl",
    repo_id="Onkar000007/spam-mlops-model",
    repo_type="model",
    token=hf_token,
    commit_message="Upload model artifacts v1.0.0"
)

print("Model & Vectorizer uploaded successfully as v1.0.0")