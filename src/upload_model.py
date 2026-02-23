# src/upload_model.py
import os
from pathlib import Path
from huggingface_hub import HfApi

# Check if model exists
model_file = Path("model.pkl")
if not model_file.exists():
    raise FileNotFoundError(f"{model_file} not found. Train the model first!")

# Get Hugging Face token from environment
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACE_TOKEN not set in environment variables!")

# Initialize Hugging Face API
api = HfApi()

# Upload the model
api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo="model.pkl",
    repo_id="Onkar000007/spam-mlops-model", 
    repo_type="model",
    token=hf_token
)

print("Model uploaded successfully!")