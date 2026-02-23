FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir huggingface_hub  # install huggingface_hub for model download

# Copy source code
COPY src/ ./src/

# Create /models folder for storing downloaded artifacts
RUN mkdir -p /models

# Expose FastAPI port
EXPOSE 8000

# Set environment variables (optional, can also pass at runtime)
# ENV HF_REPO=your-username/spam-mlops-model
# ENV HF_MODEL_FILE=model.pkl
# ENV HF_VECT_FILE=vectorizer.pkl
# ENV HUGGINGFACE_TOKEN=<your-token>

# Run FastAPI app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]