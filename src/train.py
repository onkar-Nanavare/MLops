import os
import json
import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_validate import validate_data  # import the updated validator

# -------------------------
# MLflow config
# -------------------------
MLFLOW_DIR = "./mlruns"
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
mlflow.set_experiment("spam-mlops")


def train_model():
    """
    Loads validated data, trains Logistic Regression,
    logs metrics & model to MLflow, saves artifacts locally.
    """
    # -------------------------
    # Load validated data
    # -------------------------
    df = validate_data()  # automatically uses DATA_SOURCE env variable

    # Encode labels
    X = df["text"]
    y = df["label"].map({"ham": 0, "spam": 1})

    if y.isnull().any():
        bad_labels = df.loc[y.isnull(), "label"].unique()
        raise Exception(f"Unknown labels found in dataset: {bad_labels}")

    # -------------------------
    # Split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------
    # Vectorize
    # -------------------------
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # -------------------------
    # Train model
    # -------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # -------------------------
    # Evaluate
    # -------------------------
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    # -------------------------
    # Log to MLflow
    # -------------------------
    with mlflow.start_run(run_name="spam-training"):
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    # -------------------------
    # Save artifacts locally
    # -------------------------
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    # Save metrics for CI gate
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

    print(f"Training Completed Successfully! Accuracy: {acc:.4f}")
    return model, vectorizer, acc


# -------------------------
# For standalone run
# -------------------------
if __name__ == "__main__":
    train_model()