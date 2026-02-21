import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json

# Load data (fix encoding)
df = pd.read_csv("data/clean.csv")

df = df[["label", "text"]]
df.columns = ["label", "text"]

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""]

# Features + labels
X = df["text"]
y = df["label"].map({"ham": 0, "spam": 1})

if y.isnull().any():
    bad_labels = df.loc[y.isnull(), "label"].unique()
    raise Exception(f"Unknown labels found in dataset: {bad_labels}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
preds = model.predict(X_test_vec)
acc = accuracy_score(y_test, preds)

# MLflow
mlflow.start_run()
mlflow.log_param("model", "logistic_regression")
mlflow.log_param("dataset_size", len(df))
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Save metrics for pipeline gate
with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)

print("Training done. Accuracy:", acc)
