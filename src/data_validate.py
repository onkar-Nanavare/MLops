import os
import pandas as pd
from datasets import load_dataset

def validate_data():
    """
    Loads data from local CSV or Hugging Face dataset based on DATA_SOURCE
    Validates the data and returns a clean DataFrame.
    """
    # Decide source: "local" or "hf" (huggingface)
    DATA_SOURCE = os.getenv("DATA_SOURCE", "local")  # default: local

    if DATA_SOURCE == "hf":
        # Load from Hugging Face
        dataset_name = "Onkar000007/my_retrain_data"  # your dataset repo
        # If your dataset has multiple files, you can use data_files={"train": "clean.csv"}
        dataset = load_dataset(dataset_name, split="train")
        df = pd.DataFrame(dataset)
    else:
        # Load from local folder
        # Assuming your script is in src/ and data is in ../data/
        df = pd.read_csv(
            "../data/clean.csv",
            encoding="latin1",
            engine="python",
            sep=",",
            on_bad_lines="skip"
        )

    # --- Validation Logic ---
    df = df[["label", "text"]]
    df.columns = ["label", "text"]

    before = len(df)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["text"] != ""]
    after = len(df)

    print(f"Removed {before - after} bad rows")
    print("Label counts after cleaning:")
    print(df["label"].value_counts())

    if df["label"].nunique() < 2:
        raise Exception("Need at least 2 classes (spam + ham)")

    print("Data validation passed")
    return df


# For standalone run
if __name__ == "__main__":
    validate_data()