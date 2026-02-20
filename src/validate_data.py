import pandas as pd

df = pd.read_csv("data/clean.csv", encoding="latin1")
df = df[["label", "text"]]
df.columns = ["label", "text"]

# Clean nulls
before = len(df)
df = df.dropna(subset=["label", "text"])
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""]

after = len(df)

print(f"Removed {before - after} bad rows")

# Validate
if df["label"].nunique() < 2:
    raise Exception("Need at least 2 classes (spam + ham)")

# Optional: save cleaned data for training
df.to_csv("data/clean.csv", index=False)

print("Data validation passed ")
