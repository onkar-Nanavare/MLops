import pandas as pd

df = pd.read_csv(
    "data/clean.csv",
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

df = df[["label", "text"]]
df.columns = ["label", "text"]

before = len(df)

# Drop nulls
df = df.dropna(subset=["label", "text"])

# Keep only valid labels
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["ham", "spam"])]

df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""]

after = len(df)

print(f"Removed {before - after} bad rows")

if df["label"].nunique() < 2:
    raise Exception("Need at least 2 classes (spam + ham)")

df.to_csv("data/clean.csv", index=False)

print("Data validation passed")