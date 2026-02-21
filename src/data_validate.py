import pandas as pd

df = pd.read_csv(
    "data/clean.csv",
    encoding="latin1",
    engine="python",         # more tolerant parser
    on_bad_lines="skip"      # skip broken rows
)

df = df[["label", "text"]]
df.columns = ["label", "text"]

before = len(df)

df = df.dropna(subset=["label", "text"])
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""]

after = len(df)

print(f"Removed {before - after} bad rows")

# Validate labels
if df["label"].nunique() < 2:
    raise Exception("Need at least 2 classes (spam + ham)")

df.to_csv("data/clean.csv", index=False)

print("Data validation passed")