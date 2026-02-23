import pandas as pd

df = pd.read_csv(
    "data/clean.csv",
    encoding="latin1",
    engine="python",
    sep=",",
    on_bad_lines="skip"
)

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

df.to_csv("data/clean.csv", index=False)

print("Data validation passed")