import json

with open("metrics.json") as f:
    metrics = json.load(f)

acc = metrics["accuracy"]

if acc < 0.85:
    raise Exception(f"Model rejected  Accuracy too low: {acc}")

print(f"Model approved  Accuracy: {acc}")
