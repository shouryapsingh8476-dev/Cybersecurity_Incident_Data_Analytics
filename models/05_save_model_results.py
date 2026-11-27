
import sys
import json
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# Project directory
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_data, clean_data

# Load dataset
df = clean_data(load_data())

# Create target
df["target"] = (
    df["method"]
    .str.contains("hacked", case=False, na=False)
    .astype(int)
)

# Select features
features = [
    "year",
    "sector",
    "records_lost",
    "data_sensitivity"
]

X = df[features]
y = df["target"]

# Recreate the original test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained model
MODEL_PATH = BASE_DIR / "models" / "breach_classifier.pkl"

model = joblib.load(MODEL_PATH)

# Generate predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=["Other", "Hacked"],
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(y_test, y_pred).tolist()

# Prepare evaluation results
results = {
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "classification_report": report,
    "confusion_matrix": matrix,
    "test_samples": int(len(y_test))
}

# Save metrics as JSON
RESULTS_PATH = BASE_DIR / "models" / "model_results.json"

with open(RESULTS_PATH, "w") as file:
    json.dump(results, file, indent=4)

# Save actual and predicted results
predictions = X_test.copy()

predictions["actual"] = y_test
predictions["predicted"] = y_pred

predictions["actual_label"] = predictions["actual"].map({
    0: "Other",
    1: "Hacked"
})

predictions["predicted_label"] = predictions["predicted"].map({
    0: "Other",
    1: "Hacked"
})

PREDICTIONS_PATH = BASE_DIR / "models" / "model_predictions.csv"

predictions.to_csv(PREDICTIONS_PATH, index=False)

print("\nEvaluation results saved successfully!")
print("Metrics:", RESULTS_PATH)
print("Predictions:", PREDICTIONS_PATH)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("Balanced Accuracy:", round(balanced_accuracy * 100, 2), "%")
print("Test samples:", len(y_test))
