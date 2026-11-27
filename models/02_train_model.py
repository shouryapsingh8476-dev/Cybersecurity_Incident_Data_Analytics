
import sys
from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------- PROJECT PATH ----------------

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_data, clean_data

# ---------------- LOAD DATA ----------------

df = load_data()
df = clean_data(df)

# ---------------- CREATE TARGET ----------------

df["target"] = (
    df["method"]
    .str.contains("hacked", case=False, na=False)
    .astype(int)
)

# ---------------- SELECT FEATURES ----------------

features = [
    "year",
    "sector",
    "records_lost",
    "data_sensitivity"
]

X = df[features]
y = df["target"]

numeric_features = [
    "year",
    "records_lost",
    "data_sensitivity"
]

categorical_features = ["sector"]

# ---------------- PREPROCESSING ----------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ---------------- RANDOM FOREST MODEL ----------------

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# ---------------- SPLIT DATA ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------------- TRAIN MODEL ----------------

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")

# ---------------- MODEL EVALUATION ----------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Other", "Hacked"],
    zero_division=0
))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))

# ---------------- SAVE MODEL ----------------

MODEL_PATH = BASE_DIR / "models" / "breach_classifier.pkl"

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully!")
print("Saved at:", MODEL_PATH)
