
import sys
from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

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

# Features and target
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

# Preprocessing
preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])

# Algorithms
models = {
    "Dummy Classifier": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

# Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1_macro"
        ]
    )

    results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Balanced Accuracy": scores["test_balanced_accuracy"].mean(),
        "Macro F1": scores["test_f1_macro"].mean()
    })

# Display results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Balanced Accuracy",
    ascending=False
)

print("\n--- MODEL COMPARISON RESULTS ---")
print(results_df.to_string(index=False))

