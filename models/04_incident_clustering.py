
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Project path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_data, clean_data

# Load dataset
df = clean_data(load_data())

# Select clustering features
features = [
    "year",
    "records_lost",
    "data_sensitivity",
    "sector"
]

X = df[features].copy()

# Reduce the skew in exposed-record counts
X["records_lost"] = np.log1p(
    X["records_lost"].clip(lower=0)
)

numeric_features = [
    "year",
    "records_lost",
    "data_sensitivity"
]

categorical_features = ["sector"]

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Transform features
X_processed = preprocessor.fit_transform(X)

# Compare different cluster counts
print("\n--- K-MEANS CLUSTER EVALUATION ---")

scores = {}

for k in range(2, 7):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_processed)

    score = silhouette_score(
        X_processed,
        labels
    )

    scores[k] = score

    print(f"Clusters: {k} | Silhouette Score: {score:.4f}")

# Select the best cluster count
best_k = max(scores, key=scores.get)

print("\nBest number of clusters:", best_k)

# Train final clustering model
final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = final_kmeans.fit_predict(X_processed)

# Add cluster labels to original dataset
df["cluster"] = cluster_labels

# Display cluster sizes
print("\n--- INCIDENTS PER CLUSTER ---")
print(df["cluster"].value_counts().sort_index())

# Display cluster profiles
print("\n--- CLUSTER PROFILES ---")

profiles = df.groupby("cluster").agg(
    incident_count=("cluster", "size"),
    median_year=("year", "median"),
    median_records_lost=("records_lost", "median"),
    median_sensitivity=("data_sensitivity", "median"),
    most_common_sector=(
        "sector",
        lambda x: x.mode().iloc[0]
    )
)

print(profiles)

# Save model and preprocessing
MODEL_PATH = BASE_DIR / "models" / "incident_clustering.pkl"

joblib.dump(
    {
        "preprocessor": preprocessor,
        "model": final_kmeans,
        "features": features,
        "best_k": best_k
    },
    MODEL_PATH
)

# Save dataset with cluster labels
OUTPUT_PATH = BASE_DIR / "data" / "incidents_with_clusters.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nClustering model saved:", MODEL_PATH)
print("Clustered dataset saved:", OUTPUT_PATH)
