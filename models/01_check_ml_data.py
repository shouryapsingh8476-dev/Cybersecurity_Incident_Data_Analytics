
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_data, clean_data

# Load dataset
df = load_data()
df = clean_data(df)

# Create binary target
df["target"] = df["method"].str.contains(
    "hacked",
    case=False,
    na=False
).astype(int)

# Display target distribution
print("\n--- Target Distribution ---")
print(df["target"].value_counts())

print("\n--- Target Distribution (%) ---")
print(
    df["target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Check selected feature columns
features = [
    "year",
    "sector",
    "records_lost",
    "data_sensitivity"
]

print("\n--- Feature Information ---")
print(df[features].info())

print("\n--- Missing Values ---")
print(df[features].isnull().sum())

print("\n--- Sample Data ---")
print(df[features + ["method", "target"]].head(10))

