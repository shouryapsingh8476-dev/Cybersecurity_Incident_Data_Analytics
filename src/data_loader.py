import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cyber_incidents.csv"

print("Data loader location:", Path(__file__).resolve())
print("Project directory:", BASE_DIR)
print("Expected CSV path:", DATA_PATH)
print("CSV exists:", DATA_PATH.exists())


def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    return df


def clean_data(df):

    df = df.drop_duplicates().copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["records_lost"] = pd.to_numeric(
        df["records_lost"], errors="coerce"
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()

    df = df.replace(r"^\s*$", pd.NA, regex=True)

    return df