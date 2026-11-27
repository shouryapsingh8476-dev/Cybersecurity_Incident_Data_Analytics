import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_data, clean_data

# Load and clean dataset
df = load_data()
df = clean_data(df)

print("Available columns:")
print(df.columns.tolist())

# 1. Sector-wise incident analysis
sector_counts = df["sector"].value_counts().reset_index()
sector_counts.columns = ["Sector", "Incidents"]

fig1 = px.bar(
    sector_counts,
    x="Sector",
    y="Incidents",
    title="Cybersecurity Incidents by Sector",
    color="Incidents",
    text="Incidents"
)

fig1.update_layout(xaxis_tickangle=-45)
fig1.show()

# 2. Breach method analysis
method_counts = df["method"].value_counts().reset_index()
method_counts.columns = ["Method", "Incidents"]

# 3. Data sensitivity analysis

sensitivity_counts = (
    df["data_sensitivity"]
    .value_counts()
    .sort_index()
    .reset_index()
)

sensitivity_counts.columns = ["Sensitivity", "Incidents"]

fig3 = px.bar(
    sensitivity_counts,
    x="Sensitivity",
    y="Incidents",
    title="Incidents by Data Sensitivity",
    template="plotly_dark"
)


fig3.show()

# 4. Year-wise incident analysis
year_counts = (
    df.groupby("year")
    .size()
    .reset_index(name="Incidents")
    .sort_values("year")
)

fig4 = px.line(
    year_counts,
    x="year",
    y="Incidents",
    markers=True,
    title="Cybersecurity Incidents Over the Years"
)

fig4.show()

print("EDA completed successfully!")
