import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_data, clean_data

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Cybersecurity Incident Analytics",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- DARK THEME ----------------

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
}

div[data-testid="stMetric"] {
    background-color: #1c2330;
    padding: 15px;
    border-radius: 10px;
}

h1, h2, h3 {
    color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

@st.cache_data
def get_data():
    df = load_data()
    df = clean_data(df)
    return df

df = get_data()

# ---------------- HEADER ----------------

st.title("🛡️ Cybersecurity Incident Analytics Dashboard")

st.write(
    "An interactive platform to analyze historical "
    "cybersecurity incidents and identify threat patterns."
)

st.divider()

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.title("Dashboard Filters")

years = sorted(df["year"].dropna().unique())

selected_years = st.sidebar.multiselect(
    "Select Year",
    years,
    default=years
)

sectors = sorted(df["sector"].dropna().unique())

selected_sectors = st.sidebar.multiselect(
    "Select Sector",
    sectors,
    default=sectors
)

methods = sorted(df["method"].dropna().unique())

selected_methods = st.sidebar.multiselect(
    "Select Breach Method",
    methods,
    default=methods
)

# ---------------- FILTER DATA ----------------

filtered_df = df[
    df["year"].isin(selected_years)
    & df["sector"].isin(selected_sectors)
    & df["method"].isin(selected_methods)
].copy()

# ---------------- KPI METRICS ----------------

st.subheader("Executive Overview")

total_incidents = len(filtered_df)

total_sectors = filtered_df["sector"].nunique()

total_records = filtered_df["records_lost"].sum()

hacked_incidents = filtered_df["method"].str.contains(
    "hacked", case=False, na=False
).sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Incidents", f"{total_incidents:,}")
col2.metric("Affected Sectors", f"{total_sectors:,}")
col3.metric("Records Exposed", f"{total_records:,.0f}")
col4.metric("Hacked Incidents", f"{hacked_incidents:,}")

st.divider()

# ---------------- INCIDENT TREND ----------------

st.subheader("📈 Cybersecurity Incidents Over Time")

year_data = (
    filtered_df.groupby("year")
    .size()
    .reset_index(name="Incidents")
)

fig1 = px.line(
    year_data,
    x="year",
    y="Incidents",
    markers=True,
    title="Year-wise Incident Trend",
    template="plotly_dark"
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------- SECTOR ANALYSIS ----------------

st.subheader("🏢 Incidents by Sector")

sector_data = (
    filtered_df["sector"]
    .value_counts()
    .reset_index()
)

sector_data.columns = ["Sector", "Incidents"]

fig2 = px.bar(
    sector_data,
    x="Sector",
    y="Incidents",
    title="Sector-wise Incident Distribution",
    template="plotly_dark"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- BREACH METHOD ----------------

st.subheader("🔍 Breach Method Analysis")

method_data = (
    filtered_df["method"]
    .value_counts()
    .reset_index()
)

method_data.columns = ["Method", "Incidents"]

fig3 = px.pie(
    method_data,
    names="Method",
    values="Incidents",
    title="Distribution of Breach Methods",
    hole=0.4,
    template="plotly_dark"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------- DATA SENSITIVITY ----------------

st.subheader("🔐 Data Sensitivity Analysis")

sensitivity_data = (
    filtered_df["data_sensitivity"]
    .value_counts()
    .sort_index()
    .reset_index()
)

sensitivity_data.columns = ["Sensitivity", "Incidents"]

fig4 = px.bar(
    sensitivity_data,
    x="Sensitivity",
    y="Incidents",
    title="Incidents by Data Sensitivity",
    template="plotly_dark"
)

st.plotly_chart(fig4, use_container_width=True)

# ---------------- INCIDENT EXPLORER ----------------

st.subheader("📋 Incident Explorer")

search = st.text_input(
    "Search by organisation"
)

if search:
    filtered_df = filtered_df[
        filtered_df["organisation"].str.contains(
            search, case=False, na=False
        )
    ]

display_columns = [
    "organisation",
    "year",
    "sector",
    "method",
    "records_lost",
    "data_sensitivity"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    hide_index=True
)

# ---------------- DOWNLOAD ----------------

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Incident Data (CSV)",
    data=csv,
    file_name="cybersecurity_incidents.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "Cybersecurity Incident Data Analytics | Academic Project"
)


# ==========================================
# MACHINE LEARNING INSIGHTS
# ==========================================

st.markdown("---")
st.header("🤖 Machine Learning Insights")

BASE_DIR = Path(__file__).resolve().parent

RESULTS_PATH = BASE_DIR / "models" / "model_results.json"
PREDICTIONS_PATH = BASE_DIR / "models" / "model_predictions.csv"

if RESULTS_PATH.exists() and PREDICTIONS_PATH.exists():

    with open(RESULTS_PATH, "r") as file:
        results = json.load(file)

    predictions = pd.read_csv(PREDICTIONS_PATH)

    # Model performance
    st.subheader("1. Classification Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{results['accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Balanced Accuracy",
        f"{results['balanced_accuracy'] * 100:.2f}%"
    )

    col3.metric(
        "Test Samples",
        results["test_samples"]
    )

    st.caption(
        "The Random Forest model classifies historical incidents "
        "as Hacked or Other. These results do not represent "
        "real-time attack detection."
    )

    # Confusion matrix
    st.subheader("2. Confusion Matrix")

    matrix = results["confusion_matrix"]

    confusion_df = pd.DataFrame(
        matrix,
        index=["Actual Other", "Actual Hacked"],
        columns=["Predicted Other", "Predicted Hacked"]
    )

    fig_cm = px.imshow(
        confusion_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Actual vs Predicted Classifications"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification report
    st.subheader("3. Classification Report")

    report = results["classification_report"]

    report_df = pd.DataFrame({
        "Class": ["Other", "Hacked"],
        "Precision": [
            report["Other"]["precision"],
            report["Hacked"]["precision"]
        ],
        "Recall": [
            report["Other"]["recall"],
            report["Hacked"]["recall"]
        ],
        "F1 Score": [
            report["Other"]["f1-score"],
            report["Hacked"]["f1-score"]
        ]
    })

    st.dataframe(
        report_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    # Actual vs predicted
    st.subheader("4. Actual vs Predicted Incidents")

    comparison = predictions[
        ["actual_label", "predicted_label"]
    ].copy()

    comparison_counts = (
        comparison
        .groupby(["actual_label", "predicted_label"])
        .size()
        .reset_index(name="Incidents")
    )

    fig_comparison = px.bar(
        comparison_counts,
        x="actual_label",
        y="Incidents",
        color="predicted_label",
        barmode="group",
        title="Actual and Predicted Incident Categories"
    )

    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )

    # Prediction data table
    with st.expander("View Model Prediction Details"):

        st.dataframe(
            predictions,
            use_container_width=True
        )

else:
    st.warning(
        "Machine Learning result files not found. "
        "Please run models/05_save_model_results.py first."
    )


# ==========================================
# K-MEANS INCIDENT CLUSTERING
# ==========================================

st.markdown("---")
st.header("🔍 Cybersecurity Incident Clustering")

CLUSTER_PATH = BASE_DIR / "data" / "incidents_with_clusters.csv"

if CLUSTER_PATH.exists():

    cluster_df = pd.read_csv(CLUSTER_PATH)

    st.subheader("1. Cluster Distribution")

    cluster_counts = (
        cluster_df["cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = ["Cluster", "Incidents"]

    fig_clusters = px.bar(
        cluster_counts,
        x="Cluster",
        y="Incidents",
        color="Cluster",
        title="Number of Incidents in Each Cluster",
        text="Incidents"
    )

    st.plotly_chart(
        fig_clusters,
        use_container_width=True
    )

    st.subheader("2. Cluster Characteristics")

    profile = (
        cluster_df.groupby("cluster")
        .agg(
            incident_count=("cluster", "size"),
            median_year=("year", "median"),
            median_records_lost=("records_lost", "median"),
            median_sensitivity=("data_sensitivity", "median"),
            most_common_sector=("sector", lambda x: x.mode().iloc[0])
        )
        .reset_index()
    )

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("3. Compare Clusters")

    fig_scatter = px.scatter(
        cluster_df,
        x="year",
        y="records_lost",
        color="cluster",
        hover_data=["sector", "method"],
        title="Incident Clusters: Year vs Records Lost",
        log_y=True
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

    st.caption(
        "K-Means groups historical incidents using year, "
        "records lost, data sensitivity, and sector. "
        "Clusters are exploratory and do not represent "
        "confirmed attack types."
    )

else:

    st.warning(
        "Cluster dataset not found. "
        "Please run models/04_incident_clustering.py first."
    )
