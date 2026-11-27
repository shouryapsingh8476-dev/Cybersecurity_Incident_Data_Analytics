Live url : https://cybersecurityincidentdataanalytics-6offktjqhzjfyoqucql2ju.streamlit.app/?utm_source=chatgpt.com

# Cybersecurity Incident Data Analytics and Threat Intelligence Dashboard

## 1. Introduction

Cybersecurity incidents and data breaches are increasing across different industries. Analysing historical cybersecurity incidents helps us understand common attack methods, affected sectors, and trends in exposed records.

This project presents an interactive dashboard that analyses historical cybersecurity incident data using Python, data visualization, and Machine Learning techniques.

## 2. Problem Statement

Organizations face increasing cybersecurity threats, but understanding historical breach patterns can be difficult due to scattered incident information.

This project aims to analyse historical cybersecurity incidents and present meaningful insights through an interactive dashboard and Machine Learning models.

## 3. Project Objectives

- Analyse historical cybersecurity incidents.
- Identify commonly affected industry sectors.
- Understand different data breach methods.
- Visualize yearly cybersecurity incident trends.
- Apply Machine Learning to classify historical incidents.
- Group similar incidents using K-Means clustering.
- Present insights through an interactive dashboard.

## 4. Features

- Interactive dashboard with key performance indicators.
- Year-wise cybersecurity incident analysis.
- Sector-wise incident distribution.
- Analysis of different breach methods.
- Data sensitivity analysis.
- Searchable incident explorer.
- CSV data download.
- Random Forest classification.
- K-Means incident clustering.
- Interactive charts and visualizations.

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Plotly | Interactive visualizations |
| Streamlit | Dashboard development |
| Scikit-learn | Machine Learning |
| Joblib | Model serialization |
| Git & GitHub | Version control |

## 6. Machine Learning Models

### Random Forest Classifier

Used to classify historical incidents into two categories:

- Hacked
- Other methods

**Model evaluation:**
- Accuracy: 73.08%
- Balanced Accuracy: 64.12%
- Test Samples: 104

### K-Means Clustering

Used to group historical incidents based on selected features such as year, sector, records lost, and data sensitivity.

- Number of clusters: 4
- Silhouette Score: 0.2249

The clustering results are exploratory and do not represent verified attack categories.

## 7. Project Workflow

1. Collect historical cybersecurity incident data.
2. Load and clean the dataset.
3. Perform Exploratory Data Analysis.
4. Create interactive visualizations.
5. Train and evaluate Machine Learning models.
6. Perform K-Means clustering.
7. Integrate the results into a Streamlit dashboard.

## 8. Project Structure

```text
Cybersecurity-Incident-Analysis/
│
├── app.py
├── requirements.txt
├── data/
│   ├── cyber_incidents.csv
│   └── incidents_with_clusters.csv
│
├── src/
│   ├── __init__.py
│   └── data_loader.py
│
├── models/
│   ├── breach_classifier.pkl
│   ├── incident_clustering.pkl
│   ├── model_results.json
│   └── model_predictions.csv
│
└── notebooks/
    └── 01_exploratory_data_analysis.py
```

## 9. Installation and Execution

Clone the repository:

```bash
git clone https://github.com/shouryapsingh8476-dev/Cybersecurity_Incident_Data_Analytics.git
```

Navigate to the project directory:

```bash
cd Cybersecurity_Incident_Data_Analytics
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## 10. Limitations

- The project uses historical cybersecurity incident data.
- It does not monitor live network traffic.
- The classification model has moderate predictive performance.
- Clustering results show relatively weak separation.
- Results should not be interpreted as real-time threat detection or guaranteed future attack predictions.

## 11. Future Scope

- Integrate larger and more recent cybersecurity datasets.
- Explore advanced classification algorithms.
- Improve model performance through feature engineering.
- Add threat intelligence data sources.
- Develop more detailed incident analysis and reporting features.

## 12. Conclusion

This project demonstrates how data analytics, visualization, and Machine Learning can be used to study historical cybersecurity incidents. The interactive dashboard provides a convenient way to explore breach patterns, understand affected sectors, and examine similarities between incidents.

---

**Developed as an academic project using Python, Streamlit, and Machine Learning.**
