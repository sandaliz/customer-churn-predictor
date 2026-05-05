# Customer Churn Prediction System

An end-to-end machine learning system that predicts which customers will leave and provides actionable recommendations for retention teams. Built on the IBM Telco Customer Churn dataset.

**Live Demo:** https://churncast.streamlit.app/

---

### Dataset Used
**Source:** [IBM Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/)

**Overview:**
| Feature | Value |
|---------|-------|
| Total Records | 7,043 customers |
| Features | 21 columns |
| Churn Rate | 26.6% |
| Time Range | 0-72 months tenure |

**Key Columns:**
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Account Info: tenure, Contract, PaperlessBilling, PaymentMethod
- Services: PhoneService, InternetService, OnlineSecurity, TechSupport
- Charges: MonthlyCharges, TotalCharges
- Target: Churn (Yes/No)

---

## What It Does

| Feature | Description |
|---------|-------------|
| Predict | Churn probability with 80% accuracy |
| Classify | Risk as HIGH / MEDIUM / LOW |
| Recommend | Specific retention actions for each risk level |
| Visualize | Interactive dashboard with key metrics |

---

## Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd risk-prediction-churn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the API
```bash
cd app
uvicorn api:app --reload --port 8000
```

### 4. Run the Dashboard (new terminal)
```bash
cd app
streamlit run ui.py --server.port 8502
```

### 5. Open in Browser
- Dashboard: http://localhost:8502
- API Docs: http://localhost:8000/docs

---

## Model Performance

After comparing 3 models, Logistic Regression was selected as the winner.

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 80.4% | 64.8% | 57.5% | 60.9% |
| Random Forest | 78.8% | 62.3% | 50.8% | 56.0% |
| Decision Tree | 71.6% | 46.6% | 46.0% | 46.3% |

**Confusion Matrix (Logistic Regression)**
```
                Predicted
              No     Yes
Actual  No   1034     113
        Yes    84     176
```

---

## Key Insights from EDA

| Finding | Business Impact |
|---------|-----------------|
| Month-to-month contracts | 15x higher churn risk than 2-year contracts |
| First 6 months | 109% higher churn risk (critical window) |
| Electronic check | 45% churn vs 16% for autopay |
| Monthly bills > $70 | Significantly higher churn |
| Top 3 features | TotalCharges + tenure + MonthlyCharges = 53% of prediction power |

---

## Risk Scoring System

| Risk Level | Probability | Action Required |
|------------|-------------|-----------------|
| HIGH | 70% - 100% | Call customer immediately, offer retention discount |
| MEDIUM | 40% - 70% | Send targeted email with discount offer |
| LOW | 0% - 40% | Monitor, no immediate action needed |

---

## Highest Risk Profile

- Month-to-month contract
- Tenure less than 6 months
- Monthly charges greater than $70
- Electronic check payment

Projected churn: approximately 75% (compared to 26.6% average)

---

## Business Impact

| Metric | Value |
|--------|-------|
| Annual revenue at risk | $1.66M |
| At-risk customers identified | 1,245 |
| Expected annual savings | $1.3M+ |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Data Processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Deployment | Uvicorn |

---

## Project Structure

```
risk-prediction-churn/
|
├── app/                      # API and UI
│   ├── api.py               # FastAPI backend
│   ├── ui.py                # Streamlit dashboard
│
├── data/
│   └── churn.csv            # IBM Telco dataset
│
├── models/                   # Saved model artifacts
│
├── notebooks/                # Analysis and training
│   ├── 01_eda.ipynb         # Exploratory analysis
│   ├── 02_preprocessing.ipynb # Data cleaning
│   └── 03_modeling.ipynb    # Model training
│
├── documents/                # Documentation
│
├── requirements.txt          # Main dependencies
```

---

## Skills Demonstrated

| Skill | Implementation |
|-------|----------------|
| Data Cleaning | Fixed data types, handled missing values (11 rows removed) |
| EDA | 6 actionable insights from visualizations |
| Feature Engineering | One-hot encoding (15 to 30 features), scaling, stratification |
| Model Selection | Compared 3 algorithms, business-driven selection |
| Evaluation | Precision, Recall, F1, Confusion Matrix |
| Deployment | REST API and Interactive dashboard |
| Business Translation | Risk scores to Retention actions |

---

## References

- [IBM Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
