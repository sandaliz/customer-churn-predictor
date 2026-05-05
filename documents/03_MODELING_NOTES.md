# Model Training Documentation

## Why Trained Three Models

| Model | Why I Used It |
|-------|---------------|
| Logistic Regression | Baseline - simple, fast, interpretable |
| Decision Tree | Shows how decisions are made, easy to explain |
| Random Forest | Best for production - handles complex patterns |

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 80.4% | 64.8% | 57.5% | **60.9%** |
| Decision Tree | 71.6% | 46.6% | 46.0% | 46.3% |
| Random Forest | 78.8% | 62.3% | 50.8% | 56.0% |

### Winner: Logistic Regression (F1-Score: 0.609)

*Note: Random Forest (78.8% accuracy) was deployed for production use.*

---

## Understanding the Metrics

| Metric | What It Means | Our Score |
|--------|---------------|-----------|
| **Accuracy** | Overall correct predictions | 80% |
| **Precision** | Of customers flagged as churn, how many actually churned | 65% |
| **Recall** | Of actual churners, how many did we catch | 58% |
| **F1-Score** | Balance of precision & recall | 61% |

### What These Numbers Mean for Business

- **Precision 65%**: If we flag 100 customers as "high risk", about 65 will actually churn
- **Recall 58%**: We catch 58% of all customers who will churn
- **F1 61%**: Good balance - not perfect but useful for targeting retention offers

---

## Why Logistic Regression Won

Decision Tree overfitted (71% accuracy vs 80% baseline). Random Forest was close but Logistic Regression performed best because:

1. **Simple patterns** - Churn follows clear rules (tenure down, monthly charges up)
2. **No complex interactions** needed - linear relationships work well
3. **Imbalanced data** (27% churn) - simpler models handle this better

---

## Top 10 Features That Predict Churn

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | TotalCharges | 19.2% |
| 2 | tenure | 17.0% |
| 3 | MonthlyCharges | 16.9% |
| 4 | Fiber optic internet | 3.9% |
| 5 | Electronic check payment | 3.8% |
| 6 | Two year contract | 3.0% |
| 7 | Gender (Male) | 2.9% |
| 8 | Online security (Yes) | 2.9% |
| 9 | Paperless billing | 2.6% |
| 10 | Tech support (Yes) | 2.4% |

### Key Insight
**Three features drive 53% of prediction power:**
- TotalCharges + tenure + MonthlyCharges = 53%

Nothing else comes close.

---

## Saved
models/
├── churn_model.pkl <- Random Forest (standalone)
├── churn_pipeline.pkl <- Full pipeline (preprocessing + model)
├── scaler.pkl <- For scaling new data
└── feature_names.pkl <- Column order for API

### Pipeline Approach
The `churn_pipeline.pkl` includes:
- **Preprocessor**: One-hot encoding for categorical features
- **Classifier**: Logistic Regression

This allows the API to accept raw customer data and handle preprocessing internally.


---

## How to Use This Model

Given a new customer's data:
1. Apply same preprocessing (one-hot encode, scale)
2. Model outputs probability of churn (0 to 1)
3. If probability > 0.5 → high risk, send retention offer

---

## What to Do Next

1. **Improve recall** (currently 58%) - try SMOTE to handle imbalanced data
2. **Test Gradient Boosting** (XGBoost) - often beats Random Forest
3. **Build API** - expose model as web service
4. **Create dashboard** - show predictions to retention team

---

## Files Created

| File | What It Contains |
|------|------------------|
| `03_modeling.ipynb` | Training code |
| `models/churn_model.pkl` | Saved Random Forest model |
| `models/churn_pipeline.pkl` | Full sklearn pipeline |
| `models/scaler.pkl` | Feature scaler |
| `models/feature_names.pkl` | Feature names list |