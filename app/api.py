from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any

# Initialize FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="Predict customer churn probability for telecom company",
    version="1.0.0"
)

# Load model and preprocessors at startup
model_path = "../models/churn_model.pkl"
scaler_path = "../models/scaler.pkl"
features_path = "../models/feature_names.pkl"

# Check if files exist
if not os.path.exists(model_path):
    raise Exception(f"Model not found at {model_path}. Run modeling notebook first.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(features_path)

print("Model loaded successfully")
print(f"Features: {len(feature_names)}")
print(f"Model type: {type(model).__name__}")

# Define request body structure
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Define response structure
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    recommendation: str

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Churn Prediction API",
        "status": "running",
        "endpoints": {
            "POST /predict": "Make a prediction",
            "GET /health": "Check API health"
        }
    }

# Health check
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        # Convert input to DataFrame
        input_data = customer.dict()
        df_input = pd.DataFrame([input_data])
        
        # Apply same preprocessing as training
        df_processed = preprocess_input(df_input)
        
        # Scale features
        df_scaled = scaler.transform(df_processed)
        
        # Make prediction
        probability = model.predict_proba(df_scaled)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "HIGH"
            recommendation = "Immediate retention offer - call customer"
        elif probability >= 0.4:
            risk_level = "MEDIUM"
            recommendation = "Send retention email with discount offer"
        else:
            risk_level = "LOW"
            recommendation = "Monitor but no immediate action needed"
        
        return PredictionResponse(
            churn_probability=round(probability, 4),
            churn_prediction=prediction,
            risk_level=risk_level,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Explainability endpoint
@app.post("/explain")
def explain(customer: CustomerData):
    """Returns top 3 reasons for churn prediction"""
    try:
        input_data = customer.dict()
        df_input = pd.DataFrame([input_data])
        df_processed = preprocess_input(df_input)
        df_scaled = scaler.transform(df_processed)

        probability = model.predict_proba(df_scaled)[0][1]

        # Get feature contributions (works for both Logistic Regression and Random Forest)
        if hasattr(model, 'feature_importances_'):
            # Random Forest
            importances = model.feature_importances_
        else:
            # Logistic Regression
            importances = model.coef_[0]

        feature_imp = dict(zip(feature_names, importances))

        # Sort by absolute impact
        sorted_imp = sorted(feature_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        reasons = []
        mean_importance = sum(importances) / len(importances)
        for feat, imp in sorted_imp:
            if imp > mean_importance:
                reasons.append(f"{feat}: increases churn risk")
            else:
                reasons.append(f"{feat}: reduces churn risk")

        return {"churn_probability": probability, "top_reasons": reasons}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as training"""

    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport',
                        'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']

    # Define all categories from training data (first category is baseline for drop_first)
    category_mapping = {
        'gender': ['Female', 'Male'],
        'Partner': ['No', 'Yes'],
        'Dependents': ['No', 'Yes'],
        'PhoneService': ['No', 'Yes'],
        'MultipleLines': ['No', 'No phone service', 'Yes'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'OnlineSecurity': ['No', 'No internet service', 'Yes'],
        'OnlineBackup': ['No', 'No internet service', 'Yes'],
        'DeviceProtection': ['No', 'No internet service', 'Yes'],
        'TechSupport': ['No', 'No internet service', 'Yes'],
        'StreamingTV': ['No', 'No internet service', 'Yes'],
        'StreamingMovies': ['No', 'No internet service', 'Yes'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'PaperlessBilling': ['No', 'Yes'],
        'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']
    }

    # Build encoded dataframe manually to ensure all categories are represented
    df_encoded = df.copy()

    for col in categorical_cols:
        categories = category_mapping[col]
        baseline = categories[0]

        # Create dummy columns for all categories except baseline (drop_first=True behavior)
        for cat in categories[1:]:
            dummy_col = f"{col}_{cat}"
            df_encoded[dummy_col] = (df[col] == cat).astype(int)

        # Drop original categorical column
        df_encoded = df_encoded.drop(columns=[col])

    # Add missing columns (if any)
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Ensure correct column order
    df_encoded = df_encoded[feature_names]

    return df_encoded

# Run with: uvicorn app.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)