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

# Load sklearn Pipeline at startup
pipeline_path = "../models/churn_pipeline.pkl"

if not os.path.exists(pipeline_path):
    raise Exception(f"Pipeline not found at {pipeline_path}. Run modeling notebook first.")

pipeline = joblib.load(pipeline_path)

print("Pipeline loaded successfully")
print(f"Pipeline steps: {pipeline.named_steps.keys()}")
print(f"Model type: {type(pipeline.named_steps['classifier']).__name__}")

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
        # Convert input to DataFrame - pipeline handles preprocessing
        df_input = pd.DataFrame([customer.dict()])
        
        # Make prediction using pipeline (handles preprocessing internally)
        probability = pipeline.predict_proba(df_input)[0][1]
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
        df_input = pd.DataFrame([customer.dict()])

        probability = pipeline.predict_proba(df_input)[0][1]

        # Get coefficients from Logistic Regression in pipeline
        model_step = pipeline.named_steps['classifier']
        importances = model_step.coef_[0]

        # Get feature names from pipeline's preprocessor
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
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

# Run with: uvicorn app.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)