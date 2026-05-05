import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Sample customer data (high risk)
high_risk_customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 150.0
}

# Sample customer data (low risk)
low_risk_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 48,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 45.0,
    "TotalCharges": 2160.0
}

print("="*50)
print("TESTING HIGH RISK CUSTOMER")
print("="*50)
response = requests.post(url, json=high_risk_customer)
print(json.dumps(response.json(), indent=2))

print("\n" + "="*50)
print("TESTING LOW RISK CUSTOMER")
print("="*50)
response = requests.post(url, json=low_risk_customer)
print(json.dumps(response.json(), indent=2))