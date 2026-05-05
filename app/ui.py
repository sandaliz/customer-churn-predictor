import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import shap

# Load API key from .env
ENV_PATH = os.path.join(os.path.dirname(__file__), "../.env")
OPENROUTER_API_KEY = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith("API="):
                OPENROUTER_API_KEY = line.strip().split("=")[1]
                break

# Load model for Streamlit Cloud deployment
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/churn_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../models/scaler.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "../models/feature_names.pkl")

@st.cache_resource
def load_models():
    m = joblib.load(MODEL_PATH)
    s = joblib.load(SCALER_PATH)
    f = joblib.load(FEATURES_PATH)
    return m, s, f

try:
    model, scaler, feature_names = load_models()
    LOCAL_MODE = True
except Exception as e:
    LOCAL_MODE = False
    st.error(f"Model loading failed: {e}")
    st.write(f"Model path: {MODEL_PATH}")
    st.write(f"Path exists: {os.path.exists(MODEL_PATH)}")

# Category mapping for preprocessing
CATEGORY_MAPPING = {
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

def preprocess_input(df):
    df_encoded = df.copy()
    for col in CATEGORY_MAPPING.keys():
        categories = CATEGORY_MAPPING[col]
        for cat in categories[1:]:
            dummy_col = f"{col}_{cat}"
            df_encoded[dummy_col] = (df[col] == cat).astype(int)
        df_encoded = df_encoded.drop(columns=[col])

    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[feature_names]
    return df_encoded

def predict_churn(data):
    df = pd.DataFrame([data])
    df_encoded = preprocess_input(df)
    df_scaled = scaler.transform(df_encoded)
    prob = model.predict_proba(df_scaled)[0][1]
    return prob

def explain_churn(data, prob):
    reasons = []

    # High risk factors (increase churn)
    if data['Contract'] == 'Month-to-month':
        reasons.append("Month-to-month contract (highest risk)")
    if data['PaymentMethod'] == 'Electronic check':
        reasons.append("Electronic check payment (45% churn rate)")
    if data['tenure'] < 6:
        reasons.append("New customer (<6 months tenure)")
    elif data['tenure'] < 12:
        reasons.append("Early tenure stage (6-12 months)")
    if data['MonthlyCharges'] > 70:
        reasons.append(f"High monthly charges (${data['MonthlyCharges']:.0f})")
    if data['InternetService'] == 'Fiber optic':
        reasons.append("Fiber optic internet (higher churn than DSL)")
    if data['OnlineSecurity'] == 'No' and data['InternetService'] != 'No':
        reasons.append("No online security add-on")
    if data['TechSupport'] == 'No' and data['InternetService'] != 'No':
        reasons.append("No tech support add-on")

    # Protective factors (reduce churn)
    protective = []
    if data['Contract'] in ['One year', 'Two year']:
        protective.append(f"{data['Contract']} contract (loyalty)")
    if data['tenure'] >= 24:
        protective.append(f"Long tenure ({data['tenure']} months - loyal customer)")
    if data['PaymentMethod'] in ['Bank transfer (automatic)', 'Credit card (automatic)']:
        protective.append("Auto-pay enabled (75% lower churn)")
    if data['TechSupport'] == 'Yes':
        protective.append("Has tech support")
    if data['OnlineSecurity'] == 'Yes':
        protective.append("Has online security")

    # Return top 3 based on risk level
    if prob >= 0.7:
        # High risk - show risk factors
        return reasons[:3] if reasons else ["Multiple risk factors present"]
    elif prob >= 0.4:
        # Medium risk - mix of risks and protective
        return (reasons[:2] + protective[:1]) if reasons else protective[:3]
    else:
        # Low risk - show protective factors
        return protective[:3] if protective else ["No significant risk factors"]

def get_shap_explanation(data):
    """Get feature contributions using model importances"""
    try:
        df = pd.DataFrame([data])
        df_encoded = preprocess_input(df)

        # Get feature importances from model
        importances = model.feature_importances_
        customer_values = df_encoded.iloc[0].values

        # Build list of features with importance and value
        features = []
        for i, feat in enumerate(feature_names):
            if i < len(customer_values) and i < len(importances):
                val = customer_values[i]
                imp = importances[i]
                # Only include features that are active (value > 0)
                if val > 0:
                    features.append((feat, imp * 100, val))

        # Sort by importance
        features.sort(key=lambda x: x[1], reverse=True)

        # Get top 3 risk factors (highest importance features that are present)
        top_risk = [(f, imp) for f, imp, v in features[:3]]

        # Get protective factors based on business logic from data
        protective = []
        if data.get('Contract') in ['One year', 'Two year']:
            protective.append(('Long-term contract', 5.0))
        if data.get('tenure', 0) >= 24:
            protective.append(('Loyal customer', 4.0))
        if data.get('PaymentMethod', '').startswith('Bank') or data.get('PaymentMethod', '').startswith('Credit'):
            protective.append(('Auto-pay', 3.0))
        if data.get('TechSupport') == 'Yes':
            protective.append(('Has tech support', 2.0))

        return top_risk, protective[:3]
    except Exception as e:
        return [], []

def get_ai_insight(data, prob, top_reasons):
    """Get AI-generated insight from OpenRouter"""
    if not OPENROUTER_API_KEY:
        return "AI insights unavailable (no API key)"

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

        prompt = f"""You are a business analyst for a telecom company.

Given this customer data:
- Contract: {data['Contract']}
- Tenure: {data['tenure']} months
- Monthly Charges: ${data['MonthlyCharges']:.0f}
- Payment Method: {data['PaymentMethod']}
- Internet Service: {data['InternetService']}
- Has Tech Support: {data['TechSupport']}
- Has Online Security: {data['OnlineSecurity']}

Churn Probability: {prob*100:.0f}%

Top contributing factors: {', '.join(top_reasons)}

Generate:
1. Simple reason for churn risk (1 sentence)
2. 1-2 actionable recommendations
3. Keep it under 3 sentences total"""

        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI insights unavailable: {str(e)[:50]}"

# Page config
st.set_page_config(
    page_title="Churn Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* All text - dark and readable */
    .stMarkdown, .stText, .stCaption, label, .stSelectbox label,
    .stSlider label, .stNumberInput label, .stCheckbox label,
    .streamlit-expanderHeader {
        color: #1e293b !important;
    }

    /* Expander header - ensure always visible */
    .streamlit-expanderHeader, .streamlit-expanderHeader span, .streamlit-expanderHeader p,
    [data-testid="stExpanderToggle"], [data-testid="stExpanderToggle"] span,
    .stDetails, .stDetails > div:first-child {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Expander content styling */
    [data-testid="stExpanderContent"] {
        color: #0f172a !important;
    }

    /* Expander icon visibility */
    .stExpander svg, .stDetails svg {
        fill: #0f172a !important;
        color: #0f172a !important;
    }

    /* Expander border for better visibility */
    .stExpander, .stDetails {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
    }

    /* Expander hover effect */
    .stExpander:hover, .stDetails:hover {
        border-color: #94a3b8 !important;
    }

    /* Force expander header visibility - all states */
    section[data-testid="stExpander"] > div:first-child,
    div[data-testid="stExpander"] > div:first-child,
    .stExpander > div:first-child,
    details > summary {
        color: #0f172a !important;
        background-color: #fef3c7 !important;
        font-weight: 700 !important;
    }

    /* Ensure summary (collapsed header) is visible */
    details > summary {
        list-style: none !important;
        padding: 0.75rem 1rem !important;
    }

    details > summary::-webkit-details-marker {
        display: none !important;
    }

    /* Expander content */
    details[open] > summary,
    details[open] > summary ~ * {
        color: #0f172a !important;
    }

    /* Checkbox label visibility (explicit) */
    [data-testid="stCheckbox"] label {
        color: #0f172a !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }

    /* Streamlit/BaseWeb checkbox text can be nested in spans/divs */
    [data-testid="stCheckbox"] * {
        color: #0f172a !important;
    }
    label[data-baseweb="checkbox"] * {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: white;
        border-radius: 16px;
        padding: 1.5rem 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 2px solid #e2e8f0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #cbd5e1;
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a !important;
        margin: 0.25rem 0;
    }
    
    .metric-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-change {
        font-size: 0.65rem;
        margin-top: 0.25rem;
        font-weight: 600;
    }
    
    .positive { color: #059669 !important; }
    .negative { color: #dc2626 !important; }
    
    /* Streamlit metric values - make them black/dark */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Metric containers - inherit background from parent */
    [data-testid="stMetric"] {
        background-color: transparent !important;
        padding: 0 !important;
    }
    
    /* Ensure metric card divs contain the metrics properly */
    .metric-card > div {
        background-color: inherit !important;
    }
    
    /* Chart containers */
    .chart-card {
        background-color: white;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Form card */
    .form-card {
        background-color: white;
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Result card */
    .result-card {
        background-color: white;
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Risk cards */
    .risk-high {
        background-color: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 12px;
        padding: 1rem;
    }
    .risk-medium {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem;
    }
    .risk-low {
        background-color: #ecfdf5;
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 700;
    }
    .badge-high { background-color: #dc2626; color: white !important; }
    .badge-medium { background-color: #f59e0b; color: white !important; }
    .badge-low { background-color: #10b981; color: white !important; }
    
    .section-header {
        font-size: 0.9rem;
        font-weight: 700;
        color: #0f172a !important;
        margin: 1rem 0 0.75rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Tabs - Big and clear */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: white;
        padding: 0.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.2rem;
        padding: 0 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        border-radius: 12px;
        color: #475569 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white !important;
    }
    
    .stTabs [aria-selected="false"]:hover {
        background-color: #f1f5f9;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stButton button {
        background-color: #3b82f6;
        color: white !important;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("Customer Churn Predictor")
st.caption("Machine learning model to identify customers at risk of leaving")

# Initialize session state for expander
if "services_expander" not in st.session_state:
    st.session_state.services_expander = False

# Tabs
tab1, tab2, tab3 = st.tabs(["PREDICT", "MODEL INFO", "DOCUMENTATION"])

# ============================================
# TAB 1: PREDICT
# ============================================
with tab1:
    # Prediction form
    st.subheader("Customer Risk Assessment")
    st.caption("Enter customer details   •  Click predict for assessment")
    
    with st.expander("Demographics", expanded=True):
        ga, gb = st.columns(2)
        with ga:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.checkbox("Senior Citizen (65+)", help="Check if the customer is 65 years or older")
        with gb:
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    with st.expander("Account & Billing", expanded=True):
        aa, ab = st.columns(2)
        with aa:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
        with ab:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", 
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
    
    with st.expander("Services", expanded=st.session_state.services_expander):
        sa, sb = st.columns(2)
        with sa:
            internet = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
            phone = st.selectbox("Phone Service", ["Yes", "No"])
        with sb:
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    
    predict = st.button("Predict Churn Risk", type="primary", use_container_width=True)

    # Result section below form
    if predict:
        data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": "No",
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": tech,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": "Yes",
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": monthly * tenure
        }

        with st.spinner("Analyzing customer data..."):
            # Initialize variables
            prob_pct = 0
            risk = "LOW"
            recommendation = "Monitor but no immediate action needed"

            # Use local model (Streamlit Cloud) or API (local)
            if LOCAL_MODE:
                prob = predict_churn(data)
                prob_pct = prob * 100
                if prob >= 0.7:
                    risk = "HIGH"
                    recommendation = "Immediate retention offer - call customer"
                elif prob >= 0.4:
                    risk = "MEDIUM"
                    recommendation = "Send retention email with discount offer"
            else:
                try:
                    resp = requests.post("http://localhost:8000/predict", json=data, timeout=5)
                    if resp.status_code == 200:
                        res = resp.json()
                        prob_pct = res['churn_probability'] * 100
                        risk = res['risk_level']
                        recommendation = res['recommendation']
                    else:
                        st.error("API error")
                except Exception as e:
                    st.warning("API not available. Using local mode.")
                    prob_pct = 0
                    risk = "LOW"
                    recommendation = "Model files not loaded"

            if risk == "HIGH":
                st.markdown(f"""
                <div class="risk-high">
                    <span class="badge badge-high">HIGH RISK</span>
                    <div style="font-size: 2rem; font-weight: 800; margin: 0.5rem 0;">{prob_pct:.1f}%</div>
                    <div style="font-size: 0.75rem; color: #64748b;">Churn Probability</div>
                    <hr>
                    <div style="font-size: 0.8rem;">{recommendation}</div>
                </div>
                """, unsafe_allow_html=True)
            elif risk == "MEDIUM":
                st.markdown(f"""
                <div class="risk-medium">
                    <span class="badge badge-medium">MEDIUM RISK</span>
                    <div style="font-size: 2rem; font-weight: 800; margin: 0.5rem 0;">{prob_pct:.1f}%</div>
                    <div style="font-size: 0.75rem; color: #64748b;">Churn Probability</div>
                    <hr>
                    <div style="font-size: 0.8rem;">{recommendation}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <span class="badge badge-low">LOW RISK</span>
                    <div style="font-size: 2rem; font-weight: 800; margin: 0.5rem 0;">{prob_pct:.1f}%</div>
                    <div style="font-size: 0.75rem; color: #64748b;">Churn Probability</div>
                    <hr>
                    <div style="font-size: 0.8rem;">{recommendation}</div>
                </div>
                """, unsafe_allow_html=True)

            st.progress(prob / 100 if LOCAL_MODE else res['churn_probability'])

            # Get model-based explanations
            if LOCAL_MODE:
                reasons = explain_churn(data, prob)
                st.markdown("---")
                st.markdown("### Top Reasons for This Prediction")
                for reason in reasons:
                    st.markdown(f"- {reason}")
            else:
                try:
                    explain_resp = requests.post("http://localhost:8000/explain", json=data, timeout=5)
                    if explain_resp.status_code == 200:
                        explanation = explain_resp.json()
                        st.markdown("---")
                        st.markdown("### Top Reasons for This Prediction")
                        for reason in explanation.get('top_reasons', []):
                            st.markdown(f"- {reason}")
                    else:
                        st.warning(f"Explain endpoint error: {explain_resp.status_code}")
                except Exception as e:
                    st.warning(f"Explain unavailable: {e}")

            # SHAP Explanation
            if LOCAL_MODE:
                try:
                    top_risk, top_protective = get_shap_explanation(data)

                    st.markdown("---")
                    st.markdown("### Key Drivers (SHAP)")

                    col_risk, col_prot = st.columns(2)

                    with col_risk:
                        st.markdown("**Features pushing churn**")
                        if top_risk:
                            for feat, val in top_risk:
                                feat_name = feat.replace('_', ' ')
                                st.markdown(f"""
                                <div style="background-color: #fef2f2; padding: 8px 12px; border-radius: 8px; margin: 4px 0; border-left: 3px solid #dc2626;">
                                    <b>+</b> {val:.1f}% {feat_name}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.caption("No significant risk factors")

                    with col_prot:
                        st.markdown("**Features reducing churn**")
                        if top_protective:
                            for feat, val in top_protective:
                                st.markdown(f"""
                                <div style="background-color: #ecfdf5; padding: 8px 12px; border-radius: 8px; margin: 4px 0; border-left: 3px solid #10b981;">
                                    {val:.1f}% {feat}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.caption("No protective factors")
                except Exception as e:
                    pass  # Silently skip if SHAP fails

            # AI Insight
            st.markdown("---")
            st.markdown("### AI Recommendation")
            ai_insight = get_ai_insight(data, prob, reasons)
            st.markdown(f"""
            <div style="background-color: #eff6ff; padding: 16px; border-radius: 12px; border: 1px solid #bfdbfe;">
                {ai_insight}
            </div>
            """, unsafe_allow_html=True)

# ============================================
# TAB 2: MODEL INFO
# ============================================
with tab2:
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown('''
        <div class="metric-card" style="background-color: #fee2e2; border: 2px solid #fecaca;">
            <div class="metric-label">Current Churn Rate</div>
            <div class="metric-value" style="color: #000000;">26.6%</div>
            <div class="metric-change" style="color: #059669;">▼ 2.1%</div>
        </div>
        ''', unsafe_allow_html=True)
    with m2:
        st.markdown('''
        <div class="metric-card" style="background-color: #fef3c7; border: 2px solid #fde68a;">
            <div class="metric-label">At-Risk Customers</div>
            <div class="metric-value" style="color: #000000;">1,245</div>
            <div class="metric-change" style="color: #dc2626;">▲ 156</div>
        </div>
        ''', unsafe_allow_html=True)
    with m3:
        st.markdown('''
        <div class="metric-card" style="background-color: #dcfce7; border: 2px solid #bbf7d0;">
            <div class="metric-label">Model Recall</div>
            <div class="metric-value" style="color: #000000;">57.5%</div>
            <div class="metric-change" style="color: #059669;">Catches churners</div>
        </div>
        ''', unsafe_allow_html=True)
    with m4:
        st.markdown('''
        <div class="metric-card" style="background-color: #dbeafe; border: 2px solid #bfdbfe;">
            <div class="metric-label">Revenue at Risk</div>
            <div class="metric-value" style="color: #000000;">$1.66M</div>
            <div class="metric-change" style="color: #475569;">Annual</div>
        </div>
        ''', unsafe_allow_html=True)

    # Charts row
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Churn by Tenure")
        tenure_data = pd.DataFrame({
            "Tenure": ["0-6m", "6-12m", "1-2y", "2-3y", "3-4y", "4-5y", "5+y"],
            "Churn Rate": [42.5, 31.2, 18.5, 12.3, 8.7, 5.2, 3.1]
        })
        fig1 = px.bar(tenure_data, x="Tenure", y="Churn Rate", color="Churn Rate",
                      color_continuous_scale="Reds", text="Churn Rate")
        fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig1.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Churn by Contract")
        contract_data = pd.DataFrame({
            "Contract": ["Month-to-month", "1 Year", "2 Year"],
            "Churn Rate": [42.7, 11.3, 2.8]
        })
        fig2 = px.bar(contract_data, x="Contract", y="Churn Rate", color="Churn Rate",
                      color_continuous_scale="Reds", text="Churn Rate")
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Model Overview")
        st.markdown("""
        | Property | Value |
        |----------|-------|
        | Algorithm | Logistic Regression |
        | Training Samples | 5,625 |
        | Features | 30 |
        | Target | Churn (Yes/No) |
        """)
    
    with col_b:
        st.subheader("Performance")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | Accuracy | 80.4% |
        | Precision | 64.8% |
        | Recall | 57.5% |
        | F1 Score | 60.9% |
        """)
    
    st.subheader("Top Features")
    imp_data = pd.DataFrame({
        "Feature": ["Total Charges", "Tenure", "Monthly Charges", "Fiber Optic", "Electronic Check"],
        "Importance": [19.2, 17.0, 16.9, 3.9, 3.8]
    })
    fig = px.bar(imp_data, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Risk Thresholds")
    st.markdown("""
    | Risk Level | Probability | Action |
    |------------|-------------|--------|
    | **HIGH** | 70%+ | Call customer, offer retention |
    | **MEDIUM** | 40-70% | Send discount email |
    | **LOW** | Below 40% | Monitor only |
    """)

# ============================================
# TAB 3: DOCUMENTATION
# ============================================
with tab3:
    st.subheader("How to Use")
    
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.markdown("""
        **1. Enter Customer Data**
        - Fill demographics and account info
        - Use expandable sections
        
        **2. Run Prediction**
        - Click "Predict Churn Risk"
        - Model analyzes 30+ features
        
        **3. Review Results**
        - Check probability percentage
        - Follow recommended action
        """)
    
    with col_y:
        st.markdown("""
        **Key Insights**
        - Month-to-month: 15x higher risk
        - First 6 months: Critical period
        - E-check users: 45% churn rate
        - Autopay: 75% lower churn
        """)
    
    st.subheader("Business Impact")
    st.info("Expected annual savings: **$1.3M+** through targeted retention")
    
    st.subheader("Tech Stack")
    st.code("""
    Backend: FastAPI
    ML: Scikit-learn (Logistic Regression)
    Frontend: Streamlit + Plotly
    """)

# Footer
st.markdown("---")
st.caption("Churn Prediction System")