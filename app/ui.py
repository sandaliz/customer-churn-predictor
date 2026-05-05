import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Load model for Streamlit Cloud deployment
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/churn_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../models/scaler.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "../models/feature_names.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    LOCAL_MODE = True
except:
    LOCAL_MODE = False

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
        
        with st.spinner("Analyzing..."):
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
                except:
                    st.error("API not running on port 8000")

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

            st.markdown("**Risk Factors**")
            if contract == "Month-to-month":
                st.markdown("- Month-to-month contract")
            if payment == "Electronic check":
                st.markdown("- Electronic check payment")
            if tenure < 6:
                st.markdown("- New customer (<6 months)")
            if monthly > 70:
                st.markdown("- High monthly charges")

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
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value" style="color: #000000;">80.4%</div>
            <div class="metric-change" style="color: #059669;">▲ 2.3%</div>
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