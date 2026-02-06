
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model & scaler (download these from Colab)
@st.cache_resource
def load_model():
    model = pickle.load(open('xgboost/best_model.pkl', 'rb'))
    scaler = pickle.load(open('xgboost/scaler.pkl', 'rb'))
    return model, scaler

st.title("🛡️ Telecom Customer Churn Predictor")
st.markdown("Enter customer details to predict churn probability")

# === ALL 18 INPUT FIELDS ===
col1, col2 = st.columns(2)

with col1:
    account_length = st.number_input("**Account Length** (months)", min_value=0, value=100)
    voice_mail_plan = st.selectbox("**Voice Mail Plan**", ["no", "yes"])
    day_mins = st.number_input("**Day Minutes**", min_value=0.0, value=150.0, step=0.1)
    evening_mins = st.number_input("**Evening Minutes**", min_value=0.0, value=200.0, step=0.1)
    night_mins = st.number_input("**Night Minutes**", min_value=0.0, value=180.0, step=0.1)
    international_mins = st.number_input("**Intl Minutes**", min_value=0.0, value=10.0, step=0.1)
    customer_service_calls = st.number_input("**Customer Service Calls**", min_value=0, value=1)
    day_calls = st.number_input("**Day Calls**", min_value=0, value=100)
    day_charge = st.number_input("**Day Charge** ($)", min_value=0.0, value=25.0, step=0.1)

with col2:
    voice_mail_messages = st.number_input("**Voice Mail Messages**", min_value=0, value=0)
    international_plan = st.selectbox("**International Plan**", ["no", "yes"])
    evening_calls = st.number_input("**Evening Calls**", min_value=0, value=100)
    night_calls = st.number_input("**Night Calls**", min_value=0, value=90)
    international_calls = st.number_input("**Intl Calls**", min_value=0, value=4)
    evening_charge = st.number_input("**Evening Charge** ($)", min_value=0.0, value=15.0, step=0.1)
    night_charge = st.number_input("**Night Charge** ($)", min_value=0.0, value=10.0, step=0.1)
    international_charge = st.number_input("**Intl Charge** ($)", min_value=0.0, value=3.0, step=0.1)
    total_charge = st.number_input("**Total Charge** ($)", min_value=0.0, value=55.0, step=0.1)

# Predict Button
if st.button("🔮 **Predict Churn Risk**", type="primary"):
    try:
        # Create input DataFrame (EXACT column order from training)
        input_data = {
            'account_length': [account_length],
            'voice_mail_plan': [1 if voice_mail_plan == "yes" else 0],
            'voice_mail_messages': [voice_mail_messages],
            'day_mins': [day_mins],
            'evening_mins': [evening_mins],
            'night_mins': [night_mins],
            'international_mins': [international_mins],
            'customer_service_calls': [customer_service_calls],
            'international_plan': [1 if international_plan == "yes" else 0],
            'day_calls': [day_calls],
            'day_charge': [day_charge],
            'evening_calls': [evening_calls],
            'evening_charge': [evening_charge],
            'night_calls': [night_calls],
            'night_charge': [night_charge],
            'international_calls': [international_calls],
            'international_charge': [international_charge],
            'total_charge': [total_charge]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Load model & predict
        model, scaler = load_model()
        input_scaled = scaler.transform(input_df)
        probability = model.predict_proba(input_scaled)[0][1]
        prediction = "🚨 HIGH RISK" if probability > 0.5 else "✅ LOW RISK"
        
        # Results
        st.success(f"**Prediction**: {prediction}")
        st.metric("Churn Probability", f"{probability:.1%}", delta=None)
        
        # Risk factors
        st.subheader("💡 Key Risk Factors")
        if customer_service_calls > 3:
            st.error("⚠️ High customer service calls (>3)")
        if international_plan == "yes":
            st.warning("⚠️ International plan user")
        if total_charge > 60:
            st.info("💰 High total charges")
            
    except Exception as e:
        st.error(f"Error: {e}. Make sure model/scaler files are uploaded!")

# Sidebar: Instructions
st.sidebar.header("📋 How to Deploy")
st.sidebar.markdown("**Deploy Instructions:**\n1. Download app.py + models\n2. `streamlit run app.py`")
