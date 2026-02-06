
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load trained model and scaler (train and save in Colab first)
# with open('best_model.pkl', 'rb') as f: model = pickle.load(f)
# with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)

st.title("Telecom Churn Predictor")

# Input features (match your columns)
account_length = st.number_input("Account Length", min_value=0)
customer_service_calls = st.number_input("Customer Service Calls", min_value=0)
international_plan = st.selectbox("International Plan", ["no", "yes"])
# Add other inputs similarly...

if st.button("Predict Churn"):
    input_df = pd.DataFrame({
        'account_length': [account_length],
        'customer_service_calls': [customer_service_calls],
        'international_plan': [1 if international_plan == "yes" else 0],
        # Add all features...
    })
    # input_scaled = scaler.transform(input_df)
    # pred = model.predict(input_scaled)
    # prob = model.predict_proba(input_scaled)[0][1]
    # st.success(f"Churn Probability: {prob:.2%}")

# To save models from Colab:
# import pickle
# pickle.dump(best_model, open('best_model.pkl', 'wb'))
# pickle.dump(scaler, open('scaler.pkl', 'wb'))
