import streamlit as st
import pandas as pd
import numpy as np 
import joblib

model=joblib.load("models/fraud_model.pkl")
scaler=joblib.load("models/scaler.pkl")
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💰")
st.title("Credit Card Fraud Detection")
st.markdown("Enter transaction details to predict if it's fraudulent ⤵")
st.sidebar.header("Transaction input")
amount=st.sidebar.number_input(
    "Transaction Amount ($)", 
    min_value=0.0, 
    max_value=25000.0, 
    value=100.0 #defaut val
)
time=st.sidebar.number_input(
    "Time (seconds since first transaction)", 
    min_value=0.0, 
    max_value=172792.0, 
    value=50000.0
)
st.sidebar.markdown("**V Features (PCA Components)**")
v_features = {}
for i in range(1, 29):
    v_features[f"V{i}"] = st.sidebar.slider(f"V{i}", min_value=-10.0, max_value=10.0, value=0.0,step=0.1)

if st.button("DETECT"):
    input_data=pd.DataFrame([v_features])
    amount_scaled=scaler.transform([[amount]])[0][0]
    time_scaled=scaler.transform([[time]])[0][0]

    #add scaled data
    input_data["Scaled_Amount"]=amount_scaled
    input_data["Scaled_Time"]=time_scaled
    input_data=input_data[model.feature_names_in_]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.error(f"FRAUDULENT TRANSACTION DETECTED")
        st.metric("Fraud Probability", f"{probability*100:.2f}%")
    else:
        st.success(f"LEGITIMATE TRANSACTION")
        st.metric("Fraud Probability", f"{probability*100:.2f}%")
    st.markdown("---")
    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "XGBoost")
    col2.metric("AUC-ROC Score", "0.9806")
    col3.metric("Fraud Recall", "0.89")

    st.markdown("---")
    st.caption("Dataset: ULB Credit Card Fraud Detection | 284K transactions | 0.17% fraud rate")
