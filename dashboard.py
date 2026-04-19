
 

import streamlit as st
import pandas as pd
import joblib

# ================= LOAD MODEL =================
model = joblib.load("churn_model.pkl")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Churn Intelligence Dashboard", layout="centered")

# ================= UI HEADER =================
st.title("📊 Customer Churn Intelligence Dashboard")
st.markdown("AI-powered prediction + business insights 🚀")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">📌 Telecom Churn Prediction System</p>', unsafe_allow_html=True)

# ================= INPUT SECTION =================
st.sidebar.header("Customer Inputs")

AccountWeeks = st.sidebar.slider("Account Weeks", 0, 200, 100)
ContractRenewal = st.sidebar.selectbox("Contract Renewal", [0, 1])
DataPlan = st.sidebar.selectbox("Data Plan", [0, 1])
DataUsage = st.sidebar.number_input("Data Usage", 0.0, 10.0, 2.0)
CustServCalls = st.sidebar.number_input("Customer Service Calls", 0, 10, 1)

DayMins = st.sidebar.number_input("Day Minutes", 0.0, 400.0, 200.0)
DayCalls = st.sidebar.number_input("Day Calls", 0, 200, 100)
MonthlyCharge = st.sidebar.number_input("Monthly Charge", 0.0, 200.0, 70.0)
OverageFee = st.sidebar.number_input("Overage Fee", 0.0, 50.0, 5.0)
RoamMins = st.sidebar.number_input("Roaming Minutes", 0.0, 20.0, 5.0)

# ================= DATA FRAME =================
input_df = pd.DataFrame([[
    AccountWeeks,
    ContractRenewal,
    DataPlan,
    DataUsage,
    CustServCalls,
    DayMins,
    DayCalls,
    MonthlyCharge,
    OverageFee,
    RoamMins
]], columns=[
    "AccountWeeks","ContractRenewal","DataPlan","DataUsage",
    "CustServCalls","DayMins","DayCalls","MonthlyCharge",
    "OverageFee","RoamMins"
])

# ================= PREDICTION =================
if st.button("🚀 Predict Churn Risk"):

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")

    if pred == 1:
        st.error("❌ HIGH CHURN RISK")
    else:
        st.success("✅ LOW CHURN RISK")

    # ================= METRICS =================
    st.metric("Churn Probability", f"{round(proba*100,2)}%")
    st.metric(
        "Risk Level",
        "🔴 High" if proba > 0.7 else "🟡 Medium" if proba > 0.3 else "🟢 Low"
    )

    # ================= BUSINESS INSIGHTS =================
    st.subheader("📊 Key Business Signals")

    signals = []

    if CustServCalls > 3:
        signals.append("📞 Customer is facing repeated issues")

    if MonthlyCharge > 80:
        signals.append("💰 High-value but risky customer")

    if DayMins > 250:
        signals.append("📱 Heavy usage behavior detected")

    if AccountWeeks < 50:
        signals.append("🆕 New customer (early churn risk)")

    if len(signals) == 0:
        signals.append("✅ Stable customer behavior observed")

    for s in signals:
        st.write("•", s)   





    

    

