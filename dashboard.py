
import streamlit as st
import pandas as pd
import joblib

# ================= LOAD MODEL =================
model = joblib.load("churn_model.pkl")

# ================= UI STYLE =================
st.set_page_config(page_title="Churn Dashboard", layout="centered")

st.title("📊 Customer Churn Intelligence Dashboard")
st.markdown("Predict customer churn + business insights in real time 🚀")

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

# ================= PREDICTION =================
if st.button("🚀 Predict Churn Risk"):

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

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("📌 Result")

    if pred == 1:
        st.error("❌ HIGH CHURN RISK")
    else:
        st.success("✅ LOW CHURN RISK")

    st.metric("Churn Probability", f"{round(proba*100,2)}%")

    # ================= BUSINESS INSIGHTS =================
    st.subheader("📊 Insights")

    insights = []

    if CustServCalls > 3:
        insights.append("High customer service complaints 📞")

    if MonthlyCharge > 80:
        insights.append("High monthly billing 💰")

    if DayMins > 250:
        insights.append("Heavy usage customer ⏱")

    if len(insights) == 0:
        insights.append("Stable customer behavior 👍")

    for i in insights:
        
        st.write("•", i)
        
        st.subheader("🧠 Why this prediction? (Model Explanation)")

import numpy as np

# get coefficients
try:
    model_coef = model.named_steps["model"].coef_[0]
    feature_names = input_df.columns

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Impact": model_coef
    })

    importance = importance.sort_values(by="Impact")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.barh(importance["Feature"], importance["Impact"])

    ax.set_title("Feature Impact (Logistic Regression)")
    st.pyplot(fig)

except Exception as e:
    st.write("Explanation not available, but prediction is working.")
    





    

    

