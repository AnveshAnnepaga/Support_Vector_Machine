import streamlit as st
import pandas as pd
import pickle
import os

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Loan Risk Intelligence",
    layout="centered"
)

# -------------------------------------------------
# Load model
# -------------------------------------------------
MODEL_FILES = {
    "Linear": "svm_linear.pkl",
    "Polynomial": "svm_poly.pkl",
    "RBF": "svm_rbf.pkl"
}

st.sidebar.title("⚙️ Model Settings")
kernel = st.sidebar.radio("Select SVM Kernel", list(MODEL_FILES.keys()))

model_path = MODEL_FILES[kernel]

if not os.path.exists(model_path):
    st.error(f"Missing model file: {model_path}")
    st.stop()

model = pickle.load(open(model_path, "rb"))

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("📊 Loan Risk Intelligence")

income = st.number_input("Monthly Income", min_value=0, value=4000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=120)
credit = st.selectbox("Credit History", ["Clean", "Issues"])
employment = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Evaluate Risk"):

    credit_val = 1.0 if credit == "Clean" else 0.0

    # Business rule
    if credit_val == 0 and income < 2000:
        approved = False
        confidence = 95.0
    else:
        input_df = pd.DataFrame([{
            "Gender": "Male",
            "Married": "No",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "Yes" if employment == "Self-Employed" else "No",
            "ApplicantIncome": float(income),
            "CoapplicantIncome": 0.0,
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": 360.0,
            "Credit_History": float(credit_val),
            "Property_Area": area
        }])

        pred = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df).max() * 100
        approved = pred == "Y"

    if approved:
        st.success(f"✅ LOAN APPROVED ({confidence:.1f}% confidence)")
    else:
        st.error(f"❌ LOAN REJECTED ({confidence:.1f}% confidence)")

    st.info(
        f"🧠 **Explanation:** Based on income stability, credit history, and "
        f"property profile, the applicant is "
        f"**{'low risk' if approved else 'high risk'}**."
    )
