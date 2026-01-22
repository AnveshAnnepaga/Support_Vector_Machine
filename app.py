import streamlit as st
import pickle
import numpy as np
import os

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Loan Risk Intelligence",
    layout="wide",
    page_icon="📊"
)

# =====================================================
# Sidebar – Model Settings
# =====================================================
st.sidebar.title("⚙️ Model Configuration")

kernel = st.sidebar.radio(
    "Select SVM Kernel",
    ["Linear", "Polynomial", "RBF"]
)

st.sidebar.caption("RBF handles complex non-linear credit patterns best.")

# =====================================================
# Load Models & Feature Order
# =====================================================
MODEL_FILES = {
    "Linear": "svm_linear.pkl",
    "Polynomial": "svm_poly.pkl",
    "RBF": "svm_rbf.pkl"
}

FEATURE_ORDER_FILE = "feature_order.pkl"

model_path = MODEL_FILES[kernel]

# Safety checks
if not os.path.exists(model_path):
    st.error(f"❌ Missing model file: {model_path}")
    st.stop()

if not os.path.exists(FEATURE_ORDER_FILE):
    st.error("❌ Missing feature_order.pkl. Please re-run train.py.")
    st.stop()

# Load artifacts
model = pickle.load(open(model_path, "rb"))
feature_order = pickle.load(open(FEATURE_ORDER_FILE, "rb"))

# =====================================================
# App Header
# =====================================================
st.markdown("## 📊 Loan Risk Intelligence")
st.markdown(
    "AI-assisted credit risk evaluation using **Support Vector Machines (SVM)**."
)
st.divider()

# =====================================================
# Input Section
# =====================================================
st.subheader("🔍 Applicant Information")

col1, col2 = st.columns(2)
with col1:
    income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=4000,
        help="Applicant’s monthly income"
    )

with col2:
    loan_amount = st.number_input(
        "Requested Loan Amount",
        min_value=0,
        value=120,
        help="Loan amount requested"
    )

col3, col4, col5 = st.columns(3)
with col3:
    credit = st.selectbox(
        "Credit History",
        ["Clean", "Issues"],
        help="Past credit repayment behaviour"
    )

with col4:
    employment = st.selectbox(
        "Employment Type",
        ["Salaried", "Self-Employed"]
    )

with col5:
    area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

st.divider()

# =====================================================
# Prediction Button
# =====================================================
predict_btn = st.button("🔎 Evaluate Loan Risk", use_container_width=True)

# =====================================================
# Prediction Logic (NUMPY-BASED)
# =====================================================
if predict_btn:

    credit_val = 1.0 if credit == "Clean" else 0.0

    # -------------------------------
    # Business Rule Guard
    # -------------------------------
    if credit_val == 0.0 and income < 2000:
        approved = False
        confidence = 95.0
    else:
        # Build raw input record
        record = {
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
        }

        # Convert to NumPy array in TRAINING ORDER
        X_np = np.array(
            [[record[col] for col in feature_order]],
            dtype=object
        )

        pred = model.predict(X_np)[0]
        confidence = model.predict_proba(X_np).max() * 100
        approved = pred == "Y"

    # =================================================
    # Output Section
    # =================================================
    st.subheader("📈 Risk Assessment Result")

    colA, colB = st.columns([1, 2])

    with colA:
        if approved:
            st.success("✅ LOAN APPROVED")
        else:
            st.error("❌ LOAN REJECTED")

        st.caption(f"Kernel Used: **{kernel}**")

    with colB:
        st.metric(
            label="Model Confidence",
            value=f"{confidence:.1f}%"
        )

    st.info(
        f"🧠 **Model Insight:** Based on income stability, credit behaviour, "
        f"and property profile, the applicant is assessed as "
        f"**{'low risk' if approved else 'high risk'}**."
    )

# =====================================================
# Footer
# =====================================================
st.divider()
st.caption(
    "⚠️ This system assists decision-making. Final loan approval follows institutional policies."
)
