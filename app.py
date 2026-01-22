import streamlit as st
import pandas as pd
import pickle

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="wide",
    page_icon="📊"
)

# ---------------------------------------------------
# Custom CSS (Original Theme)
# ---------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0a1020;
    color: #e2e8f0;
}
.sidebar .sidebar-content {
    background-color: #020617;
}
.panel {
    background: linear-gradient(180deg, #020617, #020617);
    border-radius: 18px;
    padding: 22px;
    border: 1px solid #1e293b;
    margin-bottom: 20px;
}
.ribbon-approve {
    background: linear-gradient(90deg, #14b8a6, #22d3ee);
    padding: 14px;
    border-radius: 14px;
    font-weight: 700;
    text-align: center;
}
.ribbon-reject {
    background: linear-gradient(90deg, #ef4444, #f97316);
    padding: 14px;
    border-radius: 14px;
    font-weight: 700;
    text-align: center;
}
.gauge {
    height: 10px;
    background-color: #1e293b;
    border-radius: 10px;
    overflow: hidden;
}
.gauge-fill {
    height: 10px;
    background: linear-gradient(90deg, #38bdf8, #14b8a6);
}
.label {
    color: #94a3b8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------
st.sidebar.markdown("## ⚙️ Model Controls")

kernel = st.sidebar.radio(
    "Kernel Strategy",
    ["Linear", "Polynomial", "RBF"]
)

st.sidebar.markdown(
    "<p class='label'>RBF adapts better to non-linear credit patterns.</p>",
    unsafe_allow_html=True
)

model_files = {
    "Linear": "svm_linear.pkl",
    "Polynomial": "svm_poly.pkl",
    "RBF": "svm_rbf.pkl"
}
model = pickle.load(open(model_files[kernel], "rb"))

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.markdown("### 📊 Smart Loan Approval System")
st.markdown(
    "<p class='label'>Decision-support system for automated credit screening</p>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------
st.markdown("## 🔍 Applicant Snapshot")

c1, c2 = st.columns(2)
with c1:
    income = st.number_input("Monthly Income ($)", value=4000)
with c2:
    loan_amount = st.number_input("Requested Loan ($K)", value=120)

c3, c4, c5 = st.columns(3)
with c3:
    credit = st.selectbox("Credit Record", ["Clean History", "Issues Present"])
with c4:
    employment = st.selectbox("Employment Category", ["Salaried", "Self-Employed"])
with c5:
    area = st.selectbox("Residence Zone", ["Urban", "Semiurban", "Rural"])

# ---------------------------------------------------
# Action
# ---------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
analyze = st.button("🔎 Evaluate Risk", use_container_width=True)

# ---------------------------------------------------
# Decision Logic
# ---------------------------------------------------
if analyze:

    credit_val = 1.0 if credit == "Clean History" else 0.0

    if credit_val == 0 and income < 2000:
        approved = False
        confidence = 92.0
    else:
        input_df = pd.DataFrame([{
            "Gender": "Male",
            "Married": "No",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "Yes" if employment == "Self-Employed" else "No",
            "ApplicantIncome": income,
            "CoapplicantIncome": 0,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": 360,
            "Credit_History": credit_val,
            "Property_Area": area
        }])

        pred = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df).max() * 100
        approved = pred == "Y"

    # ---------------------------------------------------
    # Results Layout
    # ---------------------------------------------------
    st.markdown("## 📈 Risk Summary")

    colA, colB = st.columns([1, 2])

    with colA:
        if approved:
            st.markdown(
                "<div class='ribbon-approve'>✔ APPROVED</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='ribbon-reject'>✖ REJECTED</div>",
                unsafe_allow_html=True
            )

        st.markdown(f"<p class='label'>Kernel: {kernel}</p>", unsafe_allow_html=True)

    with colB:
        st.markdown("<p class='label'>Confidence Level</p>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="gauge">
                <div class="gauge-fill" style="width:{confidence}%;"></div>
            </div>
            <p class="label">{confidence:.1f}% model certainty</p>
            """,
            unsafe_allow_html=True
        )

    # ---------------------------------------------------
    # Model Insight
    # ---------------------------------------------------
    st.markdown(
        f"""
        <div class="panel">
            <h4>🧠 Model Insight</h4>
            <p>
            The system categorized this applicant as
            <b>{"low risk" if approved else "high risk"}</b>
            based on income stability, credit behavior, and residential profile.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown(
    "<hr><p class='label' style='text-align:center;'>Predictions support decisions. Final approvals follow policy rules.</p>",
    unsafe_allow_html=True
)
