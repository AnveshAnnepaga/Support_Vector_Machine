import streamlit as st
import pandas as pd
import pickle
import os

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="wide",
    page_icon="📊"
)

# ==================================================
# Custom CSS (Original Dark FinTech UI)
# ==================================================
st.markdown("""
<style>
body { background-color:#0a1020; color:#e2e8f0; }
.panel {
    background:#020617;
    border-radius:16px;
    padding:20px;
    border:1px solid #1e293b;
    margin-bottom:20px;
}
.ribbon-ok {
    background:linear-gradient(90deg,#14b8a6,#22d3ee);
    padding:14px;
    border-radius:12px;
    font-weight:700;
    text-align:center;
}
.ribbon-no {
    background:linear-gradient(90deg,#ef4444,#f97316);
    padding:14px;
    border-radius:12px;
    font-weight:700;
    text-align:center;
}
.bar {
    height:10px;
    background:#1e293b;
    border-radius:10px;
    overflow:hidden;
}
.fill {
    height:10px;
    background:linear-gradient(90deg,#38bdf8,#14b8a6);
}
.small { color:#94a3b8; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# Sidebar – Model Selection
# ==================================================
st.sidebar.markdown("## ⚙️ Model Settings")

kernel = st.sidebar.radio(
    "Select SVM Kernel",
    ["Linear", "Polynomial", "RBF"]
)

st.sidebar.info("💡 RBF handles complex non-linear patterns best.")

# Model file paths (must exist in repo)
model_files = {
    "Linear": "svm_linear.pkl",
    "Polynomial": "svm_poly.pkl",
    "RBF": "svm_rbf.pkl"
}

model_path = model_files[kernel]

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

# Load model
model = pickle.load(open(model_path, "rb"))

# ==================================================
# Header
# ==================================================
st.markdown("### 📊 Smart Loan Approval System")
st.markdown(
    "<p class='small'>AI-assisted credit screening dashboard</p>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ==================================================
# Input Section
# ==================================================
st.markdown("## 🔍 Applicant Information")

c1, c2 = st.columns(2)
with c1:
    income = st.number_input("Monthly Income ($)", min_value=0, value=4000)
with c2:
    loan_amount = st.number_input("Requested Loan ($K)", min_value=0, value=120)

c3, c4, c5 = st.columns(3)
with c3:
    credit = st.selectbox("Credit Record", ["Clean History", "Issues Present"])
with c4:
    employment = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
with c5:
    area = st.selectbox("Residence Area", ["Urban", "Semiurban", "Rural"])

analyze = st.button("🔎 Evaluate Risk", use_container_width=True)

# ==================================================
# Prediction Logic
# ==================================================
if analyze:

    credit_val = 1.0 if credit == "Clean History" else 0.0

    # -------- Business Rule Guard --------
    if credit_val == 0.0 and income < 2000:
        approved = False
        confidence = 95.0
    else:
        # ---- dtype-safe input dataframe ----
        input_df = pd.DataFrame([{
            "Gender": str("Male"),
            "Married": str("No"),
            "Dependents": str("0"),
            "Education": str("Graduate"),
            "Self_Employed": str("Yes" if employment == "Self-Employed" else "No"),
            "ApplicantIncome": float(income),
            "CoapplicantIncome": float(0),
            "LoanAmount": float(loan_amount),
            "Loan_Amount_Term": float(360),
            "Credit_History": float(credit_val),
            "Property_Area": str(area)
        }])

        pred = model.predict(input_df)[0]
        confidence = model.predict_proba(input_df).max() * 100
        approved = pred == "Y"

    # ==================================================
    # Output Section
    # ==================================================
    st.markdown("## 📈 Risk Summary")

    left, right = st.columns([1, 2])

    with left:
        if approved:
            st.markdown(
                "<div class='ribbon-ok'>✔ APPROVED</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='ribbon-no'>✖ REJECTED</div>",
                unsafe_allow_html=True
            )

        st.markdown(
            f"<p class='small'>Kernel Used: {kernel}</p>",
            unsafe_allow_html=True
        )

    with right:
        st.markdown(
            "<p class='small'>Model Confidence</p>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="bar">
                <div class="fill" style="width:{confidence}%;"></div>
            </div>
            <p class="small">{confidence:.1f}% certainty</p>
            """,
            unsafe_allow_html=True
        )

    # ==================================================
    # Explanation Panel
    # ==================================================
    st.markdown(
        f"""
        <div class="panel">
            <h4>🧠 Model Insight</h4>
            <p>
            Based on income stability, credit behavior, and residential profile,
            the applicant is assessed as
            <b>{"low risk" if approved else "high risk"}</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================================================
# Footer
# ==================================================
st.markdown(
    "<hr><p class='small' style='text-align:center;'>ML supports decisions. Final approval follows policy rules.</p>",
    unsafe_allow_html=True
)
