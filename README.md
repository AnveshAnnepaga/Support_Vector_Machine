# 🏦 Smart Loan Approval System

An AI-powered **Loan Risk Intelligence & Approval System** built using **Support Vector Machines (SVM)** and deployed with **Streamlit**.
The system predicts whether a loan application should be **Approved or Rejected** by combining **machine learning predictions** with **real-world business rules**, presented through a modern FinTech-style dashboard.

---

## 📌 Project Overview

Financial institutions receive thousands of loan applications daily. Manual evaluation is:

* Time-consuming
* Error-prone
* Inconsistent across evaluators

This project demonstrates how **machine learning can assist loan approval decisions** while still respecting **business constraints**.

The application:

* Uses **SVM classifiers** with different kernels
* Handles missing and categorical data correctly
* Provides **explainable AI outputs**
* Prevents illogical approvals using **rule-based validation**

---

## 🎯 Problem Statement

Predict whether a loan application will be **Approved (Y)** or **Rejected (N)** based on applicant details such as:

* Income
* Credit history
* Employment status
* Property area

The solution must:

* Handle non-linear patterns
* Be robust to missing data
* Provide confidence scores
* Align predictions with business logic

---

## 🧠 Machine Learning Approach

### Model Used

* **Support Vector Machine (SVM)**

### Kernels Implemented

* **Linear SVM** – for linearly separable patterns
* **Polynomial SVM** – for moderate non-linearity
* **RBF SVM** – for complex, non-linear decision boundaries

Users can dynamically select the kernel from the UI.

---

## 🔄 ML Pipeline Design

A complete **Scikit-learn Pipeline** is used to ensure consistency between training and inference.

### Pipeline Steps

1. **Missing Value Handling**

   * Numerical: Median Imputation
   * Categorical: Most Frequent Imputation

2. **Feature Engineering**

   * Numerical Scaling: `StandardScaler`
   * Categorical Encoding: `OneHotEncoder`

3. **Model Training**

   * SVM classifier with selected kernel

4. **Serialization**

   * Entire pipeline saved using `pickle`

This guarantees that preprocessing during deployment is identical to training.

---

## ⚠️ Business Rule Integration

Pure ML predictions can sometimes produce **business-invalid results**.

To address this, a **rule-based guard** is applied:

> ❌ If **Credit History = No** and **Income < 2000**, the loan is automatically rejected.

This mirrors real-world banking systems where:

* ML assists decisions
* Business rules make the final call

---

## 🖥️ Application Features

### User Interface

* Dark FinTech-style dashboard
* Sidebar-based configuration panel
* Clean card-based layout

### Functional Highlights

* Kernel selection (Linear / Polynomial / RBF)
* Confidence score visualization
* Clear approval / rejection status
* Human-readable AI explanation

### Explainability

Each prediction includes:

* Risk category (Low / High)
* Confidence percentage
* Natural language explanation

---

## 📁 Project Structure

```
Support_Vector_Machine/
│
├── train.py                 # Model training & serialization
├── app.py                   # Streamlit application
├── svm_linear.pkl           # Trained Linear SVM pipeline
├── svm_poly.pkl             # Trained Polynomial SVM pipeline
├── svm_rbf.pkl              # Trained RBF SVM pipeline
├── train_u6lujuX_CVtuZ9i.csv # Dataset
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Models

```bash
python train.py
```

This generates the trained model files (`.pkl`).

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

Open the browser link displayed in the terminal.

---

## 📊 Dataset Information

The dataset contains historical loan application records with:

* Applicant financial details
* Credit history
* Loan approval outcome

Target variable:

* **Loan_Status** (Y / N)

---

## 🧪 Evaluation & Confidence

* Probability scores are obtained using `predict_proba()`
* Confidence reflects model certainty, **not a guarantee**

---

## 🎓 Academic & Interview Value

This project demonstrates:

* End-to-end ML lifecycle
* Proper preprocessing using pipelines
* Real-world deployment readiness
* Explainable AI concepts
* Ethical handling of ML predictions

### Strong Interview Statement

> “I built an SVM-based loan approval system using a Scikit-learn pipeline, integrated business rules to prevent invalid predictions, and deployed it with a custom Streamlit FinTech dashboard.”

---

## 🚀 Future Enhancements

* Kernel performance comparison dashboard
* Feature importance approximation
* PDF loan report generation
* Fairness & bias analysis
* Model monitoring

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
Predictions should not be used for real financial decisions without regulatory approval.

---

## 👨‍💻 Author

Developed as part of a **Machine Learning & Deployment** project using Python, Scikit-learn, and Streamlit.
