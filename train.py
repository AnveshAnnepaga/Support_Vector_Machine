import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
df.drop("Loan_ID", axis=1, inplace=True)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# SAVE FEATURE ORDER
feature_order = X.columns.tolist()
pickle.dump(feature_order, open("feature_order.pkl", "wb"))

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

kernels = {
    "linear": "svm_linear.pkl",
    "poly": "svm_poly.pkl",
    "rbf": "svm_rbf.pkl"
}

for kernel, fname in kernels.items():
    model = SVC(kernel=kernel, probability=True)
    pipe = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])
    pipe.fit(X, y)
    pickle.dump(pipe, open(fname, "wb"))

print("✅ Models and feature order saved")
