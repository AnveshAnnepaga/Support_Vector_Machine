import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

# Load data
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
df.drop("Loan_ID", axis=1, inplace=True)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

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
    pipe = Pipeline([
        ("preprocessing", preprocessor),
        ("model", SVC(kernel=kernel, probability=True))
    ])
    pipe.fit(X, y)
    pickle.dump(pipe, open(fname, "wb"))

print("✅ Models trained and saved")
