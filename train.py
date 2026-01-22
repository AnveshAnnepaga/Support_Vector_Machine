import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Drop ID column
df.drop("Loan_ID", axis=1, inplace=True)

# Features & target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Column types
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

# Pipelines
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

for kernel, filename in kernels.items():
    model = SVC(kernel=kernel, probability=True)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X, y)
    pickle.dump(pipeline, open(filename, "wb"))

    print(f"✅ {kernel.upper()} model saved as {filename}")
