

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("data/dataset.csv")

X = data.drop("Approved", axis=1)
y = data["Approved"]

# Identify column types
categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

if "ZipCode" in X.columns:
    categorical_columns.append("ZipCode")

numerical_columns = [
    col for col in X.select_dtypes(include=["number"]).columns
    if col not in categorical_columns
]

# Preprocessing
preprocessor = ColumnTransformer([
    ("numbers", StandardScaler(), numerical_columns),
    ("text", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])

# Same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create XGBoost pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    ))
])

# Train and save the complete pipeline
model.fit(X_train, y_train)
joblib.dump(model, "credit_card_model.pkl")

print("XGBoost model saved successfully!")
print("File: credit_card_model.pkl")
