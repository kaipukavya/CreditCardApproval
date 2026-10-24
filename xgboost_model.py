

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
data = pd.read_csv("data/dataset.csv")

# 2. Separate inputs and target
X = data.drop("Approved", axis=1)
y = data["Approved"]

# 3. Identify column types
categorical_columns = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

if "ZipCode" in X.columns:
    categorical_columns.append("ZipCode")

numerical_columns = [
    col for col in X.select_dtypes(include=["number"]).columns
    if col not in categorical_columns
]

# 4. Preprocessing
preprocessor = ColumnTransformer([
    ("numbers", StandardScaler(), numerical_columns),
    ("text", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
])

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Create model
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

# 7. Train model
model.fit(X_train, y_train)

# 8. Predict
predictions = model.predict(X_test)

# 9. Evaluate
print("===== XGBOOST =====")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))
