
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Separate input and target
X = data.drop("Approved", axis=1)
y = data["Approved"]

# Identify text and numerical columns
categorical_columns = X.select_dtypes(include=["object", "str"]).columns
numerical_columns = X.select_dtypes(include=["number"]).columns

# Prepare preprocessing for both types of columns
preprocessor = ColumnTransformer(
    transformers=[
        ("numbers", StandardScaler(), numerical_columns),
        ("text", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fit preprocessing on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Display results
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)
print("Preprocessing completed successfully!")
