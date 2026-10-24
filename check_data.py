
import pandas as pd

# Load the dataset
data = pd.read_csv("data/dataset.csv")

# Display first 5 rows
print(data.head())

# Display number of rows and columns
print("Dataset shape:", data.shape)

# Display column names
print("Column names:")
print(data.columns)

# Check missing values
print("Missing values:")
print(data.isnull().sum())
