


import pandas as pd

# Load the dataset
data = pd.read_csv("data/dataset.csv")

# Show dataset information
print("Dataset shape:", data.shape)

# Count approved and rejected applications
print("\nApproval counts:")
print(data["Approved"].value_counts())

# Show approval percentages
print("\nApproval percentages:")
print(data["Approved"].value_counts(normalize=True) * 100)

# Show basic statistics
print("\nDataset statistics:")
print(data.describe())

