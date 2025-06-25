import pandas as pd
from sklearn.model_selection import KFold


data = pd.read_excel("Database/Cp.xlsx")

# Assuming the first column is SMILES and the second column is Cp
smiles = data.iloc[:, 0]
properties = data.iloc[:, 1]

# Create a ten-fold cross-validator
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Create an empty list to hold the indexes of each training set and test set
train_test_splits = []

# Generate training and test sets for ten-fold cross-validation
for train_index, test_index in kf.split(smiles):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    train_test_splits.append((train_data, test_data))

# Print training set and test set sizes for each fold
for i, (train_data, test_data) in enumerate(train_test_splits):
    print(f"Fold {i+1}:")
    print(f"Training set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")

for i, (train_data, test_data) in enumerate(train_test_splits):
    train_data.to_excel(f'train_set_fold_{i+1}.xlsx', index=False)
    test_data.to_excel(f'test_set_fold_{i+1}.xlsx', index=False)
