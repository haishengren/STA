import pandas as pd
from SmilesEnumerator import SmilesEnumerator


input_file = r"Database\Extrapolated_data_in_QM9.xlsx"
output_file = 'Augmented_data.xlsx'

df = pd.read_excel(input_file)

# Initialize a SmilesEnumerator object
sme = SmilesEnumerator()

# Use sets to store the generated SMILES to avoid duplicates
unique_smiles = set()

# Create a list to store the augmented data
augmented_data = []

# Expand on each line of SMILES
for index, row in df.iterrows():
    original_smiles = row.iloc[0]  # Assuming SMILES is in the first column
    Hf = row.iloc[1]  # The second column is chemical properties

    # Generate new SMILES to ensure uniqueness
    for i in range(2000):  # Generation quantities can be adjusted as needed
        smiles = sme.randomize_smiles(original_smiles)
        if smiles not in unique_smiles:
            unique_smiles.add(smiles)
            augmented_data.append([smiles, Hf])  # Add new SMILES and original generation enthalpy to the result list

# Convert augmented data to DataFrame
augmented_df = pd.DataFrame(augmented_data, columns=['SMILES', 'Hf'])

# Save the augmented DataFrame as a new Excel file
augmented_df.to_excel(output_file, index=False)

print(f"Enhanced data saved to {output_file}")
