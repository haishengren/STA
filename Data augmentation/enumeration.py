import pandas as pd
from SmilesEnumerator import SmilesEnumerator

# SMILES enumeration of training and test sets for each fold
for i in range(1, 11):

    df = pd.read_excel(r'train_set_fold_{}.xlsx'.format(i))

    # Initialize a SmilesEnumerator object
    sme = SmilesEnumerator()

    # Use sets to store the generated SMILES to avoid duplicates
    unique_smiles = set()

    # Create a list to store the augmented data
    augmented_data = []

    # Expand on each line of SMILES
    for index, row in df.iterrows():
        original_smiles = row.iloc[0]  # Assuming SMILES is in the first column
        Cp = row.iloc[1]  # The second column is chemical properties

        # Generate new SMILES to ensure uniqueness
        for j in range(2000):  # Generation quantities can be adjusted as needed
            smiles = sme.randomize_smiles(original_smiles)
            if smiles not in unique_smiles:
                unique_smiles.add(smiles)
                augmented_data.append([smiles, Cp])

    # Convert augmented data to DataFrame
    augmented_df = pd.DataFrame(augmented_data, columns=['SMILES', 'Cp'])

    # Output file path
    output_file = f'enhanced_train_set_fold_{i}.xlsx'

    # Save the augmented DataFrame as a new Excel file
    augmented_df.to_excel(f'enhanced_train_set_fold_{i}.xlsx', index=False)

    print(f"Enhanced data saved to {output_file}")
