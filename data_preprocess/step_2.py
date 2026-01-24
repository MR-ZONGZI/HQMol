# This file is used to filter the ChEBI-20 data from PubChemSTM.
import pandas as pd

# File paths
file = '../data/PubChemSTM/PubChemSTM_original.csv'  # CSV file from which rows need to be deleted
reference_files = ['../data/ChEBI-20_data/train.csv', '../data/ChEBI-20_data/valid.csv', '../data/ChEBI-20_data/test.csv']  # Multiple CSV files used as reference
output_file = '../data/PubChemSTM/PubChemSTM_filter_chebi.csv'  # Output CSV file path

# Read the first CSV file
df1 = pd.read_csv(file)

# Initialize an empty set to store all CIDs to be removed
cids_to_remove = set()

# Read each reference file and add CIDs to the cids_to_remove set
for ref_file in reference_files:
    df_ref = pd.read_csv(ref_file)
    cids_to_remove.update(df_ref['CID'])

# Filter out CIDs present in the reference files from the first file
df_filtered = df1[~df1['CID'].isin(cids_to_remove)]

df_filtered.to_csv(output_file, index=False, quoting=2)

print(f"Filtered data has been saved to {output_file}, total {df_filtered.shape[0]} records")

