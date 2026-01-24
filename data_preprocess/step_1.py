# This file is used to convert the ChEBI-20 dataset into a CSV file.
import pandas as pd

input_file = '../data/ChEBI-20_data/train.txt'
output_file = '../data/ChEBI-20_data/train.csv'

with open(input_file, 'r') as file:
    lines = file.readlines()[1:]

data = []

for line in lines:
    fields = line.strip().split()

    if len(fields) < 3:
        continue

    cid, smiles = fields[0], fields[1]
    description = ' '.join(fields[2:])

    data.append([cid, smiles, description])

df = pd.DataFrame(data, columns=['CID', 'SMILES', 'Description'])

df.to_csv(output_file, index=False, quoting=2)

print(f"Data has been saved to {output_file}")
