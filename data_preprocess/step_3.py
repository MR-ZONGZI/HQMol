import pickle
from data_utils import MoleculeDataset
import csv
import random
import selfies as sf


def construct_instruct_question(selfies_str:str=None):
    """
    Construct instruct question for each graph
    """
    question_pools = [
        'Could you give me a brief overview of this molecule?',
        'Could you provide a description of this molecule?',
        'Describe this molecule.',
        'Please give me some details about this molecule.',
        'Provide a brief overview of this molecule.',
        'Provide a description of this molecule.',
        'What can you tell me about this molecule?'
    ]
    question = random.choice(question_pools)
    if selfies_str is not None:
        question += f" The compound SMILES sequence is: {selfies_str}."
    if random.random() < 0.5:
        question = "<image>\n" + question
    else:
        question = question + "\n<image>"
    return question


smiles_list = []
description_list = []

file_paths = [
    'Path to your CSV file',
]

for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row['SMILES'])
            description_list.append(row['Description'])

print('Have already loaded smiles list and description list.')

dataset = MoleculeDataset(smiles_list, "atom")

processed_data = []

for idx in range(len(dataset)):
    try:
        mol_graph = dataset[idx] 
    except ValueError as e:
        print(f"Skipping invalid SMILES at index {idx}: {e}")
        continue
    
    smiles = smiles_list[idx]
    description = description_list[idx] 

    try:
        selfies = sf.encoder(smiles)
    except:
        selfies = ""
    
    edge_index = mol_graph.edge_index  
    edge_feat = mol_graph.edge_attr    
    node_feat = mol_graph.x            
    num_nodes = mol_graph.num_part     

    data_dict = {
        'graph': {
            'edge_index': edge_index,
            'edge_feat': edge_feat,
            'node_feat': node_feat,
            'num_nodes': num_nodes,
        },
        'conversations': [
            {'from': 'human', 'value': construct_instruct_question(selfies)},
            {'from': 'gpt', 'value': description}
        ],
    }
    
    processed_data.append(data_dict)

data_count = len(processed_data)

output_file = f'../data/PubChemSTM/pubchem_himol_{data_count}.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(processed_data, f)

print(f"Processed data has been saved to {output_file}")
print(f"Total number of data items: {data_count}")