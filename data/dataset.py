import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import random

ATOM_TYPE_MAP = {
    'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4,
    'Cl': 5, 'Br': 6, 'P': 7, 'H': 8, 'other': 9
}

AA_TO_INDEX = {
    'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
    'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
    'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
    'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19
}

PROPERTY_PROMPTS = [
    "Generate a non-toxic molecule with molecular weight less than 400",
    "Create a brain-penetrating lipophilic kinase inhibitor",
    "Design a highly soluble antibiotic with low toxicity",
    "Generate a membrane-permeable GPCR agonist",
    "Create a drug-like molecule with good oral bioavailability",
    "Design a non-toxic inhibitor with high binding affinity",
    "Generate a small molecule with good synthetic accessibility",
    "Create a molecule with balanced hydrophilicity and lipophilicity"
]

class MolecularDataset(Dataset):
    def __init__(self, data_path, config, split='train'):
        self.data_path = data_path
        self.config = config
        self.split = split
        
        if not os.path.exists(data_path):
            print(f"Dataset path {data_path} does not exist. Generating synthetic data...")
            self.generate_synthetic_data()
        else:
            self.load_data()
    
    def generate_synthetic_data(self):
        print("Generating synthetic molecular data for demonstration...")
        num_samples = 1000 if self.split == 'train' else 200
        
        self.protein_data = []
        self.ligand_data = []
        self.text_prompts = []
        
        for i in range(num_samples):
            num_protein_atoms = random.randint(50, 150)
            protein_pos = torch.randn(num_protein_atoms, 3) * 5.0
            protein_features = torch.zeros(num_protein_atoms, 20)
            for j in range(num_protein_atoms):
                aa_type = random.randint(0, 19)
                protein_features[j, aa_type] = 1.0
            
            edge_index = []
            for j in range(num_protein_atoms):
                for k in range(j+1, min(j+10, num_protein_atoms)):
                    dist = torch.norm(protein_pos[j] - protein_pos[k])
                    if dist < self.config.distance_cutoff:
                        edge_index.append([j, k])
                        edge_index.append([k, j])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t() if edge_index else torch.zeros((2, 0), dtype=torch.long)
            
            self.protein_data.append({
                'pos': protein_pos,
                'node_features': protein_features,
                'edge_index': edge_index
            })
            
            num_ligand_atoms = random.randint(10, self.config.max_atoms)
            ligand_pos = torch.randn(num_ligand_atoms, 3) * 2.0
            ligand_atom_types = torch.randint(0, self.config.num_atom_types, (num_ligand_atoms,))
            
            self.ligand_data.append({
                'pos': ligand_pos,
                'atom_types': ligand_atom_types
            })
            
            prompt = random.choice(PROPERTY_PROMPTS)
            self.text_prompts.append(prompt)
        
        print(f"Generated {num_samples} synthetic samples for {self.split} split")
    
    def load_data(self):
        data_file = os.path.join(self.data_path, f'{self.split}.pkl')
        
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            self.protein_data = data['protein_data']
            self.ligand_data = data['ligand_data']
            self.text_prompts = data['text_prompts']
        else:
            self.generate_synthetic_data()
    
    def __len__(self):
        return len(self.protein_data)
    
    def __getitem__(self, idx):
        protein = self.protein_data[idx]
        ligand = self.ligand_data[idx]
        prompt = self.text_prompts[idx]
        
        return {
            'protein_pos': protein['pos'],
            'protein_features': protein['node_features'],
            'protein_edge_index': protein['edge_index'],
            'ligand_pos': ligand['pos'],
            'ligand_atom_types': ligand['atom_types'],
            'text_prompt': prompt
        }

def collate_fn(batch):
    batch_size = len(batch)
    
    max_protein_atoms = max([item['protein_pos'].size(0) for item in batch])
    max_ligand_atoms = max([item['ligand_pos'].size(0) for item in batch])
    
    protein_pos_batch = torch.zeros(batch_size, max_protein_atoms, 3)
    protein_features_batch = torch.zeros(batch_size, max_protein_atoms, 20)
    protein_mask = torch.zeros(batch_size, max_protein_atoms, dtype=torch.bool)
    
    ligand_pos_batch = torch.zeros(batch_size, max_ligand_atoms, 3)
    ligand_atom_types_batch = torch.zeros(batch_size, max_ligand_atoms, dtype=torch.long)
    ligand_mask = torch.zeros(batch_size, max_ligand_atoms, dtype=torch.bool)
    
    edge_index_list = []
    batch_idx = []
    text_prompts = []
    
    offset = 0
    for i, item in enumerate(batch):
        num_protein = item['protein_pos'].size(0)
        num_ligand = item['ligand_pos'].size(0)
        
        protein_pos_batch[i, :num_protein] = item['protein_pos']
        protein_features_batch[i, :num_protein] = item['protein_features']
        protein_mask[i, :num_protein] = True
        
        ligand_pos_batch[i, :num_ligand] = item['ligand_pos']
        ligand_atom_types_batch[i, :num_ligand] = item['ligand_atom_types']
        ligand_mask[i, :num_ligand] = True
        
        edge_index = item['protein_edge_index'] + offset
        edge_index_list.append(edge_index)
        batch_idx.extend([i] * num_protein)
        offset += num_protein
        
        text_prompts.append(item['text_prompt'])
    
    protein_pos_flat = protein_pos_batch[protein_mask]
    protein_features_flat = protein_features_batch[protein_mask]
    edge_index_combined = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long)
    batch_tensor = torch.tensor(batch_idx, dtype=torch.long)
    
    return {
        'protein_data': {
            'pos': protein_pos_flat,
            'node_features': protein_features_flat,
            'edge_index': edge_index_combined,
            'batch': batch_tensor
        },
        'ligand_pos': ligand_pos_batch,
        'ligand_atom_types': ligand_atom_types_batch,
        'ligand_mask': ligand_mask,
        'text_prompts': text_prompts
    }

def get_dataloader(config, split='train'):
    dataset = MolecularDataset(config.data_path, config, split=split)
    
    batch_size = config.batch_size if split == 'train' else config.batch_size * 2
    shuffle = split == 'train'
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if config.device.type == 'cuda' else False
    )
    
    return dataloader
