import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem import rdMolDescriptors
import matplotlib.pyplot as plt
import seaborn as sns

ATOM_TYPE_TO_SYMBOL = {
    0: 'C', 1: 'N', 2: 'O', 3: 'S', 4: 'F',
    5: 'Cl', 6: 'Br', 7: 'P', 8: 'H', 9: 'C'
}

def positions_to_mol(positions, atom_types):
    mol = Chem.RWMol()
    
    atom_indices = {}
    for i, (pos, atom_type) in enumerate(zip(positions, atom_types)):
        atom_symbol = ATOM_TYPE_TO_SYMBOL[atom_type.item()]
        atom = Chem.Atom(atom_symbol)
        idx = mol.AddAtom(atom)
        atom_indices[i] = idx
    
    conf = Chem.Conformer(len(positions))
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
    
    mol.AddConformer(conf)
    
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = torch.norm(positions[i] - positions[j]).item()
            if 0.5 < dist < 2.0:
                mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType.SINGLE)
    
    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None

def calculate_validity(molecules):
    valid_count = sum([1 for mol in molecules if mol is not None])
    return valid_count / len(molecules) if molecules else 0.0

def calculate_uniqueness(molecules):
    valid_mols = [mol for mol in molecules if mol is not None]
    if not valid_mols:
        return 0.0
    
    smiles_set = set()
    for mol in valid_mols:
        try:
            smiles = Chem.MolToSmiles(mol)
            smiles_set.add(smiles)
        except:
            pass
    
    return len(smiles_set) / len(valid_mols)

def calculate_novelty(molecules, training_smiles):
    valid_mols = [mol for mol in molecules if mol is not None]
    if not valid_mols:
        return 0.0
    
    novel_count = 0
    for mol in valid_mols:
        try:
            smiles = Chem.MolToSmiles(mol)
            if smiles not in training_smiles:
                novel_count += 1
        except:
            pass
    
    return novel_count / len(valid_mols)

def calculate_qed(molecules):
    valid_mols = [mol for mol in molecules if mol is not None]
    if not valid_mols:
        return 0.0
    
    qed_scores = []
    for mol in valid_mols:
        try:
            qed_score = QED.qed(mol)
            qed_scores.append(qed_score)
        except:
            pass
    
    return np.mean(qed_scores) if qed_scores else 0.0

def calculate_sa_score(molecules):
    valid_mols = [mol for mol in molecules if mol is not None]
    if not valid_mols:
        return 0.0
    
    sa_scores = []
    for mol in valid_mols:
        try:
            sa_score = 10 - rdMolDescriptors.CalcNumRotatableBonds(mol) / 5
            sa_scores.append(max(1, min(10, sa_score)))
        except:
            pass
    
    return np.mean(sa_scores) if sa_scores else 0.0

def calculate_lipinski(molecules):
    valid_mols = [mol for mol in molecules if mol is not None]
    if not valid_mols:
        return 0.0
    
    pass_count = 0
    for mol in valid_mols:
        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
                pass_count += 1
        except:
            pass
    
    return pass_count / len(valid_mols)

def evaluate_molecules(generated_positions, generated_atom_types):
    molecules = []
    
    for pos, atom_types in zip(generated_positions, generated_atom_types):
        mol = positions_to_mol(pos, atom_types)
        molecules.append(mol)
    
    metrics = {
        'validity': calculate_validity(molecules),
        'uniqueness': calculate_uniqueness(molecules),
        'qed': calculate_qed(molecules),
        'sa_score': calculate_sa_score(molecules),
        'lipinski': calculate_lipinski(molecules)
    }
    
    return metrics, molecules

def visualize_molecule(mol, save_path=None):
    if mol is None:
        print("Invalid molecule")
        return
    
    try:
        from rdkit.Chem import Draw
        img = Draw.MolToImage(mol, size=(400, 400))
        
        if save_path:
            img.save(save_path)
        
        return img
    except Exception as e:
        print(f"Error visualizing molecule: {e}")
        return None

def plot_training_curves(losses, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(losses['total'], label='Total Loss')
    axes[0, 0].plot(losses['coord'], label='Coordinate Loss')
    axes[0, 0].plot(losses['atom'], label='Atom Type Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    if 'validity' in losses:
        axes[0, 1].plot(losses['validity'], label='Validity')
        axes[0, 1].plot(losses['uniqueness'], label='Uniqueness')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score (%)')
        axes[0, 1].set_title('Validation Metrics')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    if 'qed' in losses:
        axes[1, 0].plot(losses['qed'], label='QED Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('QED Score')
        axes[1, 0].set_title('Drug-likeness (QED)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    if 'learning_rate' in losses:
        axes[1, 1].plot(losses['learning_rate'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
