import torch
import os
import sys
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import config
from models.lg3d_mol import LG3DMol
from data.dataset import get_dataloader
from utils.molecule_utils import evaluate_molecules, visualize_molecule

@torch.no_grad()
def generate_molecules(model, dataloader, device, num_samples=100):
    model.eval()
    
    all_generated_positions = []
    all_generated_atom_types = []
    all_prompts = []
    
    num_batches = (num_samples + config.batch_size - 1) // config.batch_size
    
    print(f"Generating {num_samples} molecules...")
    for i, batch in enumerate(tqdm(dataloader)):
        if i >= num_batches:
            break
        
        protein_data = {k: v.to(device) for k, v in batch['protein_data'].items()}
        text_prompts = batch['text_prompts']
        
        batch_size = min(config.batch_size, num_samples - i * config.batch_size)
        
        generated_pos, generated_atom_types = model.generate(
            protein_data,
            text_prompts[:batch_size],
            num_samples=batch_size,
            num_atoms=config.max_atoms
        )
        
        all_generated_positions.append(generated_pos.cpu())
        all_generated_atom_types.append(generated_atom_types.cpu())
        all_prompts.extend(text_prompts[:batch_size])
    
    all_generated_positions = torch.cat(all_generated_positions, dim=0)[:num_samples]
    all_generated_atom_types = torch.cat(all_generated_atom_types, dim=0)[:num_samples]
    
    return all_generated_positions, all_generated_atom_types, all_prompts[:num_samples]

def main():
    parser = argparse.ArgumentParser(description='Generate molecules using trained LG3D-Mol model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of molecules to generate')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize generated molecules')
    args = parser.parse_args()
    
    print(f"Using device: {config.device}")
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    print("Loading model...")
    model = LG3DMol(config).to(config.device)
    
    checkpoint = torch.load(args.checkpoint, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print("Previous metrics:")
        for key, value in checkpoint['metrics'].items():
            print(f"  {key}: {value:.4f}")
    
    print("\nLoading data...")
    val_loader = get_dataloader(config, split='val')
    
    generated_positions, generated_atom_types, prompts = generate_molecules(
        model, val_loader, config.device, args.num_samples
    )
    
    print("\nEvaluating generated molecules...")
    metrics, molecules = evaluate_molecules(generated_positions, generated_atom_types)
    
    print("\nGeneration Results:")
    print(f"  Validity: {metrics['validity']*100:.2f}%")
    print(f"  Uniqueness: {metrics['uniqueness']*100:.2f}%")
    print(f"  QED Score: {metrics['qed']:.4f}")
    print(f"  SA Score: {metrics['sa_score']:.4f}")
    print(f"  Lipinski Pass: {metrics['lipinski']*100:.2f}%")
    
    if args.visualize:
        os.makedirs('results/visualizations', exist_ok=True)
        print("\nVisualizing molecules...")
        
        valid_mols = [(mol, prompt) for mol, prompt in zip(molecules, prompts) if mol is not None]
        
        for i, (mol, prompt) in enumerate(valid_mols[:10]):
            img = visualize_molecule(mol, f'results/visualizations/molecule_{i}.png')
            if img:
                print(f"  Saved molecule {i}: {prompt}")
    
    results_path = os.path.join(config.results_dir, 'generation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Generated {args.num_samples} molecules\n\n")
        f.write("Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        f.write("\nSample prompts:\n")
        for i, prompt in enumerate(prompts[:10]):
            f.write(f"  {i+1}. {prompt}\n")
    
    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    main()
