import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import config
from models.lg3d_mol import LG3DMol
from data.dataset import get_dataloader
from utils.molecule_utils import evaluate_molecules, plot_training_curves

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    coord_loss_total = 0
    atom_loss_total = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in progress_bar:
        protein_data = {k: v.to(device) for k, v in batch['protein_data'].items()}
        ligand_pos = batch['ligand_pos'].to(device)
        ligand_atom_types = batch['ligand_atom_types'].to(device)
        ligand_mask = batch['ligand_mask'].to(device)
        text_prompts = batch['text_prompts']
        
        batch_size = ligand_pos.size(0)
        timesteps = torch.randint(0, config.diffusion_timesteps, (batch_size,), device=device)
        
        optimizer.zero_grad()
        
        coord_noise_pred, atom_type_logits, noise = model(
            protein_data,
            text_prompts,
            ligand_pos,
            ligand_atom_types,
            timesteps
        )
        
        coord_loss = nn.functional.mse_loss(
            coord_noise_pred[ligand_mask],
            noise[ligand_mask]
        )
        
        atom_loss = nn.functional.cross_entropy(
            atom_type_logits[ligand_mask].reshape(-1, config.num_atom_types),
            ligand_atom_types[ligand_mask].reshape(-1)
        )
        
        loss = coord_loss + atom_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        coord_loss_total += coord_loss.item()
        atom_loss_total += atom_loss.item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'coord': f'{coord_loss.item():.4f}',
            'atom': f'{atom_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_coord_loss = coord_loss_total / len(dataloader)
    avg_atom_loss = atom_loss_total / len(dataloader)
    
    return avg_loss, avg_coord_loss, avg_atom_loss

@torch.no_grad()
def evaluate(model, dataloader, device, epoch):
    model.eval()
    
    all_generated_positions = []
    all_generated_atom_types = []
    
    num_eval_batches = min(5, len(dataloader))
    
    for i, batch in enumerate(dataloader):
        if i >= num_eval_batches:
            break
        
        protein_data = {k: v.to(device) for k, v in batch['protein_data'].items()}
        text_prompts = batch['text_prompts']
        
        generated_pos, generated_atom_types = model.generate(
            protein_data,
            text_prompts,
            num_samples=batch['ligand_pos'].size(0),
            num_atoms=config.max_atoms
        )
        
        all_generated_positions.append(generated_pos.cpu())
        all_generated_atom_types.append(generated_atom_types.cpu())
    
    all_generated_positions = torch.cat(all_generated_positions, dim=0)
    all_generated_atom_types = torch.cat(all_generated_atom_types, dim=0)
    
    metrics, molecules = evaluate_molecules(all_generated_positions, all_generated_atom_types)
    
    print(f"\nEvaluation Epoch {epoch}:")
    print(f"  Validity: {metrics['validity']*100:.2f}%")
    print(f"  Uniqueness: {metrics['uniqueness']*100:.2f}%")
    print(f"  QED Score: {metrics['qed']:.4f}")
    print(f"  SA Score: {metrics['sa_score']:.4f}")
    print(f"  Lipinski Pass: {metrics['lipinski']*100:.2f}%")
    
    return metrics

def main():
    print(f"Using device: {config.device}")
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    print("Loading data...")
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    print("Initializing model...")
    model = LG3DMol(config).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=0)
    
    history = {
        'total': [], 'coord': [], 'atom': [],
        'validity': [], 'uniqueness': [], 'qed': [],
        'sa_score': [], 'lipinski': [], 'learning_rate': []
    }
    
    best_validity = 0.0
    
    print("\nStarting training...")
    for epoch in range(1, config.num_epochs + 1):
        avg_loss, avg_coord_loss, avg_atom_loss = train_epoch(
            model, train_loader, optimizer, config.device, epoch
        )
        
        history['total'].append(avg_loss)
        history['coord'].append(avg_coord_loss)
        history['atom'].append(avg_atom_loss)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  Train Loss: {avg_loss:.4f} (Coord: {avg_coord_loss:.4f}, Atom: {avg_atom_loss:.4f})")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        if epoch % config.eval_every == 0:
            metrics = evaluate(model, val_loader, config.device, epoch)
            
            history['validity'].append(metrics['validity'])
            history['uniqueness'].append(metrics['uniqueness'])
            history['qed'].append(metrics['qed'])
            history['sa_score'].append(metrics['sa_score'])
            history['lipinski'].append(metrics['lipinski'])
            
            if metrics['validity'] > best_validity:
                best_validity = metrics['validity']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': metrics,
                    'history': history
                }, os.path.join(config.checkpoint_dir, 'best_model.pt'))
                print(f"  Saved best model with validity: {best_validity*100:.2f}%")
        
        if epoch % config.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history
            }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        scheduler.step()
        
        if epoch % config.save_every == 0:
            plot_training_curves(history, os.path.join(config.results_dir, f'training_curves_epoch_{epoch}.png'))
    
    print("\nTraining completed!")
    print(f"Best validation validity: {best_validity*100:.2f}%")
    
    plot_training_curves(history, os.path.join(config.results_dir, 'final_training_curves.png'))

if __name__ == '__main__':
    main()
