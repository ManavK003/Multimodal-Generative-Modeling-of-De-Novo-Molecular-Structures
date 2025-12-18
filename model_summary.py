import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import config
from models.lg3d_mol import LG3DMol

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_model_summary(model):
    print_section("LG3D-Mol Architecture Summary")
    
    print("\n1. Protein Encoder (E(3)-Equivariant GNN)")
    print("   " + "-"*60)
    protein_encoder_params = sum(p.numel() for p in model.protein_encoder.parameters())
    print(f"   - Input: Protein 3D structure (atoms, coordinates)")
    print(f"   - Layers: {config.num_egnn_layers} EGNN layers")
    print(f"   - Hidden dim: {config.protein_hidden_dim}")
    print(f"   - Output: Global protein embedding ({config.protein_hidden_dim}-dim)")
    print(f"   - Parameters: {protein_encoder_params:,}")
    
    print("\n2. Text Encoder (ChemBERTa)")
    print("   " + "-"*60)
    text_encoder_params = sum(p.numel() for p in model.text_encoder.parameters())
    trainable_text_params = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
    print(f"   - Input: Natural language property descriptions")
    print(f"   - Model: ChemBERTa-zinc-base-v1 (pre-trained)")
    print(f"   - First 6 layers frozen for transfer learning")
    print(f"   - Output: Text property embedding ({config.text_hidden_dim}-dim)")
    print(f"   - Parameters: {text_encoder_params:,} ({trainable_text_params:,} trainable)")
    
    print("\n3. Cross-Attention Fusion")
    print("   " + "-"*60)
    fusion_params = sum(p.numel() for p in model.cross_attention_fusion.parameters())
    print(f"   - Layers: {config.num_cross_attention_layers} bidirectional attention layers")
    print(f"   - Attention heads: {config.num_attention_heads}")
    print(f"   - Input: Protein embedding + Text embedding")
    print(f"   - Output: Fused conditioning vector ({config.fusion_dim}-dim)")
    print(f"   - Parameters: {fusion_params:,}")
    
    print("\n4. Diffusion Model (Equivariant Transformer)")
    print("   " + "-"*60)
    diffusion_params = sum(p.numel() for p in model.diffusion_model.parameters())
    print(f"   - Layers: {config.num_diffusion_layers} equivariant transformer layers")
    print(f"   - Hidden dim: {config.diffusion_hidden_dim}")
    print(f"   - Timesteps: {config.diffusion_timesteps}")
    print(f"   - Schedule: {config.beta_schedule}")
    print(f"   - Atom types: {config.num_atom_types}")
    print(f"   - Max atoms: {config.max_atoms}")
    print(f"   - Parameters: {diffusion_params:,}")
    
    print_section("Parameter Summary")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nTotal parameters:      {total_params:>12,}")
    print(f"Trainable parameters:  {trainable_params:>12,}")
    print(f"Frozen parameters:     {frozen_params:>12,}")
    
    memory_mb = total_params * 4 / (1024 * 1024)
    print(f"\nEstimated memory:      {memory_mb:>12.2f} MB")
    
    print_section("Training Configuration")
    
    print(f"\nBatch size:            {config.batch_size}")
    print(f"Number of epochs:      {config.num_epochs}")
    print(f"Learning rate:         {config.learning_rate}")
    print(f"Weight decay:          {config.weight_decay}")
    print(f"Gradient clipping:     {config.gradient_clip}")
    print(f"Device:                {config.device}")
    
    print_section("Data Configuration")
    
    print(f"\nDataset:               CrossDocked2020")
    print(f"Use subset:            {config.use_subset}")
    if config.use_subset:
        print(f"Subset size:           {config.subset_size}")
    print(f"Distance cutoff:       {config.distance_cutoff} Å")
    
    print("\n" + "="*70 + "\n")

def main():
    print("\nInitializing model...")
    model = LG3DMol(config)
    
    print_model_summary(model)
    
    print("Model components:")
    for name, module in model.named_children():
        print(f"  - {name}")
    
    print("\nTo see detailed layer information, use:")
    print("  print(model)")

if __name__ == '__main__':
    main()
