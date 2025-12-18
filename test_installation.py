import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("Testing imports...")
    
    try:
        import torch
        print(f"  PyTorch: {torch.__version__} ✓")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  PyTorch: ✗ ({e})")
        return False
    
    try:
        import numpy
        print(f"  NumPy: {numpy.__version__} ✓")
    except ImportError as e:
        print(f"  NumPy: ✗ ({e})")
        return False
    
    try:
        from transformers import AutoModel
        print(f"  Transformers: ✓")
    except ImportError as e:
        print(f"  Transformers: ✗ ({e})")
        print("  Install with: pip install --break-system-packages transformers")
        return False
    
    try:
        import torch_geometric
        print(f"  PyTorch Geometric: ✓")
    except ImportError as e:
        print(f"  PyTorch Geometric: ✗ ({e})")
        print("  Install from: https://pytorch-geometric.readthedocs.io/")
    
    try:
        from rdkit import Chem
        print(f"  RDKit: ✓")
    except ImportError as e:
        print(f"  RDKit: ✗ ({e})")
        print("  Install with: conda install -c conda-forge rdkit")
    
    return True

def test_model_creation():
    print("\nTesting model creation...")
    
    try:
        from configs.config import config
        from models.lg3d_mol import LG3DMol
        
        model = LG3DMol(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model created successfully ✓")
        print(f"  Total parameters: {total_params:,}")
        return True
    except Exception as e:
        print(f"  Model creation failed: {e}")
        return False

def test_data_loading():
    print("\nTesting data loading...")
    
    try:
        from configs.config import config
        from data.dataset import get_dataloader
        
        train_loader = get_dataloader(config, split='train')
        print(f"  Data loader created successfully ✓")
        print(f"  Number of batches: {len(train_loader)}")
        
        batch = next(iter(train_loader))
        print(f"  Batch keys: {batch.keys()}")
        print(f"  Batch size: {batch['ligand_pos'].size(0)}")
        return True
    except Exception as e:
        print(f"  Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    print("\nTesting forward pass...")
    
    try:
        import torch
        from configs.config import config
        from models.lg3d_mol import LG3DMol
        from data.dataset import get_dataloader
        
        model = LG3DMol(config).to(config.device)
        model.eval()
        
        train_loader = get_dataloader(config, split='train')
        batch = next(iter(train_loader))
        
        protein_data = {k: v.to(config.device) for k, v in batch['protein_data'].items()}
        ligand_pos = batch['ligand_pos'].to(config.device)
        ligand_atom_types = batch['ligand_atom_types'].to(config.device)
        text_prompts = batch['text_prompts']
        
        batch_size = ligand_pos.size(0)
        timesteps = torch.randint(0, config.diffusion_timesteps, (batch_size,), device=config.device)
        
        with torch.no_grad():
            coord_noise_pred, atom_type_logits, noise = model(
                protein_data,
                text_prompts,
                ligand_pos,
                ligand_atom_types,
                timesteps
            )
        
        print(f"  Forward pass successful ✓")
        print(f"  Coordinate prediction shape: {coord_noise_pred.shape}")
        print(f"  Atom type logits shape: {atom_type_logits.shape}")
        return True
    except Exception as e:
        print(f"  Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("LG3D-Mol Installation Test")
    print("="*60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
        print("\n⚠ Warning: Some imports failed. Basic functionality may be limited.")
    
    if not test_model_creation():
        all_passed = False
    
    if not test_data_loading():
        all_passed = False
    
    if not test_forward_pass():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("All tests passed! ✓")
        print("You can now run training with: python train.py")
    else:
        print("Some tests failed. Please check the errors above.")
        
    print("="*60)

if __name__ == '__main__':
    main()
