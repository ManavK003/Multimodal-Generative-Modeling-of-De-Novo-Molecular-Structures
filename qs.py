import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\nError: {description} failed!")
        return False
    return True

def main():
    print("LG3D-Mol Quick Start Script")
    print("="*60)
    
    print("\nThis script will:")
    print("1. Install dependencies")
    print("2. Run a quick training demo (10 epochs)")
    print("3. Generate sample molecules")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    if not run_command(
        f"{sys.executable} -m pip install --break-system-packages torch numpy pandas scikit-learn matplotlib seaborn tqdm transformers",
        "Installing basic dependencies"
    ):
        return
    
    print("\nNote: torch-geometric and RDKit installation may require special setup.")
   
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with open('configs/config.py', 'r') as f:
        config_content = f.read()
    
    config_content = config_content.replace('self.num_epochs = 100', 'self.num_epochs = 10')
    config_content = config_content.replace('self.eval_every = 5', 'self.eval_every = 2')
    
    with open('configs/config.py', 'w') as f:
        f.write(config_content)
    
    print("\nModified config for quick demo (10 epochs)")
    
    if not run_command(
        f"{sys.executable} train.py",
        "Running training demo"
    ):
        return
    
    if os.path.exists('checkpoints/best_model.pt'):
        if not run_command(
            f"{sys.executable} generate.py --num_samples 20",
            "Generating sample molecules"
        ):
            return
    
    print("\n" + "="*60)
    print("Quick start completed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check ./results/ for training curves")
    print("2. Check ./checkpoints/ for saved models")
    print("3. Modify configs/config.py for full training")
    print("4. Run: python train.py (for full training)")
    print("5. Run: python generate.py --visualize (to generate molecules)")

if __name__ == '__main__':
    main()
