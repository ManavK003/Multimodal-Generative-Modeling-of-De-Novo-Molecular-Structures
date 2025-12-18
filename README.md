# Multimodal-Generative-Modeling-of-De-Novo-Molecular-Structures

LG3D-Mol: Language-Guided 3D Molecular Diffusion

Multimodal Generative Modeling of De Novo Molecular Structures
CSE 676 - Deep Learning Final Project
University at Buffalo

Overview

This project implements a novel multimodal deep learning system that combines 3D protein geometry with natural language descriptions to generate novel drug-like molecules. The system uses E(3)-equivariant graph neural networks, transformer-based language models, and diffusion models to enable controllable molecular generation.

Key Features

- E(3)-Equivariant GNN for protein pocket encoding
- ChemBERTa-based text encoder for property specifications
- Cross-attention fusion for multimodal conditioning
- Equivariant diffusion model for 3D molecule generation
- Comprehensive evaluation metrics (validity, uniqueness, QED, SA score)




Dataset

The project uses the CrossDocked2020 dataset (22.5M protein-ligand pairs).

For this implementation, we recommend:

 Full Dataset: Download from http://bits.csb.pitt.edu/files/crossdock2020/
   Place the data in ./data/crossdocked_subset/

Configuration

Edit configs/config.py to modify hyperparameters:

- batch_size: Training batch size (default: 16)
- num_epochs: Number of training epochs (default: 100)
- learning_rate: Initial learning rate (default: 1e-4)
- diffusion_timesteps: Number of diffusion steps (default: 1000)
- max_atoms: Maximum atoms per molecule (default: 25)






Training features:
- Automatic checkpointing every 10 epochs
- Validation every 5 epochs
- Learning rate scheduling (cosine annealing)
- Gradient clipping for stability
- Training curve visualization

The training script will:
1. Load or generate the dataset
2. Initialize the model (~50M parameters)
3. Train for specified epochs
4. Save checkpoints to ./checkpoints/
5. Generate training curves in ./results/

Generation

To generate molecules using a trained model:

python generate.py --checkpoint checkpoints/best_model.pt --num_samples 100 --visualize

Arguments:
- --checkpoint: Path to trained model checkpoint
- --num_samples: Number of molecules to generate
- --visualize: Save molecular visualizations

Evaluation Metrics

The system evaluates generated molecules using:

1. Validity: Percentage of chemically valid molecules
2. Uniqueness: Percentage of unique molecules
3. Novelty: Percentage not in training set
4. QED Score: Drug-likeness (0-1, higher is better)
5. SA Score: Synthetic accessibility (1-10, lower is better)
6. Lipinski's Rule: Percentage passing all criteria

Model Architecture

1. Protein Encoder
   - 5-layer E(3)-Equivariant GNN
   - Hidden dimension: 512
   - Preserves rotation/translation symmetry

2. Text Encoder
   - ChemBERTa-zinc-base-v1 (pre-trained)
   - First 6 layers frozen for transfer learning
   - Output dimension: 512

3. Cross-Attention Fusion
   - 4 bidirectional attention layers
   - 8 attention heads
   - Fuses protein + text → 1024-dim conditioning

4. Diffusion Model
   - 12-layer equivariant transformer
   - 1000 timesteps with cosine schedule
   - Predicts both coordinates and atom types

Training Details

- Optimizer: AdamW (weight decay 1e-5)
- Learning Rate: 1e-4 with cosine annealing
- Gradient Clipping: 1.0
- Loss: MSE (coordinates) + Cross-Entropy (atom types)
- Training Time: ~7 hours on single GPU

Expected Results

After 100 epochs, the model should achieve:
- Validity: ~87-90%
- Uniqueness: ~89-92%
- QED Score: ~0.55-0.60
- SA Score: ~3.0-3.5
- Lipinski Pass: ~85-90%

Example Text Prompts

The model accepts natural language property descriptions:
- "Generate a non-toxic molecule with molecular weight less than 400"
- "Create a brain-penetrating lipophilic kinase inhibitor"
- "Design a highly soluble antibiotic with low toxicity"
- "Generate a membrane-permeable GPCR agonist"

GPU Requirements

- Minimum: 8GB VRAM (reduced batch size)
- Recommended: 16GB+ VRAM
- CPU: Training is possible but very slow

Troubleshooting

1. CUDA out of memory:
   - Reduce batch_size in configs/config.py
   - Reduce max_atoms to 20 or less
   - Use gradient accumulation

2. RDKit import errors:
   - Install via conda: conda install -c conda-forge rdkit
   - Or use pip: pip install rdkit-pypi

3. Torch-geometric errors:
   - Install from source following official instructions
   - Match CUDA version with PyTorch

4. Slow training:
   - Reduce num_diffusion_layers to 8
   - Reduce dataset size with config.subset_size
   - Use mixed precision training (fp16)


References

- Satorras et al. "E(3)-Equivariant Graph Neural Networks" (ICML 2021)
- Ho et al. "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Schneuing et al. "Structure-Based Drug Design with Diffusion" (NeurIPS 2022)
- Chithrananda et al. "ChemBERTa" (2020)





