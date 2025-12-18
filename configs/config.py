import torch

class Config:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.data_path = './data/crossdocked_subset'
        self.use_subset = True
        self.subset_size = 10000
        
        self.protein_hidden_dim = 512
        self.text_hidden_dim = 512
        self.fusion_dim = 1024
        self.diffusion_hidden_dim = 1024
        
        self.num_egnn_layers = 5
        self.num_cross_attention_layers = 4
        self.num_attention_heads = 8
        self.num_diffusion_layers = 12
        
        self.num_atom_types = 10
        self.max_atoms = 25
        self.distance_cutoff = 10.0
        
        self.diffusion_timesteps = 1000
        self.beta_schedule = 'cosine'
        self.beta_start = 0.0001
        self.beta_end = 0.02
        
        self.batch_size = 16
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.gradient_clip = 1.0
        
        self.save_every = 10
        self.eval_every = 5
        
        self.checkpoint_dir = './checkpoints'
        self.results_dir = './results'
        
        self.use_wandb = False
        self.project_name = 'LG3D-Mol'
        
config = Config()
