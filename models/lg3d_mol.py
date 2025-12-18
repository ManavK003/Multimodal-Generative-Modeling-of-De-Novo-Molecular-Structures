import torch
import torch.nn as nn
from .protein_encoder import ProteinEncoder
from .text_encoder import TextEncoder
from .cross_attention import CrossAttentionFusion
from .diffusion_model import DiffusionModel

class LG3DMol(nn.Module):
    def __init__(self, config):
        super(LG3DMol, self).__init__()
        
        self.config = config
        
        self.protein_encoder = ProteinEncoder(
            input_dim=20,
            hidden_dim=config.protein_hidden_dim,
            num_layers=config.num_egnn_layers
        )
        
        self.text_encoder = TextEncoder(
            hidden_dim=config.text_hidden_dim,
            freeze_layers=6
        )
        
        self.cross_attention_fusion = CrossAttentionFusion(
            protein_dim=config.protein_hidden_dim,
            text_dim=config.text_hidden_dim,
            fusion_dim=config.fusion_dim,
            num_layers=config.num_cross_attention_layers,
            num_heads=config.num_attention_heads
        )
        
        self.diffusion_model = DiffusionModel(
            hidden_dim=config.diffusion_hidden_dim,
            num_atom_types=config.num_atom_types,
            num_layers=config.num_diffusion_layers,
            num_heads=config.num_attention_heads,
            num_timesteps=config.diffusion_timesteps,
            beta_schedule=config.beta_schedule
        )
        
    def forward(self, protein_data, text_prompts, molecule_pos, molecule_atom_types, timesteps):
        protein_embedding = self.protein_encoder(
            protein_data['node_features'],
            protein_data['pos'],
            protein_data['edge_index'],
            protein_data['batch']
        )
        
        text_embedding = self.text_encoder(text_prompts)
        
        conditioning = self.cross_attention_fusion(protein_embedding, text_embedding)
        
        pos_noisy, noise = self.diffusion_model.add_noise(molecule_pos, timesteps)
        
        coord_noise_pred, atom_type_logits = self.diffusion_model(
            pos_noisy,
            molecule_atom_types,
            timesteps,
            conditioning
        )
        
        return coord_noise_pred, atom_type_logits, noise
    
    @torch.no_grad()
    def generate(self, protein_data, text_prompts, num_samples=1, num_atoms=25):
        self.eval()
        
        protein_embedding = self.protein_encoder(
            protein_data['node_features'],
            protein_data['pos'],
            protein_data['edge_index'],
            protein_data['batch']
        )
        
        text_embedding = self.text_encoder(text_prompts)
        
        conditioning = self.cross_attention_fusion(protein_embedding, text_embedding)
        
        conditioning = conditioning.repeat(num_samples, 1)
        
        pos, atom_types = self.diffusion_model.sample(
            num_samples=num_samples,
            num_atoms=num_atoms,
            conditioning=conditioning,
            device=protein_data['pos'].device
        )
        
        return pos, atom_types
