import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_timesteps):
    if beta_schedule == 'linear':
        return torch.linspace(beta_start, beta_end, num_timesteps)
    elif beta_schedule == 'cosine':
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    else:
        raise ValueError(f'Unknown beta schedule: {beta_schedule}')

class EquivariantTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(EquivariantTransformerLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
    def forward(self, h, pos, conditioning):
        batch_size, num_atoms, _ = h.shape
        
        h_res = h
        h = self.norm1(h + conditioning.unsqueeze(1))
        
        Q = self.q_proj(h).view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(h).view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(h).view(batch_size, num_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_atoms, self.hidden_dim)
        attn_output = self.out_proj(attn_output)
        
        h = h_res + attn_output
        
        coord_delta = self.coord_mlp(h)
        pos = pos + coord_delta
        
        h_res = h
        h = self.norm2(h)
        h = h_res + self.ffn(h)
        
        return h, pos

class DiffusionModel(nn.Module):
    def __init__(self, hidden_dim, num_atom_types, num_layers=12, num_heads=8, num_timesteps=1000, beta_schedule='cosine'):
        super(DiffusionModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_atom_types = num_atom_types
        self.num_timesteps = num_timesteps
        
        betas = get_beta_schedule(beta_schedule, 0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        self.time_embedding = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.atom_type_embedding = nn.Embedding(num_atom_types, hidden_dim)
        
        self.coord_embedding = nn.Linear(3, hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            EquivariantTransformerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.coord_output = nn.Linear(hidden_dim, 3)
        self.atom_type_output = nn.Linear(hidden_dim, num_atom_types)
        
    def get_timestep_embedding(self, timesteps, embedding_dim=128):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, pos_noisy, atom_types, timesteps, conditioning):
        t_emb = self.get_timestep_embedding(timesteps)
        t_emb = self.time_embedding(t_emb)
        
        h_atom = self.atom_type_embedding(atom_types)
        h_coord = self.coord_embedding(pos_noisy)
        h = h_atom + h_coord
        
        pos = pos_noisy.clone()
        
        for layer in self.transformer_layers:
            h, pos = layer(h, pos, t_emb + conditioning)
        
        coord_noise_pred = self.coord_output(h)
        atom_type_logits = self.atom_type_output(h)
        
        return coord_noise_pred, atom_type_logits
    
    def add_noise(self, pos, timesteps):
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        noise = torch.randn_like(pos)
        
        while len(sqrt_alpha_cumprod.shape) < len(pos.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        
        pos_noisy = sqrt_alpha_cumprod * pos + sqrt_one_minus_alpha_cumprod * noise
        
        return pos_noisy, noise
    
    @torch.no_grad()
    def sample(self, num_samples, num_atoms, conditioning, device):
        pos = torch.randn(num_samples, num_atoms, 3, device=device)
        atom_types = torch.randint(0, self.num_atom_types, (num_samples, num_atoms), device=device)
        
        for t in reversed(range(self.num_timesteps)):
            timesteps = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            coord_noise_pred, atom_type_logits = self.forward(pos, atom_types, timesteps, conditioning)
            
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            if t > 0:
                noise = torch.randn_like(pos)
            else:
                noise = torch.zeros_like(pos)
            
            pos = (pos - beta / torch.sqrt(1.0 - alpha_cumprod) * coord_noise_pred) / torch.sqrt(alpha) + torch.sqrt(beta) * noise
            
            atom_types = torch.argmax(atom_type_logits, dim=-1)
        
        return pos, atom_types
