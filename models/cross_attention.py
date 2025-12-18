import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(CrossAttentionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        query_res = query
        query = self.norm1(query)
        key = self.norm1(key)
        value = self.norm1(value)
        
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        attn_output = self.out_proj(attn_output)
        
        query = query_res + attn_output
        
        ffn_res = query
        query = self.norm2(query)
        query = ffn_res + self.ffn(query)
        
        return query

class CrossAttentionFusion(nn.Module):
    def __init__(self, protein_dim, text_dim, fusion_dim, num_layers=4, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        
        self.protein_proj = nn.Linear(protein_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        
        self.protein_to_text_layers = nn.ModuleList([
            CrossAttentionLayer(fusion_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.text_to_protein_layers = nn.ModuleList([
            CrossAttentionLayer(fusion_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(2 * fusion_dim, fusion_dim)
        
    def forward(self, protein_embedding, text_embedding):
        protein_feat = self.protein_proj(protein_embedding).unsqueeze(1)
        text_feat = self.text_proj(text_embedding).unsqueeze(1)
        
        for p2t_layer, t2p_layer in zip(self.protein_to_text_layers, self.text_to_protein_layers):
            protein_feat = p2t_layer(protein_feat, text_feat, text_feat)
            text_feat = t2p_layer(text_feat, protein_feat, protein_feat)
        
        fused_feat = torch.cat([protein_feat.squeeze(1), text_feat.squeeze(1)], dim=-1)
        fused_feat = self.output_projection(fused_feat)
        
        return fused_feat
