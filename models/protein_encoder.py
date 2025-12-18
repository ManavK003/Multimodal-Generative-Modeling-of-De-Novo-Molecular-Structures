import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class EGNNLayer(MessagePassing):
    def __init__(self, hidden_dim, edge_feat_dim=0):
        super(EGNNLayer, self).__init__(aggr='add')
        
        self.hidden_dim = hidden_dim
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
    def forward(self, h, pos, edge_index, edge_attr=None):
        edge_index, _ = add_self_loops(edge_index, num_nodes=h.size(0))
        
        if edge_attr is not None:
            self_loop_attr = torch.zeros(h.size(0), edge_attr.size(1), device=edge_attr.device)
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        h_updated, pos_updated = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)
        
        return h_updated, pos_updated
    
    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        rel_pos = pos_i - pos_j
        dist = torch.sqrt(torch.sum(rel_pos ** 2, dim=-1, keepdim=True) + 1e-8)
        
        if edge_attr is not None:
            edge_input = torch.cat([h_i, h_j, dist, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([h_i, h_j, dist], dim=-1)
        
        edge_feat = self.edge_mlp(edge_input)
        
        coord_weights = self.coord_mlp(edge_feat)
        coord_diff = coord_weights * rel_pos
        
        return edge_feat, coord_diff
    
    def aggregate(self, inputs, index, ptr, dim_size):
        edge_feat, coord_diff = inputs
        
        h_aggr = torch.zeros(dim_size, self.hidden_dim, device=edge_feat.device)
        h_aggr = h_aggr.index_add_(0, index, edge_feat)
        
        pos_aggr = torch.zeros(dim_size, 3, device=coord_diff.device)
        pos_aggr = pos_aggr.index_add_(0, index, coord_diff)
        
        return h_aggr, pos_aggr
    
    def update(self, aggr_out, h, pos):
        h_aggr, pos_diff = aggr_out
        
        h_updated = self.node_mlp(torch.cat([h, h_aggr], dim=-1))
        h_updated = h + h_updated
        
        pos_updated = pos + pos_diff
        
        return h_updated, pos_updated

class ProteinEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=5):
        super(ProteinEncoder, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, node_features, pos, edge_index, batch):
        h = self.input_projection(node_features)
        
        for egnn_layer, layer_norm in zip(self.egnn_layers, self.layer_norms):
            h_res = h
            h, pos = egnn_layer(h, pos, edge_index)
            h = layer_norm(h + h_res)
        
        h_global = self.global_mean_pool(h, batch)
        h_global = self.output_projection(h_global)
        
        return h_global
    
    def global_mean_pool(self, x, batch):
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        
        for i in range(batch_size):
            mask = batch == i
            out[i] = x[mask].mean(dim=0)
        
        return out
