import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class QueryProjector(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(QueryProjector, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 32, input_dim))
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim, 
            num_heads=4, 
            dropout=0.0, 
            batch_first=True
        )

        self.norm = nn.LayerNorm(output_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        orig_shape = x.shape
        
        if len(orig_shape) == 2:
            x = x.unsqueeze(0) 
            
        batch_size, seq_len, _ = x.shape
        
        query = self.query.expand(batch_size, -1, -1)
        
        attn_output, _ = self.cross_attention(
            query=query,
            key=x,
            value=x
        )

        if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
            print("attn_output contains NaN values.")
        
        residual = attn_output + query
        # residual = self.norm(residual)
        output = self.mlp(residual)
        output = self.norm(output)
        
        if len(orig_shape) == 2:
            output = output.squeeze(0)

        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Output contains NaN values.")
            return attn_output.squeeze(0)
            
        return output


def build_xmodal_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'query':
        return QueryProjector(config.mm_hidden_size, config.hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')