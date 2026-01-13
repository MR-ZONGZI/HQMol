import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout):
        super(CrossAttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, representation_1, representation_2):

        query = representation_1.unsqueeze(1) 
        key = representation_2.unsqueeze(1)  
        value = representation_2.unsqueeze(1) 
        
        attended_values, _ = self.cross_attention(query=query, key=key, value=value)
        
        attended_values = attended_values.squeeze(1)  # (seq_len, batch_size, feature_dim) -> (seq_len, feature_dim)
        
        return attended_values