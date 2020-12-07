from typing import Optional

from torch import nn
from text.representation.attention import MultiheadSelfAttention
from text.representation.mlp import ResidualMLP


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        attention: Optional[MultiheadSelfAttention] = None,
        residual_mlp: Optional[ResidualMLP] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = attention or MultiheadSelfAttention(
            embedding_dim, num_heads=12
        )
        self.residual_mlp = residual_mlp or ResidualMLP(
            embedding_dim, hidden_dims=[embedding_dim * 4]
        )

        self.attention_layer_norm = nn.LayerNorm(embedding_dim)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input, key_padding_mask):
        attention = self.attention(input, key_padding_mask)
        attention = self.dropout(attention)
        biased_input = input + attention
        biased_input = self.attention_layer_norm(biased_input)

        biased = self.residual_mlp(biased_input)
        return self.final_layer_norm(biased)