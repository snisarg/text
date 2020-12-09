from typing import Optional, List

from torch import nn

from text.decoder.mlp import GeLU
from text.representation.attention import MultiheadSelfAttention


class ResidualMLP(nn.Module):
    """A square MLP component which can learn a bias on an input vector.
    This MLP in particular defaults to using GeLU as its activation function
    (this can be changed by passing a different activation function),
    and retains a residual connection to its original input to help with gradient
    propogation.

    Unlike pytext's MLPDecoder it doesn't currently allow adding a LayerNorm
    in between hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.1,
        activation=GeLU,
    ):
        super().__init__()
        modules = []
        for last_dim, dim in zip([input_dim] + hidden_dims, hidden_dims):
            modules.extend(
                [nn.Linear(last_dim, dim), activation(), nn.Dropout(dropout)]
            )

        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        # Unlike normal PyText mlp, we don't put an activation layer at the end.
        modules.extend([nn.Linear(last_dim, input_dim), nn.Dropout(dropout)])

        self.mlp = nn.Sequential(*modules)

    def forward(self, input):
        bias = self.mlp(input)
        return input + bias


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