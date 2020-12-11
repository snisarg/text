from typing import Any, Optional

import torch.nn as nn

from text.decoder.mlp import ActivationFn, MLPDecoder, ReLU
from text.representation.roberta import RobertaEncoder


class RobertaModel(nn.Module):
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 50265,
        embedding_dim: int = 768,
        num_attention_heads: int = 12,
        num_encoder_layers: int = 12,
        output_dropout: float = 0.4,
        dense_dim: int = 0,
        out_dim: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        self.encoder = RobertaEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_attention_heads=num_attention_heads,
            num_encoder_layers=num_encoder_layers,
            output_dropout=output_dropout,
            model_path=model_path,
        )
        self.decoder = MLPDecoder(
            in_dim=embedding_dim + dense_dim,
            out_dim=out_dim,
            bias=bias,
            activation=ReLU,
        )