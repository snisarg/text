from typing import Optional

from torch import nn


class RobertaEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_attention_heads: int,
        num_encoder_layers: int,
        output_dropout: float,
        model_path: Optional[str] = None,
    ):
        pass

