"""Encoder modules for multi-modal sequence models.

Provides embedding encoders for categorical, numerical, geo, and hash-categorical
features, plus positional encodings and a MultiModalEncoder that combines them all.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from michelangelo.lib.foundation_model.model.multi_hash_embedding import MultiHashEmbedding


class CategoricalEmbedding(nn.Module):
    """Standard embedding for categorical features with optional padding support.

    Args:
        vocab_size: Number of distinct categories (embedding table rows).
        embedding_dim: Size of each embedding vector.
        padding_idx: Index whose embedding is fixed to zeros. Pass ``None`` to disable.

    Input:
        x: LongTensor of arbitrary shape ``(*)``.
    Output:
        FloatTensor of shape ``(*, embedding_dim)``.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: Optional[int] = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x.long())


class NumericalMLPEmbedding(nn.Module):
    """MLP encoder that projects the last dimension of a float tensor into an embedding space.

    Architecture: Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm

    Args:
        hidden_dim: Width of the hidden layer.
        output_dim: Embedding dimension produced by the encoder.
        num_features: Size of the last input dimension. When ``None``,
            ``nn.LazyLinear`` is used so the input dimension is inferred
            on the first forward pass.
        dropout: Dropout probability between layers.

    Input:
        x: FloatTensor of shape ``(*, num_features)``.
    Output:
        FloatTensor of shape ``(*, output_dim)``.
    """

    def __init__(self, hidden_dim: int, output_dim: int, num_features: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        first_layer = nn.LazyLinear(hidden_dim) if num_features is None else nn.Linear(num_features, hidden_dim)
        self.mlp = nn.Sequential(
            first_layer,
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GeoMLPEmbedding(nn.Module):
    """3-layer MLP encoder for geospatial features (lat, lng, etc.).

    Architecture: Linear → LN → ReLU → Drop → Linear → ReLU → Drop → Linear → LN

    Args:
        hidden_dim: Width of the two hidden layers.
        output_dim: Embedding dimension produced by the encoder.
        num_geo_features: Size of the last input dimension. When ``None``,
            ``nn.LazyLinear`` is used.
        dropout: Dropout probability between layers.

    Input:
        x: FloatTensor of shape ``(*, num_geo_features)``.
    Output:
        FloatTensor of shape ``(*, output_dim)``.
    """

    def __init__(self, hidden_dim: int, output_dim: int, num_geo_features: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        first_layer = nn.LazyLinear(hidden_dim) if num_geo_features is None else nn.Linear(num_geo_features, hidden_dim)
        self.mlp = nn.Sequential(
            first_layer,
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length supported.
        dropout: Dropout probability applied after adding the encoding.

    Input:
        x: FloatTensor of shape ``(B, S, d_model)``.
    Output:
        FloatTensor of shape ``(B, S, d_model)``.
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding (BERT-style).

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length (number of learned positions).
        dropout: Dropout probability applied after adding the encoding.

    Input:
        x: FloatTensor of shape ``(B, S, d_model)`` where ``S <= max_len``.
    Output:
        FloatTensor of shape ``(B, S, d_model)``.
    """

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


class MultiModalEncoder(nn.Module):
    """Encodes multiple feature types into a unified d_model-dimensional representation.

    Accepts a flat batch dict where keys are feature names. Each feature is
    embedded by its corresponding encoder, all embeddings are concatenated,
    projected to ``d_model``, and positional encoding is applied.

    Args:
        embedding_config: Configuration dict with keys:

            - ``"hash_categoricals"``: list of ``[name, bucket_size, embed_dim]``
              or ``[name, bucket_size, embed_dim, num_hashes]``.
              Uses ``MultiHashEmbedding`` (multi-hash trick).
            - ``"categoricals"``: list of ``[name, vocab_size, embed_dim]``.
              Uses ``CategoricalEmbedding`` (standard ``nn.Embedding``).
            - ``"numerical"``: list of ``[name, hidden_dim, output_dim]``
              or ``[name, hidden_dim, output_dim, num_features]``.
              Each entry creates a ``NumericalMLPEmbedding``.
            - ``"geo"``: list of ``[name, hidden_dim, output_dim]``
              or ``[name, hidden_dim, output_dim, num_features]``.
              Each entry creates a ``GeoMLPEmbedding``.

        d_model: Output embedding dimension.
        max_len: Maximum sequence length for positional encoding.
        dropout: Dropout probability.
        pos_encoding: ``"sinusoidal"`` or ``"learned"``.

    Input:
        batch: ``dict[str, Tensor]`` mapping feature names to tensors.

            - Hash/standard categoricals: LongTensor ``(B, S)``.
            - Numerical: FloatTensor ``(B, S, num_features)`` (pre-stacked).
            - Geo: FloatTensor ``(B, S, num_geo_features)`` (pre-stacked).

    Output:
        FloatTensor of shape ``(B, S, d_model)``.
    """

    def __init__(
        self,
        embedding_config: dict,
        d_model: int,
        max_len: int = 100,
        dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",
    ):
        super().__init__()
        self.embedding_config = embedding_config
        self.d_model = d_model

        total_embed_dim = 0

        # Hash categorical embeddings
        self.hash_cat_names = []
        self.hash_cat_encoders = nn.ModuleDict()
        for entry in embedding_config.get("hash_categoricals", []):
            name, bucket_size, embed_dim = entry[0], entry[1], entry[2]
            num_hashes = entry[3] if len(entry) > 3 else 2
            self.hash_cat_names.append(name)
            self.hash_cat_encoders[name] = MultiHashEmbedding(
                num_embeddings=bucket_size + 1,
                embedding_dim=embed_dim,
                num_hashes=num_hashes,
            )
            total_embed_dim += embed_dim

        # Standard categorical embeddings
        self.cat_names = []
        self.cat_encoders = nn.ModuleDict()
        for name, vocab_size, embed_dim in embedding_config.get("categoricals", []):
            self.cat_names.append(name)
            self.cat_encoders[name] = CategoricalEmbedding(vocab_size, embed_dim)
            total_embed_dim += embed_dim

        # Numerical MLP embeddings
        self.numerical_encoders = nn.ModuleDict()
        for entry in embedding_config.get("numerical", []):
            name, hidden_dim_n, output_dim_n = entry[0], entry[1], entry[2]
            num_features = entry[3] if len(entry) > 3 else None
            self.numerical_encoders[name] = NumericalMLPEmbedding(
                hidden_dim=hidden_dim_n,
                output_dim=output_dim_n,
                num_features=num_features,
                dropout=dropout,
            )
            total_embed_dim += output_dim_n

        # Geo MLP embeddings
        self.geo_encoders = nn.ModuleDict()
        for entry in embedding_config.get("geo", []):
            name, hidden_dim_g, output_dim_g = entry[0], entry[1], entry[2]
            num_geo_features = entry[3] if len(entry) > 3 else None
            self.geo_encoders[name] = GeoMLPEmbedding(
                hidden_dim=hidden_dim_g,
                output_dim=output_dim_g,
                num_geo_features=num_geo_features,
                dropout=dropout,
            )
            total_embed_dim += output_dim_g

        self.projection = nn.Linear(total_embed_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if pos_encoding == "learned":
            self.pos_encoder = LearnedPositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

    def _collect_dense_embeddings(self, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Embed all dense features and return as a list of tensors."""
        embeddings: list[torch.Tensor] = []

        for name, encoder in self.hash_cat_encoders.items():
            embeddings.append(encoder(batch[name]))

        for name, encoder in self.cat_encoders.items():
            embeddings.append(encoder(batch[name]))

        for key, encoder in self.numerical_encoders.items():
            embeddings.append(encoder(batch[key]))

        for key, encoder in self.geo_encoders.items():
            embeddings.append(encoder(batch[key]))

        return embeddings

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = self._collect_dense_embeddings(batch)

        combined = torch.cat(embeddings, dim=-1)

        x = self.projection(combined)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.pos_encoder(x)
        return x
