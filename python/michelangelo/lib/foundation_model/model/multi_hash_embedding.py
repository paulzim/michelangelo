"""Multi-hash embedding for high-cardinality categorical features.

Based on: https://proceedings.neurips.cc/paper/2017/file/f0f6ba4b5e0000340312d33c212c3ae8-Paper.pdf

Uses multiple hash functions to reduce collision probability without
requiring an exact vocabulary, making it suitable for open-vocabulary
or high-cardinality features where a full embedding table is impractical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHashEmbedding(nn.Module):
    """Embedding layer that averages over multiple hash lookups.

    For each input token, computes ``num_hashes`` bucket indices via bit-shift
    hashing, looks up each in a shared embedding table, and sums the results
    (scaled by hash index to break symmetry). This reduces collision probability
    relative to a single hash at the same table size.

    Args:
        num_embeddings: Hash table size (number of rows in the embedding table).
            Pad one extra slot — index 0 is reserved for padding.
        embedding_dim: Dimension of each embedding vector.
        num_hashes: Number of independent hash lookups to sum. Default 2.
            Rule of thumb: 2-4 for most features; increase for very
            high-cardinality features (>10M unique values).
        normalize_output: If True, L2-normalize the summed embedding.
        sparse: If True, use sparse gradients for the embedding table.
            More memory-efficient for large tables with sparse updates.

    Input:
        x: LongTensor of shape ``(*)`` — any shape, values are hashed mod
            ``num_embeddings``.

    Output:
        FloatTensor of shape ``(*, embedding_dim)``.

    Example::

        >>> emb = MultiHashEmbedding(num_embeddings=100001, embedding_dim=32)
        >>> emb(torch.tensor([[1, 2, 3], [4, 5, 6]])).shape
        torch.Size([2, 3, 32])
    """

    def __init__(
        self,
        num_embeddings: int = 2**20,
        embedding_dim: int = 32,
        num_hashes: int = 2,
        normalize_output: bool = False,
        sparse: bool = False,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_hashes = num_hashes
        self.normalize_output = normalize_output

        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0,
            sparse=sparse,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()

        original_shape = x.shape
        if len(original_shape) == 1:
            x = x.unsqueeze(1)
            should_squeeze = True
        else:
            should_squeeze = False

        # new_zeros inherits device from input at runtime (avoids baking cpu device during tracing)
        result = x.new_zeros(x.shape[0], x.shape[1], self.embedding_dim).to(
            dtype=self.embedding_layer.weight.dtype
        )
        for i in range(self.num_hashes):
            idx = torch.remainder((x << i), self.num_embeddings)
            # Scale by (i+1) to break symmetry and reduce hash collisions from circular remainder
            result = result + self.embedding_layer(idx) * (i + 1)

        if self.normalize_output:
            result = F.normalize(result, p=2.0, dim=-1)

        if should_squeeze:
            result = result.squeeze(1)

        return result
