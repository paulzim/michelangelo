"""Packed sequence transformer with RoPE and multi-backend attention.

Purpose-built for packed sequences where multiple documents are concatenated
into a single long sequence. Supports three attention backends:

  1. FlashAttention varlen — O(1) memory masking via cu_seqlens
  2. FlexAttention — block-sparse masks via BlockMask (O(num_blocks))
  3. SDPA dense fallback — explicit (S,S) mask for local dev

Uses rotary position embeddings (RoPE) that reset at document boundaries.
See DecoderOnlyTransformer for the standard (non-packed) backbone.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask  # noqa: F401

    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False

from michelangelo.lib.foundation_model.model.transformers.rope import (
    HAS_FLASH_ATTN,
    RoPEMultiHeadAttention,
)

logger = logging.getLogger(__name__)


def _build_doc_causal_block_mask(doc_offsets, S, mask_mod, device, BLOCK_SIZE=128):
    """Build a BlockMask directly from doc_offsets — O(num_blocks), not O(S²)."""
    n_blocks = (S + BLOCK_SIZE - 1) // BLOCK_SIZE

    block_starts = torch.arange(n_blocks, device=device) * BLOCK_SIZE
    block_to_doc = torch.searchsorted(doc_offsets, block_starts, right=True) - 1

    doc_first_block = doc_offsets[:-1] // BLOCK_SIZE

    q_doc_start = doc_first_block[block_to_doc]
    num_blocks_per_q = torch.arange(n_blocks, device=device) - q_doc_start + 1

    kv_offsets = torch.arange(n_blocks, device=device)
    indices = q_doc_start.unsqueeze(1) + kv_offsets.unsqueeze(0)
    indices = indices.clamp(max=n_blocks - 1)

    return BlockMask.from_kv_blocks(
        kv_num_blocks=num_blocks_per_q[None, None, :].to(torch.int32),
        kv_indices=indices[None, None, :, :].to(torch.int32),
        BLOCK_SIZE=BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(S, S),
    )


class PackedTransformer(nn.Module):
    """Transformer backbone for packed sequences with RoPE attention.

    Supports causal (autoregressive) and bidirectional attention with
    document-boundary masking. Automatically selects the best available
    attention backend (FlashAttention > FlexAttention > SDPA dense).

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout rate.
        causal: If True, use causal (autoregressive) attention.
        use_flash: If True, prefer FlashAttention when available.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        causal: bool = True,
        use_flash: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        self.use_flash = use_flash
        self._logged_backend = False

        self.layers = nn.ModuleList([
            _RoPETransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def _generate_causal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1)

    def forward(
        self,
        src: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        doc_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through packed transformer.

        Args:
            src: Input tensor of shape (B, S, d_model). B=1 for packed sequences.
            position_ids: (S,) or (B, S) per-token position indices for RoPE.
                Resets at document boundaries for correct rotary positions.
            doc_offsets: (num_docs+1,) cumulative boundary offsets for
                building attention masks on-the-fly.

        Returns:
            Transformer output of shape (B, S, d_model).
        """
        dtype = src.dtype
        S = src.size(1)
        device = src.device

        flex_block_mask = None
        attn_mask = None

        if doc_offsets is not None and doc_offsets.dim() > 1:
            doc_offsets = doc_offsets.squeeze(0)

        if doc_offsets is not None and HAS_FLASH_ATTN and self.use_flash:
            if not self._logged_backend:
                logger.info(f"Attention backend: flash_attn_varlen (S={S}, causal={self.causal})")
                self._logged_backend = True

        elif doc_offsets is not None and HAS_FLEX_ATTENTION:
            if not self._logged_backend:
                logger.info(f"Attention backend: flex_attention (S={S}, causal={self.causal})")
                self._logged_backend = True

            document_id = torch.repeat_interleave(
                torch.arange(len(doc_offsets) - 1, device=device),
                doc_offsets[1:] - doc_offsets[:-1],
            )

            if self.causal:
                def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001
                    return (document_id[q_idx] == document_id[kv_idx]) & (kv_idx <= q_idx)
            else:
                def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001
                    return document_id[q_idx] == document_id[kv_idx]

            flex_block_mask = _build_doc_causal_block_mask(doc_offsets, S, mask_mod, device)
            doc_offsets = None

        elif doc_offsets is not None:
            if not self._logged_backend:
                logger.info(f"Attention backend: sdpa_dense (S={S}, causal={self.causal})")
                self._logged_backend = True

            document_id = torch.repeat_interleave(
                torch.arange(len(doc_offsets) - 1, device=device),
                doc_offsets[1:] - doc_offsets[:-1],
            )
            block_mask = document_id.unsqueeze(0) == document_id.unsqueeze(1)
            if self.causal:
                causal_mask = torch.ones(S, S, device=device, dtype=torch.bool).tril()
                block_mask = block_mask & causal_mask
            attn_mask = torch.where(block_mask, 0.0, float("-inf")).to(dtype)
            doc_offsets = None

        else:
            if not self._logged_backend:
                logger.info(f"Attention backend: sdpa (S={S}, causal={self.causal})")
                self._logged_backend = True
            if self.causal:
                attn_mask = self._generate_causal_mask(S, device, dtype)

        x = src
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                position_ids=position_ids,
                flex_block_mask=flex_block_mask,
                doc_offsets=doc_offsets,
                causal=self.causal,
            )
        return self.final_norm(x)

    def forward_with_mask(
        self,
        src: torch.Tensor,
        flex_block_mask: object,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled-friendly forward: BlockMask already built externally."""
        x = src
        for layer in self.layers:
            x = layer(x, flex_block_mask=flex_block_mask, position_ids=position_ids, causal=self.causal)
        return self.final_norm(x)


class _RoPETransformerLayer(nn.Module):
    """Pre-norm transformer layer with RoPE attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        flex_block_mask: object | None = None,
        doc_offsets: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            position_ids=position_ids,
            flex_block_mask=flex_block_mask,
            doc_offsets=doc_offsets,
            causal=causal,
        )
        x = x + self.ff(self.norm2(x))
        return x
