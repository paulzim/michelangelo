"""Rotary Position Embedding (RoPE) implementation.

RoPE encodes position information by rotating query and key vectors,
enabling relative position awareness without explicit position embeddings.

Key benefits:
- Relative positions: no train/serve skew with sequence packing
- Long context: no max_len limit, positions computed on-the-fly
- Proven at scale: used by LLaMA, Mistral, etc.

Reference: https://arxiv.org/abs/2104.09864
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention

    _flex_attention = torch.compile(_flex_attention)
    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Computes rotation matrices for position encoding and applies them
    to query and key tensors in attention.

    Args:
        dim: Dimension of the embedding (must be even, typically head_dim).
        max_seq_len: Maximum sequence length for precomputed frequencies.
            Cache extends automatically if sequences exceed this length.
        base: Base for frequency computation (standard RoPE default: 10000).
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to query and key.

        Args:
            q: Query tensor of shape (B, num_heads, S, head_dim).
            k: Key tensor of shape (B, num_heads, S, head_dim).
            seq_len: Sequence length.
            position_ids: Optional (S,) or (B, S) per-token position indices.
                When provided positions are looked up individually (e.g. for
                packed sequences with per-document reset). When None, uses
                contiguous 0..seq_len-1.

        Returns:
            Tuple of (rotated_q, rotated_k) with the same shapes.
        """
        if position_ids is not None:
            max_pos = int(position_ids.max().item()) + 1
            if max_pos > self.cos_cached.shape[0]:
                self._build_cache(max_pos)
            if position_ids.dim() == 2 and position_ids.size(0) != 1:
                raise ValueError(
                    f"RoPE with batched position_ids requires B=1, got B={position_ids.size(0)}. "
                    "Packed sequences must concatenate all documents into a single batch element."
                )
            ids = position_ids.reshape(-1) if position_ids.dim() == 2 else position_ids
            cos = self.cos_cached[ids].to(q.dtype)
            sin = self.sin_cached[ids].to(q.dtype)
        else:
            if seq_len > self.cos_cached.shape[0]:
                self._build_cache(seq_len)
            cos = self.cos_cached[:seq_len].to(q.dtype)
            sin = self.sin_cached[:seq_len].to(q.dtype)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding.

    Drop-in replacement for nn.MultiheadAttention with RoPE support.
    Supports FlashAttention, FlexAttention, and SDPA dense backends.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for RoPE cache.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[object] = None,
        doc_offsets: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k, S, position_ids=position_ids)

        if doc_offsets is not None and HAS_FLASH_ATTN:
            if B != 1:
                raise ValueError(f"flash_attn_varlen_func requires B=1, got B={B}.")
            cu_seqlens = doc_offsets.to(dtype=torch.int32, device=x.device)
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            q_fa = q.squeeze(0).transpose(0, 1).contiguous()
            k_fa = k.squeeze(0).transpose(0, 1).contiguous()
            v_fa = v.squeeze(0).transpose(0, 1).contiguous()
            out = flash_attn_varlen_func(
                q_fa, k_fa, v_fa,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=self.dropout.p if self.training else 0.0,
                causal=causal,
            )
            out = out.view(1, S, self.d_model)
        elif flex_block_mask is not None and HAS_FLEX_ATTENTION:
            out = _flex_attention(q, k, v, block_mask=flex_block_mask)
            out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        else:
            combined_mask = attn_mask
            if key_padding_mask is not None:
                pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                neg_inf = torch.tensor(float("-inf"), device=x.device, dtype=q.dtype)
                zero = torch.tensor(0.0, device=x.device, dtype=q.dtype)
                pad_mask = torch.where(pad_mask, neg_inf, zero)
                combined_mask = combined_mask + pad_mask if combined_mask is not None else pad_mask
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=combined_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)

        return self.out_proj(out)
