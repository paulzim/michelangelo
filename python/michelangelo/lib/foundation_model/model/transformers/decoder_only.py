"""Decoder-only GPT-style transformer backbone.

bf16-mixed precision: mask/attention dtype mismatch
----------------------------------------------------
The Michelangelo trainer defaults to ``precision="bf16-mixed"``, which wraps
forward passes in ``torch.autocast``. Inside ``nn.TransformerEncoder``, this
creates a dtype split: ``src`` enters as float32, ``_canonical_mask`` derives
a float32 mask, but Q/K/V projections are cast to bf16 by autocast, causing
``RuntimeError: Expected attn_mask dtype to match query dtype``.

Fix: pre-cast ``src`` to the active autocast dtype before entering
``TransformerEncoder``. Numerically sensitive ops (LayerNorm, Softmax) still
run in float32 — autocast exempts them regardless of input dtype.
"""

import torch
import torch.nn as nn


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only transformer backbone with causal masking.

    Uses PyTorch's TransformerEncoder with a causal attention mask to implement
    a GPT-style decoder-only architecture. No embeddings or task heads —
    pure backbone only.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def _generate_causal_mask(self, seq_len: int, reference: torch.Tensor) -> torch.Tensor:
        """Upper-triangular ``-inf`` causal mask in the dtype of *reference*.

        Uses ``new_full`` to inherit device dynamically (avoids baking cpu
        device during TorchScript tracing).
        """
        return torch.triu(reference.new_full((seq_len, seq_len), float("-inf")), diagonal=1)

    @staticmethod
    @torch.jit.unused
    def _active_dtype(fallback: torch.dtype) -> torch.dtype:
        """Return the autocast dtype if active, else ``fallback``."""
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
        if torch.is_autocast_cpu_enabled():
            return torch.get_autocast_cpu_dtype()
        return fallback

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer backbone.

        Args:
            src: Input tensor of shape (B, S, d_model).
            src_key_padding_mask: Bool mask of shape (B, S); ``True`` = padding position.

        Returns:
            Transformer output of shape (B, S, d_model).
        """
        # Pre-cast src to autocast dtype to prevent mask/Q-K-V dtype mismatch
        # (see module docstring). No-op when autocast is inactive or scripted.
        if torch.jit.is_scripting():
            dtype = src.dtype
        else:
            dtype = self._active_dtype(src.dtype)
        if dtype != src.dtype:
            src = src.to(dtype)

        mask = self._generate_causal_mask(src.size(1), src)
        return self.transformer(src, mask=mask, src_key_padding_mask=src_key_padding_mask)
