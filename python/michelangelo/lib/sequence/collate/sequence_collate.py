"""Batch collation utilities for sequence model training.

Converts numpy object arrays from Parquet (common when reading sequence columns
via Ray datasets) into typed PyTorch tensors, skipping string columns.
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SequenceCollateFn:
    """Generic collate: converts numpy arrays to torch tensors, skips strings.

    Unlike a simple float32 cast, this preserves dtypes (int stays int,
    float stays float). Handles object-dtype arrays that contain nested
    ndarrays (common when reading array columns from Parquet via Ray datasets).

    Optionally caches a vocabulary from a designated string column on first
    sight so downstream callbacks can resolve integer indices back to names.

    Args:
        vocab_key: Optional column name containing a JSON-encoded vocabulary
            (e.g. ``"_event_type_vocab_json"``). When provided, the vocabulary
            is parsed and cached on first encounter then the column is dropped
            from the output batch. When ``None``, no vocabulary caching occurs.
    """

    def __init__(self, vocab_key: Optional[str] = None):
        self._vocab_key = vocab_key
        self._vocab: Optional[dict[int, str]] = None

    @property
    def vocab(self) -> Optional[dict[int, str]]:
        """Cached vocabulary mapping index → name, or None if not yet seen."""
        return self._vocab

    @staticmethod
    def _extract_vocab_string(raw: np.ndarray) -> Optional[str]:
        """Extract a JSON vocab string from an array column.

        After entity-level aggregation the column is ``array<string>``
        (same JSON repeated per event), so in a Ray batch ``raw`` is an
        object ndarray where each element is a list/ndarray of identical
        strings.
        """
        if not isinstance(raw, np.ndarray) or len(raw) == 0:
            return None
        first = raw[0]
        if isinstance(first, str):
            return first
        if isinstance(first, (list, np.ndarray)) and len(first) > 0:
            candidate = first[0]
            if isinstance(candidate, str):
                return candidate
        return None

    @staticmethod
    def _convert_object_array(value: np.ndarray) -> Optional[np.ndarray]:
        """Convert an object-dtype ndarray to a dense numeric ndarray.

        Returns ``None`` for string columns or empty arrays (skip them).
        Raises ``ValueError`` for numeric-looking columns that fail conversion
        so bugs are caught early rather than silently dropping data.
        """
        if len(value) == 0:
            return None
        first = value[0]
        if isinstance(first, str):
            return None
        if not isinstance(first, np.ndarray):
            return np.asarray(value)
        if first.dtype.kind in ("U", "S"):
            return None
        if first.dtype.kind == "O":
            if len(first) > 0 and isinstance(first[0], str):
                return None
            result = np.stack([np.stack(v) for v in value])
        else:
            result = np.stack(value)
        if result.dtype == np.float64:
            result = result.astype(np.float32)
        return result

    def __call__(self, batch: dict) -> dict[str, torch.Tensor]:
        import json

        # Cache vocabulary from the designated column if configured
        if self._vocab is None and self._vocab_key is not None and self._vocab_key in batch:
            raw = batch[self._vocab_key]
            vocab_str = self._extract_vocab_string(raw)
            if vocab_str is not None:
                labels = json.loads(vocab_str)
                self._vocab = {i + 1: name for i, name in enumerate(labels)}
                logger.info("[collate] Cached vocab with %d labels", len(labels))

        result = {}
        for key, value in batch.items():
            # Drop the vocab column — it's a string, not a tensor feature
            if key == self._vocab_key:
                continue
            if isinstance(value, np.ndarray):
                if value.dtype.kind in ("U", "S"):
                    continue
                if value.dtype.kind == "O":
                    converted = self._convert_object_array(value)
                    if converted is None:
                        continue
                    result[key] = torch.as_tensor(converted)
                else:
                    result[key] = torch.as_tensor(value)
            elif isinstance(value, torch.Tensor):
                result[key] = value
            else:
                result[key] = torch.as_tensor(value)
        return result
