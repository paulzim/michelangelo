"""Native transform implementations.

Each transform operates on a columnar numpy batch (Ray map_batches format):
  batch = {col_name: np.ndarray, ...}

Sequence columns arrive as object-dtype arrays of shape (N,) where each
element is a Python list or ndarray of length S (max_seq_len). _to_dense()
materializes these to (N, S) float32 before arithmetic.
"""

import numpy as np
from dataclasses import dataclass, field


def _to_dense(col: np.ndarray) -> np.ndarray:
    """Materialize object-dtype column of sequences → (N, S) float32."""
    if col.dtype == object:
        return np.stack(col.tolist()).astype(np.float32)
    return col.astype(np.float32)


@dataclass
class LogTransform:
    """log(x + add_constant), applied element-wise per sequence position."""
    input_cols: list[str]
    output_cols: list[str]
    add_constant: float = 1.0

    def __call__(self, batch: dict) -> dict:
        for inp, out in zip(self.input_cols, self.output_cols):
            if inp in batch:
                batch[out] = np.log(_to_dense(batch[inp]) + self.add_constant)
        return batch


@dataclass
class Normalization:
    """(x - mean) / std z-score normalization, one mean/std per column."""
    input_cols: list[str]
    output_cols: list[str]
    mean: list[float]
    std: list[float]

    def __call__(self, batch: dict) -> dict:
        for inp, out, m, s in zip(self.input_cols, self.output_cols, self.mean, self.std):
            if inp in batch:
                batch[out] = (_to_dense(batch[inp]) - m) / (s or 1.0)
        return batch


@dataclass
class MinMax:
    """(x - min) / (max - min) normalization, one min/max per column."""
    input_cols: list[str]
    output_cols: list[str]
    min: list[float]
    max: list[float]

    def __call__(self, batch: dict) -> dict:
        for inp, out, lo, hi in zip(self.input_cols, self.output_cols, self.min, self.max):
            if inp in batch:
                r = hi - lo or 1.0
                batch[out] = (_to_dense(batch[inp]) - lo) / r
        return batch


@dataclass
class Bucketization:
    """Bin values into integer bucket indices using the given boundaries.

    Positions with value < boundaries[0] → bucket 0 (used for padding).
    boundaries=[0, 60, 300, ...] gives len(boundaries)+1 buckets.
    """
    input_cols: list[str]
    output_cols: list[str]
    boundaries: list[float]

    def __call__(self, batch: dict) -> dict:
        for inp, out in zip(self.input_cols, self.output_cols):
            if inp in batch:
                batch[out] = np.digitize(_to_dense(batch[inp]), self.boundaries).astype(np.int64)
        return batch


@dataclass
class Stack:
    """Stack multiple (N, S) columns into a single (N, S, D) column.

    All input columns must have the same (N, S) shape.
    dim=-1 stacks along the last axis → (N, S, D).
    """
    input_cols: list[str]
    output_cols: list[str]  # single element
    dim: int = -1

    def __call__(self, batch: dict) -> dict:
        cols = [_to_dense(batch[c]) for c in self.input_cols if c in batch]
        if cols:
            batch[self.output_cols[0]] = np.stack(cols, axis=self.dim).astype(np.float32)
        return batch


_TRANSFORM_REGISTRY: dict[str, type] = {
    "LogTransform": LogTransform,
    "Normalization": Normalization,
    "MinMax": MinMax,
    "Bucketization": Bucketization,
    "Stack": Stack,
}


def parse_transform(spec: dict):
    """Instantiate a transform from a spec dict (mirrors YAML structure)."""
    name = spec["transform_name"]
    cls = _TRANSFORM_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown transform: {name!r}. Available: {list(_TRANSFORM_REGISTRY)}")
    kwargs = {k: v for k, v in spec.items() if k != "transform_name"}
    # strip dtype string — Bucketization output is always int64
    kwargs.pop("dtype", None)
    return cls(**kwargs)
