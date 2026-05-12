"""Native transform runner.

Composes a list of transform specs into a single map_batches function
and applies it to a Ray dataset, optionally projecting to columns_to_keep.
"""

from typing import Optional

from michelangelo.lib.native_transform.transforms import parse_transform


def apply_native_transforms(ray_dataset, transform_specs: list[dict], columns_to_keep: Optional[list[str]] = None):
    """Apply a sequence of native transforms to a Ray dataset via map_batches.

    Args:
        ray_dataset: Input Ray dataset (columnar, Parquet-backed).
        transform_specs: List of transform spec dicts matching native_transform_specs.yaml
            structure (keys: transform_name, input_cols, output_cols, and transform params).
        columns_to_keep: If provided, drop all columns not in this list after transforms.

    Returns:
        Transformed Ray dataset (lazy — execution happens on iteration).
    """
    transforms = [parse_transform(spec) for spec in transform_specs]

    def _apply(batch: dict) -> dict:
        for t in transforms:
            batch = t(batch)
        if columns_to_keep is not None:
            batch = {k: v for k, v in batch.items() if k in columns_to_keep}
        return batch

    return ray_dataset.map_batches(_apply, batch_format="numpy")
