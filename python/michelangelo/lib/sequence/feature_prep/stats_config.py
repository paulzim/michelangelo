"""Feature statistics configuration for post-processor."""

from pydantic import BaseModel


class FeatureStatsConfig(BaseModel):
    """Configuration for optional event-level feature statistics computation.

    When enabled, stats are computed on a sampled event-level DataFrame after
    all feature engineering but before sequence aggregation. Results are logged.

    Attributes:
        enabled: Whether to compute stats.
        columns: Columns to include. Empty = auto-detect all numeric/string columns.
        columns_to_skip: Columns to exclude (takes precedence over ``columns``).
        sample_fraction: Fraction of events to sample (0.0, 1.0].
        log1p_transform_columns: Columns for which to additionally compute
            mean/std of ``ln(1 + max(0, x))``.
    """

    enabled: bool = False
    columns: list[str] = []
    columns_to_skip: list[str] = []
    sample_fraction: float = 0.1
    log1p_transform_columns: list[str] = []
