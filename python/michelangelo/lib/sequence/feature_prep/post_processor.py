"""Base post-processing module for tabular feature preparation.

Wraps Spark-based feature engineering logic into the interface expected by
the Michelangelo tabular_feature_prep task:

    def __call__(df: DataFrame, params: dict) -> DataFrame

Subclasses override ``_pre_aggregation_hook`` to inject domain-specific
features before the entity-level aggregation step.

Processing steps (in order):
  1. Base column schema casting
  2. Optional JSON parsing + struct flattening
  3. Coalesce columns
  4. Event sequence numbering
  5. StringIndexer for categorical features
  6. ``_pre_aggregation_hook`` (subclass extension point)
  7. Optional feature statistics logging
  8. Entity-level aggregation (struct-based, null-preserving)
  9. Empty array → null conversion
  10. Timestamp string column creation
"""

import json
import logging
from typing import Optional

from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

from michelangelo.lib.sequence.feature_prep.constants import (
    DATESTR,
    EVENT_SEQUENCE_NUMBER,
    EVENT_TIMESTAMP,
    EVENT_TYPE,
    STATS_METADATA_COLS,
    STATS_NUMERIC_TYPES,
    STATS_PERCENTILES,
)

logger = logging.getLogger(__name__)

# Column written by StringIndexer to store the event_type vocabulary JSON
EVENT_TYPE_VOCAB_JSON = "_event_type_vocab_json"


def _fmt_float(v: object) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "N/A"


class FeaturePrepPostProcessor:
    """Callable post-processor for the tabular_feature_prep task.

    The SDK expects a callable class with signature:
        def __call__(df: DataFrame, params: dict) -> DataFrame

    This class wraps all feature preparation logic into that interface.

    Args:
        config: Optional pre-constructed configuration object. Its fields are
            merged with ``params`` at call time; ``params`` takes precedence.
    """

    def __init__(self, config=None):
        self.config = config

    def __call__(self, df: DataFrame, params: dict) -> DataFrame:
        """Apply feature preparation transformations.

        Args:
            df: Already-split DataFrame from tabular_feature_prep.
            params: Configuration dictionary. When ``self.config`` is also
                provided, its values are used as defaults that ``params``
                can override.

        Returns:
            Transformed DataFrame ready for tabular_transform.
        """
        _ = SparkSession.getActiveSession()

        if self.config:
            config_dict = self.config.model_dump()
            full_params = {**config_dict, **params}
        else:
            full_params = params

        base_columns_schema = full_params.get("base_columns_schema", {})
        event_value_json_col = full_params.get("event_value_json_col")
        event_value_schema = full_params.get("event_value_schema")
        struct_flatten_rules = full_params.get("struct_flatten_rules")
        flattened_columns_schema = full_params.get("flattened_columns_schema")
        coalesce_columns = full_params.get("coalesce_columns")
        indexed_cat_features = full_params.get("indexed_cat_features")
        event_timestamp_column = full_params.get("event_timestamp_column", EVENT_TIMESTAMP)
        sequence_id_column = full_params.get("sequence_id_column")
        max_sequence_length = full_params.get("max_sequence_length")
        aggregate_to_sequence = full_params.get("aggregate_to_sequence", True)
        empty_array_to_null_columns = full_params.get("empty_array_to_null_columns")

        if aggregate_to_sequence and not sequence_id_column:
            raise ValueError(
                "sequence_id_column is required when aggregate_to_sequence=True."
            )

        logger.info("[PostProcessor] Starting feature preparation...")

        # Step 1: schema casting
        if base_columns_schema:
            for col_name, col_type in base_columns_schema.items():
                if col_name in df.columns:
                    df = df.withColumn(col_name, F.col(col_name).cast(col_type))

        # Step 2: JSON parsing + struct flattening
        if event_value_json_col and event_value_schema:
            df = df.withColumn("event_value_parsed", F.from_json(F.col(event_value_json_col), event_value_schema))
            base_columns = list(base_columns_schema.keys())
            non_json_columns = [col for col in base_columns if col != event_value_json_col]
            df = df.select(*non_json_columns, "event_value_parsed.*")

            if struct_flatten_rules:
                for struct_col, field_mappings in struct_flatten_rules.items():
                    if struct_col in df.columns:
                        for struct_field, output_col in field_mappings.items():
                            df = df.withColumn(output_col, F.col(f"{struct_col}.{struct_field}"))
                        df = df.drop(struct_col)
                if flattened_columns_schema:
                    for col_name, col_type in flattened_columns_schema.items():
                        if col_name in df.columns:
                            df = df.withColumn(col_name, F.col(col_name).cast(col_type))

        # Step 3: coalesce columns
        if coalesce_columns:
            cols_to_drop = []
            for output_col, source_cols in coalesce_columns.items():
                existing = [c for c in source_cols if c in df.columns]
                if existing:
                    df = df.withColumn(output_col, F.coalesce(*[F.col(c) for c in existing]))
                    cols_to_drop.extend(existing)
            for c in set(cols_to_drop):
                if c in df.columns:
                    df = df.drop(c)

        # Step 4: event sequence numbering
        window = Window.partitionBy(sequence_id_column).orderBy(
            F.col(event_timestamp_column).asc(), F.col(EVENT_TYPE).asc()
        )
        df = df.withColumn(EVENT_SEQUENCE_NUMBER, F.row_number().over(window))

        if max_sequence_length:
            df = df.filter(F.col(EVENT_SEQUENCE_NUMBER) <= max_sequence_length)

        # Step 5: StringIndexer for categorical columns
        if indexed_cat_features:
            for col_name in indexed_cat_features:
                if col_name in df.columns:
                    indexer = StringIndexer(
                        inputCol=col_name,
                        outputCol=f"{col_name}_indexed",
                        handleInvalid="keep",
                    )
                    indexer_model = indexer.fit(df.select(col_name))
                    df = indexer_model.transform(df)
                    if col_name == EVENT_TYPE:
                        labels_json = json.dumps(list(indexer_model.labels))
                        df = df.withColumn(EVENT_TYPE_VOCAB_JSON, F.lit(labels_json))

        # Step 6: hook for domain-specific features before aggregation
        df = self._pre_aggregation_hook(df, window, sequence_id_column, event_timestamp_column)

        # Step 7: optional feature statistics logging
        feature_stats_cfg = full_params.get("feature_stats")
        if isinstance(feature_stats_cfg, dict):
            from michelangelo.lib.sequence.feature_prep.stats_config import FeatureStatsConfig
            feature_stats_cfg = FeatureStatsConfig(**feature_stats_cfg)
        if feature_stats_cfg and getattr(feature_stats_cfg, "enabled", False):
            self._compute_feature_stats(
                df, feature_stats_cfg, sequence_id_column, event_timestamp_column,
                event_value_json_col=full_params.get("event_value_json_col"),
            )

        # Step 8: aggregate to entity-level
        if aggregate_to_sequence:
            df = self._aggregate_to_entity_level(df, sequence_id_column, event_timestamp_column)
            self._log_sequence_length_stats(df, sequence_id_column)

        # Step 9: empty array → null
        if aggregate_to_sequence and empty_array_to_null_columns:
            type_lookup = {**base_columns_schema, **(flattened_columns_schema or {})}
            for col in empty_array_to_null_columns:
                if col in df.columns:
                    base_type = type_lookup.get(col, "STRING").lower()
                    arr_type = f"array<{base_type}>"
                    df = df.withColumn(
                        col,
                        F.when(F.size(F.col(col)) == 0, F.lit(None).cast(arr_type)).otherwise(F.col(col)),
                    )

        # Step 10: timestamp string column
        if event_timestamp_column in df.columns:
            ts_str_col = f"{event_timestamp_column}_string"
            if aggregate_to_sequence:
                df = df.withColumn(
                    ts_str_col,
                    F.transform(
                        F.col(event_timestamp_column),
                        lambda x: F.coalesce(F.date_format(x, "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"), F.lit("null")),
                    ).cast("array<string>"),
                )
            else:
                df = df.withColumn(
                    ts_str_col,
                    F.coalesce(F.date_format(F.col(event_timestamp_column), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"), F.lit("null")),
                )

        logger.info("[PostProcessor] Feature preparation complete")
        return df

    def _pre_aggregation_hook(
        self, df: DataFrame, window, sequence_id_column: str, event_timestamp_column: str
    ) -> DataFrame:
        """Extension point for subclasses to add columns before aggregation.

        Called after StringIndexer (Step 5) and before aggregation (Step 8).
        The window spec is already partitioned and ordered so subclasses can
        use ``F.lag`` / ``F.lead`` without rebuilding it.

        Args:
            df: Event-level DataFrame (one row per event, not yet aggregated).
            window: ``partitionBy(sequence_id_column).orderBy(event_timestamp_column)``.
            sequence_id_column: Column used for partitioning.
            event_timestamp_column: Timestamp column name.

        Returns:
            DataFrame unchanged (override in subclasses to add columns).
        """
        return df

    def _compute_feature_stats(
        self,
        df: DataFrame,
        cfg,
        sequence_id_column: str,
        event_timestamp_column: str,
        event_value_json_col: Optional[str] = None,
    ) -> None:
        """Compute and log event-level feature statistics in a single aggregation pass."""
        _METADATA_COLS = STATS_METADATA_COLS | {sequence_id_column, event_timestamp_column}
        skip = set(getattr(cfg, "columns_to_skip", []))
        if event_value_json_col:
            skip.add(event_value_json_col)

        columns = getattr(cfg, "columns", [])
        if columns:
            schema_map = {f.name: f.dataType for f in df.schema.fields}
            candidate_fields = [(c, schema_map[c]) for c in columns if c in schema_map and c not in skip]
        else:
            candidate_fields = [
                (f.name, f.dataType)
                for f in df.schema.fields
                if f.name not in _METADATA_COLS and f.name not in skip
            ]

        numeric_cols = [c for c, t in candidate_fields if isinstance(t, STATS_NUMERIC_TYPES)]
        string_cols = [c for c, t in candidate_fields if isinstance(t, StringType)]
        sample_fraction = getattr(cfg, "sample_fraction", 0.1)
        log1p_cols = getattr(cfg, "log1p_transform_columns", [])

        if not numeric_cols and not string_cols:
            logger.info("[PostProcessor] Feature stats: no numeric or string columns found.")
            return

        sample_df = df.sample(fraction=min(sample_fraction, 1.0), seed=42)

        agg_exprs = [F.count(F.lit(1)).alias("__total__")]
        for c in numeric_cols:
            agg_exprs += [
                F.mean(c).alias(f"{c}__mean"),
                F.stddev_samp(c).alias(f"{c}__std"),
                F.min(c).alias(f"{c}__min"),
                F.max(c).alias(f"{c}__max"),
                F.count(F.when(F.col(c).isNull(), 1)).alias(f"{c}__nulls"),
                F.percentile_approx(c, STATS_PERCENTILES, 100).alias(f"{c}__pct"),
            ]
        for c in [n for n in log1p_cols if n in numeric_cols]:
            log_expr = F.log(F.greatest(F.col(c), F.lit(0.0)) + F.lit(1.0))
            agg_exprs += [F.mean(log_expr).alias(f"{c}__log_mean"), F.stddev_samp(log_expr).alias(f"{c}__log_std")]
        for c in string_cols:
            agg_exprs += [
                F.approx_count_distinct(c, rsd=0.05).alias(f"{c}__approx_distinct"),
                F.count(F.when(F.col(c).isNull(), 1)).alias(f"{c}__nulls"),
            ]

        row = sample_df.agg(*agg_exprs).collect()[0]
        total = row["__total__"] or 1
        logger.info(f"[PostProcessor] Feature stats ({sample_fraction*100:.0f}% sample, {total} rows):")

        for c in numeric_cols:
            null_pct = _fmt_float(100.0 * (row[f"{c}__nulls"] or 0) / total)
            pct_vals = row[f"{c}__pct"] or []
            pct_str = ", ".join(f"p{int(p*100)}={_fmt_float(v)}" for p, v in zip(STATS_PERCENTILES, pct_vals))
            logger.info(
                f"  {c}: mean={_fmt_float(row[f'{c}__mean'])}, std={_fmt_float(row[f'{c}__std'])}, "
                f"min={_fmt_float(row[f'{c}__min'])}, max={_fmt_float(row[f'{c}__max'])}, null={null_pct}%"
            )
            logger.info(f"    pct=[{pct_str}]")

        for c in string_cols:
            null_pct = _fmt_float(100.0 * (row[f"{c}__nulls"] or 0) / total)
            logger.info(f"  {c}: approx_distinct={row[f'{c}__approx_distinct']}, null={null_pct}%")

    def _log_sequence_length_stats(self, df: DataFrame, sequence_id_col: str) -> None:
        list_col = next((c for c in df.columns if c not in (sequence_id_col, DATESTR)), None)
        if list_col is None:
            return
        lengths = df.select(F.size(F.col(list_col)).alias("seq_len"))
        stats = lengths.agg(
            F.min("seq_len").alias("min"),
            F.max("seq_len").alias("max"),
            F.mean("seq_len").alias("mean"),
            F.expr("percentile(seq_len, 0.50)").alias("p50"),
            F.expr("percentile(seq_len, 0.90)").alias("p90"),
            F.expr("percentile(seq_len, 0.99)").alias("p99"),
        ).collect()[0]
        logger.info(
            f"[PostProcessor] Sequence lengths: min={stats['min']}, p50={stats['p50']}, "
            f"p90={stats['p90']}, p99={stats['p99']}, max={stats['max']}, "
            f"mean={_fmt_float(stats['mean'])}"
        )

    def _aggregate_to_entity_level(
        self, df: DataFrame, sequence_id_column: str, event_timestamp_column: str
    ) -> DataFrame:
        """Convert event-level DataFrame to sequence-level with list-valued columns.

        Uses struct-based collection to preserve NULLs during aggregation —
        plain ``collect_list`` drops NULLs, misaligning features at different
        positions. Wrapping all features in a struct before collection and
        extracting individual arrays afterwards guarantees equal-length arrays.
        """
        metadata_cols = [sequence_id_column, EVENT_SEQUENCE_NUMBER]
        if EVENT_TYPE in df.columns:
            metadata_cols.append(EVENT_TYPE)
        if event_timestamp_column in df.columns:
            metadata_cols.append(event_timestamp_column)
        if DATESTR in df.columns:
            metadata_cols.append(DATESTR)
        if EVENT_TYPE_VOCAB_JSON in df.columns:
            metadata_cols.append(EVENT_TYPE_VOCAB_JSON)

        feature_cols = [col for col in df.columns if col not in metadata_cols]

        agg_exprs = []
        if EVENT_TYPE in df.columns:
            agg_exprs.append(F.collect_list(EVENT_TYPE).alias(EVENT_TYPE))
        if event_timestamp_column in df.columns:
            agg_exprs.append(F.collect_list(event_timestamp_column).alias(event_timestamp_column))
        if DATESTR in df.columns:
            agg_exprs.append(F.min(DATESTR).alias(DATESTR))
        if EVENT_TYPE_VOCAB_JSON in df.columns:
            agg_exprs.append(F.first(EVENT_TYPE_VOCAB_JSON).alias(EVENT_TYPE_VOCAB_JSON))

        events_struct_col = "_events_struct"
        features_struct = F.struct(*[F.col(c) for c in feature_cols])
        agg_exprs.append(F.collect_list(features_struct).alias(events_struct_col))

        df_aggregated = df.groupBy(sequence_id_column).agg(*agg_exprs)

        for col_name in feature_cols:
            df_aggregated = df_aggregated.withColumn(col_name, F.col(f"{events_struct_col}.{col_name}"))
        df_aggregated = df_aggregated.drop(events_struct_col)

        # Validate equal-length arrays
        ref_col = EVENT_TYPE if EVENT_TYPE in df_aggregated.columns else event_timestamp_column
        ref_len = F.size(ref_col)
        mismatch_flags = [F.when(F.size(c) != ref_len, F.lit(1)).otherwise(F.lit(0)) for c in feature_cols]
        if mismatch_flags:
            mismatch_count = df_aggregated.select(
                F.sum(F.greatest(*mismatch_flags)).alias("n")
            ).collect()[0]["n"] or 0
            if mismatch_count > 0:
                raise ValueError(
                    f"Feature array length mismatch: {mismatch_count} sequences have unequal array lengths. "
                    "This indicates NULLs were dropped during aggregation."
                )

        return df_aggregated
