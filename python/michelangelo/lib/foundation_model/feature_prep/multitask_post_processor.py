"""MultitaskPostProcessor: extends FeaturePrepPostProcessor with IDLE_DAY injection.

Adds earner-specific sequential features and targets before entity-level
aggregation, and injects synthetic IDLE_DAY events for inactive calendar days.
"""

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from michelangelo.lib.sequence.feature_prep.constants import EVENT_TYPE
from michelangelo.lib.sequence.feature_prep.post_processor import (
    EVENT_TIMESTAMP,
    FeaturePrepPostProcessor,
)

logger = logging.getLogger(__name__)


class MultitaskPostProcessor(FeaturePrepPostProcessor):
    """Post-processor with multitask sequential features, targets, and IDLE_DAY injection.

    Before sequence numbering, injects synthetic IDLE_DAY events for calendar
    days with no earner activity. This expands the sequence to include
    inactivity signals as explicit tokens.

    Adds event-level columns before entity-level aggregation:
      - ``time_since_last_seconds``: seconds since the previous event (0.0 for first).
      - ``event_number_norm``: normalized position in sequence [0.0, 1.0].
      - ``next_event_type_raw``: event_type of the following event.
      - ``next_event_type_indexed``: StringIndexer index of the following event.
      - ``time_to_next_seconds``: seconds until the next event; for the last event
        uses ``dataset_end_ts - current_ts`` as a churn supervision signal.
      - ``seconds_since_supply_online``: cumulative seconds since last supply_online.
      - ``consecutive_idle_days``: 0 for real events; days since last real event
        for IDLE_DAY tokens.
    """

    def __init__(self):
        super().__init__(config=None)

    @staticmethod
    def _inject_idle_day_events(df: DataFrame, sequence_id_column: str, event_timestamp_column: str) -> DataFrame:
        """Inject IDLE_DAY synthetic events for calendar days with no earner activity.

        Uses a single groupBy to collect per-earner date ranges and active-date sets,
        then filters with ``array_contains`` instead of an anti-join, reducing shuffles.
        """
        earner_info = df.groupBy(sequence_id_column).agg(
            F.to_date(F.min(event_timestamp_column)).alias("first_active_date"),
            F.to_date(F.max(event_timestamp_column)).alias("last_active_date"),
            F.collect_set(F.to_date(event_timestamp_column)).alias("active_dates"),
            F.first(F.coalesce(F.col("churned"), F.lit(-1))).alias("_idle_churned"),
        )

        idle_days = (
            earner_info
            .select(
                sequence_id_column,
                F.explode(
                    F.sequence(F.col("first_active_date"), F.col("last_active_date"), F.expr("INTERVAL 1 DAY"))
                ).alias("date"),
                "active_dates",
                F.col("_idle_churned").alias("churned"),
            )
            .filter(~F.array_contains(F.col("active_dates"), F.col("date")))
            .drop("active_dates")
        )

        idle_events = idle_days.select(
            F.col(sequence_id_column),
            F.lit("IDLE_DAY").alias(EVENT_TYPE),
            F.col("date").cast("timestamp").alias(event_timestamp_column),
            F.col("churned"),
        )

        result = df.unionByName(idle_events, allowMissingColumns=True)
        logger.info("[PostProcessor] Injected IDLE_DAY events for calendar days with no activity")
        return result

    def __call__(self, df: DataFrame, params: dict) -> DataFrame:
        """Inject IDLE_DAY events before the parent's processing pipeline."""
        sequence_id_column = params.get("sequence_id_column")
        event_timestamp_column = params.get("event_timestamp_column", EVENT_TIMESTAMP)

        if sequence_id_column and "churned" in df.columns:
            logger.info("[PostProcessor] Injecting IDLE_DAY synthetic events...")
            df = self._inject_idle_day_events(df, sequence_id_column, event_timestamp_column)
            df = df.cache()

        return super().__call__(df, params)

    def _pre_aggregation_hook(
        self, df: DataFrame, window, sequence_id_column: str, event_timestamp_column: str
    ) -> DataFrame:
        """Add sequential features and multitask targets before aggregation.

        Args:
            df: Event-level DataFrame with ``event_sequence_number`` already computed.
            window: ``partitionBy(sequence_id_column).orderBy(event_timestamp_column)``.
            sequence_id_column: Column used for partitioning.
            event_timestamp_column: Timestamp column name.

        Returns:
            DataFrame with sequential feature columns added.
        """
        partition_window = Window.partitionBy(sequence_id_column)

        # 1. time_since_last_seconds
        df = df.withColumn(
            "time_since_last_seconds",
            (
                F.unix_timestamp(F.col(event_timestamp_column))
                - F.unix_timestamp(F.lag(event_timestamp_column, 1).over(window))
            ).cast("double"),
        )

        # 2. event_number_norm: position normalised to [0, 1]
        from michelangelo.lib.sequence.feature_prep.constants import EVENT_SEQUENCE_NUMBER
        max_seq_num = F.max(EVENT_SEQUENCE_NUMBER).over(partition_window)
        df = df.withColumn(
            "event_number_norm",
            F.when(
                max_seq_num > 1,
                (F.col(EVENT_SEQUENCE_NUMBER) - 1).cast("double") / (max_seq_num - 1).cast("double"),
            ).otherwise(F.lit(0.0)),
        )

        # 3. next_event_type_raw + next_event_type_indexed
        df = df.withColumn("next_event_type_raw", F.lead(EVENT_TYPE, 1).over(window))
        df = df.withColumn(
            "next_event_type_indexed",
            F.lead("event_type_indexed", 1).over(window).cast("long"),
        )

        # 4. time_to_next_seconds (churn supervision signal for last event)
        lead_ts = F.unix_timestamp(F.lead(event_timestamp_column, 1).over(window))
        current_ts = F.unix_timestamp(F.col(event_timestamp_column))
        dataset_end_ts = df.agg(F.max(F.unix_timestamp(F.col(event_timestamp_column)))).collect()[0][0]
        dataset_end_lit = F.lit(float(dataset_end_ts))
        df = df.withColumn(
            "time_to_next_seconds",
            F.greatest(
                F.lit(0.0),
                F.coalesce(lead_ts - current_ts, dataset_end_lit - current_ts).cast("double"),
            ),
        )

        # 5. seconds_since_supply_online
        last_supply_online_ts = F.last(
            F.when(F.col(EVENT_TYPE) == "supply_online", F.col(event_timestamp_column)),
            ignorenulls=True,
        ).over(window)
        df = df.withColumn(
            "seconds_since_supply_online",
            F.coalesce(
                (F.unix_timestamp(F.col(event_timestamp_column)) - F.unix_timestamp(last_supply_online_ts)).cast("double"),
                F.lit(0.0),
            ),
        )

        # 6. consecutive_idle_days: 0 for real events, datediff for IDLE_DAY tokens
        last_real_ts = F.last(
            F.when(F.col(EVENT_TYPE) != "IDLE_DAY", F.col(event_timestamp_column)),
            ignorenulls=True,
        ).over(window)
        df = df.withColumn(
            "consecutive_idle_days",
            F.when(F.col(EVENT_TYPE) != "IDLE_DAY", F.lit(0)).otherwise(
                F.datediff(F.to_date(F.col(event_timestamp_column)), F.to_date(last_real_ts))
            ).cast("double"),
        )

        return df
