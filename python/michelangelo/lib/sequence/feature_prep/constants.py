"""Column name constants for the feature preparation pipeline."""

from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType

EVENT_TYPE = "event_type"
EVENT_SEQUENCE_NUMBER = "event_sequence_number"
EVENT_TIMESTAMP = "event_timestamp"
DATESTR = "datestr"

# Spark types treated as numeric for feature statistics computation
STATS_NUMERIC_TYPES = (DoubleType, FloatType, IntegerType, LongType)

# Percentiles computed for each numeric column during stats computation
STATS_PERCENTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Static metadata columns excluded from auto-detected stats
STATS_METADATA_COLS = frozenset({EVENT_SEQUENCE_NUMBER, DATESTR})
