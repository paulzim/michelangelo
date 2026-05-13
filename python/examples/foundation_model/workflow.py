"""Earner foundation model training workflow.

Production pipeline steps (reference: pipeline_conf.yaml):
  1. tabular_feature_prep      — Spark + MultitaskPostProcessor (internal)
  2. tabular_transform         — DSL feature engineering (internal)
  3. tabular_native_transform  — Ray: LogTransform, Normalization, Stack, etc.
  4. tabular_trainer           — Ray: train MultitaskSequenceLightning
  5. tabular_assembler / pusher (internal)

This workflow covers steps 3 and 4.

## Running locally

    cd python/
    python examples/foundation_model/workflow.py local-run

Synthetic data is generated in the pre-native-transform schema (raw *_padded
columns) so the full transform → train code path matches production.

## Running on a cluster

    mactl submit examples/foundation_model/workflow.py \\
        --storage-url s3://your-bucket/uf_storage

## Testing the model without Spark/Ray

    python examples/foundation_model/test_model.py
"""

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.core.decorator import workflow
from michelangelo.workflow.variables import DatasetVariable
from examples.foundation_model.tasks.native_transform_task import native_transform_task
from examples.foundation_model.tasks.train_task import train_task
from examples.foundation_model.tasks.upload_task import upload_task

from examples.foundation_model.config import NATIVE_TRANSFORM_CONFIG, TRAIN_CONFIG, MAX_LEN


@workflow()
def train_workflow(train_data: DatasetVariable, val_data: DatasetVariable):
    """Native transform → train → upload."""
    # Step 1: Apply native transforms (LogTransform, Normalization, Stack, etc.)
    transformed = native_transform_task(
        config=NATIVE_TRANSFORM_CONFIG,
        train_data=train_data,
        val_data=val_data,
    )

    # Step 2: Train MultitaskSequenceLightning (evaluation handled by callbacks)
    trained = train_task(
        config=TRAIN_CONFIG,
        train_data=transformed["train_dataset"],
        val_data=transformed["val_dataset"],
    )

    # Step 3: Upload checkpoint
    upload_task(config={"train_config": TRAIN_CONFIG}, train_result=trained)


def _make_synthetic_dataset(n_earners: int, split: str) -> DatasetVariable:
    """Generate synthetic data in the pre-native-transform schema.

    Column schema matches production output of tabular_transform (DSL step):
    raw *_padded float sequences + hash/standard categoricals + labels.
    The native_transform_task will produce derived_numerical_stacked etc.
    """
    import random
    from pyspark.sql import SparkSession
    from examples.foundation_model.config import EVENT_TYPE_VOCAB_SIZE

    spark = SparkSession.builder.master("local[*]").appName("efm_local").getOrCreate()
    random.seed(42 if split == "train" else 99)

    rows = []
    for _ in range(n_earners):
        seq_len = random.randint(10, MAX_LEN)

        def seq_float(lo=0.0, hi=100.0):
            # Padded float sequences: real values then zeros for padding
            return [random.uniform(lo, hi) for _ in range(seq_len)] + [0.0] * (MAX_LEN - seq_len)

        def seq_int(lo, hi):
            return [random.randint(lo, hi) for _ in range(seq_len)] + [0] * (MAX_LEN - seq_len)

        rows.append({
            # Hash categoricals (pre-computed by DSL)
            "derived_city_id_hashed": seq_int(0, 999),
            "derived_earner_primary_lob_hashed": seq_int(0, 9),
            "derived_supply_pause_reason_hashed": seq_int(0, 9),
            "derived_offer_lob_hashed": seq_int(0, 9),
            "derived_offer_type_hashed": seq_int(0, 9),
            "derived_offer_canceled_feedback_type_id_hashed": seq_int(0, 9),
            "derived_offer_canceled_reason_hashed": seq_int(0, 9),
            "derived_offer_canceled_status_hashed": seq_int(0, 9),
            "derived_pickup_h9_hashed": seq_int(0, 9999),
            "derived_dropoff_h9_hashed": seq_int(0, 9999),
            "derived_event_type_hashed": seq_int(0, 19),
            "derived_pickup_geohash_long": seq_int(0, 9999),
            # Standard categoricals (indexed by DSL StringIndexer)
            "derived_event_type_indexed": seq_int(0, EVENT_TYPE_VOCAB_SIZE - 1),
            "derived_hour_of_day_local": seq_int(0, 23),
            "derived_day_of_week_local": seq_int(0, 6),
            "derived_minute_of_hour_local": seq_int(0, 59),
            # Raw padded numerical features (inputs to native transform)
            "derived_offer_earner_eta_padded": seq_float(0, 3600),
            "eats_fare_meal_subtotal_padded": seq_float(0, 200),
            "eats_fare_courier_gross_fare_padded": seq_float(0, 50),
            "eats_fare_co_funded_promo_uber_amt_padded": seq_float(0, 30),
            "eats_fare_delivery_fee_premium_padded": seq_float(0, 20),
            "rides_fare_driver_upfront_fare_padded": seq_float(0, 100),
            "rides_fare_driver_surge_padded": seq_float(0, 5),
            "earner_tenure_padded": seq_float(0, 1000),
            "earner_app_rating_padded": seq_float(0, 5),
            "seconds_since_supply_online_padded": seq_float(0, 86400),
            "consecutive_idle_days_padded": seq_float(0, 30),
            # Raw padded geo features (inputs to native transform)
            "derived_pickup_lat_padded": seq_float(-90, 90),
            "derived_pickup_lng_padded": seq_float(-180, 180),
            "derived_dropoff_lat_padded": seq_float(-90, 90),
            "derived_dropoff_lng_padded": seq_float(-180, 180),
            "derived_event_lat_padded": seq_float(-90, 90),
            "derived_event_lng_padded": seq_float(-180, 180),
            # Time features for bucketization
            "derived_time_to_next_seconds_padded": seq_float(-1, 604800),
            "time_since_last_seconds_padded": seq_float(-1, 604800),
            "event_number_norm_padded": seq_float(0, 1),
            # Scalar
            "derived_sequence_length": seq_len,
            # Labels
            "response_next_event_type_indexed": seq_int(0, EVENT_TYPE_VOCAB_SIZE - 1),
            "response_churned": [random.choice([-1, 0, 1]) for _ in range(MAX_LEN)],
        })

    df = spark.createDataFrame(rows)
    dv = DatasetVariable.create(df)
    dv.save_spark_dataframe()
    return dv


if __name__ == "__main__":
    import os
    import tempfile

    ctx = uniflow.create_context()
    ctx.environ["MA_NAMESPACE"] = "local"

    _local_storage = os.path.join(tempfile.gettempdir(), "uf_storage_efm")
    os.environ["UF_STORAGE_URL"] = f"file://{_local_storage}"
    os.makedirs(_local_storage, exist_ok=True)

    if ctx.is_local_run():
        TRAIN_CONFIG.save_model_config.model_dir = os.path.expanduser("~/efm_checkpoints")
        train_dv = _make_synthetic_dataset(n_earners=200, split="train")
        val_dv = _make_synthetic_dataset(n_earners=50, split="val")
    else:
        raise RuntimeError(
            "Remote-run: pass train_data and val_data from the DSL/tabular_transform output."
        )

    ctx.run(train_workflow, train_data=train_dv, val_data=val_dv)
