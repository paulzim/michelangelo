"""Feature preparation Spark task for the foundation model.

Builds vocabularies, computes numerical statistics, encodes features,
creates fixed-length sequences with multitask targets, and splits
train/validation datasets.
"""

import json
import logging
from typing import Any

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window

from michelangelo.uniflow.core.decorator import task
from michelangelo.uniflow.plugins.spark.task import SparkTask
from michelangelo.workflow.variables import DatasetVariable

logger = logging.getLogger(__name__)


def _multitask_prep_aux_paths(vocab_var_path: str) -> tuple[str, str]:
    base = str(vocab_var_path).rstrip("/")
    aux = f"{base}_prep_aux"
    return f"{aux}/numerical_stats", f"{aux}/geo_bounds"


def _save_prep_auxiliary_artifacts(vocab_var_path, stats_df, geo_bounds, spark):
    stats_path, geo_path = _multitask_prep_aux_paths(vocab_var_path)
    stats_df.write.mode("overwrite").parquet(stats_path)
    rows = [(str(k), float(v) if v is not None else None) for k, v in geo_bounds.items()]
    schema = StructType([
        StructField("bound_key", StringType(), False),
        StructField("bound_value", DoubleType(), True),
    ])
    spark.createDataFrame(rows, schema=schema).write.mode("overwrite").parquet(geo_path)
    logger.info("Saved prep auxiliary artifacts: stats=%s geo=%s", stats_path, geo_path)


def _load_prep_auxiliary_artifacts(spark, vocab_var_path):
    stats_path, geo_path = _multitask_prep_aux_paths(vocab_var_path)
    stats_df = spark.read.parquet(stats_path)
    geo_bounds = {row.bound_key: row.bound_value for row in spark.read.parquet(geo_path).collect()}
    return stats_df, geo_bounds


def calculate_embedding_dim(vocab_size: int, min_dim: int = 8, max_dim: int = 256) -> int:
    if vocab_size > 10000:
        dim = min(max_dim, vocab_size // 50)
    elif vocab_size > 1000:
        dim = min(128, vocab_size // 20)
    elif vocab_size > 100:
        dim = min(64, vocab_size // 5)
    else:
        dim = min(32, max(min_dim, vocab_size // 2))
    return 2 ** int(np.log2(max(dim, 1)))


def _inject_idle_day_events(df, config, dataset_end_date):
    dataset_end_dt = F.to_date(F.lit(str(dataset_end_date)))
    active_dates = df.select(
        config.earner_column, F.to_date(config.timestamp_column).alias("date")
    ).distinct()
    earner_meta = df.groupBy(config.earner_column).agg(
        F.to_date(F.min(config.timestamp_column)).alias("first_active_date"),
        F.first(F.coalesce(F.col("churned"), F.lit(-1))).alias("churned"),
    )
    all_days = earner_meta.select(
        config.earner_column,
        F.explode(F.sequence(F.col("first_active_date"), dataset_end_dt, F.expr("INTERVAL 1 DAY"))).alias("date"),
        F.col("churned"),
    )
    idle_days = all_days.join(active_dates, on=[config.earner_column, "date"], how="left_anti")
    idle_events = idle_days.select(
        F.col(config.earner_column),
        F.lit("IDLE_DAY").alias("event_type"),
        F.col("date").cast("timestamp").alias(config.timestamp_column),
        F.col("churned"),
    )
    return df.unionByName(idle_events, allowMissingColumns=True)


def _build_vocabularies(df, config):
    spark = SparkSession.getActiveSession()
    vocab_rows = []

    for feat in config.categorical_features:
        if feat not in df.columns:
            continue
        unique_df = (
            df.select(feat).filter(F.col(feat).isNotNull())
            .groupBy(feat).count().orderBy(F.desc("count"))
        )
        unique_count = unique_df.count()
        if unique_count > 100000:
            unique_df = unique_df.limit(100000)
        unique_values = [row[feat] for row in unique_df.collect()]

        vocab = dict(config.special_tokens)
        for idx, value in enumerate(unique_values, start=len(config.special_tokens)):
            vocab[str(value)] = idx

        vocab_size = len(vocab)
        embed_dim = calculate_embedding_dim(vocab_size)
        vocab_rows.append((feat, json.dumps(vocab), vocab_size, embed_dim, "categorical"))
        logger.info("  %s: %d tokens, embed_dim=%d", feat, vocab_size, embed_dim)

    # Numerical embedding config
    if config.numerical_features:
        all_num_names = list(config.numerical_features)
        for derived in ["seconds_since_supply_online", "consecutive_idle_days"]:
            if derived not in all_num_names:
                all_num_names.append(derived)
        emb_cfg = {
            "type": "numerical",
            "num_features": len(all_num_names),
            "hidden_dim": config.embedding_dims.numerical_hidden,
            "output_dim": config.embedding_dims.numerical_output,
            "feature_names": all_num_names,
        }
        vocab_rows.append(("numerical", json.dumps(emb_cfg), 0, config.embedding_dims.numerical_output, "numerical"))

    # Geo embedding config
    total_geo = sum(len(cfg.features) for cfg in config.geo_features.values())
    if total_geo > 0:
        geo_names = [f for cfg in config.geo_features.values() for f in cfg.features]
        emb_cfg = {
            "type": "geo",
            "num_features": total_geo,
            "hidden_dim": config.embedding_dims.geo_hidden,
            "output_dim": config.embedding_dims.geo_output,
            "feature_names": geo_names,
        }
        vocab_rows.append(("geo", json.dumps(emb_cfg), 0, config.embedding_dims.geo_output, "geo"))

    schema = StructType([
        StructField("feature_name", StringType(), False),
        StructField("vocab_json", StringType(), False),
        StructField("vocab_size", IntegerType(), False),
        StructField("embedding_dim", IntegerType(), False),
        StructField("feature_type", StringType(), False),
    ])
    return spark.createDataFrame(vocab_rows, schema)


def _compute_numerical_stats(df, numerical_features):
    spark = SparkSession.getActiveSession()
    agg_exprs = []
    for feat in numerical_features:
        if feat in df.columns:
            agg_exprs.extend([F.mean(feat).alias(f"{feat}_mean"), F.stddev(feat).alias(f"{feat}_std")])
    if not agg_exprs:
        return spark.createDataFrame([], StructType([
            StructField("feature_name", StringType()),
            StructField("mean", DoubleType()),
            StructField("std", DoubleType()),
        ]))
    stats = df.select(*agg_exprs).collect()[0]
    rows = []
    for feat in numerical_features:
        if feat in df.columns:
            mean = stats[f"{feat}_mean"] or 0.0
            std = stats[f"{feat}_std"] if stats[f"{feat}_std"] and stats[f"{feat}_std"] > 0 else 1.0
            rows.append((feat, float(mean), float(std)))
    return spark.createDataFrame(rows, StructType([
        StructField("feature_name", StringType(), False),
        StructField("mean", DoubleType(), False),
        StructField("std", DoubleType(), False),
    ]))


def _compute_geo_bounds(df, geo_features_config):
    lat_cols, lng_cols = [], []
    for group_config in geo_features_config.values():
        for feat in group_config.features:
            if "lat" in feat and feat in df.columns:
                lat_cols.append(feat)
            elif "lng" in feat and feat in df.columns:
                lng_cols.append(feat)
    if not lat_cols and not lng_cols:
        return {}
    agg_exprs = []
    for col in lat_cols + lng_cols:
        agg_exprs += [
            F.expr(f"percentile_approx({col}, 0.01)").alias(f"{col}_min"),
            F.expr(f"percentile_approx({col}, 0.99)").alias(f"{col}_max"),
        ]
    bounds_result = df.select(*agg_exprs).collect()[0]
    return {f"{col}_{side}": bounds_result[f"{col}_{side}"] for col in lat_cols + lng_cols for side in ("min", "max")}


def _encode_features(df, vocabularies_df, stats_df, geo_bounds, config):
    vocab_map = {row.feature_name: json.loads(row.vocab_json) for row in vocabularies_df.collect()}
    vocab_bc = df.sql_ctx.sparkSession.sparkContext.broadcast(vocab_map)
    stats_map = {row.feature_name: {"mean": row.mean, "std": row.std} for row in stats_df.collect()}

    @F.udf(returnType=IntegerType())
    def encode_cat(value, feature_name):
        if value is None:
            return vocab_bc.value.get(feature_name, {}).get("PAD", 0)
        vocab = vocab_bc.value.get(feature_name, {})
        return vocab.get(str(value), vocab.get("UNK", 1))

    for feat in config.categorical_features:
        if feat in df.columns:
            df = df.withColumn(f"{feat}_encoded", encode_cat(F.col(feat), F.lit(feat)))

    for feat in config.numerical_features:
        if feat in df.columns:
            s = stats_map.get(feat, {"mean": 0.0, "std": 1.0})
            df = df.withColumn(
                f"{feat}_normalized",
                F.when(
                    F.col("event_type") != "IDLE_DAY",
                    (F.coalesce(F.col(feat), F.lit(0.0)) - F.lit(s["mean"])) / F.lit(s["std"]),
                ).otherwise(F.lit(0.0)),
            )

    for group_config in config.geo_features.values():
        for feat in group_config.features:
            if feat not in df.columns:
                continue
            if "lat" in feat or "lng" in feat:
                min_val = geo_bounds.get(f"{feat}_min", 0.0)
                max_val = geo_bounds.get(f"{feat}_max", 1.0)
                range_val = max_val - min_val if max_val > min_val else 1.0
                df = df.withColumn(
                    f"{feat}_normalized",
                    F.when(F.col(feat).isNotNull(), (F.col(feat) - F.lit(min_val)) / F.lit(range_val)).otherwise(0.0),
                )

    # Sequential features
    window_spec = Window.partitionBy(config.earner_column).orderBy(config.timestamp_column)
    df = df.withColumn("row_num", F.row_number().over(window_spec))
    max_row = F.max("row_num").over(Window.partitionBy(config.earner_column))
    df = df.withColumn(
        "event_number_norm",
        F.when(max_row > 1, (F.col("row_num") - 1) / (max_row - 1)).otherwise(0.0),
    )

    # seconds_since_supply_online
    is_supply = F.when(F.col("event_type") == "supply_online", F.lit(1)).otherwise(F.lit(0))
    session_group = F.sum(is_supply).over(window_spec)
    session_window = Window.partitionBy(config.earner_column, session_group).orderBy(config.timestamp_column)
    session_start_ts = F.first(config.timestamp_column).over(session_window)
    df = df.withColumn(
        "seconds_since_supply_online_normalized",
        F.coalesce(
            F.log1p((F.unix_timestamp(F.col(config.timestamp_column)) - F.unix_timestamp(session_start_ts)).cast("double")),
            F.lit(0.0),
        ),
    )

    # consecutive_idle_days
    last_real_ts = F.last(
        F.when(F.col("event_type") != "IDLE_DAY", F.col(config.timestamp_column)), ignorenulls=True
    ).over(window_spec)
    df = df.withColumn(
        "consecutive_idle_days_normalized",
        F.coalesce(
            F.log1p(F.when(F.col("event_type") != "IDLE_DAY", F.lit(0)).otherwise(
                F.datediff(F.to_date(F.col(config.timestamp_column)), F.to_date(last_real_ts)).cast("double")
            )),
            F.lit(0.0),
        ),
    )

    # next_event_type target
    df = df.withColumn("next_event_type_encoded", F.lead("event_type_encoded", 1).over(window_spec))
    return df


def _create_sequences(df, config):
    window_desc = Window.partitionBy(config.earner_column).orderBy(F.desc(config.timestamp_column))
    df = df.withColumn("event_rank", F.row_number().over(window_desc))
    df = df.filter(F.col("event_rank") <= config.max_events_per_earner)

    window_asc = Window.partitionBy(config.earner_column).orderBy(config.timestamp_column)
    df = df.withColumn("seq_order", F.row_number().over(window_asc))

    agg_exprs = []
    for feat in config.categorical_features:
        col_name = f"{feat}_encoded"
        if col_name in df.columns:
            agg_exprs.append(F.collect_list(col_name).alias(f"{feat}_seq"))

    num_cols = list(config.numerical_features) + ["seconds_since_supply_online", "consecutive_idle_days"]
    for feat in num_cols:
        col_name = f"{feat}_normalized"
        if col_name in df.columns:
            agg_exprs.append(F.collect_list(col_name).alias(f"{feat}_seq"))

    # Collect geo feature sequences
    geo_cols = [feat for group_cfg in config.geo_features.values() for feat in group_cfg.features]
    for feat in geo_cols:
        col_name = f"{feat}_normalized"
        if col_name in df.columns:
            agg_exprs.append(F.collect_list(col_name).alias(f"{feat}_seq"))

    if "next_event_type_encoded" in df.columns:
        agg_exprs.append(F.collect_list("next_event_type_encoded").alias("next_event_type_seq"))
    if "churned" in df.columns:
        agg_exprs.append(F.first(F.coalesce(F.col("churned"), F.lit(-1))).alias("churn_label"))
    agg_exprs.append(F.count("*").alias("total_events"))

    df = df.orderBy(config.earner_column, "seq_order")
    earner_data = df.groupBy(config.earner_column).agg(*agg_exprs)
    sequences_df = earner_data.filter(F.col("total_events") >= config.min_seq_length)

    PAD = config.special_tokens.get("PAD", 0)
    max_len = config.max_seq_length

    @F.udf(returnType=ArrayType(LongType()))
    def pad_int(arr, pad_value, target_length):
        arr_list = list(arr) if arr else []
        arr_list = arr_list[-target_length:]
        return arr_list + [pad_value] * (target_length - len(arr_list))

    @F.udf(returnType=ArrayType(DoubleType()))
    def pad_float(arr, pad_value, target_length):
        arr_list = list(arr) if arr else []
        arr_list = arr_list[-target_length:]
        return arr_list + [float(pad_value)] * (target_length - len(arr_list))

    for feat in config.categorical_features:
        col_name = f"{feat}_seq"
        if col_name in sequences_df.columns:
            sequences_df = sequences_df.withColumn(
                feat, pad_int(F.col(col_name), F.lit(PAD), F.lit(max_len))
            ).drop(col_name)

    for feat in num_cols:
        col_name = f"{feat}_seq"
        if col_name in sequences_df.columns:
            sequences_df = sequences_df.withColumn(
                feat, pad_float(F.col(col_name), F.lit(0.0), F.lit(max_len))
            ).drop(col_name)

    for feat in geo_cols:
        col_name = f"{feat}_seq"
        if col_name in sequences_df.columns:
            sequences_df = sequences_df.withColumn(
                feat, pad_float(F.col(col_name), F.lit(0.0), F.lit(max_len))
            ).drop(col_name)

    if "next_event_type_seq" in sequences_df.columns:
        sequences_df = sequences_df.withColumn(
            "next_event_type_target", pad_int(F.col("next_event_type_seq"), F.lit(PAD), F.lit(max_len))
        ).drop("next_event_type_seq")

    sequences_df = sequences_df.withColumn(
        "attention_mask",
        F.expr(f"transform(sequence(1, {max_len}), x -> CASE WHEN x <= total_events THEN 1 ELSE 0 END)"),
    )
    sequences_df = sequences_df.withColumn("sequence_length", F.col("total_events"))
    sequences_df = sequences_df.withColumnRenamed("sequence_length", "derived_sequence_length")
    return sequences_df


def _split_train_val(sequences_df, config):
    sequences_df = sequences_df.withColumn("split_hash", F.abs(F.hash(F.col(config.earner_column))) % 100)
    threshold = int(config.train_split_ratio * 100)
    return (
        sequences_df.filter(F.col("split_hash") < threshold).drop("split_hash"),
        sequences_df.filter(F.col("split_hash") >= threshold).drop("split_hash"),
    )


@task(config=SparkTask())
def feature_prep_task(config, event_data: DatasetVariable) -> dict[str, Any]:
    """Spark task: vocabulary building, feature encoding, sequence creation, train/val split."""
    logger.info("=" * 60)
    logger.info("FOUNDATION MODEL FEATURE PREPARATION")
    logger.info("=" * 60)

    event_data.load_spark_dataframe()
    df = event_data.value
    logger.info("Loaded %d events", df.count())

    dataset_end_date = df.select(F.max("event_timestamp")).collect()[0][0]
    df = _inject_idle_day_events(df, config, dataset_end_date)

    vocabularies_df = _build_vocabularies(df, config)
    vocabularies_df.cache()

    stats_df = _compute_numerical_stats(df, config.numerical_features)
    stats_df.cache()

    geo_bounds = _compute_geo_bounds(df, config.geo_features)

    df_encoded = _encode_features(df, vocabularies_df, stats_df, geo_bounds, config)
    df_encoded.cache()

    sequences_df = _create_sequences(df_encoded, config)
    train_df, val_df = _split_train_val(sequences_df, config)

    logger.info("Train: %d sequences  Val: %d sequences", train_df.count(), val_df.count())

    vocab_var = DatasetVariable.create(vocabularies_df)
    vocab_var.save_spark_dataframe()

    spark = SparkSession.getActiveSession()
    _save_prep_auxiliary_artifacts(vocab_var.path, stats_df, geo_bounds, spark)

    train_var = DatasetVariable.create(train_df)
    train_var.save_spark_dataframe()

    val_var = DatasetVariable.create(val_df)
    val_var.save_spark_dataframe()

    return {"vocabularies": vocab_var, "train_dataset": train_var, "val_dataset": val_var}
