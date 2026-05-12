"""Pipeline configuration for the earner foundation model.

Mirrors the production pipeline_conf.yaml + native_transform_specs.yaml.
All concrete values are defined here and passed to the tasks.

Production reference:
  uber/product/earner/earner_access_ml/earner_foundation/pipelines/train/
  earner_foundation_model/v1/pipeline_conf.yaml
  earner_foundation_model/v1/native_transform_specs.yaml
"""

from pydantic import BaseModel


class TrainParams(BaseModel):
    batch_size: int = 1024
    num_epochs: int = 5
    learning_rate: float = 1e-3
    warmup_steps: int = 195
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    early_stopping_patience: int = 5


class SaveModelConfig(BaseModel):
    model_dir: str
    project_name: str
    experiment_name: str


class TrainConfig(BaseModel):
    embedding_config: dict
    architecture_config: dict
    task_config: dict
    forward_output_fields: list[str]
    train_params: TrainParams
    save_model_config: SaveModelConfig
    eval_callback_fn: str | None = None
    eval_callback_every_n_epochs: int = 1


# ---------------------------------------------------------------------------
# Shared constants (local-run values; production uses max_len=900)
# ---------------------------------------------------------------------------
MAX_LEN = 100
EVENT_TYPE_VOCAB_SIZE = 25  # StringIndexer distinct values + 1 (0 = padding)
TIME_BUCKET_COUNT = 8       # 7 real bins + 1 padding bin (padding=-1 → bucket 0)
NUM_NUMERICAL = 11
NUM_GEO = 6

# ---------------------------------------------------------------------------
# Native transform config  (mirrors native_transform_specs.yaml)
# ---------------------------------------------------------------------------
# Boundaries for time-based buckets (seconds): 0=padding, 1-7=real intervals
_TIME_BOUNDARIES = [0.0, 60.0, 300.0, 1800.0, 7200.0, 86400.0, 604800.0]
# Boundaries for event_number_norm (10 equal-width bins)
_NORM_BOUNDARIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

NATIVE_TRANSFORM_CONFIG = {
    "transform_specs": [
        # --- LogTransform numerical features ---
        {
            "transform_name": "LogTransform",
            "input_cols": [
                "derived_offer_earner_eta_padded",
                "eats_fare_meal_subtotal_padded",
                "eats_fare_courier_gross_fare_padded",
                "eats_fare_co_funded_promo_uber_amt_padded",
                "eats_fare_delivery_fee_premium_padded",
                "rides_fare_driver_upfront_fare_padded",
                "rides_fare_driver_surge_padded",
                "earner_tenure_padded",
                "earner_app_rating_padded",
                "seconds_since_supply_online_padded",
                "consecutive_idle_days_padded",
            ],
            "output_cols": [
                "derived_offer_earner_eta_log",
                "derived_eats_fare_meal_subtotal_log",
                "derived_eats_fare_courier_gross_fare_log",
                "derived_eats_fare_co_funded_promo_uber_amt_log",
                "derived_eats_fare_delivery_fee_premium_log",
                "derived_rides_fare_driver_upfront_fare_log",
                "derived_rides_fare_driver_surge_log",
                "derived_earner_tenure_log",
                "derived_earner_app_rating_log",
                "derived_seconds_since_supply_online_log",
                "derived_consecutive_idle_days_log",
            ],
            "add_constant": 1.0,
        },
        # --- Normalization (one per column) ---
        {"transform_name": "Normalization", "input_cols": ["derived_offer_earner_eta_log"],
         "output_cols": ["derived_offer_earner_eta_log_scaled"], "mean": [3.75], "std": [1.5]},
        {"transform_name": "Normalization", "input_cols": ["derived_eats_fare_meal_subtotal_log"],
         "output_cols": ["derived_eats_fare_meal_subtotal_log_scaled"], "mean": [2.65], "std": [1.0]},
        {"transform_name": "Normalization", "input_cols": ["derived_eats_fare_courier_gross_fare_log"],
         "output_cols": ["derived_eats_fare_courier_gross_fare_log_scaled"], "mean": [1.95], "std": [0.8]},
        {"transform_name": "Normalization", "input_cols": ["derived_eats_fare_co_funded_promo_uber_amt_log"],
         "output_cols": ["derived_eats_fare_co_funded_promo_uber_amt_log_scaled"], "mean": [1.75], "std": [0.7]},
        {"transform_name": "Normalization", "input_cols": ["derived_eats_fare_delivery_fee_premium_log"],
         "output_cols": ["derived_eats_fare_delivery_fee_premium_log_scaled"], "mean": [1.55], "std": [0.6]},
        {"transform_name": "Normalization", "input_cols": ["derived_rides_fare_driver_upfront_fare_log"],
         "output_cols": ["derived_rides_fare_driver_upfront_fare_log_scaled"], "mean": [2.5], "std": [1.0]},
        {"transform_name": "Normalization", "input_cols": ["derived_rides_fare_driver_surge_log"],
         "output_cols": ["derived_rides_fare_driver_surge_log_scaled"], "mean": [0.5], "std": [0.5]},
        {"transform_name": "Normalization", "input_cols": ["derived_earner_tenure_log"],
         "output_cols": ["derived_earner_tenure_log_scaled"], "mean": [5.5], "std": [1.5]},
        {"transform_name": "Normalization", "input_cols": ["derived_earner_app_rating_log"],
         "output_cols": ["derived_earner_app_rating_log_scaled"], "mean": [1.6], "std": [0.2]},
        {"transform_name": "Normalization", "input_cols": ["derived_seconds_since_supply_online_log"],
         "output_cols": ["derived_seconds_since_supply_online_log_scaled"], "mean": [4.0], "std": [2.0]},
        {"transform_name": "Normalization", "input_cols": ["derived_consecutive_idle_days_log"],
         "output_cols": ["derived_consecutive_idle_days_log_scaled"], "mean": [0.5], "std": [0.5]},
        # --- Stack all 11 numerical features ---
        {
            "transform_name": "Stack",
            "input_cols": [
                "derived_offer_earner_eta_log_scaled",
                "derived_eats_fare_meal_subtotal_log_scaled",
                "derived_eats_fare_courier_gross_fare_log_scaled",
                "derived_eats_fare_co_funded_promo_uber_amt_log_scaled",
                "derived_eats_fare_delivery_fee_premium_log_scaled",
                "derived_rides_fare_driver_upfront_fare_log_scaled",
                "derived_rides_fare_driver_surge_log_scaled",
                "derived_earner_tenure_log_scaled",
                "derived_earner_app_rating_log_scaled",
                "derived_seconds_since_supply_online_log_scaled",
                "derived_consecutive_idle_days_log_scaled",
            ],
            "output_cols": ["derived_numerical_stacked"],
            "dim": -1,
        },
        # --- MinMax geo lat/lng ---
        {"transform_name": "MinMax", "input_cols": ["derived_pickup_lat_padded"],
         "output_cols": ["derived_pickup_lat_scaled"], "min": [-90.0], "max": [90.0]},
        {"transform_name": "MinMax", "input_cols": ["derived_pickup_lng_padded"],
         "output_cols": ["derived_pickup_lng_scaled"], "min": [-180.0], "max": [180.0]},
        {"transform_name": "MinMax", "input_cols": ["derived_dropoff_lat_padded"],
         "output_cols": ["derived_dropoff_lat_scaled"], "min": [-90.0], "max": [90.0]},
        {"transform_name": "MinMax", "input_cols": ["derived_dropoff_lng_padded"],
         "output_cols": ["derived_dropoff_lng_scaled"], "min": [-180.0], "max": [180.0]},
        {"transform_name": "MinMax", "input_cols": ["derived_event_lat_padded"],
         "output_cols": ["derived_event_lat_scaled"], "min": [-90.0], "max": [90.0]},
        {"transform_name": "MinMax", "input_cols": ["derived_event_lng_padded"],
         "output_cols": ["derived_event_lng_scaled"], "min": [-180.0], "max": [180.0]},
        # --- Stack 6 geo features ---
        {
            "transform_name": "Stack",
            "input_cols": [
                "derived_pickup_lat_scaled", "derived_pickup_lng_scaled",
                "derived_dropoff_lat_scaled", "derived_dropoff_lng_scaled",
                "derived_event_lat_scaled", "derived_event_lng_scaled",
            ],
            "output_cols": ["derived_geo_stacked"],
            "dim": -1,
        },
        # --- Bucketize time targets and positional features ---
        {
            "transform_name": "Bucketization",
            "input_cols": ["derived_time_to_next_seconds_padded"],
            "output_cols": ["response_time_to_next_event_bucket"],
            "boundaries": _TIME_BOUNDARIES,
        },
        {
            "transform_name": "Bucketization",
            "input_cols": ["time_since_last_seconds_padded"],
            "output_cols": ["derived_time_since_last_event_bucket"],
            "boundaries": _TIME_BOUNDARIES,
        },
        {
            "transform_name": "Bucketization",
            "input_cols": ["event_number_norm_padded"],
            "output_cols": ["derived_event_number_norm_bucket"],
            "boundaries": _NORM_BOUNDARIES,
        },
    ],
    # Keep only training-relevant columns; drops all *_padded intermediates.
    "columns_to_keep": [
        # Hash categoricals
        "derived_city_id_hashed",
        "derived_earner_primary_lob_hashed",
        "derived_supply_pause_reason_hashed",
        "derived_offer_lob_hashed",
        "derived_offer_type_hashed",
        "derived_offer_canceled_feedback_type_id_hashed",
        "derived_offer_canceled_reason_hashed",
        "derived_offer_canceled_status_hashed",
        "derived_pickup_h9_hashed",
        "derived_dropoff_h9_hashed",
        "derived_event_type_hashed",
        "derived_pickup_geohash_long",
        # Standard categoricals
        "derived_event_type_indexed",
        "derived_hour_of_day_local",
        "derived_day_of_week_local",
        "derived_minute_of_hour_local",
        "derived_time_since_last_event_bucket",
        "derived_event_number_norm_bucket",
        # Stacked features (output of transforms)
        "derived_numerical_stacked",
        "derived_geo_stacked",
        # Scalar
        "derived_sequence_length",
        # Labels
        "response_next_event_type_indexed",
        "response_time_to_next_event_bucket",
        "response_churned",
    ],
}

# ---------------------------------------------------------------------------
# Trainer config  (mirrors pipeline_conf.yaml tabular_trainer section)
# ---------------------------------------------------------------------------
TRAIN_CONFIG = TrainConfig(
    embedding_config={
        "hash_categoricals": [
            ["derived_city_id_hashed", 1000, 64],
            ["derived_earner_primary_lob_hashed", 10, 64],
            ["derived_supply_pause_reason_hashed", 10, 64],
            ["derived_offer_lob_hashed", 10, 64],
            ["derived_offer_type_hashed", 10, 64],
            ["derived_offer_canceled_feedback_type_id_hashed", 10, 64],
            ["derived_offer_canceled_reason_hashed", 10, 64],
            ["derived_offer_canceled_status_hashed", 10, 64],
            ["derived_pickup_h9_hashed", 10000, 64],
            ["derived_dropoff_h9_hashed", 10000, 64],
            ["derived_event_type_hashed", 20, 64],
            ["derived_pickup_geohash_long", 10000, 64],
        ],
        "categoricals": [
            ["derived_event_type_indexed", EVENT_TYPE_VOCAB_SIZE, 16],
            ["derived_hour_of_day_local", 25, 8],
            ["derived_day_of_week_local", 10, 4],
            ["derived_minute_of_hour_local", 61, 8],
            ["derived_time_since_last_event_bucket", TIME_BUCKET_COUNT, 8],
            ["derived_event_number_norm_bucket", 11, 16],
        ],
        "numerical": [["derived_numerical_stacked", 64, 32, NUM_NUMERICAL]],
        "geo": [["derived_geo_stacked", 64, 32, NUM_GEO]],
    },
    architecture_config={
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "dropout": 0.1,
        "max_len": MAX_LEN,
        "pos_encoding": "sinusoidal",
    },
    task_config={
        "next_event_type": {
            "task_type": "classification",
            "mask_type": "token",
            "target_key": "response_next_event_type_indexed",
            "num_classes": EVENT_TYPE_VOCAB_SIZE,
            "hidden_dims": [64],
            "weight": 1.0,
        },
        "time_to_next_event": {
            "task_type": "classification",
            "mask_type": "padding",
            "target_key": "response_time_to_next_event_bucket",
            "num_classes": TIME_BUCKET_COUNT,
            "hidden_dims": [32, 16],
            "weight": 0.5,
        },
        "churn": {
            "task_type": "classification",
            "mask_type": "sequence",
            "target_key": "response_churned",
            "num_classes": 2,
            "hidden_dims": [32],
            "weight": 0.5,
        },
    },
    forward_output_fields=[
        "pred_churn_logits",
        "pred_embedding",
        "pred_next_event_type_indexed_logits",
        "pred_time_to_next_event_bucket_logits",
    ],
    train_params=TrainParams(
        batch_size=1024,
        num_epochs=5,
        learning_rate=1e-3,
        warmup_steps=195,
        weight_decay=0.01,
        gradient_clip=1.0,
        early_stopping_patience=5,
    ),
    save_model_config=SaveModelConfig(
        model_dir="s3://your-bucket/models",
        project_name="earner-foundation-model",
        experiment_name="v1",
    ),
    eval_callback_fn="michelangelo.lib.foundation_model.callbacks.acceptance_rate.acceptance_rate_eval",
    eval_callback_every_n_epochs=5,
)
