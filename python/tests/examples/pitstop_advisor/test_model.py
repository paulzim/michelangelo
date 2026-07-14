"""Save/load/predict round-trip test for PitStopAdvisorModel.

Trains tiny boosters directly with xgboost (not via train.py's `train` task,
which also does S3 upload + model registry calls that need a live sandbox)
to isolate the round trip this test actually cares about: does
PitStopAdvisorModel.save() -> PitStopAdvisorModel.load() -> .predict()
reproduce the same predictions as the in-memory model.
"""

from __future__ import annotations

import numpy as np
import xgboost

from examples.pipelines.pitstop_advisor.model import PitStopAdvisorModel


def _train_tiny_booster(seed: int) -> xgboost.Booster:
    rng = np.random.default_rng(seed)
    features = rng.uniform(0.0, 1.0, size=(20, 2)).astype(np.float32)
    labels = (features[:, 0] * 10 + features[:, 1] * 5).astype(np.float32)
    dtrain = xgboost.DMatrix(features, label=labels)
    return xgboost.train(
        {"objective": "reg:squarederror", "max_depth": 2},
        dtrain=dtrain,
        num_boost_round=5,
    )


def test_save_load_predict_round_trip(tmp_path):
    """save() -> load() -> predict() reproduces the pre-save prediction."""
    speed_booster = _train_tiny_booster(seed=1)
    caution_booster = _train_tiny_booster(seed=2)
    model = PitStopAdvisorModel(
        speed_booster=speed_booster, caution_booster=caution_booster
    )

    features = np.array([1.0, 0.75], dtype=np.float32)
    expected = model.predict({"features": features})

    model.save(str(tmp_path))
    loaded = PitStopAdvisorModel.load(str(tmp_path))
    actual = loaded.predict({"features": features})

    assert actual.keys() == expected.keys()
    np.testing.assert_allclose(actual["settings"], expected["settings"])


def test_predict_output_shape_and_dtype():
    """predict() returns a float32 "settings" array of shape (2,)."""
    model = PitStopAdvisorModel(
        speed_booster=_train_tiny_booster(seed=3),
        caution_booster=_train_tiny_booster(seed=4),
    )

    result = model.predict({"features": np.array([0.0, 0.5], dtype=np.float32)})

    assert result["settings"].shape == (2,)
    assert result["settings"].dtype == np.float32
