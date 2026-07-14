"""Pit Crew Advisor model: recommends pit-lane settings for the KubeCon demo.

Implements the ``Model`` interface (save/load/predict) that
``CustomTritonPackager`` packages for serving. The input/output tensor
contract here — a single ``"features"`` input of shape ``[2]`` and a single
``"settings"`` output of shape ``[2]`` — must exactly match the constants
hardcoded in ``go/components/lanerun/controller.go`` (``advisorInputName``,
``advisorOutputName``), since the LaneRun controller builds its KServe v2
request against this contract, not against the schema at runtime.
"""

from __future__ import annotations

import os

import numpy as np

from michelangelo.lib.model_manager.interface.custom_model import Model

__all__ = ["PitStopAdvisorModel"]

_SPEED_MODEL_FILENAME = "speed_cap_cms.ubj"
_CAUTION_MODEL_FILENAME = "caution_buffer_cm.ubj"


class PitStopAdvisorModel(Model):
    """Two independent XGBoost regressors sharing one [lane_feature, track_grip] input.

    speed_cap_cms and caution_buffer_cm are trained as separate single-output
    regressors rather than one multi-output model — simpler to save/load with
    plain XGBoost booster checkpoints and avoids depending on a specific
    XGBoost version's multi-output support.
    """

    def __init__(self, speed_booster=None, caution_booster=None) -> None:
        """Wrap two pre-trained boosters, or leave both None until `load()`."""
        self._speed_booster = speed_booster
        self._caution_booster = caution_booster

    def save(self, path: str) -> None:
        """Persist both boosters as XGBoost's binary (.ubj) checkpoint format."""
        os.makedirs(path, exist_ok=True)
        self._speed_booster.save_model(os.path.join(path, _SPEED_MODEL_FILENAME))
        self._caution_booster.save_model(os.path.join(path, _CAUTION_MODEL_FILENAME))

    @classmethod
    def load(cls, path: str) -> PitStopAdvisorModel:
        """Load both boosters from artifacts under `path`."""
        import xgboost

        speed_booster = xgboost.Booster()
        speed_booster.load_model(os.path.join(path, _SPEED_MODEL_FILENAME))

        caution_booster = xgboost.Booster()
        caution_booster.load_model(os.path.join(path, _CAUTION_MODEL_FILENAME))

        return cls(speed_booster=speed_booster, caution_booster=caution_booster)

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Predict [speed_cap_cms, caution_buffer_cm] from a [lane, track_grip] input.

        `inputs["features"]` arrives with shape (2,) — CustomTritonPackager
        strips the batch dimension per-sample by default (custom_batch_processing
        is left at its default False), matching the model schema's shape=[2].
        """
        import xgboost

        features = np.asarray(inputs["features"], dtype=np.float32).reshape(1, -1)
        dmatrix = xgboost.DMatrix(features)

        speed_cap_cms = float(self._speed_booster.predict(dmatrix)[0])
        caution_buffer_cm = float(self._caution_booster.predict(dmatrix)[0])

        return {
            "settings": np.array([speed_cap_cms, caution_buffer_cm], dtype=np.float32)
        }
