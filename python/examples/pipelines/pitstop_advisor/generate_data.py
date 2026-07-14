"""Synthetic lap data generation for the Pit Crew Advisor demo model.

Ground truth is a known closed-form function of (lane, track_grip) with
noise added — turns training into an ordinary supervised regression problem
(context -> optimal settings) rather than a full RL/bandit setup, which is
plenty realistic for a demo model while staying simple to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import michelangelo.uniflow.core as uniflow
from michelangelo.uniflow.plugins.ray import RayTask

__all__ = ["GeneratedData", "generate_data"]

# Mirrors go/components/lanerun/controller.go's laneFeatureValue():
# LANE_A -> 0, LANE_B -> 1.
_LANE_A = 0.0
_LANE_B = 1.0


@dataclass
class GeneratedData:
    """Synthetic (lane, track_grip) -> (speed_cap_cms, caution_buffer_cm) rows.

    Plain list[float] fields, not numpy arrays: uniflow's task-boundary codec
    (michelangelo/uniflow/core/codec.py) has no numpy codec, and json-dumping
    an ndarray field fails with `TypeError: Object of type ndarray is not
    JSON serializable`.

    Attributes:
        features: Rows of [lane, track_grip], matching controller.go's
            advisorInputName ("features") input layout exactly.
        speed_cap_cms: Target speed cap in cm/s for each row.
        caution_buffer_cm: Target caution buffer in cm for each row.
    """

    features: list[list[float]] = field(default_factory=list)
    speed_cap_cms: list[float] = field(default_factory=list)
    caution_buffer_cm: list[float] = field(default_factory=list)


@uniflow.task(
    config=RayTask(
        head_cpu=1,
        head_memory="1Gi",
        worker_instances=0,
    ),
)
def generate_data(num_samples: int = 2000, seed: int = 0) -> GeneratedData:
    """Generate synthetic lap data spanning the full track_grip range.

    Args:
        num_samples: Number of synthetic rows to generate.
        seed: RNG seed, for reproducible datasets across pipeline runs.

    Returns:
        GeneratedData with num_samples rows.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    lanes = rng.choice([_LANE_A, _LANE_B], size=num_samples)
    track_grip = rng.uniform(0.0, 1.0, size=num_samples)
    lane_is_b = lanes == _LANE_B

    # Higher grip supports a higher safe speed cap; lane B is the tighter
    # (inside) lane so it gets a lower cap and a larger caution buffer.
    speed_cap = 150.0 + track_grip * 250.0
    speed_cap -= np.where(lane_is_b, 20.0, 0.0)
    speed_cap += rng.normal(0.0, 10.0, size=num_samples)
    speed_cap = np.clip(speed_cap, 50.0, 450.0)

    caution_buffer = 30.0 - track_grip * 20.0
    caution_buffer += np.where(lane_is_b, 5.0, 0.0)
    caution_buffer += rng.normal(0.0, 2.0, size=num_samples)
    caution_buffer = np.clip(caution_buffer, 5.0, 40.0)

    features = np.stack([lanes, track_grip], axis=1)

    return GeneratedData(
        features=features.tolist(),
        speed_cap_cms=speed_cap.tolist(),
        caution_buffer_cm=caution_buffer.tolist(),
    )
