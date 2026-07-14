"""Unit tests for pitstop_advisor's synthetic data generator.

Calls `generate_data.fn(...)` (the raw wrapped function) rather than
`generate_data(...)` (the TaskFunction) — going through the TaskFunction
triggers RayTask.pre_run()/post_run(), which starts and tears down a real
local Ray instance per call. That's appropriate for an actual pipeline run,
but far too slow/heavy for a unit test of pure numpy logic.
"""

from __future__ import annotations

import statistics

from examples.pipelines.pitstop_advisor.generate_data import (
    _LANE_A,
    _LANE_B,
    generate_data,
)


def test_shapes_and_ranges():
    """Feature/label rows have the expected count, shape, and value ranges."""
    num_samples = 200
    data = generate_data.fn(num_samples=num_samples, seed=0)

    assert len(data.features) == num_samples
    assert len(data.speed_cap_cms) == num_samples
    assert len(data.caution_buffer_cm) == num_samples

    for row in data.features:
        assert len(row) == 2
        lane, track_grip = row
        assert lane in (_LANE_A, _LANE_B)
        assert 0.0 <= track_grip <= 1.0

    for speed_cap in data.speed_cap_cms:
        assert 50.0 <= speed_cap <= 450.0

    for caution_buffer in data.caution_buffer_cm:
        assert 5.0 <= caution_buffer <= 40.0


def test_reproducible_given_same_seed():
    """Same seed produces byte-identical rows across calls."""
    first = generate_data.fn(num_samples=100, seed=42)
    second = generate_data.fn(num_samples=100, seed=42)

    assert first.features == second.features
    assert first.speed_cap_cms == second.speed_cap_cms
    assert first.caution_buffer_cm == second.caution_buffer_cm


def test_different_seeds_differ():
    """Different seeds produce different synthetic rows."""
    first = generate_data.fn(num_samples=100, seed=1)
    second = generate_data.fn(num_samples=100, seed=2)

    assert first.features != second.features


def test_lane_b_has_lower_speed_cap_and_larger_caution_buffer_on_average():
    """Lane B (inside lane) skews toward a lower speed cap, larger caution buffer."""
    data = generate_data.fn(num_samples=5000, seed=7)

    lane_a_speed, lane_b_speed = [], []
    lane_a_buffer, lane_b_buffer = [], []
    for (lane, _track_grip), speed_cap, caution_buffer in zip(
        data.features, data.speed_cap_cms, data.caution_buffer_cm
    ):
        if lane == _LANE_A:
            lane_a_speed.append(speed_cap)
            lane_a_buffer.append(caution_buffer)
        else:
            lane_b_speed.append(speed_cap)
            lane_b_buffer.append(caution_buffer)

    avg = statistics.fmean
    assert avg(lane_b_speed) < avg(lane_a_speed)
    assert avg(lane_b_buffer) > avg(lane_a_buffer)
