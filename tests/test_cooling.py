import numpy as np
import pytest

from models.cooling import simulate_cooling


def test_initial_temperature_equals_t0():
    """The simulated temperature should start at T0."""
    k = 0.08
    T_env = 22
    T0 = 90

    t, temperature = simulate_cooling(k, T_env, T0, (0, 60), 100)

    assert t[0] == pytest.approx(0)
    assert temperature[0] == pytest.approx(T0)


def test_hot_object_temperature_decreases_toward_t_env():
    """A hot object should cool down toward the environment temperature."""
    k = 0.08
    T_env = 22
    T0 = 90

    _, temperature = simulate_cooling(k, T_env, T0, (0, 60), 200)

    assert temperature[-1] < temperature[0]
    assert temperature[-1] == pytest.approx(T_env, rel=0.1)


def test_temperature_difference_at_one_time_constant():
    """At t = 1/k, the remaining temperature difference should be exp(-1)."""
    k = 0.08
    T_env = 22
    T0 = 90
    tau = 1 / k

    _, temperature = simulate_cooling(k, T_env, T0, (0, tau), 50)

    initial_difference = T0 - T_env
    remaining_difference = temperature[-1] - T_env

    assert remaining_difference == pytest.approx(
        np.exp(-1) * initial_difference,
        rel=0.01,
    )


def test_cold_object_warms_toward_t_env():
    """A cold object should warm up toward the environment temperature."""
    k = 0.08
    T_env = 22
    T0 = 5

    _, temperature = simulate_cooling(k, T_env, T0, (0, 60), 200)

    assert temperature[-1] > temperature[0]
    assert temperature[-1] == pytest.approx(T_env, rel=0.1)
