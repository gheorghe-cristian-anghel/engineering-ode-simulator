import numpy as np
import pytest

from models.rl_circuit import (
    analytical_rl,
    simulate_rl,
    steady_state_current,
    time_constant,
)


def test_time_constant():
    """For R = 10 ohms and L = 2 H, tau should be 0.2 seconds."""
    assert time_constant(10, 2) == pytest.approx(0.2)


def test_steady_state_current():
    """Steady-state current should be Vin / R."""
    assert steady_state_current(10, 5) == pytest.approx(0.5)


def test_initial_simulated_current_equals_i0():
    """The simulated current should start at i0."""
    R = 10
    L = 2
    Vin = 5
    i0 = 0.2

    t, current = simulate_rl(R, L, Vin, i0, (0, 1.5), 100)

    assert t[0] == pytest.approx(0)
    assert current[0] == pytest.approx(i0)


def test_analytical_current_at_one_time_constant():
    """At t = tau and i0 = 0, current reaches about 63.2% of steady state."""
    R = 10
    L = 2
    Vin = 5
    i0 = 0
    tau = time_constant(R, L)
    current_ss = steady_state_current(R, Vin)

    current_at_tau = analytical_rl(tau, R, L, Vin, i0)

    assert current_at_tau == pytest.approx((1 - np.exp(-1)) * current_ss)


def test_numerical_and_analytical_solutions_match():
    """Numerical and analytical RL current responses should closely agree."""
    R = 10
    L = 2
    Vin = 5
    i0 = 0

    t, numerical_current = simulate_rl(R, L, Vin, i0, (0, 1.5), 1000)
    analytical_current = analytical_rl(t, R, L, Vin, i0)

    assert numerical_current == pytest.approx(analytical_current, rel=0.01)


def test_invalid_r_raises_value_error():
    """Resistance must be positive."""
    for resistance in [0.0, -1.0]:
        with pytest.raises(ValueError):
            simulate_rl(resistance, 2, 5, 0, (0, 1.5), 100)


def test_invalid_l_raises_value_error():
    """Inductance must be positive."""
    for inductance in [0.0, -1.0]:
        with pytest.raises(ValueError):
            simulate_rl(10, inductance, 5, 0, (0, 1.5), 100)
