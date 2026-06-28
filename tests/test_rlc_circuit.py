import numpy as np
import pytest

from models.rlc_circuit import (
    damping_ratio,
    natural_frequency,
    simulate_rlc,
)


def test_natural_frequency():
    """For L = 1 H and C = 0.25 F, natural frequency should be 2 rad/s."""
    assert natural_frequency(1, 0.25) == pytest.approx(2)


def test_damping_ratio():
    """For R = 2, L = 1, and C = 0.25, damping ratio should be 0.5."""
    assert damping_ratio(2, 1, 0.25) == pytest.approx(0.5)


def test_initial_capacitor_voltage_matches_vc0():
    """The simulated capacitor voltage should start at Vc0."""
    t, capacitor_voltage, _ = simulate_rlc(2, 1, 0.25, 5, 1, 0, (0, 10), 200)

    assert t[0] == pytest.approx(0)
    assert capacitor_voltage[0] == pytest.approx(1)


def test_initial_current_matches_i0():
    """The simulated current should start at i0."""
    _, _, current = simulate_rlc(2, 1, 0.25, 5, 0, 0.5, (0, 10), 200)

    assert current[0] == pytest.approx(0.5)


def test_final_capacitor_voltage_approaches_vin():
    """With a DC input, capacitor voltage should approach Vin."""
    _, capacitor_voltage, _ = simulate_rlc(2, 1, 0.25, 5, 0, 0, (0, 20), 2000)

    assert capacitor_voltage[-1] == pytest.approx(5, rel=0.01)


def test_final_current_approaches_zero():
    """With a DC input, final current should approach zero."""
    _, _, current = simulate_rlc(2, 1, 0.25, 5, 0, 0, (0, 20), 2000)

    assert current[-1] == pytest.approx(0, abs=0.01)


def test_underdamped_example_overshoots_vin():
    """The underdamped example should overshoot the input voltage."""
    _, capacitor_voltage, _ = simulate_rlc(2, 1, 0.25, 5, 0, 0, (0, 10), 2000)

    assert np.max(capacitor_voltage) > 5


def test_invalid_negative_r_raises_value_error():
    """Resistance must be nonnegative."""
    with pytest.raises(ValueError):
        simulate_rlc(-1, 1, 0.25, 5, 0, 0, (0, 10), 200)


def test_invalid_zero_l_raises_value_error():
    """Inductance must be positive."""
    with pytest.raises(ValueError):
        simulate_rlc(2, 0, 0.25, 5, 0, 0, (0, 10), 200)


def test_invalid_zero_c_raises_value_error():
    """Capacitance must be positive."""
    with pytest.raises(ValueError):
        simulate_rlc(2, 1, 0, 5, 0, 0, (0, 10), 200)
