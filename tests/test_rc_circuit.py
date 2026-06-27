import pytest

from models.rc_circuit import simulate_rc


def test_initial_voltage_equals_v0():
    """The simulated capacitor voltage should start at V0."""
    R = 1000
    C = 0.001
    Vin = 5
    V0 = 2

    t, voltage = simulate_rc(R, C, Vin, V0, (0, 5), 100)

    assert t[0] == pytest.approx(0)
    assert voltage[0] == pytest.approx(V0)


def test_voltage_approaches_vin():
    """After several time constants, the capacitor voltage should be near Vin."""
    R = 1000
    C = 0.001
    Vin = 5
    V0 = 0

    _, voltage = simulate_rc(R, C, Vin, V0, (0, 10), 200)

    assert voltage[-1] == pytest.approx(Vin, rel=0.01)


def test_voltage_at_one_time_constant_is_63_percent_of_vin():
    """At t = R*C and V0 = 0, Vc should be about 63.2% of Vin."""
    R = 1000
    C = 0.001
    Vin = 5
    V0 = 0
    tau = R * C

    _, voltage = simulate_rc(R, C, Vin, V0, (0, tau), 50)

    assert voltage[-1] == pytest.approx(0.632 * Vin, rel=0.01)
